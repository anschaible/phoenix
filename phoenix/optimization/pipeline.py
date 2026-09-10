"""
Inference/optimization pipeline: fit potential + distribution-function parameters
so that the Phoenix surrogate reproduces an observed galaxy (mass, v_rot, sigma
maps) while remaining dynamically self-consistent (i.e. the density implied by
the sampled DF matches the density that sources the assumed potential via the
Poisson equation).
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax

from phoenix.optimization.observables import (
    sample_and_map_particles,
    project_to_sky,
    bin_maps,
    blur_maps,
    generate_edge_on_maps,
    apply_mass_to_light,
)
from phoenix.optimization.poisson_penalty import compute_poisson_penalty
from phoenix.potentials.potentials import nfw_potential, plummer_potential, miyamoto_nagai_potential


# ==============================================================================
# POTENTIAL HELPER
# ==============================================================================

def total_potential_raw(x, y, z, M_halo, a_halo, M_disk, a_disk, b_disk, M_bulge, a_bulge):
    """Same combined potential used everywhere else, but with parameters passed
    positionally instead of via a dict closure — this is the signature
    `compute_poisson_penalty` expects (potential_fn(x, y, z, *params))."""
    return (nfw_potential(x, y, z, M_halo, a_halo) +
            miyamoto_nagai_potential(x, y, z, M_disk, a_disk, b_disk) +
            plummer_potential(x, y, z, M_bulge, a_bulge))

def baryonic_potential_raw(x, y, z, M_halo, a_halo, M_disk, a_disk, b_disk, M_bulge, a_bulge):
    """The disk + bulge (baryonic) part of the combined potential only, with the same
    positional signature as `total_potential_raw` so it is a drop-in for
    `compute_poisson_penalty`.

    This is the correct potential for the Poisson self-consistency check. The tracer
    population sampled from the disk and bulge DFs carries a total mass of
    M_disk + M_bulge, so its density can only ever match the density sourcing the
    *baryonic* components. Comparing it against the total potential additionally
    demands that it reproduce the dark halo's density -- and the halo dominates the
    mass budget (e.g. M_halo = 1e12 against 6e10 in tracers), so that comparison can
    never be satisfied and biases the penalty even at the true parameters.

    `M_halo` and `a_halo` are accepted but unused, to keep the signature identical.
    """
    return (miyamoto_nagai_potential(x, y, z, M_disk, a_disk, b_disk) +
            plummer_potential(x, y, z, M_bulge, a_bulge))


def pot_params_to_tuple(pot_params: dict) -> tuple:
    """Converts a potential parameter dictionary into a tuple for jax.vmap."""
    return (
        pot_params['M_halo'], pot_params['a_halo'],
        pot_params['M_disk'], pot_params['a_disk'], pot_params['b_disk'],
        pot_params['M_bulge'], pot_params['a_bulge']
    )


# ==============================================================================
# LOG-SPACE PARAMETER TRANSFORM
# ==============================================================================
# Optimizing strictly positive parameters spanning many orders of magnitude 
# (e.g., M_halo ~1e12 vs. a_disk ~2.0) is numerically unstable. By transforming 
# everything into log-space, unconstrained optimizers (like Adam) can safely 
# update them while automatically enforcing positivity.
def params_to_log(pot_params: dict, disk_df_params: dict, bulge_df_params: dict) -> dict:
    return {
        'pot': {k: jnp.log(jnp.asarray(v, dtype=jnp.float32)) for k, v in pot_params.items()},
        'disk': {k: jnp.log(jnp.asarray(v, dtype=jnp.float32)) for k, v in disk_df_params.items()},
        'bulge': {k: jnp.log(jnp.asarray(v, dtype=jnp.float32)) for k, v in bulge_df_params.items()},
    }


def log_to_params(params_log: dict) -> tuple:
    pot_params = {k: jnp.exp(v) for k, v in params_log['pot'].items()}
    disk_df_params = {k: jnp.exp(v) for k, v in params_log['disk'].items()}
    bulge_df_params = {k: jnp.exp(v) for k, v in params_log['bulge'].items()}
    return pot_params, disk_df_params, bulge_df_params


# Physical bounds prevent parameters from drifting into unphysical or degenerate 
# regimes (e.g., DF power laws growing infinitely) which causes float32 overflow.
DEFAULT_PARAM_BOUNDS = {
    'pot': {
        'M_halo': (1e9, 1e13), 'a_halo': (0.5, 100.0),
        'M_disk': (1e8, 5e11), 'a_disk': (0.2, 20.0), 'b_disk': (0.02, 3.0),
        'M_bulge': (1e7, 5e11), 'a_bulge': (0.05, 10.0),
    },
    'disk': {
        'R0': (1.0, 30.0), 'Rd': (0.5, 20.0), 'Sigma0': (1.0, 1e5),
        'RsigR': (0.5, 30.0), 'RsigZ': (0.5, 30.0),
        'sigmaR0_R0': (1.0, 300.0), 'sigmaz0_R0': (1.0, 300.0),
        'L0': (0.5, 200.0), 'Rinit_for_Rc': (1.0, 30.0),
    },
    'bulge': {
        'N0_spheroid': (1e6, 1e13), 'J0_spheroid': (1.0, 2000.0),
        'Gamma_spheroid': (0.0, 2.8), 'Beta_spheroid': (3.2, 12.0),
        'eta_spheroid': (0.3, 5.0),
    },
}


def log_bounds_tree(params_log: dict, param_bounds: dict = None) -> tuple:
    """Builds identical pytrees containing the log-space lower and upper bounds."""
    param_bounds = param_bounds or DEFAULT_PARAM_BOUNDS
    log_lo, log_hi = {}, {}
    for group, group_log in params_log.items():
        log_lo[group] = {}
        log_hi[group] = {}
        for k in group_log:
            lo_v, hi_v = param_bounds[group][k]
            log_lo[group][k] = jnp.log(lo_v)
            log_hi[group][k] = jnp.log(hi_v)
    return log_lo, log_hi


# ==============================================================================
# SYNTHETIC OBSERVATION
# ==============================================================================
def make_observation(
    mapper, 
    pot_params: dict, 
    disk_df_params: dict, 
    bulge_df_params: dict,
    N_disk: int = 100_000, 
    N_bulge: int = 100_000, 
    grid_size: int = 30,
    extent_x: float = 15.0, 
    extent_z: float = 10.0, 
    prng_seed: int = 0,
    inclination_deg: float = 90.0,
    ml_ratios=None,
    sampler: str = "soft",
) -> dict:
    """
    Generates a pristine synthetic 'observed' galaxy from a ground-truth parameter set.
    
    This acts as a stand-in for real telescopic data. Once real data is available,
    replace the output of this function with actual mass/v_rot/sigma maps binned 
    on the exact same grid setup.
    
    Args:
        mapper: The Phoenix neural network surrogate mapping actions to phase space.
        pot_params (dict): Ground truth gravitational potential parameters.
        disk_df_params (dict): Ground truth disk DF parameters.
        bulge_df_params (dict): Ground truth bulge DF parameters.
        N_disk (int, optional): Number of disk tracer particles. Defaults to 100,000.
        N_bulge (int, optional): Number of bulge tracer particles. Defaults to 100,000.
        grid_size (int, optional): Resolution of the output maps. Defaults to 30.
        extent_x (float, optional): Physical half-width of the spatial grid. Defaults to 15.0.
        extent_z (float, optional): Physical half-height of the spatial grid. Defaults to 10.0.
        prng_seed (int, optional): Seed for particle generation. Defaults to 0.
        ml_ratios (tuple, optional): `(Upsilon_disk, Upsilon_bulge)` mass-to-light
            ratios in solar units. When given, the mock is a surface-BRIGHTNESS map
            with LIGHT-weighted kinematics rather than a mass map -- the quantity a
            real integral-field survey delivers. The same `ml_ratios` must then be
            passed to `make_loss_fn`/`fit`, or the model and the data are not
            measuring the same thing. See `observables.apply_mass_to_light`.
        sampler (str, optional): 'soft' (default, the original behaviour) or
            'importance'; see `observables.sample_and_map_particles`. This MUST match
            the sampler the subsequent `fit`/`make_loss_fn` uses. The two draw
            genuinely different populations from the same DF -- under 'soft' the DF
            parameters barely reach the tracers at all -- so a mock built with one and
            fitted with the other has a mismatch at the true parameters that no
            parameter choice can remove.
        
    Returns:
        dict: JAX arrays containing the pristine mock observation maps ('mass', 'v_rot', 'sigma').
            'mass' holds surface luminosity when `ml_ratios` is given.
    """
    maps = generate_edge_on_maps(
        mapper, 
        pot_params, 
        disk_df_params, 
        bulge_df_params,
        N_disk=N_disk, 
        N_bulge=N_bulge, 
        grid_size=grid_size,
        extent_x=extent_x, 
        extent_z=extent_z, 
        prng_seed=prng_seed,
        inclination_deg=inclination_deg,
        ml_ratios=ml_ratios,
        sampler=sampler,
    )
    
    # Return as explicitly cast JAX arrays
    return {k: jnp.asarray(v) for k, v in maps.items()}


# ==============================================================================
# LOSS: DATA FIT + DYNAMICAL SELF-CONSISTENCY (POISSON EQUATION)
# ==============================================================================
from jax.flatten_util import ravel_pytree

# ==============================================================================
# LOSS: DATA FIT + DYNAMICAL SELF-CONSISTENCY (POISSON EQUATION)
# ==============================================================================
def data_fit_loss(model_maps: dict, obs_maps: dict, mass_floor: float = 1e-3, mask=None,
                  pixel_weights=None, weight_mass: bool = False):
    """
    Computes the Mean Squared Error (MSE) between the model and observation maps.
    
    Evaluates only on pixels where the observed mass exceeds `mass_floor` to 
    mimic realistic telescope signal-to-noise thresholds. Mass is compared in 
    log-space, while velocities and dispersions are compared in normalized 
    linear space.

    Args:
        model_maps (dict): Dictionary of model maps ('mass', 'v_rot', 'sigma').
        obs_maps (dict): Dictionary of observed maps ('mass', 'v_rot', 'sigma').
        mass_floor (float, optional): Minimum mass threshold for a pixel to be
            included in the loss calculation. Defaults to 1e-3.
        mask (Array, optional): Explicit boolean mask of pixels to evaluate. Pass this
            whenever `obs_maps` is a *blurred* version of the real observation: blurring
            spreads flux into pixels that were never observed, so a mask derived from
            the blurred map would score the model against fabricated data. Defaults to
            None, which derives the mask from `obs_maps` as before.
        pixel_weights (Array, optional): Per-pixel weights, same shape as the maps.
            The losses become weighted means instead of plain means. Use this when the
            features that matter occupy few pixels: an unweighted mean over a large
            footprint dilutes them to irrelevance. For NGC 5010 the rotation gradient
            and the central dispersion peak live in the |z| < 0.5 kpc midplane, ~15%
            of the 558 observed cells, so at equal weight the optimizer trades them
            away for a marginally better fit off-plane. `None` (default) weights every
            pixel equally, reproducing the previous behaviour.

    Returns:
        tuple: (mass_loss, vrot_loss, sigma_loss)
    """
    if mask is None:
        mask = obs_maps['mass'] > mass_floor

    w = jnp.where(mask, 1.0 if pixel_weights is None else pixel_weights, 0.0)
    # Prevent division by zero if the mask is entirely empty
    n = jnp.maximum(jnp.sum(w), 1e-12)

    # Mass keeps flat (mask-only) weights unless explicitly asked otherwise.
    w_m = w if (weight_mass or pixel_weights is None) else jnp.where(mask, 1.0, 0.0)
    n_m = jnp.maximum(jnp.sum(w_m), 1e-12)

    # Log-space mass loss (masked mean: sum over detected pixels / #detected pixels)
    log_mass_res = jnp.log10(model_maps['mass'] + mass_floor) - jnp.log10(obs_maps['mass'] + mass_floor)
    mass_loss = jnp.sum(w_m * log_mass_res**2) / n_m

    # Normalized linear-space kinematic losses (typical scales: 200 km/s and 100 km/s)
    vrot_res = (model_maps['v_rot'] - obs_maps['v_rot']) / 200.0
    vrot_loss = jnp.sum(w * vrot_res**2) / n

    sigma_res = (model_maps['sigma'] - obs_maps['sigma']) / 100.0
    sigma_loss = jnp.sum(w * sigma_res**2) / n

    return mass_loss, vrot_loss, sigma_loss


def feature_weight_map(grid_size: int, extent_x: float, extent_z: float,
                       midplane_z: float = 0.6, w_midplane: float = 4.0,
                       center_x: float = 2.5, w_center: float = 2.0):
    """
    Builds a per-pixel weight map that emphasizes the midplane and the central region.

    The two observational signatures of a rotation-supported disk with a hot core --
    the steep central velocity gradient and the central-to-disk dispersion contrast --
    both live in a thin midplane strip near the centre. That strip is a small minority
    of the observed footprint, so an unweighted mean lets the optimizer trade it away:
    with equal weights the disk scale length ran to its bound rather than becoming
    concentrated enough to produce the dispersion contrast. Up-weighting the strip
    makes the objective actually reward the features being modelled.

    Weights multiply, so a central midplane pixel gets `w_midplane * w_center`.

    Args:
        grid_size (int): Map resolution (must match the maps being compared).
        extent_x, extent_z (float): Grid half-width/half-height (kpc).
        midplane_z (float): Half-thickness of the up-weighted midplane strip (kpc).
        w_midplane (float): Weight applied inside that strip.
        center_x (float): Half-width of the up-weighted central region (kpc).
        w_center (float): Weight applied inside it.

    Returns:
        Array: (grid_size, grid_size) weights, 1.0 outside the emphasized regions.
    """
    dx = 2.0 * extent_x / grid_size
    dz = 2.0 * extent_z / grid_size
    xc = jnp.linspace(-extent_x + dx / 2, extent_x - dx / 2, grid_size)
    zc = jnp.linspace(-extent_z + dz / 2, extent_z - dz / 2, grid_size)
    w = jnp.ones((grid_size, grid_size))
    w = w * jnp.where(jnp.abs(zc)[:, None] <= midplane_z, w_midplane, 1.0)
    w = w * jnp.where(jnp.abs(xc)[None, :] <= center_x, w_center, 1.0)
    return w


def make_loss_fn(
    mapper, 
    obs_maps: dict,
    N_disk: int = 5_000, 
    N_bulge: int = 5_000, 
    grid_size: int = 16,
    extent_x: float = 15.0, 
    extent_z: float = 10.0, 
    prng_seed: int = 123,
    loss_weights: tuple = (1.0, 1.0, 1.0, 0.1),
    poisson_kwargs: dict = None,
    spheroid_corotation: float = 0.5,
    inclination_deg: float = 90.0,
    sampler: str = "soft",
    obs_bandwidth: float = None,
    reg_weight: float = 0.0,
    reg_center_log: dict = None,
    mass_floor: float = 1e-3,
    pixel_weights=None,
    weight_mass: bool = False,
    extra_mask=None,
    ml_ratios=None,
):
    """
    Builds a JAX-compatible, deterministic loss function closure for optimization.

    Combines observational data-fit losses (mass, velocity, dispersion) with a 
    physics-based Poisson self-consistency penalty.

    Args:
        mapper: The Phoenix neural network surrogate.
        obs_maps (dict): The target observed maps to fit against.
        N_disk (int, optional): Number of disk tracer particles. Defaults to 5_000.
        N_bulge (int, optional): Number of bulge tracer particles. Defaults to 5_000.
        grid_size (int, optional): Resolution of the 2D evaluation grid. Defaults to 16.
        extent_x (float, optional): Grid half-width (kpc). Defaults to 15.0.
        extent_z (float, optional): Grid half-height (kpc). Defaults to 10.0.
        prng_seed (int, optional): Fixed random seed for particle sampling to ensure 
            deterministic gradients. Defaults to 123.
        loss_weights (tuple, optional): Weights for (mass, v_rot, sigma, poisson). 
            Defaults to (1.0, 1.0, 1.0, 0.1).
        poisson_kwargs (dict, optional): Extra arguments for the Poisson penalty.
        spheroid_corotation (float, optional): Bulge rotation fraction. Defaults to 0.5.
        inclination_deg (float, optional): Viewing inclination in degrees at which the
            model is projected before binning. 90 (default) is edge-on. Note that in a
            kinematic-only fit this is close to degenerate with the total mass, since
            v_los scales as sin(i); see `observables.project_to_sky`.
        sampler (str, optional): 'soft' (default, original behaviour) or
            'importance'. With 'soft' the DF parameters barely reach the sampled
            tracers, so a fit can only adjust the potential; see
            `observables.sample_and_map_particles`.
        obs_bandwidth (float, optional): The KDE bandwidth `obs_maps` was binned
            with (kpc). Set this whenever the caller anneals `soft_bin_h`: the
            observation is then blurred to the model's current resolution so the
            objective stays minimized at the true parameters at every annealing
            stage. Without it, only the model is smoothed and the wide-bandwidth
            stages pull the fit away from the truth. `None` (default) disables the
            correction, reproducing the previous behaviour.
        ml_ratios (tuple, optional): `(Upsilon_disk, Upsilon_bulge)` mass-to-light
            ratios in solar units, applied to the tracer weights before binning so the
            model produces a surface-BRIGHTNESS map with LIGHT-weighted kinematics.
            Set this when `obs_maps` is photometric (a real IFU cube, or a mock built
            with the same argument); leaving it `None` against luminosity data
            compares a mass map to a light map and the log-mass term absorbs the
            mismatch as a spurious offset.

            Only the observables are converted. The Poisson penalty keeps the MASS
            weights, because self-consistency is the statement that the tracers'
            *mass* sources the potential -- light does not gravitate.

            Note that a spatially CONSTANT Upsilon (Upsilon_disk == Upsilon_bulge)
            changes nothing: the mass term is a log-space residual, so a common factor
            cancels between model and data, and the kinematic maps are ratios of
            weighted sums, so it cancels there too. The fit only sees a difference
            when the two components have different Upsilon, which is what re-weights
            disk against bulge.

    Returns:
        Callable: A function `loss_fn(params_log, soft_bin_h)` returning `(loss, aux_dict)`.
    """
    poisson_kwargs = poisson_kwargs or {}
    w_mass, w_vrot, w_sigma, w_poisson = loss_weights

    use_reg = reg_weight > 0 and reg_center_log is not None
    if use_reg:
        reg_center_vec, _ = ravel_pytree(reg_center_log)

    # Mask of pixels that actually contain observations, taken from the UNBLURRED maps
    # and held fixed for the whole run. This matters as soon as `obs_bandwidth` is used:
    # blurring the observation to the model's resolution spreads flux into pixels that
    # were never observed, so a mask re-derived from the blurred map grows with the
    # bandwidth and the fit gets scored against fabricated data. On a real GECKOS
    # pointing covering 473 of 1600 cells, the wide-bandwidth stages inflated the mask
    # to all 1600 -- i.e. ~1100 invented pixels -- and made the loss vary ~8x
    # non-monotonically with bandwidth at fixed parameters.
    data_mask = obs_maps['mass'] > mass_floor
    # Drop caller-supplied bad pixels (e.g. a dust lane) from the fit entirely.
    if extra_mask is not None:
        data_mask = data_mask & jnp.logical_not(jnp.asarray(extra_mask))

    def loss_fn(params_log, soft_bin_h=None):
        pot_params, disk_df_params, bulge_df_params = log_to_params(params_log)

        # Generate fully differentiable phase space population
        x, y, z, vx, vy, vz, w = sample_and_map_particles(
            mapper, pot_params, disk_df_params, bulge_df_params,
            N_disk=N_disk, N_bulge=N_bulge, prng_seed=prng_seed,
            spheroid_corotation=spheroid_corotation, sampler=sampler,
        )
        
        # 1. Observational Data Fit
        # The maps are built from LIGHT weights when `ml_ratios` is set; `w` itself
        # stays the mass weighting and is what the Poisson penalty below uses.
        w_obs = apply_mass_to_light(w, N_disk, ml_ratios)
        x_sky, z_sky, v_los = project_to_sky(x, y, z, vx, vy, vz, inclination_deg)
        model_maps = bin_maps(
            x_sky, z_sky, v_los, w_obs, 
            grid_size=grid_size, 
            extent_x=extent_x,
            extent_z=extent_z, 
            soft_bin_h=soft_bin_h
        )
        # Match the observation's resolution to the model's current bandwidth so
        # that bandwidth annealing compares like with like (see blur_maps).
        target_maps = obs_maps
        if obs_bandwidth is not None and soft_bin_h is not None:
            blur_h = jnp.sqrt(jnp.maximum(soft_bin_h**2 - obs_bandwidth**2, 0.0))
            target_maps = blur_maps(
                obs_maps, blur_h, grid_size=grid_size,
                extent_x=extent_x, extent_z=extent_z,
            )

        mass_loss, vrot_loss, sigma_loss = data_fit_loss(
            model_maps, target_maps, mass_floor=mass_floor, mask=data_mask,
            pixel_weights=pixel_weights, weight_mass=weight_mass)
        
        # 2. Physical Self-Consistency (Poisson Penalty)
        # Compared against the BARYONIC potential: the tracers carry M_disk + M_bulge,
        # so they cannot reproduce the dark halo's density (see baryonic_potential_raw).
        poisson_penalty = compute_poisson_penalty(
            x, y, z, w,
            baryonic_potential_raw,
            pot_params_to_tuple(pot_params),
            **poisson_kwargs
        )

        # 3. Combine Core Losses
        loss = (w_mass * mass_loss +
                w_vrot * vrot_loss +
                w_sigma * sigma_loss +
                w_poisson * poisson_penalty)

        # 4. Optional Tikhonov (ridge) prior toward `reg_center_log`
        reg = 0.0
        if use_reg:
            u_vec, _ = ravel_pytree(params_log)
            reg = reg_weight * jnp.mean((u_vec - reg_center_vec) ** 2)
            loss = loss + reg

        # Package auxiliary metrics for telemetry
        aux = {
            'mass_loss': mass_loss,
            'vrot_loss': vrot_loss,
            'sigma_loss': sigma_loss,
            'poisson_penalty': poisson_penalty,
            'reg': reg,
            'model_maps': model_maps,
        }
        return loss, aux

    return loss_fn

# ==============================================================================
# OPTIMIZATION LOOP
# ==============================================================================
def fit(
    mapper, 
    obs_maps: dict,
    init_pot_params: dict, 
    init_disk_df_params: dict, 
    init_bulge_df_params: dict,
    N_disk: int = 5_000, 
    N_bulge: int = 5_000, 
    grid_size: int = 16,
    extent_x: float = 15.0, 
    extent_z: float = 10.0, 
    prng_seed: int = 123,
    loss_weights: tuple = (1.0, 1.0, 1.0, 0.1), 
    poisson_kwargs: dict = None,
    learning_rate: float = 0.02, 
    n_steps: int = 300, 
    grad_clip_norm: float = 1.0,
    param_bounds: dict = None,
    anneal_bandwidth: tuple = None,
    spheroid_corotation: float = 0.5,
    inclination_deg: float = 90.0,
    sampler: str = "soft",
    obs_bandwidth: float = None,
    frozen_params: tuple = (),
    reg_weight: float = 0.0,
    reg_center_log: dict = None,
    anneal_hold_frac: float = 0.35,
    anneal_min_blur_pixels: float = 0.5,
    pixel_weights=None,
    weight_mass: bool = False,
    extra_mask=None,
    ml_ratios=None,
    mass_floor: float = 1e-3,
):
    """
    Fits the gravitational potential and distribution function parameters to 
    observed maps using gradient descent (Adam optimizer).

    Optimization is performed entirely in log-space to ensure parameters remain 
    strictly positive and well-conditioned. Two stabilizers are applied per step:
      1. Gradient clipping (global norm) to prevent catastrophic overshoots on 
         steep loss surfaces.
      2. Parameter clipping (via `param_bounds`) to prevent unconstrained "sloppy" 
         parameters from drifting to infinity and causing float32 overflow.

    Args:
        mapper: The Phoenix neural network surrogate.
        obs_maps (dict): The target observed maps ('mass', 'v_rot', 'sigma').
        init_pot_params (dict): Initial guess for potential parameters.
        init_disk_df_params (dict): Initial guess for disk DF parameters.
        init_bulge_df_params (dict): Initial guess for bulge DF parameters.
        N_disk (int, optional): Tracer particles for the disk. Defaults to 5_000.
        N_bulge (int, optional): Tracer particles for the bulge. Defaults to 5_000.
        grid_size (int, optional): Resolution of the 2D evaluation grids. Defaults to 16.
        extent_x (float, optional): Grid half-width (kpc). Defaults to 15.0.
        extent_z (float, optional): Grid half-height (kpc). Defaults to 10.0.
        prng_seed (int, optional): Fixed random seed for determinism. Defaults to 123.
        loss_weights (tuple, optional): Weights for (mass, v_rot, sigma, poisson). 
            Defaults to (1.0, 1.0, 1.0, 0.1).
        poisson_kwargs (dict, optional): Extra arguments for the Poisson penalty.
        learning_rate (float, optional): Adam optimizer learning rate. Defaults to 0.02.
        n_steps (int, optional): Total optimization iterations. Defaults to 300.
        grad_clip_norm (float, optional): Maximum gradient norm. Defaults to 1.0.
        param_bounds (dict, optional): Pytree of (min, max) physical bounds.
        anneal_bandwidth (tuple, optional): (h_start, h_end) in kpc. Enables 
            coarse-to-fine optimization by log-linearly decaying the KDE bandwidth. 
            Prevents flat gradients when the initial model is far from the data. 
            Defaults to None (fixed bandwidth).
        spheroid_corotation (float, optional): Bulge rotation fraction. Defaults to 0.5.
        obs_bandwidth (float, optional): The KDE bandwidth `obs_maps` was binned
            with (kpc). Strongly recommended together with `anneal_bandwidth`: it
            blurs the observation to the model's current resolution so each
            annealing stage compares maps at matched resolution. Without it the
            wide-bandwidth stages optimize a biased objective whose minimum is not
            the true parameters. Note there is no point annealing below
            `obs_bandwidth` — the data carries no information on finer scales.
        frozen_params (tuple, optional): Names of parameters to hold at their
            initial values (their gradients are zeroed). Use this for parameters
            the observables cannot constrain: because the tracer weights are
            renormalized (`w / sum(w) * M_disk`), the DF amplitudes `Sigma0` and
            `N0_spheroid` cancel exactly and have zero gradient, as do `L0` and
            `Rinit_for_Rc` for these maps. Fitting them only lets them drift to
            their bounds and inflates any parameter-recovery metric.
        reg_weight (float, optional): Weight of a Tikhonov (ridge) prior in log-space,
            `reg_weight * mean((log p - log p_ref)**2)`, pulling the parameters toward
            `reg_center_log`. This is a MAP prior and is mainly useful on real data:
            parameters the observables barely constrain (bulge DF shape, disk
            thickness) otherwise drift to their bounds while hardly changing the fit,
            which produces a map-matching but unphysical model. Data-constrained
            parameters are almost unaffected. 0 (default) disables it.
        reg_center_log (dict, optional): The `params_log` pytree the prior pulls
            toward. Defaults to the (physically motivated) initial guess.
        anneal_hold_frac (float, optional): Fraction of the run spent held at
            `anneal_bandwidth[1]` instead of annealing all the way down to it. Only
            active when `obs_bandwidth` is set. This exists because matching the
            observation's resolution requires blurring it by
            `sqrt(h**2 - obs_bandwidth**2)`, and `blur_maps` works on the binned grid,
            so it cannot represent a blur much narrower than a pixel: the kernel
            collapses toward the identity, the data stops being blurred while the
            model's KDE genuinely is smoother, and a systematic mismatch appears that
            does NOT vanish at the true parameters. Measured on the mock (2.14 kpc
            pixels, obs_bandwidth 0.536), the data-fit loss at the *true* parameters
            rises from a 0.013 floor to 0.35 when the required blur is ~0.17 pixels,
            then drops to exactly 0 once the blur reaches 0. Crawling through that band
            therefore optimizes a corrupted objective and injects noisy gradients.
            Annealing to a well-resolved bandwidth and then jumping straight to the
            final one skips it. Set 0 to restore a plain log-linear schedule.
        anneal_min_blur_pixels (float, optional): The switch point, in pixels: the ramp
            stops once the required blur would fall below this many pixels. Defaults to
            0.5, comfortably above the regime where the discrete kernel degrades.
        ml_ratios (tuple, optional): `(Upsilon_disk, Upsilon_bulge)` mass-to-light
            ratios, forwarded to `make_loss_fn`. Pass this whenever `obs_maps` is
            photometric rather than a mass map. See `make_loss_fn`.
        mass_floor (float, optional): Detection floor for the first map, in that map's
            own units, forwarded to `make_loss_fn`. It sets both which pixels enter the
            loss and the softening inside the log-space residual. The default 1e-3 is
            an ABSOLUTE value, which is only meaningful for a mass map: switching to
            luminosity rescales the whole map by 1/Upsilon and the same absolute floor
            then cuts at a different physical depth. Pass a floor scaled to the map
            (e.g. `1e-4 * obs_maps['mass'].max()`) whenever mass- and light-based fits
            are being compared, so both see the same footprint.

    Returns:
        dict: The final optimized parameters and full training history:
            - 'pot_params': Final potential parameters.
            - 'disk_df_params': Final disk parameters.
            - 'bulge_df_params': Final bulge parameters.
            - 'history': Per-iteration telemetry (losses and parameter states).
    """
    
    # Default the prior's anchor to the (physically motivated) initial guess.
    if reg_weight > 0 and reg_center_log is None:
        reg_center_log = params_to_log(init_pot_params, init_disk_df_params,
                                       init_bulge_df_params)

    # 1. Initialize Loss Function
    loss_fn = make_loss_fn(
        mapper, obs_maps, N_disk=N_disk, N_bulge=N_bulge, grid_size=grid_size,
        extent_x=extent_x, extent_z=extent_z, prng_seed=prng_seed,
        loss_weights=loss_weights, poisson_kwargs=poisson_kwargs,
        spheroid_corotation=spheroid_corotation, inclination_deg=inclination_deg,
        sampler=sampler, obs_bandwidth=obs_bandwidth,
        reg_weight=reg_weight, reg_center_log=reg_center_log,
        pixel_weights=pixel_weights, weight_mass=weight_mass, extra_mask=extra_mask,
        ml_ratios=ml_ratios, mass_floor=mass_floor,
    )

    # 2. Setup KDE Bandwidth Annealing Schedule
    if anneal_bandwidth is not None:
        h_start, h_end = anneal_bandwidth

        # Decide whether to skip the band where the observation's required blur is
        # sub-pixel and `blur_maps` therefore under-blurs it (see anneal_hold_frac).
        h_switch = None
        if obs_bandwidth is not None and anneal_hold_frac > 0:
            pixel = max(2.0 * extent_x / grid_size, 2.0 * extent_z / grid_size)
            # bandwidth whose required blur is exactly anneal_min_blur_pixels pixels
            h_candidate = float(np.sqrt(obs_bandwidth**2
                                        + (anneal_min_blur_pixels * pixel) ** 2))
            # only useful if it actually lies inside the annealing range
            if h_end < h_candidate < h_start:
                h_switch = h_candidate

        if h_switch is None:
            # Plain log-linear ramp (also the path when no blur matching is in use).
            bandwidth_schedule = jnp.exp(
                jnp.linspace(jnp.log(h_start), jnp.log(h_end), n_steps))
        else:
            n_hold = max(1, int(round(anneal_hold_frac * n_steps)))
            n_ramp = max(1, n_steps - n_hold)
            ramp = jnp.exp(jnp.linspace(jnp.log(h_start), jnp.log(h_switch), n_ramp))
            # Jump straight to the final bandwidth: there the required blur is zero,
            # so the objective is exact, and the remaining steps refine against it.
            hold = jnp.full((n_steps - n_ramp,), h_end)
            bandwidth_schedule = jnp.concatenate([ramp, hold])
    else:
        bandwidth_schedule = None

    # 3. Setup Log-Space Optimization State
    params_log = params_to_log(init_pot_params, init_disk_df_params, init_bulge_df_params)
    log_lo, log_hi = log_bounds_tree(params_log, param_bounds)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params_log)

    # 4. JIT-Compiled Optimizer Step
    @jax.jit
    def step(params_log, opt_state, soft_bin_h):
        # Forward pass & Gradients
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_log, soft_bin_h)

        # Sanitize non-finite gradient entries. Partway through a run the
        # optimizer can reach a parameter region where the forward loss is still
        # finite but a boundary op in the sampling/binning graph produces a
        # NaN/inf gradient for a few parameters. Left untouched, that entry flows
        # through clip_by_global_norm/Adam (a single inf makes the global norm inf,
        # so every entry becomes 0*inf=NaN) and the parameter clip below preserves
        # it (clip(NaN)=NaN) — one bad gradient therefore poisons the parameters
        # and every subsequent step, which is how an annealed fit suddenly
        # "breaks" and reports NaN for the rest of the run. Zeroing only the
        # offending entries drops their contribution for that step while the
        # finite components keep driving the descent.
        #
        # NOTE: this is not specific to a small KDE bandwidth. Empirically the
        # blowup depends on which parameter region the trajectory reaches, not on
        # the bandwidth value: annealing to h_end=1.0 (sub-pixel) can be clean
        # while h_end=1.5 (larger than a pixel) fails earlier. Flooring the
        # bandwidth therefore does not reliably prevent it, and costs resolution.
        n_nan = sum(jnp.sum(jnp.logical_not(jnp.isfinite(g)))
                    for grp in grads.values() for g in grp.values())
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
        )

        # Hold requested parameters fixed by zeroing their gradients.
        if frozen_params:
            grads = {
                group: {
                    k: (jnp.zeros_like(v) if k in frozen_params else v)
                    for k, v in group_grads.items()
                }
                for group, group_grads in grads.items()
            }

        # Apply Optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params_log)
        params_log = optax.apply_updates(params_log, updates)

        # Enforce physical boundaries
        params_log = jax.tree_util.tree_map(jnp.clip, params_log, log_lo, log_hi)

        return params_log, opt_state, loss, aux, n_nan

    # 5. Training Loop
    history = {
        'loss': [], 'mass_loss': [], 'vrot_loss': [], 'sigma_loss': [], 'poisson_penalty': [],
        'reg': [], 'pot_params': [], 'disk_df_params': [], 'bulge_df_params': [],
    }

    nan_steps = 0
    nan_grads_total = 0
    for i in range(n_steps):
        # Fetch current bandwidth if annealing, otherwise pass None
        soft_bin_h = None if bandwidth_schedule is None else bandwidth_schedule[i]
        
        # Execute Step
        params_log, opt_state, loss, aux, n_nan = step(params_log, opt_state, soft_bin_h)
        n_nan = int(n_nan)
        if n_nan:
            nan_steps += 1
            nan_grads_total += n_nan
        
        # Convert parameters back to linear space for logging
        pot_params, disk_df_params, bulge_df_params = log_to_params(params_log)
        
        # Record Telemetry
        history['loss'].append(float(loss))
        history['mass_loss'].append(float(aux['mass_loss']))
        history['vrot_loss'].append(float(aux['vrot_loss']))
        history['sigma_loss'].append(float(aux['sigma_loss']))
        history['poisson_penalty'].append(float(aux['poisson_penalty']))
        history['reg'].append(float(aux['reg']))
        
        history['pot_params'].append({k: float(v) for k, v in pot_params.items()})
        history['disk_df_params'].append({k: float(v) for k, v in disk_df_params.items()})
        history['bulge_df_params'].append({k: float(v) for k, v in bulge_df_params.items()})

    # 6. Extract Final Parameters
    final_pot_params, final_disk_df_params, final_bulge_df_params = log_to_params(params_log)

    # Non-finite gradients are zeroed above so a single bad step cannot destroy the
    # run, but that silently FREEZES the affected parameters. If it happens on most
    # steps the result looks converged while some parameters never moved from their
    # initial values at all, so it has to be surfaced rather than swallowed.
    if nan_steps:
        frac = 100.0 * nan_steps / max(n_steps, 1)
        msg = (f"non-finite gradients on {nan_steps}/{n_steps} steps ({frac:.0f}%); "
               f"{nan_grads_total} parameter-gradients were zeroed. Affected "
               f"parameters are held at whatever value they had, so a 'converged' "
               f"result here may simply be the initial guess.")
        if frac > 10.0:
            warnings.warn("fit: " + msg, RuntimeWarning, stacklevel=2)
        else:
            print(f"    [fit] note: {msg}")

    return {
        'pot_params': final_pot_params,
        'disk_df_params': final_disk_df_params,
        'bulge_df_params': final_bulge_df_params,
        'history': history,
        'nan_steps': nan_steps,
        'nan_grads_total': nan_grads_total,
    }


# ==============================================================================
# SELF-CONSISTENT GROUND TRUTH
# ==============================================================================
def make_self_consistent_truth(
    mapper,
    pot_params: dict,
    disk_df_params: dict,
    bulge_df_params: dict,
    tune_params: tuple = ('M_disk', 'a_disk', 'b_disk', 'M_bulge', 'a_bulge'),
    N_disk: int = 6_000,
    N_bulge: int = 6_000,
    prng_seed: int = 0,
    learning_rate: float = 0.02,
    n_steps: int = 200,
    grad_clip_norm: float = 1.0,
    param_bounds: dict = None,
    poisson_kwargs: dict = None,
    spheroid_corotation: float = 0.5,
    sampler: str = "soft",
):
    """
    Adjusts selected parameters until the model is dynamically self-consistent, i.e.
    until the density of the DF-sampled tracer population reproduces the density that
    sources the baryonic potential via the Poisson equation.

    Why this is needed: choosing `pot_params` and the DF parameters independently does
    NOT give a self-consistent galaxy. The resulting model has a real Poisson residual,
    so a fit that includes the self-consistency penalty is pulled away from those
    parameters -- the "truth" is not a minimum of the objective. Generating the mock
    observation from the output of this function removes that inconsistency, which is
    what makes it legitimate to switch the physics term on during fitting.

    Only the *baryonic* components are constrained (see `baryonic_potential_raw`): the
    tracers carry M_disk + M_bulge, so they cannot and should not be asked to
    reproduce the dark halo's density. The halo parameters are therefore left alone
    by default.

    IMPORTANT: pass the SAME `poisson_kwargs` here that you pass to `fit`. Self
    consistency is only defined relative to a particular penalty definition (grid,
    bandwidth, masking, whether the analytic density is kernel-matched), so tuning
    against one definition and fitting against another reintroduces the bias.

    Args:
        mapper: The Phoenix neural network surrogate.
        pot_params (dict): Starting potential parameters.
        disk_df_params (dict): Starting disk DF parameters.
        bulge_df_params (dict): Starting bulge DF parameters.
        tune_params (tuple, optional): Names of the parameters allowed to move. The
            default adjusts only the baryonic potential (the disk and bulge mass and
            shape). Add DF shape parameters (e.g. 'Rd', 'RsigZ', 'sigmaz0_R0') to give
            the solver more freedom if the residual plateaus too high.
        N_disk, N_bulge (int, optional): Tracer counts used to evaluate the density.
            Use the same values as the subsequent fit so the shot noise matches.
        prng_seed (int, optional): Sampling seed; keep it equal to the fit's seed.
        learning_rate (float, optional): Adam learning rate. Defaults to 0.02.
        n_steps (int, optional): Number of iterations. Defaults to 200.
        grad_clip_norm (float, optional): Gradient clipping norm. Defaults to 1.0.
        param_bounds (dict, optional): Physical bounds pytree.
        poisson_kwargs (dict, optional): Passed to `compute_poisson_penalty`.
        spheroid_corotation (float, optional): Bulge rotation fraction.
        sampler (str, optional): 'soft' (default) or 'importance'; see
            `observables.sample_and_map_particles`. Like `poisson_kwargs` and
            `prng_seed`, this must match what `make_observation` and `fit` use:
            self-consistency is a property of the sampled population, so solving for
            it under one sampler and then fitting under another leaves a residual at
            the "truth" and reintroduces exactly the bias this function exists to
            remove.

    Returns:
        dict: The self-consistent parameters plus diagnostics:
            - 'pot_params', 'disk_df_params', 'bulge_df_params': tuned parameters.
            - 'penalty_initial', 'penalty_final': Poisson residual before and after.
            - 'history': per-iteration penalty values.
    """
    poisson_kwargs = poisson_kwargs or {}

    params_log = params_to_log(pot_params, disk_df_params, bulge_df_params)
    log_lo, log_hi = log_bounds_tree(params_log, param_bounds)

    def penalty_fn(params_log):
        pp, dd, bb = log_to_params(params_log)
        x, y, z, vx, vy, vz, w = sample_and_map_particles(
            mapper, pp, dd, bb, N_disk=N_disk, N_bulge=N_bulge,
            prng_seed=prng_seed, spheroid_corotation=spheroid_corotation,
            sampler=sampler,
        )
        return compute_poisson_penalty(
            x, y, z, w, baryonic_potential_raw, pot_params_to_tuple(pp),
            **poisson_kwargs
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params_log)

    @jax.jit
    def step(params_log, opt_state):
        penalty, grads = jax.value_and_grad(penalty_fn)(params_log)

        # Same non-finite guard as in `fit` (see the comment there).
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
        )
        # Only the requested parameters are allowed to move.
        grads = {
            group: {
                k: (v if k in tune_params else jnp.zeros_like(v))
                for k, v in group_grads.items()
            }
            for group, group_grads in grads.items()
        }

        updates, opt_state = optimizer.update(grads, opt_state, params_log)
        params_log = optax.apply_updates(params_log, updates)
        params_log = jax.tree_util.tree_map(jnp.clip, params_log, log_lo, log_hi)
        return params_log, opt_state, penalty

    history = []
    penalty_initial = float(penalty_fn(params_log))
    for _ in range(n_steps):
        params_log, opt_state, penalty = step(params_log, opt_state)
        history.append(float(penalty))

    final_pot, final_disk, final_bulge = log_to_params(params_log)
    return {
        'pot_params': {k: float(v) for k, v in final_pot.items()},
        'disk_df_params': {k: float(v) for k, v in final_disk.items()},
        'bulge_df_params': {k: float(v) for k, v in final_bulge.items()},
        'penalty_initial': penalty_initial,
        'penalty_final': float(penalty_fn(params_log)),
        'history': history,
    }


# ==============================================================================
# MULTI-START OPTIMIZATION
# ==============================================================================
def _scatter_params(params: dict, log_sigma: float, rng: "np.random.Generator") -> dict:
    """Multiplicatively scatter a parameter dict by lognormal noise (a fixed
    fractional perturbation in log-space), used to seed diverse restarts."""
    return {k: float(v) * float(np.exp(rng.normal(0.0, log_sigma))) for k, v in params.items()}


def fit_multistart(
    mapper, obs_maps: dict,
    init_pot_params: dict, init_disk_df_params: dict, init_bulge_df_params: dict,
    n_restarts: int = 6, restart_log_sigma: float = 0.4, restart_seed: int = 0,
    include_given_init: bool = True,
    **fit_kwargs,
):
    """
    Runs `fit` from several scattered initial guesses and returns the run with the
    lowest final loss. This is the practical remedy for the identifiability
    problem that makes a single far-off start unreliable: a single edge-on view
    under-constrains some parameter combinations, so from one bad guess the
    optimizer can settle into a different-but-almost-equally-good-fitting minimum.
    Sampling several starts and keeping the best final loss makes recovery far more
    robust, and the spread across restarts is itself a useful diagnostic of how
    well each parameter is constrained by the data.

    Parameters
    ----------
    n_restarts : total number of optimization runs.
    restart_log_sigma : fractional (log-space) scatter applied to the supplied
        initial guess to seed each additional restart. ~0.4 corresponds to a
        typical multiplicative perturbation of e^0.4 ~ 1.5x.
    include_given_init : if True, the first run uses the supplied initial guess
        verbatim (unscattered); the remaining runs are scattered around it.

    Returns the best run's result dict (same schema as `fit`), augmented with:
        'all_runs'      : list of every run's result dict
        'final_losses'  : list of each run's final loss
        'best_index'    : index of the returned (best) run
    """
    rng = np.random.default_rng(restart_seed)
    all_runs, final_losses = [], []

    for i in range(n_restarts):
        if i == 0 and include_given_init:
            p, d, b = init_pot_params, init_disk_df_params, init_bulge_df_params
        else:
            p = _scatter_params(init_pot_params, restart_log_sigma, rng)
            d = _scatter_params(init_disk_df_params, restart_log_sigma, rng)
            b = _scatter_params(init_bulge_df_params, restart_log_sigma, rng)

        res = fit(mapper, obs_maps, p, d, b, **fit_kwargs)
        all_runs.append(res)
        # Guard against a diverged run (NaN final loss) so it can never be "best".
        fl = res['history']['loss'][-1]
        final_losses.append(fl if fl == fl else float('inf'))

    best_index = int(np.argmin(final_losses))
    best = dict(all_runs[best_index])
    best['all_runs'] = all_runs
    best['final_losses'] = final_losses
    best['best_index'] = best_index
    return best