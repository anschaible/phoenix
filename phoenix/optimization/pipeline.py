"""
Inference/optimization pipeline: fit potential + distribution-function parameters
so that the Phoenix surrogate reproduces an observed galaxy (mass, v_rot, sigma
maps) while remaining dynamically self-consistent (i.e. the density implied by
the sampled DF matches the density that sources the assumed potential via the
Poisson equation).
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from phoenix.optimization.observables import (
    sample_and_map_particles,
    bin_maps,
    generate_edge_on_maps,
)
from phoenix.optimization.poisson_penalty import compute_poisson_penalty
from phoenix.potentials.potentials import nfw_potential, plummer_potential, miyamoto_nagai_potential

# ==============================================================================
# POTENTIAL (explicit-args form, required by compute_poisson_penalty)
# ==============================================================================
def total_potential_raw(x, y, z, M_halo, a_halo, M_disk, a_disk, b_disk, M_bulge, a_bulge):
    """Same combined potential used everywhere else, but with parameters passed
    positionally instead of via a dict closure — this is the signature
    `compute_poisson_penalty` expects (potential_fn(x, y, z, *params))."""
    return (nfw_potential(x, y, z, M_halo, a_halo) +
            miyamoto_nagai_potential(x, y, z, M_disk, a_disk, b_disk) +
            plummer_potential(x, y, z, M_bulge, a_bulge))


def pot_params_to_tuple(pot_params: dict) -> tuple:
    return (pot_params['M_halo'], pot_params['a_halo'],
            pot_params['M_disk'], pot_params['a_disk'], pot_params['b_disk'],
            pot_params['M_bulge'], pot_params['a_bulge'])


# ==============================================================================
# LOG-SPACE PARAMETER TRANSFORM
# ==============================================================================
# Every parameter here (masses, scale radii, dispersions, ...) is a strictly
# positive physical quantity, so optimizing log(param) with unconstrained Adam
# both keeps updates well-conditioned across ~7 orders of magnitude (M_halo ~1e12
# vs. Gamma_spheroid ~1.5) and makes positivity automatic.
def params_to_log(pot_params: dict, disk_df_params: dict, bulge_df_params: dict) -> dict:
    return {
        'pot': {k: jnp.log(jnp.asarray(v, dtype=jnp.float32)) for k, v in pot_params.items()},
        'disk': {k: jnp.log(jnp.asarray(v, dtype=jnp.float32)) for k, v in disk_df_params.items()},
        'bulge': {k: jnp.log(jnp.asarray(v, dtype=jnp.float32)) for k, v in bulge_df_params.items()},
    }


def log_to_params(params_log: dict):
    pot_params = {k: jnp.exp(v) for k, v in params_log['pot'].items()}
    disk_df_params = {k: jnp.exp(v) for k, v in params_log['disk'].items()}
    bulge_df_params = {k: jnp.exp(v) for k, v in params_log['bulge'].items()}
    return pot_params, disk_df_params, bulge_df_params


# Physical bounds for each parameter. These serve two purposes: (1) they keep
# every parameter in a physically sane range (e.g. a normalizable double-power-law
# spheroid DF requires Gamma_spheroid < 3 and Beta_spheroid > 3), and (2) they
# prevent runaway "sloppy" directions — e.g. Beta_spheroid and eta_spheroid can
# both grow together, nearly canceling in their effect on the DF shape, and drift
# to infinity (float32 overflow -> NaN) under unconstrained gradient descent.
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


def log_bounds_tree(params_log: dict, param_bounds: dict = None):
    """Builds (log_lo, log_hi) pytrees matching params_log's structure, for
    clipping log-space parameters to `param_bounds` after each optimizer step."""
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
# SYNTHETIC OBSERVATION (stand-in for a real mass/velocity/dispersion map)
# ==============================================================================
def make_observation(
    mapper, pot_params: dict, disk_df_params: dict, bulge_df_params: dict,
    N_disk: int = 100_000, N_bulge: int = 100_000, grid_size: int = 30,
    extent_x: float = 15.0, extent_z: float = 10.0, prng_seed: int = 0,
    noise_frac: float = 0.0, noise_seed: int = 1,
):
    """
    Generates a synthetic 'observed' galaxy from a ground-truth parameter set.
    Stands in for a real observation for now — once real data is available,
    replace this with the observed mass/v_rot/sigma maps on the same grid
    (`extent_x`, `extent_z`, `grid_size`) and skip straight to `fit`.
    """
    maps = generate_edge_on_maps(
        mapper, pot_params, disk_df_params, bulge_df_params,
        N_disk=N_disk, N_bulge=N_bulge, grid_size=grid_size,
        extent_x=extent_x, extent_z=extent_z, prng_seed=prng_seed,
    )
    obs = {k: jnp.array(v) for k, v in maps.items()}

    if noise_frac > 0:
        key = jax.random.PRNGKey(noise_seed)
        k_mass, k_vrot, k_sigma = jax.random.split(key, 3)
        mask = obs['mass'] > 1e-3
        mass_noisy = obs['mass'] * (1.0 + noise_frac * jax.random.normal(k_mass, obs['mass'].shape))
        obs['mass'] = jnp.where(mask, jnp.maximum(mass_noisy, 0.0), obs['mass'])
        obs['v_rot'] = jnp.where(
            mask, obs['v_rot'] + noise_frac * 200.0 * jax.random.normal(k_vrot, obs['v_rot'].shape), obs['v_rot']
        )
        sigma_noisy = obs['sigma'] + noise_frac * 100.0 * jax.random.normal(k_sigma, obs['sigma'].shape)
        obs['sigma'] = jnp.where(mask, jnp.maximum(sigma_noisy, 0.0), obs['sigma'])

    return obs


# ==============================================================================
# LOSS: DATA FIT + DYNAMICAL SELF-CONSISTENCY (POISSON EQUATION)
# ==============================================================================
def data_fit_loss(model_maps: dict, obs_maps: dict, mass_floor: float = 1e-3):
    """
    Compares model maps to the observation, restricted to pixels where the
    observation has detected mass (mirrors how real surface-brightness/kinematic
    maps are only meaningful above a noise floor). Mass is compared in log space
    since it spans many orders of magnitude; v_rot/sigma are compared in linear
    space, normalized by a typical velocity/dispersion scale (200 / 100 km/s).
    """
    mask = obs_maps['mass'] > mass_floor
    n = jnp.maximum(jnp.sum(mask), 1)

    log_mass_res = jnp.log10(model_maps['mass'] + mass_floor) - jnp.log10(obs_maps['mass'] + mass_floor)
    mass_loss = jnp.sum(jnp.where(mask, log_mass_res**2, 0.0)) / n

    vrot_res = (model_maps['v_rot'] - obs_maps['v_rot']) / 200.0
    vrot_loss = jnp.sum(jnp.where(mask, vrot_res**2, 0.0)) / n

    sigma_res = (model_maps['sigma'] - obs_maps['sigma']) / 100.0
    sigma_loss = jnp.sum(jnp.where(mask, sigma_res**2, 0.0)) / n

    return mass_loss, vrot_loss, sigma_loss


def make_loss_fn(
    mapper, obs_maps: dict,
    N_disk: int = 5_000, N_bulge: int = 5_000, grid_size: int = 16,
    extent_x: float = 15.0, extent_z: float = 10.0, prng_seed: int = 123,
    loss_weights=(1.0, 1.0, 1.0, 0.1), poisson_kwargs: dict = None,
    spheroid_corotation: float = 0.5,
    reg_weight: float = 0.0, reg_center_log: dict = None,
):
    """
    Builds a scalar loss(params_log) -> (loss, aux) closure combining:
      - data-fit terms (mass, v_rot, sigma) against `obs_maps`
      - a Poisson-equation self-consistency penalty: the density implied by the
        sampled DF tracer population must match the density that sources
        `total_potential_raw` at the current potential parameters.

    The sampling PRNG seed is held FIXED across all calls (same candidate
    actions/angles every evaluation, "common random numbers"): this turns the
    loss into a smooth deterministic function of the parameters instead of a
    noisy Monte-Carlo estimate that resamples every step, which is essential
    for stable gradient-based optimization.

    The returned closure takes an optional `soft_bin_h` (KDE smoothing bandwidth)
    so a caller can anneal it during optimization (graduated / coarse-to-fine):
    with a wide bandwidth the model and observed maps overlap even when the model
    galaxy starts far off, giving the masked log-mass term a non-vanishing
    gradient; shrinking it later recovers sharp maps. `None` uses bin_maps' default.

    `reg_weight` adds a Tikhonov (ridge) prior penalty
        reg_weight * mean_i (log p_i - log p_ref_i)^2
    over all parameters, pulling them in log-space toward the reference values
    `reg_center_log` (a params_log pytree; typically the physically-motivated
    initial guess). This is a MAP prior: data-constrained parameters barely move,
    but degenerate ones (bulge DF shape, DF normalizations) are anchored to sane
    values instead of drifting to their bounds — which is what makes an
    otherwise map-matching fit look unphysical.
    """
    poisson_kwargs = dict(poisson_kwargs or {})
    w_mass, w_vrot, w_sigma, w_poisson = loss_weights
    if reg_weight > 0 and reg_center_log is not None:
        from jax.flatten_util import ravel_pytree
        reg_center_vec, _ = ravel_pytree(reg_center_log)

    def loss_fn(params_log, soft_bin_h=None):
        pot_params, disk_df_params, bulge_df_params = log_to_params(params_log)

        x, y, z, vy, w = sample_and_map_particles(
            mapper, pot_params, disk_df_params, bulge_df_params,
            N_disk=N_disk, N_bulge=N_bulge, prng_seed=prng_seed,
            spheroid_corotation=spheroid_corotation,
        )
        model_maps = bin_maps(x, z, vy, w, grid_size=grid_size, extent_x=extent_x,
                              extent_z=extent_z, soft_bin_h=soft_bin_h)

        mass_loss, vrot_loss, sigma_loss = data_fit_loss(model_maps, obs_maps)
        poisson_penalty = compute_poisson_penalty(
            x, y, z, w, total_potential_raw, pot_params_to_tuple(pot_params), **poisson_kwargs
        )

        loss = w_mass * mass_loss + w_vrot * vrot_loss + w_sigma * sigma_loss + w_poisson * poisson_penalty

        reg = 0.0
        if reg_weight > 0 and reg_center_log is not None:
            from jax.flatten_util import ravel_pytree
            u_vec, _ = ravel_pytree(params_log)
            reg = reg_weight * jnp.mean((u_vec - reg_center_vec) ** 2)
            loss = loss + reg

        aux = {
            'mass_loss': mass_loss, 'vrot_loss': vrot_loss, 'sigma_loss': sigma_loss,
            'poisson_penalty': poisson_penalty, 'reg': reg, 'model_maps': model_maps,
        }
        return loss, aux

    return loss_fn


# ==============================================================================
# OPTIMIZATION LOOP
# ==============================================================================
def fit(
    mapper, obs_maps: dict,
    init_pot_params: dict, init_disk_df_params: dict, init_bulge_df_params: dict,
    N_disk: int = 5_000, N_bulge: int = 5_000, grid_size: int = 16,
    extent_x: float = 15.0, extent_z: float = 10.0, prng_seed: int = 123,
    loss_weights=(1.0, 1.0, 1.0, 0.1), poisson_kwargs: dict = None,
    learning_rate: float = 0.02, n_steps: int = 300, grad_clip_norm: float = 1.0,
    param_bounds: dict = None,
    anneal_bandwidth=None,
    spheroid_corotation: float = 0.5,
    reg_weight: float = 0.0, reg_center_log: dict = None,
):
    """
    Fits pot_params/disk_df_params/bulge_df_params to `obs_maps` via Adam on the
    combined data-fit + Poisson self-consistency loss. Returns the fitted
    parameters and a history dict (per-iteration losses and parameter values)
    for diagnostics/plotting.

    `reg_weight` > 0 adds a Tikhonov prior penalty in log-space toward
    `reg_center_log` (default: the initial guess) — see make_loss_fn. Use it to
    keep the degenerate parameters physically sensible instead of letting them
    drift to their bounds.

    Two stabilizers are applied every step, both needed in practice:
      - Gradients are clipped by global norm before the Adam update: near a sharp
        minimum (e.g. once the data-fit term is nearly noise-free) the loss surface
        in log-parameter space is steep enough that unclipped Adam steps can
        overshoot into an unphysical region and diverge to NaN.
      - Parameters are clipped to `param_bounds` (default: DEFAULT_PARAM_BOUNDS) in
        log-space after the update: some parameter combinations (e.g.
        Beta_spheroid and eta_spheroid growing together) barely change the DF
        shape, so gradient descent can drift along this "sloppy" direction
        indefinitely and eventually overflow. Bounding also enforces physical
        validity (e.g. a normalizable spheroid DF needs Gamma_spheroid < 3 <
        Beta_spheroid).

    anneal_bandwidth : (h_start, h_end) tuple, optional. Enables graduated /
        coarse-to-fine optimization: the KDE map-smoothing bandwidth is decayed
        log-linearly from `h_start` (kpc) down to `h_end` over the run. This is the
        remedy for far-off starts: when the model galaxy starts far from the data,
        sharp maps barely overlap and the masked log-mass term has a vanishing
        gradient (a high, flat plateau the optimizer can't descend). A wide initial
        bandwidth makes both maps broad and overlapping so there is always a
        gradient pulling the model toward the data; shrinking it recovers a sharp
        final fit. If None (default), the loss uses bin_maps' fixed default bandwidth.
    """
    # default the regularization anchor to the (physically-motivated) initial guess
    if reg_weight > 0 and reg_center_log is None:
        reg_center_log = params_to_log(init_pot_params, init_disk_df_params, init_bulge_df_params)

    loss_fn = make_loss_fn(
        mapper, obs_maps, N_disk=N_disk, N_bulge=N_bulge, grid_size=grid_size,
        extent_x=extent_x, extent_z=extent_z, prng_seed=prng_seed,
        loss_weights=loss_weights, poisson_kwargs=poisson_kwargs,
        spheroid_corotation=spheroid_corotation,
        reg_weight=reg_weight, reg_center_log=reg_center_log,
    )

    if anneal_bandwidth is not None:
        h_start, h_end = anneal_bandwidth
        # log-linear decay; jnp array so it can be indexed inside the jitted step
        bandwidth_schedule = jnp.exp(jnp.linspace(jnp.log(h_start), jnp.log(h_end), n_steps))
    else:
        bandwidth_schedule = None

    params_log = params_to_log(init_pot_params, init_disk_df_params, init_bulge_df_params)
    log_lo, log_hi = log_bounds_tree(params_log, param_bounds)
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params_log)

    @jax.jit
    def step(params_log, opt_state, soft_bin_h):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_log, soft_bin_h)
        updates, opt_state = optimizer.update(grads, opt_state, params_log)
        params_log = optax.apply_updates(params_log, updates)
        params_log = jax.tree_util.tree_map(jnp.clip, params_log, log_lo, log_hi)
        return params_log, opt_state, loss, aux

    history = {
        'loss': [], 'mass_loss': [], 'vrot_loss': [], 'sigma_loss': [], 'poisson_penalty': [], 'reg': [],
        'pot_params': [], 'disk_df_params': [], 'bulge_df_params': [],
    }

    for i in range(n_steps):
        soft_bin_h = None if bandwidth_schedule is None else bandwidth_schedule[i]
        params_log, opt_state, loss, aux = step(params_log, opt_state, soft_bin_h)
        pot_params, disk_df_params, bulge_df_params = log_to_params(params_log)
        history['loss'].append(float(loss))
        history['mass_loss'].append(float(aux['mass_loss']))
        history['vrot_loss'].append(float(aux['vrot_loss']))
        history['sigma_loss'].append(float(aux['sigma_loss']))
        history['poisson_penalty'].append(float(aux['poisson_penalty']))
        history['reg'].append(float(aux['reg']))
        history['pot_params'].append({k: float(v) for k, v in pot_params.items()})
        history['disk_df_params'].append({k: float(v) for k, v in disk_df_params.items()})
        history['bulge_df_params'].append({k: float(v) for k, v in bulge_df_params.items()})

    final_pot_params, final_disk_df_params, final_bulge_df_params = log_to_params(params_log)
    return {
        'pot_params': final_pot_params,
        'disk_df_params': final_disk_df_params,
        'bulge_df_params': final_bulge_df_params,
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


# ==============================================================================
# VARIATIONAL INFERENCE (mean-field Gaussian posterior over log-parameters)
# ==============================================================================
def fit_vi(
    mapper, obs_maps: dict,
    init_pot_params: dict, init_disk_df_params: dict, init_bulge_df_params: dict,
    N_disk: int = 5_000, N_bulge: int = 5_000, grid_size: int = 16,
    extent_x: float = 15.0, extent_z: float = 10.0, prng_seed: int = 123,
    use_terms=('v_rot', 'sigma'), noise_v: float = 10.0, noise_sigma: float = 10.0,
    spheroid_corotation: float = 0.5, soft_bin_h: float = None, mass_floor: float = 1e-3,
    prior_log_std: float = 2.0, init_post_std: float = 0.05,
    n_mc: int = 4, learning_rate: float = 0.02, n_steps: int = 400, vi_seed: int = 0,
    param_bounds: dict = None,
):
    """
    Approximate the Bayesian posterior over the (log-)parameters with a mean-field
    Gaussian, fitted by maximizing the ELBO (stochastic VI + reparameterization).
    Where `fit` returns a single best-fit point, this returns a *distribution* —
    a posterior mean and 1-sigma uncertainty for every parameter — which is what
    actually matters given the strong parameter degeneracies (well-constrained
    parameters get tight posteriors; degenerate ones come out broad).

    Model:
      - unconstrained variables u = log(params) (same log-space as `fit`);
      - Gaussian likelihood on the kinematic maps: for each term in `use_terms`,
        chi2 = sum_pixels (model - obs)^2 / noise^2 over the observed footprint,
        log L = -0.5 * chi2. `noise_v`/`noise_sigma` are the assumed per-pixel
        measurement errors (km/s) and set the scale of the posterior width;
      - a broad Gaussian prior on u centred at the initial guess (weak, width
        `prior_log_std` in log-space) to keep parameters in a sane range;
      - variational posterior q(u) = N(mu, diag(sigma^2)), sigma = softplus(rho).

    The DF-sampling PRNG seed is fixed across the whole run (common random numbers),
    so the likelihood is a smooth deterministic function of u — essential for
    low-variance reparameterization gradients. Initialise `mu` at (or near) the
    point-estimate MAP for a clean local posterior.

    Returns a dict with per-parameter posterior means/stds (physical units, via the
    log-normal transform), the raw log-space mu/sigma, and the ELBO history.
    """
    from jax.flatten_util import ravel_pytree

    use_v = 'v_rot' in use_terms
    use_s = 'sigma' in use_terms
    obs_v = jnp.asarray(obs_maps['v_rot'])
    obs_s = jnp.asarray(obs_maps['sigma'])
    mask = jnp.asarray(obs_maps['mass']) > mass_floor

    def data_loglik(params_log):
        pot_params, disk_df_params, bulge_df_params = log_to_params(params_log)
        x, y, z, vy, w = sample_and_map_particles(
            mapper, pot_params, disk_df_params, bulge_df_params,
            N_disk=N_disk, N_bulge=N_bulge, prng_seed=prng_seed,
            spheroid_corotation=spheroid_corotation,
        )
        m = bin_maps(x, z, vy, w, grid_size=grid_size, extent_x=extent_x,
                     extent_z=extent_z, soft_bin_h=soft_bin_h)
        ll = 0.0
        if use_v:
            ll = ll - 0.5 * jnp.sum(jnp.where(mask, (m['v_rot'] - obs_v) ** 2, 0.0)) / noise_v ** 2
        if use_s:
            ll = ll - 0.5 * jnp.sum(jnp.where(mask, (m['sigma'] - obs_s) ** 2, 0.0)) / noise_sigma ** 2
        return ll

    # flatten the parameter pytree to a vector for the VI algebra
    mu0_tree = params_to_log(init_pot_params, init_disk_df_params, init_bulge_df_params)
    mu0_vec, unravel = ravel_pytree(mu0_tree)
    prior_mean = mu0_vec
    D = mu0_vec.shape[0]

    # Optional physical support: clip each sampled parameter to its log-bounds
    # before evaluating the model. Without this a wide variational sample can push
    # a parameter into a regime where the sampler/surrogate returns non-finite
    # values and the ELBO goes NaN; clipping makes the posterior a truncated
    # Gaussian on the physical range and keeps VI stable.
    if param_bounds is not None:
        log_lo_tree, log_hi_tree = log_bounds_tree(mu0_tree, param_bounds)
        log_lo_vec, _ = ravel_pytree(log_lo_tree)
        log_hi_vec, _ = ravel_pytree(log_hi_tree)
    else:
        log_lo_vec = log_hi_vec = None

    def logjoint(u_vec):
        if log_lo_vec is not None:
            u_vec = jnp.clip(u_vec, log_lo_vec, log_hi_vec)
        lp = -0.5 * jnp.sum(((u_vec - prior_mean) / prior_log_std) ** 2) - D * jnp.log(prior_log_std) \
             - 0.5 * D * jnp.log(2 * jnp.pi)
        return data_loglik(unravel(u_vec)) + lp

    def neg_elbo(vi_params, eps):
        mu, rho = vi_params['mu'], vi_params['rho']
        sigma = jax.nn.softplus(rho)
        # reparameterized MC estimate of E_q[logjoint]; loop over samples (low memory)
        lj = 0.0
        for k in range(n_mc):
            lj = lj + logjoint(mu + sigma * eps[k])
        lj = lj / n_mc
        entropy = jnp.sum(jnp.log(sigma) + 0.5 * (1.0 + jnp.log(2 * jnp.pi)))
        return -(lj + entropy)

    rho0 = jnp.full((D,), jnp.log(jnp.expm1(init_post_std)))  # softplus^{-1}(init_post_std)
    vi_params = {'mu': mu0_vec, 'rho': rho0}
    optimizer = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(learning_rate))
    opt_state = optimizer.init(vi_params)

    @jax.jit
    def step(vi_params, opt_state, key):
        eps = jax.random.normal(key, (n_mc, D))
        loss, grads = jax.value_and_grad(neg_elbo)(vi_params, eps)
        # NaN-safe: a bad MC sample can produce non-finite grads; zero them so one
        # unlucky draw can't poison the whole run (belt-and-braces with the
        # sample clipping above).
        grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, vi_params)
        vi_params = optax.apply_updates(vi_params, updates)
        return vi_params, opt_state, loss

    key = jax.random.PRNGKey(vi_seed)
    elbo_hist = []
    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        vi_params, opt_state, loss = step(vi_params, opt_state, sub)
        elbo_hist.append(float(-loss))

    mu = vi_params['mu']
    sigma = np.asarray(jax.nn.softplus(vi_params['rho']))
    mu_np = np.asarray(mu)
    # posterior mean/std in physical units (log-normal): median exp(mu),
    # and +/-1sigma band exp(mu +/- sigma).
    mu_tree = unravel(jnp.asarray(mu_np))
    sig_tree = unravel(jnp.asarray(sigma))
    posterior = {}
    for group in ('pot', 'disk', 'bulge'):
        posterior[group] = {}
        for kk in mu_tree[group]:
            m_log = float(mu_tree[group][kk]); s_log = float(sig_tree[group][kk])
            posterior[group][kk] = {
                'median': float(np.exp(m_log)),
                'lo': float(np.exp(m_log - s_log)),
                'hi': float(np.exp(m_log + s_log)),
                'log_std': s_log,   # ~ fractional uncertainty
            }

    pot_params, disk_df_params, bulge_df_params = log_to_params(mu_tree)
    return {
        'posterior': posterior,
        'pot_params': {k: float(v) for k, v in pot_params.items()},
        'disk_df_params': {k: float(v) for k, v in disk_df_params.items()},
        'bulge_df_params': {k: float(v) for k, v in bulge_df_params.items()},
        'mu_log': mu_tree, 'sigma_log': sig_tree,
        'elbo_hist': elbo_hist,
    }
