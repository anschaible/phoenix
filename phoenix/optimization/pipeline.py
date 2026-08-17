"""
Inference/optimization pipeline: fit potential + distribution-function parameters
so that the Phoenix surrogate reproduces an observed galaxy (mass, v_rot, sigma
maps) while remaining dynamically self-consistent (i.e. the density implied by
the sampled DF matches the density that sources the assumed potential via the
Poisson equation).
"""
from build.lib.build.lib.build.lib.build.lib.build.lib.phoenix.potentials import nfw_potential
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
# POTENTIAL HELPER
# ==============================================================================

def total_potential_raw(x, y, z, M_halo, a_halo, M_disk, a_disk, b_disk, M_bulge, a_bulge):
    """Same combined potential used everywhere else, but with parameters passed
    positionally instead of via a dict closure — this is the signature
    `compute_poisson_penalty` expects (potential_fn(x, y, z, *params))."""
    return (nfw_potential(x, y, z, M_halo, a_halo) +
            miyamoto_nagai_potential(x, y, z, M_disk, a_disk, b_disk) +
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
    prng_seed: int = 0
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
        
    Returns:
        dict: JAX arrays containing the pristine mock observation maps ('mass', 'v_rot', 'sigma').
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
def data_fit_loss(model_maps: dict, obs_maps: dict, mass_floor: float = 1e-3):
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

    Returns:
        tuple: (mass_loss, vrot_loss, sigma_loss)
    """
    mask = obs_maps['mass'] > mass_floor
    
    # Prevent division by zero if the mask is entirely empty
    n = jnp.maximum(jnp.sum(mask), 1)

    # Log-space mass loss
    log_mass_res = jnp.log10(model_maps['mass'] + mass_floor) - jnp.log10(obs_maps['mass'] + mass_floor)
    mass_loss = jnp.mean(jnp.where(mask, log_mass_res**2, 0.0)) / n

    # Normalized linear-space kinematic losses (typical scales: 200 km/s and 100 km/s)
    vrot_res = (model_maps['v_rot'] - obs_maps['v_rot']) / 200.0
    vrot_loss = jnp.mean(jnp.where(mask, vrot_res**2, 0.0)) / n

    sigma_res = (model_maps['sigma'] - obs_maps['sigma']) / 100.0
    sigma_loss = jnp.mean(jnp.where(mask, sigma_res**2, 0.0)) / n
    #sigma_loss = jnp.sum(jnp.where(mask, sigma_res**2, 0.0)) / n

    return mass_loss, vrot_loss, sigma_loss


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

    Returns:
        Callable: A function `loss_fn(params_log, soft_bin_h)` returning `(loss, aux_dict)`.
    """
    poisson_kwargs = poisson_kwargs or {}
    w_mass, w_vrot, w_sigma, w_poisson = loss_weights

    def loss_fn(params_log, soft_bin_h=None):
        pot_params, disk_df_params, bulge_df_params = log_to_params(params_log)

        # Generate fully differentiable phase space population
        x, y, z, vx, vy, vz, w = sample_and_map_particles(
            mapper, pot_params, disk_df_params, bulge_df_params,
            N_disk=N_disk, N_bulge=N_bulge, prng_seed=prng_seed,
            spheroid_corotation=spheroid_corotation,
        )
        
        # 1. Observational Data Fit
        model_maps = bin_maps(
            x, z, vy, w, 
            grid_size=grid_size, 
            extent_x=extent_x,
            extent_z=extent_z, 
            soft_bin_h=soft_bin_h
        )
        mass_loss, vrot_loss, sigma_loss = data_fit_loss(model_maps, obs_maps)
        
        # 2. Physical Self-Consistency (Poisson Penalty)
        poisson_penalty = compute_poisson_penalty(
            x, y, z, w, 
            total_potential_raw, 
            pot_params_to_tuple(pot_params), 
            **poisson_kwargs
        )

        # 3. Combine Core Losses
        loss = (w_mass * mass_loss + 
                w_vrot * vrot_loss + 
                w_sigma * sigma_loss + 
                w_poisson * poisson_penalty)

        # Package auxiliary metrics for telemetry
        aux = {
            'mass_loss': mass_loss, 
            'vrot_loss': vrot_loss, 
            'sigma_loss': sigma_loss,
            'poisson_penalty': poisson_penalty, 
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

    Returns:
        dict: The final optimized parameters and full training history:
            - 'pot_params': Final potential parameters.
            - 'disk_df_params': Final disk parameters.
            - 'bulge_df_params': Final bulge parameters.
            - 'history': Per-iteration telemetry (losses and parameter states).
    """
    
    # 1. Initialize Loss Function
    loss_fn = make_loss_fn(
        mapper, obs_maps, N_disk=N_disk, N_bulge=N_bulge, grid_size=grid_size,
        extent_x=extent_x, extent_z=extent_z, prng_seed=prng_seed,
        loss_weights=loss_weights, poisson_kwargs=poisson_kwargs,
        spheroid_corotation=spheroid_corotation,
    )

    # 2. Setup KDE Bandwidth Annealing Schedule
    if anneal_bandwidth is not None:
        h_start, h_end = anneal_bandwidth
        bandwidth_schedule = jnp.exp(jnp.linspace(jnp.log(h_start), jnp.log(h_end), n_steps))
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
        
        # Apply Optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params_log)
        params_log = optax.apply_updates(params_log, updates)
        
        # Enforce physical boundaries
        params_log = jax.tree_util.tree_map(jnp.clip, params_log, log_lo, log_hi)
        
        return params_log, opt_state, loss, aux

    # 5. Training Loop
    history = {
        'loss': [], 'mass_loss': [], 'vrot_loss': [], 'sigma_loss': [], 'poisson_penalty': [],
        'pot_params': [], 'disk_df_params': [], 'bulge_df_params': [],
    }

    for i in range(n_steps):
        # Fetch current bandwidth if annealing, otherwise pass None
        soft_bin_h = None if bandwidth_schedule is None else bandwidth_schedule[i]
        
        # Execute Step
        params_log, opt_state, loss, aux = step(params_log, opt_state, soft_bin_h)
        
        # Convert parameters back to linear space for logging
        pot_params, disk_df_params, bulge_df_params = log_to_params(params_log)
        
        # Record Telemetry
        history['loss'].append(float(loss))
        history['mass_loss'].append(float(aux['mass_loss']))
        history['vrot_loss'].append(float(aux['vrot_loss']))
        history['sigma_loss'].append(float(aux['sigma_loss']))
        history['poisson_penalty'].append(float(aux['poisson_penalty']))
        
        history['pot_params'].append({k: float(v) for k, v in pot_params.items()})
        history['disk_df_params'].append({k: float(v) for k, v in disk_df_params.items()})
        history['bulge_df_params'].append({k: float(v) for k, v in bulge_df_params.items()})

    # 6. Extract Final Parameters
    final_pot_params, final_disk_df_params, final_bulge_df_params = log_to_params(params_log)
    
    return {
        'pot_params': final_pot_params,
        'disk_df_params': final_disk_df_params,
        'bulge_df_params': final_bulge_df_params,
        'history': history,
    }