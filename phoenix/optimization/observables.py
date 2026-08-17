import jax
import jax.numpy as jnp

from phoenix.distribution_functions.sampling import sample_df_potential
from phoenix.distribution_functions.disk import f_disc_from_params
from phoenix.distribution_functions.spheroid import f_double_power_law
from phoenix.potentials.potentials import nfw_potential, plummer_potential, miyamoto_nagai_potential



# HELPER FUNCTIONS
def spheroid_df_wrapper(Jr, Jz, Jphi, Phi_xyz, theta, params):
    return f_double_power_law(Jr, Jz, Jphi, params)


def sample_and_map_particles(
    mapper,
    pot_params: dict,
    disk_df_params: dict,
    bulge_df_params: dict,
    N_disk: int = 100_000,
    N_bulge: int = 100_000,
    prng_seed: int = 42,
    spheroid_corotation: float = 0.5,
):
    """
    Samples the disk and bulge distribution functions (DFs) in the given gravitational 
    potential and maps the resulting actions to 3D phase space via the Phoenix surrogate.

    This function performs differentiable rejection sampling of the actions, queries the 
    neural network to obtain phase-space coordinates, and applies a dynamic net-rotation 
    correction to the bulge population.

    Args:
        mapper: The Phoenix neural network surrogate used to map actions to phase space.
        pot_params (dict): Gravitational potential parameters. Expected keys:
            'M_halo', 'a_halo', 'M_disk', 'a_disk', 'b_disk', 'M_bulge', 'a_bulge'.
        disk_df_params (dict): Parameters governing the disk's distribution function.
        bulge_df_params (dict): Parameters governing the bulge's distribution function.
        N_disk (int, optional): Number of disk candidates to sample. Defaults to 100,000.
        N_bulge (int, optional): Number of bulge candidates to sample. Defaults to 100,000.
        prng_seed (int, optional): Seed for the JAX pseudo-random number generator. Defaults to 42.
        spheroid_corotation (float, optional): Fraction of bulge/spheroid orbits assigned 
            as PROGRADE (co-rotating with the disk). Because the spheroid DF is even in J_phi, 
            a sign must be manually assigned to its azimuthal motion. 
            - 0.5 (default) => Equal prograde/retrograde => Non-rotating, pressure-supported.
            - 1.0 => Fully co-rotating central component.

    Returns:
        tuple: A fully differentiable tracer population containing (x, y, z, vx, vy, vz, weights).
            - x, y, z (Array): 3D spatial coordinates (kpc).
            - vx, vy (Array): In-plane velocities, dynamically reconstructed to account 
              for the applied bulge prograde/retrograde rotation mask (km/s).
            - vz (Array): Vertical velocities (km/s).
            - all_weights (Array): Particle weights, properly scaled by M_disk and M_bulge.
    """
    # 1. Unpack Potential Parameters
    M_halo, a_halo = pot_params['M_halo'], pot_params['a_halo']
    M_disk, a_disk, b_disk = pot_params['M_disk'], pot_params['a_disk'], pot_params['b_disk']
    M_bulge, a_bulge = pot_params['M_bulge'], pot_params['a_bulge']

    # 2. Define the Local Potential
    def total_potential(x, y, z):
        return (nfw_potential(x, y, z, M_halo, a_halo) +
                miyamoto_nagai_potential(x, y, z, M_disk, a_disk, b_disk) +
                plummer_potential(x, y, z, M_bulge, a_bulge))
    # 3. Differentiable Sampling
    key = jax.random.PRNGKey(prng_seed)

    # Disk
    # envelope_max=None -> the sampler auto-calibrates the rejection envelope from
    # the candidates' own max DF value. This is essential for optimization from
    # far-off starts: a fixed envelope computed at a single hard-coded action point
    # underflows to zero when the DF parameters move, dividing by zero and poisoning
    # everything with NaNs. Auto-calibration tracks the DF as the parameters change.
    key, subkey = jax.random.split(key)
    cand_disk, w_disk = sample_df_potential(
        df=f_disc_from_params, key=subkey, params=disk_df_params, Phi_xyz=total_potential,
        theta=(), n_candidates=N_disk, envelope_max=None,
        J_bounds=(100.0, 50.0, 3000.0), tau=0.05
    )
    key, subkey = jax.random.split(key)
    angles_disk = jax.random.uniform(subkey, shape=(N_disk, 3), minval=0.0, maxval=2*jnp.pi)

    # Bulge
    key, subkey = jax.random.split(key)
    cand_bulge, w_bulge = sample_df_potential(
        df=spheroid_df_wrapper, key=subkey, params=bulge_df_params, Phi_xyz=total_potential,
        theta=(), n_candidates=N_bulge, envelope_max=None,
        J_bounds=(500.0, 500.0, 500.0), tau=0.05
    )
    key, subkey = jax.random.split(key)
    angles_bulge = jax.random.uniform(subkey, shape=(N_bulge, 3), minval=0.0, maxval=2*jnp.pi)

    # 4. Combine and Format Weights
    w_disk_scaled = w_disk / jnp.sum(w_disk) * M_disk
    w_bulge_scaled = w_bulge / jnp.sum(w_bulge) * M_bulge
    all_weights = jnp.concatenate([w_disk_scaled, w_bulge_scaled])

    all_candidates = jnp.vstack([cand_disk, cand_bulge])
    all_angles = jnp.vstack([angles_disk, angles_bulge])

    # Scale potential params for the network
    nn_potentials = jnp.array([M_halo/1e11, a_halo, M_disk/1e11, a_disk, b_disk, M_bulge/1e11, a_bulge])
    potentials_batch = jnp.tile(nn_potentials, (N_disk + N_bulge, 1))

    # 5. Network Mapping
    phase_space = mapper.map_to_phase_space(all_candidates, all_angles, potentials_batch)
    x_raw, y_raw, z_raw = phase_space[:, 0], phase_space[:, 1], phase_space[:, 2]
    vx_raw, vy_raw, vz_raw = phase_space[:, 3], phase_space[:, 4], phase_space[:, 5]

    # 6. Cylindrical Decomposition from Raw Space (Skipping Averages)
    # Safety buffer of 0.05 prevents velocities from exploding near the origin
    R = jnp.maximum(jnp.sqrt(x_raw**2 + y_raw**2), 0.05)
    v_R = (x_raw * vx_raw + y_raw * vy_raw) / R
    v_phi = (x_raw * vy_raw - y_raw * vx_raw) / R

    # 7. Bulge retro-flip (Using JAX random)
    # Assign each bulge orbit prograde (+1) with probability `spheroid_corotation`,
    # retrograde (-1) otherwise.
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(N_bulge,))
    bulge_flip_mask = jnp.where(u < spheroid_corotation, 1.0, -1.0)
    
    # Apply the flip mask ONLY to the bulge indices (from N_disk to the end)
    v_phi = v_phi.at[N_disk:].multiply(bulge_flip_mask)

    # 8. Reconstruct Cartesian Velocities 
    # (Keeping original x, y, z and mapping the flipped v_phi back to vx, vy)
    cos_phi = x_raw / R
    sin_phi = y_raw / R
    
    vx_new = v_R * cos_phi - v_phi * sin_phi
    vy_new = v_R * sin_phi + v_phi * cos_phi

    # Return original coordinates and vz_raw, but updated vx and vy
    return x_raw, y_raw, z_raw, vx_new, vy_new, vz_raw, all_weights


def bin_maps(
    x, z, vy, all_weights,
    grid_size: int = 30,
    extent_x: float = 15.0,
    extent_z: float = 10.0,
    soft_bin_h: float = None,
):
    """
    Bins a tracer population into differentiable, edge-on observable maps using 
    Gaussian Soft-Binning (Kernel Density Estimation).

    Instead of dropping particles into hard discrete bins (which breaks gradients), 
    each particle is smeared into a 2D Gaussian blob. The grid evaluates the sum 
    of all blobs at each pixel center.

    Args:
        x: Array of particle x-coordinates (shape: [N]).
        z: Array of particle z-coordinates (shape: [N]).
        vy: Array of particle line-of-sight velocities (shape: [N]).
        all_weights: Array of particle masses/weights (shape: [N]).
        grid_size: Number of pixels along one spatial dimension (grid is N x N).
        extent_x: Physical half-width of the grid (from -extent_x to +extent_x).
        extent_z: Physical half-height of the grid (from -extent_z to +extent_z).
        soft_bin_h: Gaussian bandwidth (smoothing scale). If None, defaults to 
            0.25x the pixel width.

    Returns:
        dict: Containing the smoothed 2D maps ('mass', 'v_rot', 'sigma') and 
        the grid edges ('x_edges', 'z_edges') for plotting.
    """
    
    # 1. Grid Definition
    dx = 2.0 * extent_x / grid_size
    dz = 2.0 * extent_z / grid_size

    # Set default smoothing scale (bandwidth)
    if soft_bin_h is None:
        soft_bin_h = jnp.maximum(dx, dz) * 0.25

    # 1D arrays of pixel centers
    X_centers = jnp.linspace(-extent_x + dx/2, extent_x - dx/2, grid_size)
    Z_centers = jnp.linspace(-extent_z + dz/2, extent_z - dz/2, grid_size)

    # 2. Distance Matrix Calculation via Broadcasting
    # We reshape the arrays to create a 3D matrix of shape: (grid_z, grid_x, N_particles)
    # This calculates the distance from EVERY grid pixel to EVERY particle simultaneously.
    
    # Shape: (grid_z, 1, 1) - (1, 1, N_particles) -> (grid_z, 1, N_particles)
    dz_arr = Z_centers[:, None, None] - z[None, None, :]
    
    # Shape: (1, grid_x, 1) - (1, 1, N_particles) -> (1, grid_x, N_particles)
    dx_arr = X_centers[None, :, None] - x[None, None, :]

    # Sum of squared distances. Shape becomes: (grid_z, grid_x, N_particles)
    dist_sq = dx_arr**2 + dz_arr**2

    # 3. Gaussian Kernel Application
    # Calculate the normalized 2D Gaussian density. We multiply by pixel area (dx * dz)
    # so that integrating over the grid recovers the exact original particle mass.
    normalization = (dx * dz) / (2.0 * jnp.pi * soft_bin_h**2)
    kernel = normalization * jnp.exp(-0.5 * dist_sq / soft_bin_h**2)

    # Scale the kernel by the mass/weight of each particle
    w_kernel = all_weights[None, None, :] * kernel

    # 4. Map Construction
    # Sum across the particle axis (-1) to collapse down to a 2D map: (grid_z, grid_x)
    mass_map = jnp.sum(w_kernel, axis=-1)

    # Protect against Division-by-Zero in empty pixels during gradient calculations
    mass_safe = jnp.maximum(mass_map, 1e-12)

    # Calculate Velocity (First Moment: Mean)
    # We ignore pixels with negligible mass (< 1e-5) to prevent numerical instability
    momentum_map = jnp.sum(w_kernel * vy[None, None, :], axis=-1)
    v_rot_map = jnp.where(mass_map > 1e-5, momentum_map / mass_safe, 0.0)

    # Calculate Velocity Dispersion (Second Moment: Variance -> Sigma)
    # Variance = E[v^2] - (E[v])^2
    v2_momentum_map = jnp.sum(w_kernel * (vy**2)[None, None, :], axis=-1)
    v2_map = jnp.where(mass_map > 1e-5, v2_momentum_map / mass_safe, 0.0)
    
    variance_map = v2_map - v_rot_map**2
    sigma_map = jnp.sqrt(jnp.maximum(variance_map, 1e-12))

    # 5. Output Packaging
    x_edges = jnp.linspace(-extent_x, extent_x, grid_size + 1)
    z_edges = jnp.linspace(-extent_z, extent_z, grid_size + 1)

    return {
        'mass': mass_map,
        'v_rot': v_rot_map,
        'sigma': sigma_map,
        'x_edges': x_edges,
        'z_edges': z_edges
    }


def generate_edge_on_maps(
    mapper,
    pot_params: dict,
    disk_df_params: dict,
    bulge_df_params: dict,
    N_disk: int = 100_000,
    N_bulge: int = 100_000,
    grid_size: int = 30,
    extent_x: float = 15.0,
    extent_z: float = 10.0,
    prng_seed: int = 42,
    soft_bin_h: float = None, # Configurable smoothing bandwidth
    spheroid_corotation: float = 0.5,
):
    """
    End-to-end pipeline for generating fully differentiable edge-on observable kinematic maps 
    of a galaxy model using Gaussian Soft-Binning (KDE).

    This function combines two steps:
    1. Samples tracer particles from the combined disk and bulge distribution 
       functions and maps them to 3D phase space via the Phoenix neural network.
    2. Projects these particles into an edge-on view (x-z plane with line-of-sight 
       velocity vy) and bins them into mass and kinematic maps using Gaussian 
       soft-binning (KDE) to preserve differentiability.

    Args:
        mapper: The Phoenix neural network surrogate used to map actions to phase space.
        pot_params (dict): Gravitational potential parameters (halo, disk, and bulge masses/scales).
        disk_df_params (dict): Parameters governing the disk's distribution function.
        bulge_df_params (dict): Parameters governing the bulge's distribution function.
        N_disk (int, optional): Number of disk particles to sample. Defaults to 100,000.
        N_bulge (int, optional): Number of bulge particles to sample. Defaults to 100,000.
        grid_size (int, optional): The resolution of the output 2D grids (grid_size x grid_size). Defaults to 30.
        extent_x (float, optional): Physical half-width of the spatial grid in the x-direction (kpc). Defaults to 15.0.
        extent_z (float, optional): Physical half-height of the spatial grid in the z-direction (kpc). Defaults to 10.0.
        prng_seed (int, optional): Seed for the JAX pseudo-random number generator. Defaults to 42.
        soft_bin_h (float, optional): Gaussian smoothing bandwidth for KDE. If None, 
            auto-calculates to 0.25x the pixel width.
        spheroid_corotation (float, optional): Fraction of bulge orbits assigned as 
            prograde (co-rotating). 0.5 = non-rotating bulge, 1.0 = fully co-rotating. Defaults to 0.5.

    Returns:
        dict: A dictionary containing the smoothed edge-on maps and grid edges:
            - 'mass' (Array): 2D map of projected mass density.
            - 'v_rot' (Array): 2D map of line-of-sight rotational velocity.
            - 'sigma' (Array): 2D map of line-of-sight velocity dispersion.
            - 'x_edges' (Array): 1D array of spatial bin edges along the x-axis.
            - 'z_edges' (Array): 1D array of spatial bin edges along the z-axis.
    """
    x, y, z, vx, vy, vz, all_weights = sample_and_map_particles(
        mapper, pot_params, disk_df_params, bulge_df_params,
        N_disk=N_disk, N_bulge=N_bulge, prng_seed=prng_seed,
        spheroid_corotation=spheroid_corotation,
    )
    return bin_maps(x, z, vy, all_weights, grid_size=grid_size, extent_x=extent_x,
                     extent_z=extent_z, soft_bin_h=soft_bin_h)

