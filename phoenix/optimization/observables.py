import jax
import jax.numpy as jnp

from phoenix.distribution_functions.sampling import sample_df_potential
from phoenix.distribution_functions.disk import f_disc_from_params
from phoenix.distribution_functions.spheroid import f_double_power_law
from phoenix.potentials.potentials import nfw_potential, plummer_potential, miyamoto_nagai_potential

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def spheroid_df_wrapper(Jr, Jz, Jphi, Phi_xyz, theta, params):
    return f_double_power_law(Jr, Jz, Jphi, params)

# ==============================================================================
# MAIN OBSERVABLE FUNCTION
# ==============================================================================
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
    soft_bin_h: float = None # Configurable smoothing bandwidth
):
    """
    Generates fully differentiable edge-on mass and kinematic maps using 
    Gaussian Soft-Binning (KDE).
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
    test_disk = f_disc_from_params(10.0, 5.0, 2000.0, total_potential, (), disk_df_params)
    env_max_disk = float(test_disk) * 2.0 
    key, subkey = jax.random.split(key)
    cand_disk, w_disk = sample_df_potential(
        df=f_disc_from_params, key=subkey, params=disk_df_params, Phi_xyz=total_potential,
        theta=(), n_candidates=N_disk, envelope_max=env_max_disk, 
        J_bounds=(100.0, 50.0, 3000.0), tau=0.05
    )
    key, subkey = jax.random.split(key)
    angles_disk = jax.random.uniform(subkey, shape=(N_disk, 3), minval=0.0, maxval=2*jnp.pi)

    # Bulge
    test_bulge = spheroid_df_wrapper(1.0, 1.0, 1.0, total_potential, (), bulge_df_params)
    env_max_bulge = float(test_bulge) * 2.0 
    key, subkey = jax.random.split(key)
    cand_bulge, w_bulge = sample_df_potential(
        df=spheroid_df_wrapper, key=subkey, params=bulge_df_params, Phi_xyz=total_potential,
        theta=(), n_candidates=N_bulge, envelope_max=env_max_bulge, 
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

    # 6. Center and Symmetrize (Fully JAX compatible)
    avg_x_disk = jnp.sum(x_raw[:N_disk] * w_disk_scaled) / jnp.sum(w_disk_scaled)
    avg_y_disk = jnp.sum(y_raw[:N_disk] * w_disk_scaled) / jnp.sum(w_disk_scaled)
    avg_z_disk = jnp.sum(z_raw[:N_disk] * w_disk_scaled) / jnp.sum(w_disk_scaled)
    avg_vx_disk = jnp.sum(vx_raw[:N_disk] * w_disk_scaled) / jnp.sum(w_disk_scaled)
    avg_vy_disk = jnp.sum(vy_raw[:N_disk] * w_disk_scaled) / jnp.sum(w_disk_scaled)
    avg_vz_disk = jnp.sum(vz_raw[:N_disk] * w_disk_scaled) / jnp.sum(w_disk_scaled)

    avg_x_bulge = jnp.sum(x_raw[N_disk:] * w_bulge_scaled) / jnp.sum(w_bulge_scaled)
    avg_y_bulge = jnp.sum(y_raw[N_disk:] * w_bulge_scaled) / jnp.sum(w_bulge_scaled)
    avg_z_bulge = jnp.sum(z_raw[N_disk:] * w_bulge_scaled) / jnp.sum(w_bulge_scaled)
    avg_vx_bulge = jnp.sum(vx_raw[N_disk:] * w_bulge_scaled) / jnp.sum(w_bulge_scaled)
    avg_vy_bulge = jnp.sum(vy_raw[N_disk:] * w_bulge_scaled) / jnp.sum(w_bulge_scaled)
    avg_vz_bulge = jnp.sum(vz_raw[N_disk:] * w_bulge_scaled) / jnp.sum(w_bulge_scaled)

    x_centered = jnp.concatenate([x_raw[:N_disk] - avg_x_disk, x_raw[N_disk:] - avg_x_bulge])
    y_centered = jnp.concatenate([y_raw[:N_disk] - avg_y_disk, y_raw[N_disk:] - avg_y_bulge])
    z_centered = jnp.concatenate([z_raw[:N_disk] - avg_z_disk, z_raw[N_disk:] - avg_z_bulge])
    vx_centered = jnp.concatenate([vx_raw[:N_disk] - avg_vx_disk, vx_raw[N_disk:] - avg_vx_bulge])
    vy_centered = jnp.concatenate([vy_raw[:N_disk] - avg_vy_disk, vy_raw[N_disk:] - avg_vy_bulge])
    vz_centered = jnp.concatenate([vz_raw[:N_disk] - avg_vz_disk, vz_raw[N_disk:] - avg_vz_bulge])

    # Safety buffer of 0.05 prevents velocities from exploding near the origin;
    # crucial here because the soft-binning kernel below smears any outlier
    # velocity across the *entire* grid, not just one pixel like a histogram would.
    R = jnp.maximum(jnp.sqrt(x_centered**2 + y_centered**2), 0.05)
    v_R = (x_centered * vx_centered + y_centered * vy_centered) / R
    v_phi = (x_centered * vy_centered - y_centered * vx_centered) / R

    # Bulge retro-flip (Using JAX random)
    key, subkey = jax.random.split(key)
    bulge_flip_mask = jax.random.choice(subkey, jnp.array([1.0, -1.0]), shape=(N_bulge,))
    v_phi = v_phi.at[N_disk:].multiply(bulge_flip_mask)

    key, subkey = jax.random.split(key)
    phi_new = jax.random.uniform(subkey, shape=(N_disk + N_bulge,), minval=0.0, maxval=2*jnp.pi)
    x = R * jnp.cos(phi_new)
    vy = v_R * jnp.sin(phi_new) + v_phi * jnp.cos(phi_new)
    
    key, subkey = jax.random.split(key)
    z_flip = jax.random.choice(subkey, jnp.array([1.0, -1.0]), shape=(N_disk + N_bulge,))
    z = z_centered * z_flip

    # 7. Differentiable Soft-Binning (KDE)
    dx = 2.0 * extent_x / grid_size
    dz = 2.0 * extent_z / grid_size

    # Default bandwidth is 0.25x the pixel width: wide enough to stay smooth/
    # differentiable, narrow enough not to blur out the velocity sign-flip near
    # the disk center or the density cusp at the bulge center.
    if soft_bin_h is None:
        soft_bin_h = jnp.maximum(dx, dz) * 0.25

    # Define pixel centers
    X_centers = jnp.linspace(-extent_x + dx/2, extent_x - dx/2, grid_size)
    Z_centers = jnp.linspace(-extent_z + dz/2, extent_z - dz/2, grid_size)

    # Vectorized broadcasting to shape: (grid_z, grid_x, N_particles)
    # This evaluates the distance of EVERY star to EVERY pixel instantly
    dz_arr = Z_centers[:, None, None] - z[None, None, :]
    dx_arr = X_centers[None, :, None] - x[None, None, :]

    dist_sq = dx_arr**2 + dz_arr**2
    # Normalized 2D Gaussian kernel (density per unit area) times pixel area,
    # so that summing over the whole grid recovers each star's weight exactly
    # once (mass-conserving), matching the histogram convention.
    kernel = (dx * dz) / (2.0 * jnp.pi * soft_bin_h**2) * jnp.exp(-0.5 * dist_sq / soft_bin_h**2)

    # Apply soft acceptance weights and physical masses to the kernel
    w_kernel = all_weights[None, None, :] * kernel
    
    # Calculate Maps
    mass_map = jnp.sum(w_kernel, axis=-1)
    
    # Protect against Division-by-Zero in empty pixels
    mass_safe = jnp.maximum(mass_map, 1e-12)
    
    # Differentiable Moment calculations
    # Where mass is negligible (< 1e-5), return 0.0 to prevent gradient poisoning
    v_rot_map = jnp.where(mass_map > 1e-5, jnp.sum(w_kernel * vy[None, None, :], axis=-1) / mass_safe, 0.0)
    
    v2_map = jnp.where(mass_map > 1e-5, jnp.sum(w_kernel * (vy**2)[None, None, :], axis=-1) / mass_safe, 0.0)
    variance_map = v2_map - v_rot_map**2
    sigma_map = jnp.sqrt(jnp.maximum(variance_map, 1e-12))
    
    # Return edges for plotting consistency
    x_edges = jnp.linspace(-extent_x, extent_x, grid_size + 1)
    z_edges = jnp.linspace(-extent_z, extent_z, grid_size + 1)

    return {
        'mass': mass_map,
        'v_rot': v_rot_map,
        'sigma': sigma_map,
        'x_edges': x_edges,
        'z_edges': z_edges
    }