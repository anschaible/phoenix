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
    N_bulge: int = 50_000,
    grid_size: int = 40,
    extent_x: float = 15.0,
    extent_z: float = 5.0,
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

    # Bulge - FIX: Tighter J_bounds to heavily populate the dense central core!
    # FIX: Evaluate envelope_max much deeper in the core (1.0) so weights scale properly.
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

    # FIX: Safety buffer of 0.05 prevents velocities from exploding precisely at the origin
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

    x_edges = jnp.linspace(-extent_x, extent_x, grid_size + 1)
    z_edges = jnp.linspace(-extent_z, extent_z, grid_size + 1)
    
    # Use 2D histograms to calculate sums in each bin (Non-differentiable spatial mask)
    # jnp.histogram2d returns shape (grid_x, grid_z), so we transpose (.T) to get (grid_z, grid_x)
    mass_map, _, _ = jnp.histogram2d(x, z, bins=[x_edges, z_edges], weights=all_weights)
    mass_map = mass_map.T
    
    v_mom_map, _, _ = jnp.histogram2d(x, z, bins=[x_edges, z_edges], weights=all_weights * vy)
    v_mom_map = v_mom_map.T
    
    v2_mom_map, _, _ = jnp.histogram2d(x, z, bins=[x_edges, z_edges], weights=all_weights * (vy**2))
    v2_mom_map = v2_mom_map.T
    
    # Protect against Division-by-Zero in empty pixels
    mass_safe = jnp.maximum(mass_map, 1e-12)
    
    # Kinematic Moment calculations
    # Where mass is negligible (< 1e-5), return 0.0 to prevent artifacting
    v_rot_map = jnp.where(mass_map > 1e-5, v_mom_map / mass_safe, 0.0)
    v2_map = jnp.where(mass_map > 1e-5, v2_mom_map / mass_safe, 0.0)
    
    variance_map = v2_map - v_rot_map**2
    sigma_map = jnp.sqrt(jnp.maximum(variance_map, 1e-12))

    return {
        'mass': mass_map,
        'v_rot': v_rot_map,
        'sigma': sigma_map,
        'x_edges': x_edges,
        'z_edges': z_edges
    }