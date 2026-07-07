import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit

# Import your custom modules
from sampling import sample_df_potential
from disk_df import f_disc_from_params
from distribution_functions import f_double_power_law
from potentials import nfw_potential, plummer_potential, miyamoto_nagai_potential

# ==============================================================================
# HELPER FUNCTIONS: GAUSS-HERMITE KINEMATICS
# ==============================================================================
def gauss_hermite(v, v_rot, v_disp, h3, h4):
    y = np.asarray((np.asarray(v) - v_rot) / (v_disp))
    return (np.exp(-0.5 * y**2)/(v_disp*np.sqrt(2*np.pi)) )* (
        1 + h3 * ((2 * np.sqrt(2) * y**3 - 3 * np.sqrt(2) * y) / np.sqrt(6))
          + h4 * ((4 * y**4 - 12 * y**2 + 3) / np.sqrt(24))
    )

def fit_losvd_weighted(v_bin, w_bin):
    if len(v_bin) < 10 or np.sum(w_bin) < 1e-5:
        return np.nan, np.nan, np.nan, np.nan
        
    hist, bins = np.histogram(v_bin, bins=30, density=True, weights=w_bin)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    V_guess = np.average(v_bin, weights=w_bin)
    variance = np.average((v_bin - V_guess)**2, weights=w_bin)
    sigma_guess = np.sqrt(variance)
    
    if sigma_guess == 0:
        return V_guess, 0.0, 0.0, 0.0
        
    p0 = [V_guess, sigma_guess, 0.1, 0.1]
    try:
        popt, _ = curve_fit(gauss_hermite, bin_centers, hist, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError, TypeError):
        return np.nan, np.nan, np.nan, np.nan
    return popt

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
    grid_size: int = 30,
    extent_x: float = 15.0,
    extent_z: float = 5.0,
    prng_seed: int = 42
):
    """
    Generates edge-on (XZ plane, Vy line-of-sight) mass and kinematic maps.
    
    Returns:
        dict: Containing 'mass', 'v_rot', 'sigma', 'h3', 'h4', 'x_edges', 'z_edges'
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
    test_bulge = spheroid_df_wrapper(10.0, 10.0, 10.0, total_potential, (), bulge_df_params)
    env_max_bulge = float(test_bulge) * 2.0 
    key, subkey = jax.random.split(key)
    cand_bulge, w_bulge = sample_df_potential(
        df=spheroid_df_wrapper, key=subkey, params=bulge_df_params, Phi_xyz=total_potential,
        theta=(), n_candidates=N_bulge, envelope_max=env_max_bulge, 
        J_bounds=(500.0, 500.0, 500.0), tau=0.05
    )
    key, subkey = jax.random.split(key)
    angles_bulge = jax.random.uniform(subkey, shape=(N_bulge, 3), minval=0.0, maxval=2*jnp.pi)

    # 4. Combine and Format
    w_disk_scaled = np.array(w_disk) / np.sum(w_disk) * M_disk
    w_bulge_scaled = np.array(w_bulge) / np.sum(w_bulge) * M_bulge
    all_weights = np.concatenate([w_disk_scaled, w_bulge_scaled])

    cand_bulge_np = np.array(cand_bulge)
    all_candidates = np.vstack([np.array(cand_disk), cand_bulge_np])
    all_angles = np.vstack([np.array(angles_disk), np.array(angles_bulge)])
    
    # Scale potential params for the network
    nn_potentials = np.array([M_halo/1e11, a_halo, M_disk/1e11, a_disk, b_disk, M_bulge/1e11, a_bulge])
    potentials_batch = np.tile(nn_potentials, (N_disk + N_bulge, 1))

    # 5. Network Mapping
    phase_space = mapper.map_to_phase_space(all_candidates, all_angles, potentials_batch)
    x_raw, y_raw, z_raw = phase_space[:, 0], phase_space[:, 1], phase_space[:, 2]
    vx_raw, vy_raw, vz_raw = phase_space[:, 3], phase_space[:, 4], phase_space[:, 5]

    # 6. Center and Symmetrize
    for arr in (x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw):
        arr[:N_disk] -= np.average(arr[:N_disk], weights=w_disk_scaled)
        arr[N_disk:] -= np.average(arr[N_disk:], weights=w_bulge_scaled)

    R = np.maximum(np.sqrt(x_raw**2 + y_raw**2), 1e-12)
    v_R = (x_raw * vx_raw + y_raw * vy_raw) / R
    v_phi = (x_raw * vy_raw - y_raw * vx_raw) / R

    # Bulge retro-flip
    bulge_flip_mask = np.random.choice([1.0, -1.0], size=N_bulge)
    v_phi[N_disk:] *= bulge_flip_mask

    phi_new = np.random.uniform(0, 2 * np.pi, size=N_disk + N_bulge)
    x = R * np.cos(phi_new)
    y = R * np.sin(phi_new)
    vx = v_R * np.cos(phi_new) - v_phi * np.sin(phi_new)
    vy = v_R * np.sin(phi_new) + v_phi * np.cos(phi_new)
    
    z_flip = np.random.choice([1.0, -1.0], size=N_disk + N_bulge)
    z = z_raw * z_flip
    vz = vz_raw * z_flip

    # 7. Grid Binning
    x_edges = np.linspace(-extent_x, extent_x, grid_size + 1)
    z_edges = np.linspace(-extent_z, extent_z, grid_size + 1)
    
    mass_map = np.zeros((grid_size, grid_size))
    V_map = np.zeros((grid_size, grid_size))
    sigma_map = np.zeros((grid_size, grid_size))
    h3_map = np.zeros((grid_size, grid_size))
    h4_map = np.zeros((grid_size, grid_size))

    # Mass Map (2D Histogram)
    mass_map, _, _ = np.histogram2d(x, z, bins=[x_edges, z_edges], weights=all_weights)
    
    # Kinematics Loop
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (x >= x_edges[i]) & (x < x_edges[i+1]) & (z >= z_edges[j]) & (z < z_edges[j+1])
            w_bin = all_weights[mask]
            
            if np.sum(w_bin) < 1e-5 or len(vy[mask]) < 10: 
                V_map[j, i], sigma_map[j, i], h3_map[j, i], h4_map[j, i] = np.nan, np.nan, np.nan, np.nan
                continue
                
            v_rot, v_disp, h3, h4 = fit_losvd_weighted(vy[mask], w_bin)
            V_map[j, i] = v_rot
            sigma_map[j, i] = v_disp
            h3_map[j, i] = h3
            h4_map[j, i] = h4

    # Note: returning transpose for Mass to match standard imshow (Y, X) orientation
    return {
        'mass': mass_map.T,
        'v_rot': V_map,
        'sigma': sigma_map,
        'h3': h3_map,
        'h4': h4_map,
        'x_edges': x_edges,
        'z_edges': z_edges
    }