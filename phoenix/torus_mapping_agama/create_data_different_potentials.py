import os
import agama
import numpy as np
import h5py

# Ensure reproducibility across runs
np.random.seed(42)

# Initialize Agama units (M_sun, kpc, km/s)
agama.setUnits(mass=1, length=1, velocity=1)

# ==============================================================================
# 1. TUNING PARAMETERS (Adjust these based on your RAM and storage limits)
# ==============================================================================
N_potentials = 100         # Number of unique galaxy configurations
N_tori_per_pot = 500        # Unique action combinations per galaxy
N_angles_per_torus = 200   # Angle realizations per torus
# Total dataset size = 100 * 50 * 100 = 500,000 phase space rows

output_dir = "./data_mapping"
os.makedirs(output_dir, exist_ok=True)

# Lists to collect our final training columns
all_actions = []
all_angles = []
all_pot_params = []
all_phase_space = []

print(f"Starting pipeline: Generating {N_potentials * N_tori_per_pot * N_angles_per_torus} samples...")

# ==============================================================================
# 2. OUTER LOOP: GENERATING DIFFERENT GALACTIC POTENTIALS
# ==============================================================================
for p_idx in range(N_potentials):
    print(f"[{p_idx+1}/{N_potentials}] Generating unique potential parameters...")
    
    # --- Sample Physical Potential Parameters ---
    # Dark Matter Halo (NFW)
    M_halo = np.random.uniform(5e11, 3e12)
    R_halo = np.random.uniform(10.0, 40.0)
    
    # Stellar-to-Halo Mass Ratio (Baryons are 1% to 5% of dark matter)
    SHMR = np.random.uniform(0.01, 0.05)
    M_total_stars = M_halo * SHMR
    
    # Split stars between Bulge (5% to 40%) and Disk
    bulge_fraction = np.random.uniform(0.05, 0.40)
    M_bulge = M_total_stars * bulge_fraction
    M_disk = M_total_stars * (1 - bulge_fraction)
    
    # Structural scale bounds
    R_disk = np.random.uniform(1.5, 6.0)
    H_disk = np.random.uniform(0.2, 1.5)
    R_bulge = np.random.uniform(0.5, 3.0)
    R_bulge = min(R_bulge, R_disk * 0.5) # Physics guardrails
    
    # Convert total disk mass to surface density for Agama
    Sigma_disk = M_disk / (2 * np.pi * (R_disk**2))
    
    # --- Instantiate Agama Potentials ---
    try:
        halo = agama.Potential(type='NFW', mass=M_halo, scaleRadius=R_halo)
        
        disk = agama.Potential(
            type='Disk', 
            surfaceDensity=Sigma_disk, 
            scaleRadius=R_disk, 
            scaleHeight=H_disk
        )
        
        bulge = agama.Potential(
            type='Spheroid', 
            mass=M_bulge, 
            scaleRadius=R_bulge, 
            gamma=1, beta=4, alpha=1
        )
        
        total_potential = agama.Potential(halo, disk, bulge)
        mapper = agama.ActionMapper(total_potential)
        
    except RuntimeError as potential_error:
        print(f"Skipping potential generation step due to instantiation error: {potential_error}")
        continue

    # Pack parameters into a feature vector for the ML model.
    # TIP: Storing raw masses (e.g., 1e12) forces neural networks to handle chaotic gradients.
    # We divide by 1e11 here to keep the features scaled near 0.1 - 30.0 for easier training.
    current_pot_features = np.array([
        M_halo / 1e11, R_halo,
        M_disk / 1e11, R_disk, H_disk,
        M_bulge / 1e11, R_bulge
    ])

    # ==============================================================================
    # 3. INNER LOOP: SAMPLING ACTION-ANGLE TORI FOR THIS GALACTIC CONFIGURATION
    # ==============================================================================
    # Sample Actions spanning from inner bulge out to the extended disk
    J_R = np.random.uniform(0.1, 200.0, N_tori_per_pot)   
    J_z = np.random.uniform(0.1, 100.0, N_tori_per_pot)    
    J_phi = np.random.uniform(10.0, 4000.0, N_tori_per_pot) 
    J_phi *= np.random.choice([-1, 1], size=N_tori_per_pot) # Prograde & Retrograde
    
    unique_actions = np.vstack([J_R, J_z, J_phi]).T

    for J in unique_actions:
        # Uniformly sample angles U(0, 2pi)
        angles = np.random.uniform(0, 2 * np.pi, (N_angles_per_torus, 3))
        
        # Broadcast Actions and Potential Features to match the angle rows
        J_batch = np.tile(J, (N_angles_per_torus, 1))
        pot_batch = np.tile(current_pot_features, (N_angles_per_torus, 1))
        
        action_angles = np.hstack([J_batch, angles])

        # --- Execute Torus Mapping ---
        try:
            xv = mapper(action_angles)
        except RuntimeError:
            # Safely skip singular, resonant, or unbound orbit edge cases
            continue
            
        all_actions.append(J_batch)
        all_angles.append(angles)
        all_pot_params.append(pot_batch)
        all_phase_space.append(xv)

# ==============================================================================
# 4. CONSOLIDATE AND SAVE
# ==============================================================================
print("Consolidating all generated data blocks...")
X_actions = np.vstack(all_actions)
X_angles = np.vstack(all_angles)
X_potentials = np.vstack(all_pot_params)
Y_phase_space = np.vstack(all_phase_space)

output_path = os.path.join(output_dir, 'phoenix_conditional_torus_data.npz')
np.savez(output_path, 
         actions=X_actions, 
         angles=X_angles, 
         potentials=X_potentials, 
         phase_space=Y_phase_space)

print(f"\nDataset generation complete!")
print(f"Final shape of inputs -> Actions: {X_actions.shape}, Angles: {X_angles.shape}, Potentials: {X_potentials.shape}")
print(f"Final shape of target -> Phase Space: {Y_phase_space.shape}")
print(f"Saved successfully to: {output_path}")