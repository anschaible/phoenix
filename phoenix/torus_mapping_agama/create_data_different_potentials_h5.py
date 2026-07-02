import os
import agama
import numpy as np
import h5py

# Ensure reproducibility and set units
np.random.seed(42)
agama.setUnits(mass=1, length=1, velocity=1)

# Config
N_potentials = 100         
N_tori_per_pot = 500       
N_angles_per_torus = 200   
total_estimated_rows = N_potentials * N_tori_per_pot * N_angles_per_torus

output_dir = "./data_mapping"
os.makedirs(output_dir, exist_ok=True)
h5_path = os.path.join(output_dir, 'phoenix_conditional_torus_data.h5')

print(f"Creating HDF5 file at {h5_path}...")
print(f"Pre-allocating space for {total_estimated_rows:,} potential rows...")

with h5py.File(h5_path, 'w') as f:
    # 1. FIX: Pre-allocate the full estimated shape. 
    # maxshape=(None, X) ensures we are still allowed to shrink the arrays at the end.
    dset_actions = f.create_dataset('actions', shape=(total_estimated_rows, 3), maxshape=(None, 3), dtype='float32', chunks=(10000, 3), compression="gzip")
    dset_angles = f.create_dataset('angles', shape=(total_estimated_rows, 3), maxshape=(None, 3), dtype='float32', chunks=(10000, 3), compression="gzip")
    dset_potentials = f.create_dataset('potentials', shape=(total_estimated_rows, 7), maxshape=(None, 7), dtype='float32', chunks=(10000, 7), compression="gzip")
    dset_phase_space = f.create_dataset('phase_space', shape=(total_estimated_rows, 6), maxshape=(None, 6), dtype='float32', chunks=(10000, 6), compression="gzip")

    written_rows = 0

    for p_idx in range(N_potentials):
        # --- Generate unique potential parameters ---
        M_halo = np.random.uniform(5e11, 3e12)
        R_halo = np.random.uniform(10.0, 40.0)
        SHMR = np.random.uniform(0.01, 0.05)
        M_total_stars = M_halo * SHMR
        bulge_fraction = np.random.uniform(0.05, 0.40)
        M_bulge = M_total_stars * bulge_fraction
        M_disk = M_total_stars * (1 - bulge_fraction)
        R_disk = np.random.uniform(1.5, 6.0)
        H_disk = np.random.uniform(0.2, 1.5)
        R_bulge = min(np.random.uniform(0.5, 3.0), R_disk * 0.5)
        Sigma_disk = M_disk / (2 * np.pi * (R_disk**2))
        
        try:
            halo = agama.Potential(type='NFW', mass=M_halo, scaleRadius=R_halo)
            disk = agama.Potential(type='Disk', surfaceDensity=Sigma_disk, scaleRadius=R_disk, scaleHeight=H_disk)
            bulge = agama.Potential(type='Spheroid', mass=M_bulge, scaleRadius=R_bulge, gamma=1, beta=4, alpha=1)
            total_potential = agama.Potential(halo, disk, bulge)
            mapper = agama.ActionMapper(total_potential)
        except RuntimeError:
            continue

        current_pot_features = np.array([M_halo/1e11, R_halo, M_disk/1e11, R_disk, H_disk, M_bulge/1e11, R_bulge], dtype='float32')

        # --- Sample Actions for this potential ---
        J_R = np.random.uniform(0.1, 200.0, N_tori_per_pot)   
        J_z = np.random.uniform(0.1, 100.0, N_tori_per_pot)    
        J_phi = np.random.uniform(10.0, 4000.0, N_tori_per_pot) * np.random.choice([-1, 1], size=N_tori_per_pot)
        unique_actions = np.vstack([J_R, J_z, J_phi]).T

        pot_actions, pot_angles, pot_features, pot_xv = [], [], [], []

        for J in unique_actions:
            angles = np.random.uniform(0, 2 * np.pi, (N_angles_per_torus, 3))
            J_batch = np.tile(J, (N_angles_per_torus, 1))
            action_angles = np.hstack([J_batch, angles])

            try:
                xv = mapper(action_angles)
            except RuntimeError:
                continue
                
            pot_actions.append(J_batch)
            pot_angles.append(angles)
            pot_features.append(np.tile(current_pot_features, (N_angles_per_torus, 1)))
            pot_xv.append(xv)

        if not pot_actions:
            continue

        # Flatten this potential's block
        block_actions = np.vstack(pot_actions).astype('float32')
        block_angles = np.vstack(pot_angles).astype('float32')
        block_features = np.vstack(pot_features).astype('float32')
        block_xv = np.vstack(pot_xv).astype('float32')
        n_new_rows = block_actions.shape[0]

        # 2. FIX: Write directly into the pre-allocated slice boundaries
        dset_actions[written_rows : written_rows + n_new_rows] = block_actions
        dset_angles[written_rows : written_rows + n_new_rows] = block_angles
        dset_potentials[written_rows : written_rows + n_new_rows] = block_features
        dset_phase_space[written_rows : written_rows + n_new_rows] = block_xv

        written_rows += n_new_rows
        
        if (p_idx + 1) % 10 == 0 or p_idx == N_potentials - 1:
            print(f"Progress: [{p_idx+1}/{N_potentials}] potentials processed. Total rows saved: {written_rows:,}")

    # 3. FIX: Once the loops are entirely finished, trim off the empty pre-allocated rows
    print("Trimming unused pre-allocated rows...")
    dset_actions.resize((written_rows, 3))
    dset_angles.resize((written_rows, 3))
    dset_potentials.resize((written_rows, 7))
    dset_phase_space.resize((written_rows, 6))

print(f"\nSuccessfully created HDF5 dataset! Total mapped rows: {written_rows:,}")