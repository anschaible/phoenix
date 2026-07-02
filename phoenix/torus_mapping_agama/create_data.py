import agama
import numpy as np

# Ensure reproducibility
np.random.seed(42)

# 1. Define the Galactic Potential
# Torus mapping requires an axisymmetric or spherical potential.
# We will use a standard Miyamoto-Nagai disk.
agama.setUnits(length=1, mass=1, velocity=1)
pot = agama.Potential(type='MiyamotoNagai', mass=1e10, scaleRadius=3.0, scaleHeight=0.3)

# 2. Initialize the Action Mapper
# In Agama >= Feb 2024, initialize the mapper with the potential. 
# It caches Torus objects for unique action triplets under the hood.
mapper = agama.ActionMapper(pot)

# 3. Define Sampling Grid Parameters
N_tori = 500              # Number of unique action coordinates (Tori)
N_angles_per_torus = 200  # Number of angle samples per torus
# Total training samples = 100,000

print("Sampling realistic Galactic actions...")

# Sample actions with realistic galactic scales (kpc * km/s)
# J_R and J_z should be positive.
J_R = np.random.uniform(10.0, 200.0, N_tori)   # Radial oscillations
J_z = np.random.uniform(5.0, 100.0, N_tori)    # Vertical oscillations
# Sample angular momentum, keeping it away from zero to avoid z-axis singularities
J_phi = np.random.uniform(100.0, 2000.0, N_tori) 
# Optional: If your ML model needs to learn counter-rotating orbits, 
# randomly flip the sign of some J_phi values (while still avoiding 0)
J_phi *= np.random.choice([-1, 1], size=N_tori)
unique_actions = np.vstack([J_R, J_z, J_phi]).T

all_actions = []
all_angles = []
all_phase_space = []

print(f"Generating {N_tori * N_angles_per_torus} samples for Phoenix...")

for J in unique_actions:
    # Uniformly sample angles U(0, 2pi) for this specific torus
    angles = np.random.uniform(0, 2 * np.pi, (N_angles_per_torus, 3))
    
    # Broadcast J to match the shape of the angle samples
    J_batch = np.tile(J, (N_angles_per_torus, 1))
    
    action_angles = np.hstack([J_batch, angles])

    # 4. Execute Torus Mapping: (J, theta) -> (x, v)
    try:
        # Returns an (N, 6) array: [:, 0:3] is pos, [:, 3:6] is vel
        xv = mapper(action_angles)
    except RuntimeError as e:
        # Torus fitting occasionally fails on highly resonant or chaotic orbits.
        # Safely catch and skip to keep the pipeline moving.
        print(f"Mapping failed for J={J}: {e}")
        continue
        
    all_actions.append(J_batch)
    all_angles.append(angles)
    all_phase_space.append(xv)

# 5. Consolidate Data
X_actions = np.vstack(all_actions)
X_angles = np.vstack(all_angles)
Y_phase_space = np.vstack(all_phase_space)

# Save to disk for ingestion into PyTorch/JAX
np.savez('./data_mapping/phoenix_torus_training_data.npz', 
         actions=X_actions, 
         angles=X_angles, 
         phase_space=Y_phase_space)

print("Dataset generation complete. Ready for ML.")