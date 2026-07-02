# 1. Define the Galactic Potential
# Torus mapping requires an axisymmetric or spherical potential.
# We will use a standard Miyamoto-Nagai disk.
import agama
import numpy as np

# Ensure reproducibility
np.random.seed(42)

# Initialize Agama units (e.g., M_sun, kpc, km/s)
agama.setUnits(mass=1, length=1, velocity=1)

# 1. Dark Matter Halo (NFW Profile)
# gamma=1 (inner slope), beta=3 (outer slope), alpha=1 (transition)
# 1. Dark Matter Halo (Built-in NFW Profile)
halo = agama.Potential(
    type='NFW',
    mass=1e12,         # Agama interprets this as Virial Mass
    scaleRadius=20.0
)

# 2. Stellar Disk (Double-Exponential)
disk = agama.Potential(
    type='Disk',
    surfaceDensity=5e10 / (2 * 3.14159 * 3.0**2), # Approx normalization
    scaleRadius=3.0,      # Radial scale length (kpc)
    scaleHeight=0.3       # Vertical scale height (kpc)
)

# 3. Bulge (Hernquist Profile)
# gamma=1, beta=4, alpha=1
bulge = agama.Potential(
    type='Spheroid',
    mass=1e10,            # Bulge mass
    scaleRadius=1.0,      # Bulge effective radius (kpc)
    gamma=1, beta=4, alpha=1
)

# 4. The Composite Potential
# Passing a list/tuple of potentials to Agama naturally sums them
total_potential = agama.Potential(halo, disk, bulge)

# You can now pass 'total_potential' to the ActionMapper
mapper = agama.ActionMapper(total_potential)
# 2. Initialize the Action Mapper
# In Agama >= Feb 2024, initialize the mapper with the potential. 
# It caches Torus objects for unique action triplets under the hood.
mapper = agama.ActionMapper(total_potential)

# 3. Define Sampling Grid Parameters
N_tori = 500              # Number of unique action coordinates (Tori)
N_angles_per_torus = 200  # Number of angle samples per torus
# Total training samples = 100,000

print("Sampling realistic Galactic actions...")

# Sample actions with realistic galactic scales (kpc * km/s)
# Lower bounds brought near zero to capture the ultra-thin disk
# J_R and J_z should be positive.
J_R = np.random.uniform(0.1, 200.0, N_tori)   # Radial oscillations
J_z = np.random.uniform(0.1, 100.0, N_tori)    # Vertical oscillations
# Sample angular momentum, keeping it away from zero to avoid z-axis singularities
# J_phi expanded to 4000 to cover the outer edges of the disk (~15-20 kpc)
J_phi = np.random.uniform(10.0, 4000.0, N_tori) 
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