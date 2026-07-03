import os
# Force JAX to use the CPU for inference (avoids CUDA errors on non-GPU nodes)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.serialization import msgpack_restore
import functools

# ==============================================================================
# 1. RECREATE THE EXACT ARCHITECTURE
# ==============================================================================
class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.features)(x)
        x = nn.silu(x)
        x = nn.Dense(self.features)(x)
        return nn.silu(x + residual)

class PhoenixSurrogate(nn.Module):
    hidden_features: int = 256
    num_blocks: int = 4
    output_features: int = 6

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_features)(x)
        x = nn.silu(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(features=self.hidden_features)(x)
        x = nn.Dense(self.output_features)(x)
        return x

# ==============================================================================
# 2. PHOENIX MAPPER CLASS
# ==============================================================================
class PhoenixMapper:
    def __init__(self, weights_path=None, stats_path=None):
        # Anchor the paths to the absolute location of this script file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Resolve default paths relative to the script's location
        if weights_path is None:
            weights_path = os.path.join(current_dir, "..", "torus_mapping_neural_network", "phoenix_weights.msgpack")
        if stats_path is None:
            stats_path = os.path.join(current_dir, "..", "torus_mapping_neural_network", "phoenix_norm_stats.npz")
            
        print(f"Loading Phoenix Surrogate Model...")
        print(f" -> Weights: {os.path.normpath(weights_path)}")
        print(f" -> Stats:   {os.path.normpath(stats_path)}")
        
        # 1. Load Normalization Stats
        stats = np.load(stats_path)
        self.in_mu = jnp.array(stats['in_mu'])
        self.in_std = jnp.array(stats['in_std'])
        self.out_mu = jnp.array(stats['out_mu'])
        self.out_std = jnp.array(stats['out_std'])
        
        # 2. Initialize Empty Model
        self.model = PhoenixSurrogate()
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 16))
        
        # We need an empty parameter tree to act as a template
        empty_params = self.model.init(rng, dummy_input)['params']
        
        # 3. Load Trained Weights
        with open(weights_path, "rb") as f:
            self.params = msgpack_restore(f.read())
            
        print("Model loaded and ready for inference!")

    @functools.partial(jax.jit, static_argnums=(0,))
    def _forward_pass(self, normalized_inputs):
        """JIT-compiled forward pass for maximum speed."""
        return self.model.apply({'params': self.params}, normalized_inputs)

    def map_to_phase_space(self, actions, angles, potentials):
        """
        Maps physical inputs to phase space coordinates (x, y, z, vx, vy, vz).
        
        Parameters:
        - actions: numpy array of shape (N, 3) -> [J_R, J_z, J_phi]
        - angles: numpy array of shape (N, 3) -> [theta_R, theta_z, theta_phi]
        - potentials: numpy array of shape (N, 7) -> [M_halo, R_halo, M_disk, R_disk, H_disk, M_bulge, R_bulge]
        
        Returns:
        - phase_space: numpy array of shape (N, 6)
        """
        # 1. Engineer Features
        sin_angles = jnp.sin(angles)
        cos_angles = jnp.cos(angles)
        
        # Concatenate into shape (N, 16)
        inputs = jnp.hstack([actions, sin_angles, cos_angles, potentials])
        
        # 2. Normalize Inputs
        norm_inputs = (inputs - self.in_mu) / self.in_std
        
        # 3. Predict (Forward Pass)
        norm_preds = self._forward_pass(norm_inputs)
        
        # 4. Un-normalize Outputs
        phase_space = (norm_preds * self.out_std) + self.out_mu
        
        # Return as a standard NumPy array for easy handling
        return np.array(phase_space)

# ==============================================================================
# 3. EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    import functools # Required for the JIT decorator above
    
    # Initialize the mapper
    mapper = PhoenixMapper()
    
    # --- Create some dummy physical inputs to test ---
    N_samples = 5
    
    # Actions: [J_R, J_z, J_phi]
    test_actions = np.array([
        [10.0, 5.0, 1000.0],
        [20.0, 10.0, 1500.0],
        [5.0, 2.0, 500.0],
        [50.0, 30.0, 2500.0],
        [100.0, 50.0, 3500.0]
    ])
    
    # Angles: Randomly drawn from 0 to 2pi
    test_angles = np.random.uniform(0, 2 * np.pi, (N_samples, 3))
    
    # Potential Parameters: Must match the scaling used during training!
    # Remember: Masses were divided by 1e11 during dataset generation.
    # [M_halo/1e11, R_halo, M_disk/1e11, R_disk, H_disk, M_bulge/1e11, R_bulge]
    single_potential = np.array([10.0, 20.0, 0.5, 3.0, 0.3, 0.1, 1.0])
    test_potentials = np.tile(single_potential, (N_samples, 1))
    
    # --- Run the Mapping ---
    print("\nRunning Inference...")
    phase_space_out = mapper.map_to_phase_space(test_actions, test_angles, test_potentials)
    
    # --- Display Results ---
    np.set_printoptions(precision=3, suppress=True)
    for i in range(N_samples):
        print(f"\nSample {i+1}:")
        print(f"  Actions (J)   : {test_actions[i]}")
        print(f"  Angles (θ)    : {test_angles[i]}")
        print(f"  --> Pos (kpc) : {phase_space_out[i, 0:3]}")
        print(f"  --> Vel (km/s): {phase_space_out[i, 3:6]}")