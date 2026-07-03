import os
# Force JAX memory fraction and configure logging behavior if needed
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate
import optax
from flax.serialization import msgpack_serialize
import functools

# ==============================================================================
# 1. ARCHITECTURE DEFINITION
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
# 2. DATA ENGINEERING & STREAMING PIPELINE
# ==============================================================================
def process_raw_chunk(actions, angles, potentials, phase_space):
    """Engineers trigonometric features and pairs them with physical targets."""
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    inputs = np.hstack([actions, sin_angles, cos_angles, potentials])
    return inputs, phase_space

def compute_normalization_stats(h5_file, split_assignments, required_rows=1_000_000, chunk_size=500_000):
    """Calculates Mean and Std Dev exclusively from the training split."""
    print(f"Gathering ~{required_rows:,} training rows for normalization stats...")
    total_rows = h5_file['actions'].shape[0]
    gathered_inputs, gathered_targets = [], []
    rows_gathered = 0
    
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_mask = split_assignments[chunk_start:chunk_end] == 0  # 0 = Train
        
        if not np.any(chunk_mask):
            continue
            
        actions = h5_file['actions'][chunk_start:chunk_end][chunk_mask]
        angles = h5_file['angles'][chunk_start:chunk_end][chunk_mask]
        potentials = h5_file['potentials'][chunk_start:chunk_end][chunk_mask]
        phase_space = h5_file['phase_space'][chunk_start:chunk_end][chunk_mask]
        
        inputs, targets = process_raw_chunk(actions, angles, potentials, phase_space)
        
        # FILTER OUT NaN and Infinity: Prevents bad Agama outputs from ruining the stats
        valid_mask = np.isfinite(inputs).all(axis=1) & np.isfinite(targets).all(axis=1)
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        gathered_inputs.append(inputs)
        gathered_targets.append(targets)
        rows_gathered += inputs.shape[0]
        
        if rows_gathered >= required_rows:
            break
            
    final_inputs = np.vstack(gathered_inputs)
    final_targets = np.vstack(gathered_targets)
    
    in_mu, in_std = np.mean(final_inputs, axis=0), np.std(final_inputs, axis=0)
    out_mu, out_std = np.mean(final_targets, axis=0), np.std(final_targets, axis=0)
    
    in_std[in_std == 0] = 1e-6
    out_std[out_std == 0] = 1e-6
    return in_mu, in_std, out_mu, out_std

def chunked_dataloader(h5_file, split_assignments, target_split_id, batch_size, num_devices, norm_stats, chunk_size=500_000):
    """Streams fast contiguous reads from disk and shapes batches for multi-GPU pmap."""
    in_mu, in_std, out_mu, out_std = norm_stats
    total_rows = h5_file['actions'].shape[0]
    
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_mask = split_assignments[chunk_start:chunk_end] == target_split_id
        
        if not np.any(chunk_mask):
            continue
        
        # Continuous Read + RAM Masking
        actions = h5_file['actions'][chunk_start:chunk_end][chunk_mask]
        angles = h5_file['angles'][chunk_start:chunk_end][chunk_mask]
        potentials = h5_file['potentials'][chunk_start:chunk_end][chunk_mask]
        targets = h5_file['phase_space'][chunk_start:chunk_end][chunk_mask]
        
        inputs, targets = process_raw_chunk(actions, angles, potentials, targets)
        
        # FILTER OUT NaN and Infinity: Prevent bad data from hitting the GPU
        valid_mask = np.isfinite(inputs).all(axis=1) & np.isfinite(targets).all(axis=1)
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        inputs = (inputs - in_mu) / in_std
        targets = (targets - out_mu) / out_std
        
        # Shuffle rows in RAM
        ram_shuffle = np.random.permutation(inputs.shape[0])
        inputs = inputs[ram_shuffle]
        targets = targets[ram_shuffle]
        
        # Yield sharded multi-GPU batches
        for b in range(0, inputs.shape[0], batch_size):
            x_b = inputs[b:b+batch_size]
            y_b = targets[b:b+batch_size]
            
            # Drop trailing batch fragments that don't match static pmap shape rules
            if x_b.shape[0] < batch_size:
                continue
                
            # Reshape from (BatchSize, Features) to (NumGPUs, LocalBatchSize, Features)
            x_sharded = x_b.reshape(num_devices, -1, 16)
            y_sharded = y_b.reshape(num_devices, -1, 6)
            
            yield x_sharded, y_sharded

# ==============================================================================
# 3. PARALLELIZED MULTI-GPU STEP FUNCTIONS (PMAP)
# ==============================================================================
@functools.partial(jax.pmap, axis_name='batch')
def parallel_train_step(state, batch_inputs, batch_targets):
    """Executes forward/backward pass simultaneously across all 8 GPUs."""
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch_inputs)
        loss = jnp.mean((preds - batch_targets) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, list_grads = grad_fn(state.params)
    
    # Synchronize and average losses/gradients across all GPUs over the interconnect
    synchronized_grads = jax.lax.pmean(list_grads, axis_name='batch')
    synchronized_loss = jax.lax.pmean(loss, axis_name='batch')
    
    new_state = state.apply_gradients(grads=synchronized_grads)
    return new_state, synchronized_loss

@functools.partial(jax.pmap, axis_name='batch')
def parallel_eval_step(state, batch_inputs, batch_targets):
    """Evaluates cross-validation accuracy synchronously across all GPUs."""
    preds = state.apply_fn({'params': state.params}, batch_inputs)
    loss = jnp.mean((preds - batch_targets) ** 2)
    return jax.lax.pmean(loss, axis_name='batch')

# ==============================================================================
# 4. EXECUTION DRIVER
# ==============================================================================
if __name__ == "__main__":
    data_path = '../torus_mapping_agama/data_mapping/phoenix_conditional_torus_data.h5'
    global_batch_size = 8192
    learning_rate = 1e-4  # Lowered slightly for stability
    epochs = 20

    # Device Setup Verification
    devices = jax.local_devices()
    num_devices = len(devices)
    print(f"JAX successfully linked to {num_devices} local devices: {devices}")
    assert global_batch_size % num_devices == 0, "Global batch size must be perfectly divisible by the GPU count."

    # Establish HDF5 Data Streams
    f = h5py.File(data_path, 'r')
    total_rows = f['actions'].shape[0]
    
    # Seeded Global Partition Assignments (0=Train, 1=Val, 2=Test)
    print("Partitioning master dataset indices (80/10/10 ratio)...")
    np.random.seed(42)
    split_assignments = np.random.choice([0, 1, 2], size=total_rows, p=[0.8, 0.1, 0.1])
    
    print(f"Total Rows: {total_rows:,}")
    print(f"  -> Training Split:   {np.sum(split_assignments == 0):,}")
    print(f"  -> Validation Split: {np.sum(split_assignments == 1):,}")
    print(f"  -> Testing Split:    {np.sum(split_assignments == 2):,}")

    # Compute Normalization Bounds
    norm_stats = compute_normalization_stats(f, split_assignments, required_rows=1_000_000)
    np.savez('phoenix_norm_stats.npz', in_mu=norm_stats[0], in_std=norm_stats[1], 
             out_mu=norm_stats[2], out_std=norm_stats[3])
    print("Saved normalizer constraints to 'phoenix_norm_stats.npz'")

    # Initialize Base Model State
    rng = jax.random.PRNGKey(42)
    model = PhoenixSurrogate()
    dummy_input = jnp.ones((1, 16))
    initial_params = model.init(rng, dummy_input)['params']
    
    # Chain Gradient Clipping with the Adam Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Prevents exploding gradients
        optax.adam(learning_rate)
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=initial_params, tx=optimizer)

    # Replicate Model Weights across the VRAM of all Quadro GPUs
    print("Replicating model network parameters across all devices...")
    replicated_state = replicate(state)

    print("\nInitializing Distributed Training Run...")
    for epoch in range(epochs):
        train_loss_accumulator = 0.0
        train_step_count = 0
        
        # Parallel Multi-GPU Training Loop
        for x_shard, y_shard in chunked_dataloader(f, split_assignments, 0, global_batch_size, num_devices, norm_stats):
            replicated_state, hardware_loss = parallel_train_step(
                replicated_state, jnp.array(x_shard), jnp.array(y_shard)
            )
            # Pull loss back from the first device for diagnostic tracking
            train_loss_accumulator += hardware_loss[0]
            train_step_count += 1
            
        epoch_train_loss = train_loss_accumulator / train_step_count
        
        # Parallel Multi-GPU Validation Loop
        val_loss_accumulator = 0.0
        val_step_count = 0
        for x_shard, y_shard in chunked_dataloader(f, split_assignments, 1, global_batch_size, num_devices, norm_stats):
            hardware_val_loss = parallel_eval_step(
                replicated_state, jnp.array(x_shard), jnp.array(y_shard)
            )
            val_loss_accumulator += hardware_val_loss[0]
            val_step_count += 1
            
        epoch_val_loss = val_loss_accumulator / val_step_count
        print(f"Epoch {epoch+1:02d}/{epochs} | Average Train MSE: {epoch_train_loss:.6f} | Average Val MSE: {epoch_val_loss:.6f}")

    # Out-of-Sample Test Evaluation 
    print("\nRunning unbiased validation over locked out Test Set...")
    test_loss_accumulator = 0.0
    test_step_count = 0
    for x_shard, y_shard in chunked_dataloader(f, split_assignments, 2, global_batch_size, num_devices, norm_stats):
        hardware_test_loss = parallel_eval_step(
            replicated_state, jnp.array(x_shard), jnp.array(y_shard)
        )
        test_loss_accumulator += hardware_test_loss[0]
        test_step_count += 1
        
    print(f"==> VERIFIED TEST SET LOSS (MSE): {test_loss_accumulator / test_step_count:.6f}")

    # Un-replicate model weights back to a single device structure before exporting
    print("\nCollapsing parallel parameters back into a unified weight topology...")
    final_standalone_params = unreplicate(replicated_state.params)

    # Save finalized model configuration to disk
    serialized_weights = msgpack_serialize(final_standalone_params)
    with open("phoenix_weights.msgpack", "wb") as output_file:
        output_file.write(serialized_weights)
        
    print("Export complete! Weights saved securely to 'phoenix_weights.msgpack'. Ready for local CPU deployment.")
    f.close()