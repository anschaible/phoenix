import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from flax.serialization import msgpack_serialize

# ==============================================================================
# 1. NETWORK ARCHITECTURE
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

class TorusMappingSurrogate(nn.Module):
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
# 2. DATA PIPELINE & NORMALIZATION
# ==============================================================================
def process_raw_chunk(actions, angles, potentials, phase_space):
    """Engineers the features (sin/cos) and concats into a single input vector."""
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    inputs = np.hstack([actions, sin_angles, cos_angles, potentials])
    return inputs, phase_space

def compute_normalization_stats(h5_file, split_assignments, required_rows=1_000_000, chunk_size=500_000):
    """Computes Mean and Std Dev STRICTLY on the training set to avoid data leakage."""
    print(f"Gathering ~{required_rows:,} training rows for normalization stats...")
    
    total_rows = h5_file['actions'].shape[0]
    gathered_inputs = []
    gathered_targets = []
    rows_gathered = 0
    
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        
        # Identify which rows in this chunk belong to the training set
        chunk_mask = split_assignments[chunk_start:chunk_end] == 0  # 0 = Train
        
        if not np.any(chunk_mask):
            continue
            
        actions = h5_file['actions'][chunk_start:chunk_end][chunk_mask]
        angles = h5_file['angles'][chunk_start:chunk_end][chunk_mask]
        potentials = h5_file['potentials'][chunk_start:chunk_end][chunk_mask]
        phase_space = h5_file['phase_space'][chunk_start:chunk_end][chunk_mask]
        
        inputs, targets = process_raw_chunk(actions, angles, potentials, phase_space)
        
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

def chunked_dataloader(h5_file, split_assignments, target_split_id, batch_size, norm_stats, chunk_size=500_000):
    """Reads contiguous chunks from disk, extracts the correct split, and yields GPU batches."""
    in_mu, in_std, out_mu, out_std = norm_stats
    total_rows = h5_file['actions'].shape[0]
    
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        
        # Mask out only the rows that belong to train/val/test
        chunk_mask = split_assignments[chunk_start:chunk_end] == target_split_id
        
        if not np.any(chunk_mask):
            continue
        
        # Read contiguous HDF5 block, then apply boolean mask in RAM (Fastest method)
        actions = h5_file['actions'][chunk_start:chunk_end][chunk_mask]
        angles = h5_file['angles'][chunk_start:chunk_end][chunk_mask]
        potentials = h5_file['potentials'][chunk_start:chunk_end][chunk_mask]
        targets = h5_file['phase_space'][chunk_start:chunk_end][chunk_mask]
        
        inputs, targets = process_raw_chunk(actions, angles, potentials, targets)
        
        # Normalize
        inputs = (inputs - in_mu) / in_std
        targets = (targets - out_mu) / out_std
        
        # Shuffle this filtered chunk in RAM to ensure stochasticity
        chunk_len = inputs.shape[0]
        ram_shuffle = np.random.permutation(chunk_len)
        inputs = inputs[ram_shuffle]
        targets = targets[ram_shuffle]
        
        # Yield mini-batches
        for b in range(0, chunk_len, batch_size):
            # Ensure we only yield full batches or the remainder
            yield inputs[b:b+batch_size], targets[b:b+batch_size]

# ==============================================================================
# 3. JAX TRAINING LOGIC
# ==============================================================================
@jax.jit
def train_step(state, batch_inputs, batch_targets):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch_inputs)
        loss = jnp.mean((preds - batch_targets) ** 2) 
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch_inputs, batch_targets):
    preds = state.apply_fn({'params': state.params}, batch_inputs)
    loss = jnp.mean((preds - batch_targets) ** 2)
    return loss

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Settings
    data_path = '../torus_mapping_agama/data_mapping/phoenix_conditional_torus_data.h5'
    batch_size = 8192
    learning_rate = 1e-3
    epochs = 20

    print("Checking JAX devices:", jax.devices())

    # Open dataset
    f = h5py.File(data_path, 'r')
    total_rows = f['actions'].shape[0]
    
    # Generate the global Random Split Array (0=Train, 1=Val, 2=Test)
    print("Generating global random split mask (80/10/10)...")
    np.random.seed(42) # Ensure reproducible splits across runs
    split_assignments = np.random.choice([0, 1, 2], size=total_rows, p=[0.8, 0.1, 0.1])
    
    train_count = np.sum(split_assignments == 0)
    val_count = np.sum(split_assignments == 1)
    test_count = np.sum(split_assignments == 2)
    print(f"Total Data: {total_rows:,} rows")
    print(f"Random Split -> Train: {train_count:,} | Val: {val_count:,} | Test: {test_count:,}")

    # 1. Compute and Save Normalization Stats (Train Set Only!)
    norm_stats = compute_normalization_stats(f, split_assignments, required_rows=1_000_000)
    np.savez('phoenix_norm_stats.npz', in_mu=norm_stats[0], in_std=norm_stats[1], 
             out_mu=norm_stats[2], out_std=norm_stats[3])
    print("Saved normalization stats to 'phoenix_norm_stats.npz'")

    # 2. Initialize Model
    rng = jax.random.PRNGKey(42)
    model = TorusMappingSurrogate()
    dummy_input = jnp.ones((1, 16)) 
    params = model.init(rng, dummy_input)['params']
    
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # 3. Training Loop
    print("\nStarting Training...")
    for epoch in range(epochs):
        train_loss = 0.0
        train_batches = 0
        
        # Train pass (Target ID = 0)
        for x_batch, y_batch in chunked_dataloader(f, split_assignments, target_split_id=0, batch_size=batch_size, norm_stats=norm_stats):
            state, loss = train_step(state, jnp.array(x_batch), jnp.array(y_batch))
            train_loss += loss
            train_batches += 1
            
        avg_train_loss = train_loss / train_batches
        
        # Validation pass (Target ID = 1)
        val_loss = 0.0
        val_batches = 0
        for x_batch, y_batch in chunked_dataloader(f, split_assignments, target_split_id=1, batch_size=batch_size, norm_stats=norm_stats):
            loss = eval_step(state, jnp.array(x_batch), jnp.array(y_batch))
            val_loss += loss
            val_batches += 1
            
        avg_val_loss = val_loss / val_batches
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 4. Final Evaluation on Locked Test Set
    print("\nTraining Complete. Evaluating on unseen Test Set...")
    test_loss = 0.0
    test_batches = 0
    # Test pass (Target ID = 2)
    for x_batch, y_batch in chunked_dataloader(f, split_assignments, target_split_id=2, batch_size=batch_size, norm_stats=norm_stats):
        loss = eval_step(state, jnp.array(x_batch), jnp.array(y_batch))
        test_loss += loss
        test_batches += 1
        
    avg_test_loss = test_loss / test_batches
    print(f"FINAL TEST LOSS (MSE): {avg_test_loss:.6f}")

    # 5. Save Weights
    bytes_output = msgpack_serialize(state.params)
    with open("phoenix_weights.msgpack", "wb") as file:
        file.write(bytes_output)
    
    print("\nSaved model weights to 'phoenix_weights.msgpack'.")
    f.close()