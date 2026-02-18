import jax
import jax.numpy as jnp
import optax

from phoenix.constants import G
from phoenix import sampling, actions_to_phase_space
from phoenix.potentials import miyamoto_nagai_potential, plummer_potential
from phoenix.distributionfunctions_spheroidal import f_double_power_law

def compute_density_jax(x, y, z, weights, r_bins, z_bins):
    # 1. Compute radial distance r
    r = jnp.sqrt(x**2 + y**2)

    # 2. Use histogram2d to accumulate weights efficiently
    # This replaces the entire fori_loop and digitize logic
    counts, _, _ = jnp.histogram2d(
        r, z, 
        bins=[r_bins, z_bins], 
        weights=weights
    )

    # 3. Compute the volume of each bin
    # Use midpoint r for volume calculation to be more accurate
    r_mid = (r_bins[:-1] + r_bins[1:]) / 2
    r_widths = jnp.diff(r_bins)
    z_widths = jnp.diff(z_bins)
    
    # Calculate volume of cylindrical shell: V = 2 * pi * r * dr * dz
    bin_volumes = 2 * jnp.pi * jnp.outer(r_mid * r_widths, z_widths)

    # 4. Normalize
    density = counts / bin_volumes
    valid_mask = (density > 0).astype(jnp.float32)
    
    return density, valid_mask

def compute_laplacian_on_grid(potential, theta, r_bins, z_bins):
    """
    Computes nabla^2 Phi (Laplacian) on the centers of the R-Z bins.
    """
    
    # 1. Define the Laplacian for a SINGLE point (x, y, z)
    def potential_wrapper(x, y, z):
        return potential(x, y, z, *theta)

    def laplacian_fn(x, y, z):
        # jax.hessian returns a 3x3 matrix of second derivatives
        # argnums=(0, 1, 2) makes it return a nested tuple structure relative to x,y,z
        # However, it's easier to treat input as a vector for Hessian logic,
        # but since our function takes scalars, we can just sum unmixed 2nd derivatives.
        
        # Method A: Direct curvature calculation (faster/simpler than full Hessian)
        # We use simple nested grad or specific derivative operators
        d2dx2 = jax.grad(lambda x_: jax.grad(potential_wrapper, argnums=0)(x_, y, z))(x)
        d2dy2 = jax.grad(lambda y_: jax.grad(potential_wrapper, argnums=1)(x, y_, z))(y)
        d2dz2 = jax.grad(lambda z_: jax.grad(potential_wrapper, argnums=2)(x, y, z_))(z)
        
        return d2dx2 + d2dy2 + d2dz2
    
    # 2. Vectorize the Laplacian function
    # Maps over inputs x, y, z
    laplacian_vmap = jax.vmap(laplacian_fn, in_axes=(0, 0, 0))

    # 3. Setup Grid (Bin Centers)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    
    # Create Meshgrid (indexing='ij' matches density shape)
    R_grid, Z_grid = jnp.meshgrid(r_centers, z_centers, indexing='ij')

    # 4. Convert R-Z to Cartesian X-Y-Z
    # We evaluate at y=0, so x=R
    X_flat = R_grid.flatten()
    Y_flat = jnp.zeros_like(X_flat)
    Z_flat = Z_grid.flatten()

    # 5. Compute
    laplacian_flat = laplacian_vmap(X_flat, Y_flat, Z_flat)
    
    # Reshape back to (N_r, N_z)
    laplacian_grid = laplacian_flat.reshape(R_grid.shape)
    
    return laplacian_grid

def compute_loss(key, params, Phi, theta, n_candidates, envelope_max, rbin, zbin):
    candidates, samples, soft_weights = sampling.sample_df_potential(f_double_power_law, key, params, Phi, theta, n_candidates, envelope_max, tau=0.01)
    phase_space_coords = actions_to_phase_space.map_actions_to_phase_space(samples, params, key, Phi, theta)
    x = phase_space_coords[:, 0]
    y = phase_space_coords[:, 1]
    z = phase_space_coords[:, 2]
    density, mask = compute_density_jax(x, y, z, soft_weights, r_bins=rbin, z_bins=zbin)
    nabla2_Phi = compute_laplacian_on_grid(Phi, theta, r_bins=rbin, z_bins=zbin)
    lossplane = (4*jnp.pi*G*density - nabla2_Phi)
    lossplane = lossplane * mask

    return lossplane

def compute_lossvalue(key, params, Phi, theta, n_candidates, envelope_max, rbin, zbin):
    candidates, samples, soft_weights = sampling.sample_df_potential(f_double_power_law, key, params, Phi, theta, n_candidates, envelope_max, tau=0.01)
    phase_space_coords = actions_to_phase_space.map_actions_to_phase_space(samples, params, key, Phi, theta)
    x = phase_space_coords[:, 0]
    y = phase_space_coords[:, 1]
    z = phase_space_coords[:, 2]
    density, mask = compute_density_jax(x, y, z, soft_weights, r_bins=rbin, z_bins=zbin)
    nabla2_Phi = compute_laplacian_on_grid(Phi, theta, r_bins=rbin, z_bins=zbin)
    lossplane = (4*jnp.pi*G*density - nabla2_Phi)
    lossplane = lossplane * mask
    loss = jnp.sum(lossplane**2)

    return loss

def adam_optimizer_spheroid(key, params, Phi_spheroid, theta_init, n_candidates, envelope_max, rbin, zbin, learning, num_iterations=2000):
    current_log_theta = jnp.log(theta_init)
    
    optimizer = optax.adam(learning)
    opt_state = optimizer.init(current_log_theta)

    @jax.jit
    def step(log_theta, state, step_key):
        def loss_wrapper(lt):
            t_phys = jnp.exp(lt)
            return compute_lossvalue(step_key, params, Phi_spheroid, t_phys, 
                                     n_candidates, envelope_max, rbin, zbin)
        
        # Now we differentiate w.r.t 'log_theta' directly
        loss_val, grads = jax.value_and_grad(loss_wrapper)(log_theta)
        
        updates, new_state = optimizer.update(grads, state, log_theta)
        new_log_theta = optax.apply_updates(log_theta, updates)
        
        return new_log_theta, new_state, loss_val
    loss_history = []
    theta_history = []

    for i in range(num_iterations):
        key, subkey = jax.random.split(key)
        
        current_log_theta, opt_state, loss_val = step(current_log_theta, opt_state, subkey)
        
        # Convert back to physical space for your records
        phys_theta = jnp.exp(current_log_theta)
        
        loss_history.append(float(loss_val))
        theta_history.append(phys_theta)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss_val:.4f}, Theta: {phys_theta}")
            #pbar.set_postfix({"Loss": f"{loss_val:.2e}", "Theta": [f"{t:.2e}" for t in phys_theta]})

    return jnp.exp(current_log_theta), loss_history, theta_history