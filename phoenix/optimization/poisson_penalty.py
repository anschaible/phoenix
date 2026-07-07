import jax
import jax.numpy as jnp
from typing import Callable
import functools

# Import your physical Gravitational constant (approx 4.3e-6)
from phoenix.constants import G as G_phoenix

@jax.jit
def _single_point_kde(test_x, test_y, test_z, x, y, z, weights, h):
    """Calculates the 3D density at a single test point using a Gaussian Kernel."""
    dist_sq = (x - test_x)**2 + (y - test_y)**2 + (z - test_z)**2
    
    # 3D Gaussian kernel
    kernel = jnp.exp(-0.5 * dist_sq / h**2) / ((h * jnp.sqrt(2 * jnp.pi))**3)
    
    # The density is the sum of the weighted kernels
    return jnp.sum(weights * kernel)

def get_density_from_potential(potential_fn: Callable, G: float = G_phoenix):
    """
    Returns a JAX-jittable function that computes the exact analytical density 
    rho(x, y, z) from any given gravitational potential Phi(x, y, z) using 
    the Poisson equation: rho = Laplace(Phi) / (4 * pi * G)
    """
    def rho_fn(x, y, z, *params):
        # Wrap the coordinates into a single vector for the Hessian
        def phi_vec(pos):
            return potential_fn(pos[0], pos[1], pos[2], *params)
        
        # jax.hessian computes the 3x3 matrix of second spatial derivatives
        H = jax.hessian(phi_vec)(jnp.array([x, y, z]))
        
        # Laplace(Phi) is the trace of the Hessian matrix (d2x + d2y + d2z)
        laplacian = jnp.trace(H)
        
        # Return physical density using the correct astrophysical G
        return laplacian / (4.0 * jnp.pi * G)
        
    return rho_fn

# We use static_argnums=(4,) so JAX knows the potential function won't change between calls
@functools.partial(jax.jit, static_argnums=(4,))
def compute_poisson_penalty(
    x: jax.Array, 
    y: jax.Array, 
    z: jax.Array, 
    weights: jax.Array, 
    potential_fn: Callable, 
    potential_params: tuple,
    G: float = G_phoenix,
    h: float = 0.8
) -> float:
    """
    Computes a differentiable penalty forcing the Neural Network's mapped DF density 
    to match the exact density generating the potential (via the Poisson equation).
    
    Parameters:
    -----------
    x, y, z : JAX arrays from the Neural Network mapped phase-space
    weights : JAX arrays containing the soft acceptance probabilities
    potential_fn : Callable from `potentials.py` (e.g., total_potential)
    potential_params : Tuple of parameters to pass to the potential function
    G : float, Gravitational constant (default 1.0 for Agama units)
    h : float, bandwidth of the Gaussian KDE
    """
    
    # 1. Define physical anchor points to check the density
    R_anchors = jnp.linspace(1.0, 15.0, 15)  # 1 to 15 kpc radially
    Z_anchors = jnp.linspace(0.5, 5.0, 10)   # 0.5 to 5 kpc vertically
    
    # 2. Calculate DF density at the anchor points (using KDE on mapped particles)
    vmap_kde = jax.vmap(_single_point_kde, in_axes=(0, 0, 0, None, None, None, None, None))
    
    # Radial checking (z=0)
    df_rho_R = vmap_kde(R_anchors, jnp.zeros_like(R_anchors), jnp.zeros_like(R_anchors), x, y, z, weights, h)
    # Vertical checking (x=8 kpc, moving up in z)
    df_rho_Z = vmap_kde(jnp.full_like(Z_anchors, 8.0), jnp.zeros_like(Z_anchors), Z_anchors, x, y, z, weights, h)
    
    # 3. Calculate True Physics Density directly from the Potential!
    rho_fn = get_density_from_potential(potential_fn, G)
    
    analytic_rho_R = jax.vmap(lambda r: rho_fn(r, 0.0, 0.0, *potential_params))(R_anchors)
    analytic_rho_Z = jax.vmap(lambda z_val: rho_fn(8.0, 0.0, z_val, *potential_params))(Z_anchors)
    
    # 4. Density Shape Penalty
    # Using Log10 to handle exponential drop-offs smoothly. 
    # Because we check the absolute magnitude of log10(rho), this automatically 
    # forces both the SHAPE and the TOTAL MASS to align!
    eps = 1e-10
    profile_penalty_R = jnp.mean(jnp.square(jnp.log10(df_rho_R + eps) - jnp.log10(analytic_rho_R + eps)))
    profile_penalty_Z = jnp.mean(jnp.square(jnp.log10(df_rho_Z + eps) - jnp.log10(analytic_rho_Z + eps)))
    
    total_penalty = profile_penalty_R + profile_penalty_Z
    
    return total_penalty