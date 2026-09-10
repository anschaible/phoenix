import jax
import jax.numpy as jnp
from typing import Callable
import numpy as np
import functools

# Import physical Gravitational constant
from phoenix.constants import G as G_phoenix

@jax.jit
def _single_point_kde(test_x, test_y, test_z, x, y, z, weights, h):
    """
    Calculates the smooth 3D mass density at a single test coordinate using 
    Gaussian Kernel Density Estimation (KDE).

    This function smears every discrete particle into a 3D Gaussian sphere 
    and sums their overlapping contributions at the target location. It is 
    JIT-compiled for high performance.

    Args:
        test_x (float or Array): The x-coordinate of the point to evaluate.
        test_y (float or Array): The y-coordinate of the point to evaluate.
        test_z (float or Array): The z-coordinate of the point to evaluate.
        x (Array): 1D array of particle x-coordinates (shape: [N]).
        y (Array): 1D array of particle y-coordinates (shape: [N]).
        z (Array): 1D array of particle z-coordinates (shape: [N]).
        weights (Array): 1D array of particle masses/weights (shape: [N]).
        h (float): The smoothing length (Gaussian bandwidth). Controls how 
            "fuzzy" or smeared out the particles are.

    Returns:
        float or Array: The scalar 3D mass density at the requested test point.
    """
    dist_sq = (x - test_x)**2 + (y - test_y)**2 + (z - test_z)**2
    
    # 3D Gaussian kernel
    kernel = jnp.exp(-0.5 * dist_sq / h**2) / ((h * jnp.sqrt(2 * jnp.pi))**3)
    
    # The density is the sum of the weighted kernels
    return jnp.sum(weights * kernel)

def get_density_from_potential(potential_fn: Callable, G: float = G_phoenix):
    """
    Creates a JAX-jittable function that computes the exact analytical mass 
    density rho(x, y, z) for any arbitrary gravitational potential Phi(x, y, z).

    This uses automatic differentiation to solve Poisson's equation:
    rho = Laplace(Phi) / (4 * pi * G).

    Args:
        potential_fn (Callable): A function computing the gravitational potential. 
            Must take signature `(x, y, z, *params)` and return a scalar.
        G (float, optional): The gravitational constant in the simulation's 
            internal unit system. Defaults to `G_phoenix`.

    Returns:
        Callable: A new function with signature `rho_fn(x, y, z, *params)` that 
        evaluates the exact mass density at the given coordinates.
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

# static_argnames is cleaner than static_argnums. JAX needs to know if 
# grid_size changes, because changing grid dimensions requires recompilation.
@functools.partial(jax.jit, static_argnames=('potential_fn', 'grid_size', 'shape_only',
                                             'match_kernel', 'n_quad'))
def compute_poisson_penalty(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    weights: jax.Array,
    potential_fn: Callable,
    potential_params: tuple,
    G: float = G_phoenix,
    h: float = 0.8,
    grid_size: int = 30,
    extent_x: float = 15.0,
    extent_z: float = 10.0,
    shape_only: bool = True,
    density_floor_frac: float = 1e-2,
    match_kernel: bool = False,
    n_quad: int = 5,
) -> float:
    """
    Computes a differentiable physics penalty forcing the neural network's mapped 
    phase-space density to match the true analytical density generating the potential.

    Instead of 1D anchors, this evaluates the 3D volume density over a full 2D 
    cross-sectional grid (slice plane at y=0) in the x-z plane.

    Args:
        x, y, z (Array): Arrays of particle coordinates from the NN mapped phase-space.
        weights (Array): Array of particle masses/weights.
        potential_fn (Callable): The gravitational potential function.
        potential_params (tuple): Parameters to pass into the potential function.
        G (float, optional): Gravitational constant. Defaults to the physical
            `phoenix.constants.G`, so the analytic density comes out in the same
            mass units as the tracer weights. (The previous default of 1.0 left an
            enormous unit mismatch that only went unnoticed because `shape_only`
            silently absorbs any constant factor.)
        h (float, optional): Bandwidth of the 3D Gaussian KDE. Defaults to 0.8.
        grid_size (int, optional): Resolution of the grid. Defaults to 30.
        extent_x (float, optional): Physical half-width of the grid. Defaults to 15.0.
        extent_z (float, optional): Physical half-height of the grid. Defaults to 10.0.
        shape_only (bool, optional): If True, compares only the shape of the log-density
            by subtracting the mean offset, ignoring KDE blurring biases. Defaults to True.
        density_floor_frac (float, optional): Only grid pixels whose tracer density
            exceeds `density_floor_frac * max(tracer density)` enter the penalty.
            This matters enormously: most of the grid lies outside the galaxy, where
            the KDE is exactly zero, and comparing `log10(0 + eps)` against a small
            but finite analytic density manufactures multi-dex residuals that
            dominate the mean and have nothing to do with self-consistency.
            Defaults to 1e-2.
        match_kernel (bool, optional): If True, the analytic density is convolved
            with the same Gaussian kernel used for the tracer KDE (via Gauss-Hermite
            quadrature), so the two sides are compared at matched resolution. The KDE
            bandwidth is typically much larger than a thin disk's scale height, so an
            unconvolved comparison biases the penalty even for a perfectly
            self-consistent model. More rigorous but costs `n_quad**3` extra density
            evaluations per pixel. Defaults to False.
        n_quad (int, optional): Gauss-Hermite nodes per dimension when
            `match_kernel=True`. Defaults to 5.

    Returns:
        float: The scalar penalty (Mean Squared Error in log-space) over the pixels
        where the tracer population actually has support.
    """

    # 1. Grid Definition (Slice at y=0)
    dx = 2.0 * extent_x / grid_size
    dz = 2.0 * extent_z / grid_size

    X_centers = jnp.linspace(-extent_x + dx/2, extent_x - dx/2, grid_size)
    Z_centers = jnp.linspace(-extent_z + dz/2, extent_z - dz/2, grid_size)

    # 2. Calculate DF Density on the Grid using Vectorized 3D KDE
    # Broadcast shapes: (grid_z, grid_x, N_particles)
    # Evaluates distance from every pixel on the y=0 plane to EVERY 3D particle
    dx_arr = X_centers[None, :, None] - x[None, None, :]
    dz_arr = Z_centers[:, None, None] - z[None, None, :]
    dy_arr = 0.0 - y[None, None, :]  # Evaluating at the y=0 cross-section

    dist_sq = dx_arr**2 + dy_arr**2 + dz_arr**2

    # 3D Gaussian kernel normalization: 1 / (h * sqrt(2pi))^3
    norm = 1.0 / ((h * jnp.sqrt(2.0 * jnp.pi)) ** 3)
    kernel = norm * jnp.exp(-0.5 * dist_sq / h**2)

    # Summing over particles (axis -1) gives a (grid_z, grid_x) map of volume densities
    df_rho_grid = jnp.sum(weights[None, None, :] * kernel, axis=-1)

    # 3. Calculate True Physics Density directly from the Potential (Poisson equation)
    rho_fn = get_density_from_potential(potential_fn, G)

    # Create a 2D meshgrid of coordinates for the analytic evaluation
    X_mesh, Z_mesh = jnp.meshgrid(X_centers, Z_centers)

    if match_kernel:
        # Convolve the analytic density with the SAME Gaussian kernel used above,
        # so both sides are compared at matched resolution. Gauss-Hermite nodes
        # for a Gaussian of width h: offsets sqrt(2)*h*t_i with weights w_i/sqrt(pi).
        t_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
        offsets = np.sqrt(2.0) * h * t_nodes
        w_nodes = w_nodes / np.sqrt(np.pi)
        ox, oy, oz = np.meshgrid(offsets, offsets, offsets, indexing='ij')
        wx, wy, wz = np.meshgrid(w_nodes, w_nodes, w_nodes, indexing='ij')
        node_off = jnp.asarray(np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1))
        node_w = jnp.asarray((wx * wy * wz).ravel())

        def _rho_shifted(off):
            return jax.vmap(
                lambda x_val, z_val: rho_fn(x_val + off[0], off[1], z_val + off[2],
                                            *potential_params)
            )(X_mesh.flatten(), Z_mesh.flatten())

        shifted = jax.vmap(_rho_shifted)(node_off)              # (n_quad**3, grid**2)
        analytic_rho_flat = jnp.sum(node_w[:, None] * shifted, axis=0)
    else:
        # vmap the analytic density function over the flattened grid coordinates
        vmap_rho = jax.vmap(lambda x_val, z_val: rho_fn(x_val, 0.0, z_val, *potential_params))
        analytic_rho_flat = vmap_rho(X_mesh.flatten(), Z_mesh.flatten())

    # Reshape back to (grid_size, grid_size). Clamp at zero: a combined potential can
    # yield a slightly negative Laplacian in places, and log10 of a negative value is
    # NaN, which would poison the gradient for the whole run.
    analytic_rho_grid = jnp.maximum(analytic_rho_flat.reshape(grid_size, grid_size), 0.0)
    df_rho_grid = jnp.maximum(df_rho_grid, 0.0)

    # 4. Density Shape Penalty (Log-Space MSE) over the tracer-supported region
    # Restrict to pixels where the tracer population actually has mass. Outside the
    # galaxy the KDE is identically zero and any floor turns log10 into a large
    # constant, so unmasked pixels would dominate the penalty with pure artifacts.
    mask = df_rho_grid > density_floor_frac * jnp.max(df_rho_grid)
    n_pix = jnp.maximum(jnp.sum(mask), 1)

    # Relative (not absolute) floors, so the regularization scales with the data
    # instead of depending on the physical units of the density.
    eps_model = 1e-6 * jnp.max(df_rho_grid)
    eps_target = 1e-6 * jnp.max(analytic_rho_grid)
    diff_grid = jnp.log10(df_rho_grid + eps_model) - jnp.log10(analytic_rho_grid + eps_target)

    if shape_only:
        # Scale-invariant loss: penalize profile SHAPE only. The offset must be the
        # mean over the *masked* pixels, otherwise excluded pixels still bias it.
        mean_diff = jnp.sum(jnp.where(mask, diff_grid, 0.0)) / n_pix
        diff_grid = diff_grid - mean_diff

    total_penalty = jnp.sum(jnp.where(mask, jnp.square(diff_grid), 0.0)) / n_pix

    return total_penalty