import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
import optax

# 1. The "Smoother" (KDE)
# This converts your discrete coordinates into a continuous density field
def get_density_from_particles(R_test, z_test, particle_coords, weights, bandwidth=0.2):
    """
    Calculates the density at a specific point (R_test, z_test) 
    by summing contributions from all particles.
    """
    # Extract particle positions
    # particle_coords shape: (N, 6) -> (x, y, z, vx, vy, vz)
    x_p = particle_coords[:, 0]
    y_p = particle_coords[:, 1]
    z_p = particle_coords[:, 2]
    
    # Convert particles to Cylindrical R
    R_p = jnp.sqrt(x_p**2 + y_p**2)
    
    # Calculate squared distance from the Test Point to every Particle
    # (We assume axisymmetry, so we only care about distance in R and z)
    dist_sq = (R_p - R_test)**2 + (z_p - z_test)**2
    
    # Gaussian Kernel: exp(-distance^2 / 2h^2)
    # bandwidth (h) controls how "smeared" the particles are. 
    # ~0.2 kpc is a good starting size for a disk.
    kernel = jnp.exp(-0.5 * dist_sq / bandwidth**2)
    
    # Sum weighted contributions
    # Normalization factor ensures the units are correct (mass/volume)
    # Note: precise normalization is tricky, but for self-consistency 
    # we often just need the SHAPE to match, so we solve for a scaling factor later.
    raw_density = jnp.sum(weights * kernel)
    
    return raw_density

# ==========================================
# 2. POTENTIAL DENSITY (The Target)
# ==========================================
def get_rho_poisson(R, z, Phi_xyz, theta):
    # (Same Poisson calculation as before)
    def pot_func(r, z_val):
        return Phi_Rz_from_xyz(Phi_xyz, r, z_val, *theta)

    dPhi_dR  = grad(pot_func, argnums=0)
    d2Phi_dR2 = grad(dPhi_dR, argnums=0)
    d2Phi_dz2 = grad(grad(pot_func, argnums=1), argnums=1)
    
    laplacian = d2Phi_dR2(R, z) + (1.0/jnp.clip(R, 1e-5)) * dPhi_dR(R, z) + d2Phi_dz2(R, z)
    return laplacian / (4.0 * jnp.pi) # Assuming G=1

# 2. The Loss Function
@jax.jit
def consistency_loss(theta_optim, params, key):
    
    # A. GENERATE COORDINATES (Forward Pass)
    # We pass 'theta_optim' so JAX can track how changing the potential moves the stars
    phase_coords, soft_weights = params_to_phasespace(
        params, theta=theta_optim, key=key, n_candidates=20000 
    )
    
    # B. DEFINE TEST POINTS
    # Where do we want to check consistency? 
    # Let's check a vertical slice at the solar radius (R=8)
    z_test_points = jnp.linspace(0.0, 2.0, 10) # 0 to 2 kpc height
    R_test_points = jnp.full_like(z_test_points, 8.0) # Fixed at R=8
    
    # C. CALCULATE DENSITIES
    
    # 1. Density from Potential (The Goal)
    # (Using the Poisson helper we discussed earlier)
    rho_pot = vmap(lambda z: get_rho_poisson(8.0, z, Phi, theta_optim))(z_test_points)
    
    # 2. Density from Particles (The Simulation)
    rho_sim = vmap(lambda r, z: get_density_from_particles(
        r, z, phase_coords, soft_weights, bandwidth=0.3
    ))(R_test_points, z_test_points)
    
    # D. NORMALIZE
    # Since the simulation has arbitrary total mass (n_candidates), 
    # we scale it so the central density matches. We only care if the PROFILE matches.
    scale_factor = rho_pot[0] / jnp.clip(rho_sim[0], 1e-10)
    rho_sim_scaled = rho_sim * scale_factor
    
    # E. CALCULATE ERROR
    # Mean Squared Error of the log-density (log makes it care about the tails too)
    return jnp.mean((jnp.log(rho_pot) - jnp.log(rho_sim_scaled))**2)