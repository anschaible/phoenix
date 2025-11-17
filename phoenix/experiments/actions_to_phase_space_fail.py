#This is work done by Nihat Oguz
"""
This module provides a function to map actions to phase-space
coordinates for stars. The conversion uses the guiding center 
approximation and incorporates frequencies derived from the gravitational potentials.

The function is implemented with JAX and uses random sampling of the angles to 
generate a distribution of positions and velocities.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from phoenix.distributionfunctions import Rc_from_Lz, kappa, nu, Omega, vcirc
from jax import grad

def actions_to_phase_space(Jr, Jz, Lz, params, key, Phi_xyz, theta):
    """
    Converts actions into phase-space coordinates.
    
    Parameters:
      - Jr: Radial action
      - Jz: Vertical action
      - Lz: Angular momentum along z
      - params: Dictionary of parameters that must include at least:
          * v0: Circular velocity (km/s)
      - key: PRNGKey for random sampling
    
    Returns:
      A tuple (x, y, z, v_x, v_y, v_z) representing the star's Cartesian coordinates
      and velocity components.
    """
    R0 = params["R0"]
    R_init = params["Rinit"]
    #v0 = params["v0"]
    #Rc_val = Rc_from_Lz(Lz, R_init=R0)
    Rc_val = Rc_from_Lz(Phi_xyz, Lz, R_init, *theta)
    #Guiding center radius from angular momentum
    epsilon = 0.1 
    #Rc_val = jnp.maximum(Lz / params["v0"], epsilon)
    Rc_val = jnp.maximum(Rc_val, epsilon)


    #Obtain dynamical frequencies from potentials
    kap = jnp.maximum(kappa(Phi_xyz, Rc_val, *theta), epsilon)
    nu_val = jnp.maximum(nu(Phi_xyz, Rc_val, *theta), epsilon)

    #Amplitudes for oscillations in the radial and vertical directions
    A_R = jnp.sqrt(2.0 * Jr / jnp.maximum(kap, epsilon))
    A_z = jnp.sqrt(2.0 * Jz / jnp.maximum(nu_val, epsilon))

    #Dynamical frequencies
    Omega = vcirc(Phi_xyz, Rc_val, *theta) / jnp.maximum(Rc_val, 1e-8)
    dO_dR = grad(lambda r: vcirc(Phi_xyz, r, *theta) / jnp.maximum(r, 1e-8))(Rc_val)
    
    #Sample random angles uniformly in [0, 2π]
    key, subkey = random.split(key)
    theta_R = random.uniform(subkey, minval=0.0, maxval=2 * jnp.pi)
    key, subkey = random.split(key)
    theta_z = random.uniform(subkey, minval=0.0, maxval=2 * jnp.pi)
    key, subkey = random.split(key)
    theta_phi = random.uniform(subkey, minval=0.0, maxval=2 * jnp.pi)
    
    #Compute the instantaneous radial position:
    R = Rc_val + A_R * jnp.cos(theta_R)
    #Azimuthal angle is given by the random angle theta_phi.
    phi = theta_phi
    #Vertical position: oscillates with amplitude A_z
    z = A_z * jnp.cos(theta_z)
    
    #Transform from cylindrical to Cartesian coordinates
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    
    #Radial velocity due to epicyclic motion
    v_R = -A_R * kap * jnp.sin(theta_R)
    #Azimuthal velocity: for a simple model, we use Lz/R 
    delta_vphi = - (2 * Omega / kap) * A_R * jnp.sin(theta_R) \
                + (R - Rc_val) * dO_dR
    v_phi     = Omega * R + delta_vphi
    #Vertical velocity due to vertical oscillation
    v_z = -A_z * nu_val * jnp.sin(theta_z)
    
    #Transform the radial and azimuthal velocities into Cartesian coordinates
    v_x = v_R * jnp.cos(phi) - v_phi * jnp.sin(phi)
    v_y = v_R * jnp.sin(phi) + v_phi * jnp.cos(phi)
    
    return x, y, z, v_x, v_y, v_z

#Helper function to map an array of action triplets to phase space coordinates
def map_actions_to_phase_space(candidates, params, key, Phi_xyz, theta):
    """
    Applies actions_to_phase_space to a batch of action candidates.
    
    Parameters:
      - candidates: A JAX array of shape (N, 3) where each row is (Jr, Jz, Lz)
      - params: Dictionary of parameters
      - key: PRNGKey for random sampling
      
    Returns:
      A JAX array of shape (N, 6) where each row is (x, y, z, v_x, v_y, v_z).
    """
    num = candidates.shape[0]
    keys = random.split(key, num)
    mapped = vmap(lambda cand, k: actions_to_phase_space(cand[0], cand[1], cand[2], params, k, Phi_xyz, theta))(
        candidates, keys)
    
    #mapped is a tuple of 6 arrays, each of shape (N,)
    #we use jnp.column_stack to combine them into a (N, 6) array
    return jnp.column_stack(mapped)
