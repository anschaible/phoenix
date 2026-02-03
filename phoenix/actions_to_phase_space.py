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
from phoenix.frequencies import Rc_from_Lz, nu, vcirc, kappa
from jax import grad

def actions_to_phase_space(Jr, Jz, Lz, params, key, Phi_xyz, theta):
    """
    Converts actions into phase-space coordinates.
    
    Parameters
    ----------
    Jr : float
        Radial action
    Jz : float
        Vertical action
    Lz : float
        Angular momentum about the z-axis
    params : dict
        Dictionary of parameters, must include 'R0' (initial guess for Rc)
    key : jax.random.PRNGKey
        Random key for sampling angles
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    theta : tuple
        Additional parameters for the potential   

    Returns
    -------
    x, y, z : float
        Cartesian positions
    v_x, v_y, v_z : float
        Cartesian velocities
    """
    R0 = params["R0"]
    #v0 = 200.0 #params["v0"]
    #Rc_val = Rc_of_Lz(Lz, v0)
    Rc_val = Rc_from_Lz(Phi_xyz, Lz, R0, *theta)
    #Guiding center radius from angular momentum
    epsilon = 1e-6
    #Rc_val = jnp.maximum(Lz / params["v0"], epsilon)
    Rc_val = jnp.maximum(Rc_val, epsilon)

    Jr = jnp.sqrt(Jr**2 + epsilon**2)
    Jz = jnp.sqrt(Jz**2 + epsilon**2)


    #Obtain dynamical frequencies from potentials
    kap = kappa(Phi_xyz, Rc_val, *theta)
    #kap = kappa(Rc_val)
    nu_val = nu(Phi_xyz, Rc_val, *theta)

    #Amplitudes for oscillations in the radial and vertical directions
    A_R = jnp.sqrt(2.0 * Jr / jnp.maximum(kap, epsilon))
    A_z = jnp.sqrt(2.0 * Jz / jnp.maximum(nu_val, epsilon))

    #Dynamical frequencies
    Omega = vcirc(Phi_xyz, Rc_val, *theta) / Rc_val
    dO_dR = grad(lambda r: vcirc(Phi_xyz, Rc_val, *theta) / r)(Rc_val)
    
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