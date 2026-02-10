#This code is work done by Nihat Oguz
"""
This module implements a differentiable sampling pipeline using a soft acceptance approach.
A sigmoid function is applied to assign acceptance weights to candidate actions generated from the total DF. These weighted candidate actions
can then be transformed into phase-space coordinates.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.nn import sigmoid

def soft_acceptance(df_vals, rand_vals, envelope_max, tau=0.01):
    """
    Computes a soft acceptance mask using a sigmoid function.

    Parameters
    ----------
    df_vals : array
        An array of DF values for each candidate.
    rand_vals : array
        An array of uniformly generated random values between 0 and 1.
    envelope_max : float
        The normalization factor used in the hard acceptance comparison.
    tau : float, optional
        The temperature parameter controlling the softness of the transition (smaller values result in a harder transition). Default is 0.01.

    Returns
    -------
    Sigmoid(x) : An array of values between 0 and 1 representing the "softness" of the acceptance, 
    where values close to 1 indicate high acceptance and values close to 0 indicate low acceptance.
    """
    return sigmoid((df_vals / envelope_max - rand_vals) / tau)


def sample_df_potential(df, key, params, Phi_xyz, theta, n_candidates, envelope_max, tau=0.01, **kwargs):
    """
    Differentiable version of the sampling pipeline.
    
    Instead of a hard acceptance/rejection, returns weighted candidate actions.
    
    Parameters
    ----------
    df : Function
        The distribution function to sample from.
    key : PRNGKey
        A JAX PRNGKey for random number generation.
    params : dict
        Dictionary with DF parameters.
    Phi_xyz : Function
        Potential function Phi(x, y, z).
    theta : dict
        Additional parameters for the potential.
    n_candidates : int
        Number of candidate samples to generate.
    envelope_max : float
        Normalization factor.
    tau : float, optional
        Temperature parameter for soft acceptance. Default is 0.01.
      
    Returns
    -------
    candidates, weighted_candidates, soft_weights : tuple
        Candidates (n_candidates, 3), weighted candidates (n_candidates, 3), and their soft acceptance weights (n_candidates,).
    """
    key, subkey = random.split(key)
    Jr_candidates = random.uniform(subkey, shape=(n_candidates,), minval=0.0, maxval=200.0)
    key, subkey = random.split(key)
    Jz_candidates = random.uniform(subkey, shape=(n_candidates,), minval=0.0, maxval=200.0)
    key, subkey = random.split(key)
    Lz_candidates = random.uniform(subkey, shape=(n_candidates,), minval=0.0, maxval=6000.0)
    
    candidates = jnp.stack([Jr_candidates, Jz_candidates, Lz_candidates], axis=1)
    
    #Evaluate the total DF for each candidate
    df_total_vec = jit(vmap(lambda cand: df(cand[0], cand[1], cand[2], Phi_xyz, theta, params)))
    df_vals = df_total_vec(candidates)
    
    #Generate uniform random numbers
    key, subkey = random.split(key)
    rand_vals = random.uniform(subkey, shape=(n_candidates,))
    
    #Compute the soft acceptance weights
    soft_weights = soft_acceptance(df_vals, rand_vals, envelope_max, tau)
    
    #here we multiply each candidate by its weight
    weighted_candidates = candidates * soft_weights[:, None]
    
    return candidates, weighted_candidates, soft_weights