#This code is work done by Nihat Oguz
"""
This module implements a differentiable sampling pipeline using a soft acceptance approach.
A sigmoid function is applied to assign acceptance weights to candidate actions generated from the total DF. 
These soft weights are used downstream to compute weighted mock observables or likelihoods.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.nn import sigmoid
from typing import Callable, Dict, Tuple
from jax.typing import ArrayLike

@jax.jit
def soft_acceptance(df_vals: ArrayLike, rand_vals: ArrayLike, envelope_max: float, tau: float = 0.01) -> jax.Array:
    """
    Computes a soft acceptance mask using a sigmoid function.

    Parameters
    ----------
    df_vals : array
        An array of DF values for each candidate.
    rand_vals : array
        An array of uniformly generated random values between 0 and 1.
    envelope_max : float
        The normalization factor used in the hard acceptance comparison (max value of DF).
    tau : float, optional
        The temperature parameter controlling the softness of the transition. Default is 0.01.

    Returns
    -------
    soft_weights : array
        Values between 0 and 1 representing the "softness" of the acceptance.
    """
    return sigmoid((df_vals / envelope_max - rand_vals) / tau)


def sample_df_potential(
    df: Callable,
    key: jax.Array,
    params: Dict,
    Phi_xyz: Callable,
    theta: tuple,
    n_candidates: int,
    envelope_max: float = None,
    J_bounds: Tuple[float, float, float] = (200.0, 200.0, 6000.0),
    tau: float = 0.01
) -> Tuple[jax.Array, jax.Array]:
    """
    Differentiable version of the sampling pipeline.
    
    Instead of a hard acceptance/rejection, returns the original candidates and their continuous acceptance weights.
    
    Parameters
    ----------
    df : Callable
        The distribution function to sample from. Signature must be: df(Jr, Jz, Lz, Phi_xyz, theta, params)
    key : PRNGKey
        A JAX PRNGKey for random number generation.
    params : dict
        Dictionary with DF parameters.
    Phi_xyz : Callable
        Potential function Phi(x, y, z).
    theta : tuple
        Additional parameters for the potential.
    n_candidates : int
        Number of candidate samples to generate.
    envelope_max : float, optional
        Normalization factor (maximum expected value of the DF in the sampled
        volume). If None (default), it is auto-calibrated as the maximum DF value
        over the drawn candidates — the textbook rejection-sampling envelope.
        This is far more robust than a fixed constant: a hand-picked envelope
        computed from one fixed action point can underflow to zero when the DF
        parameters shift (e.g. a far-off optimization start), which would make
        `df_vals / envelope_max` divide by zero and poison everything downstream
        with NaNs. Auto-calibration tracks the DF as the parameters move.
    J_bounds : tuple
        Maximum sampling boundaries for (J_r, J_z, L_z). Default is (200.0, 200.0, 6000.0).
    tau : float, optional
        Temperature parameter for soft acceptance. Default is 0.01.
      
    Returns
    -------
    candidates, soft_weights : tuple
        - candidates: The exact sampled actions (n_candidates, 3).
        - soft_weights: The differentiable acceptance probability for each candidate (n_candidates,).
    """
    
    Jr_max, Jz_max, Lz_max = J_bounds
    
    # Generate uniform action candidates
    key, kr, kz, kphi, krand = random.split(key, 5)
    
    Jr_candidates = random.uniform(kr, shape=(n_candidates,), minval=0.0, maxval=Jr_max)
    Jz_candidates = random.uniform(kz, shape=(n_candidates,), minval=0.0, maxval=Jz_max)
    Lz_candidates = random.uniform(kphi, shape=(n_candidates,), minval=0.0, maxval=Lz_max)
    
    candidates = jnp.stack([Jr_candidates, Jz_candidates, Lz_candidates], axis=1)
    
    # Evaluate the total DF for each candidate using vmap (Removed the inner 'jit')
    df_vmap = vmap(lambda c: df(c[0], c[1], c[2], Phi_xyz, theta, params))
    df_vals = df_vmap(candidates)

    # Auto-calibrate the envelope from the candidates' own maximum DF value when
    # not supplied. Floored away from zero so an all-but-vanishing DF can never
    # cause a division-by-zero (NaN) in soft_acceptance.
    if envelope_max is None:
        envelope_max = jnp.maximum(jnp.max(df_vals), 1e-30)

    # Generate uniform random numbers for rejection comparison
    rand_vals = random.uniform(krand, shape=(n_candidates,))

    # Compute the soft acceptance weights
    soft_weights = soft_acceptance(df_vals, rand_vals, envelope_max, tau)
    
    # Return the un-altered candidates and their corresponding weights
    return candidates, soft_weights