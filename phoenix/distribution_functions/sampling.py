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

def sample_df_importance(
    df: Callable,
    key: jax.Array,
    params: Dict,
    Phi_xyz: Callable,
    theta: tuple,
    n_candidates: int,
    J_scales: Tuple[float, float, float],
) -> Tuple[jax.Array, jax.Array]:
    """
    Importance sampling of a DF in action space. Unbiased, tuning-free, and the
    correct estimator for the weighted moments this package actually computes.

    Why not `sample_df_potential`
    -----------------------------
    That function relaxes rejection sampling into
    `w = sigmoid((f/f_max - u) / tau)` so it stays differentiable. The relaxation
    is only faithful while `tau` is small compared with the typical value of
    `f/f_max` -- and it is not. Action-space DFs span many orders of magnitude
    over the sampled box: for the spheroid double power law with J0 = 60 and the
    default J_bounds = (500, 500, 500), the median `f/f_max` is ~8e-6 against
    tau = 0.05. Then `f/f_max - u ~ -u` for essentially every candidate and the
    weight collapses to `sigmoid(-u/tau)`, a function of the *random draw alone*:
    measured corr(w, f) = 0.06, corr(w, u) = -0.58. The sampled population stops
    responding to the DF at all -- J0_spheroid 25 -> 400 and Beta_spheroid
    3.5 -> 10 gave bit-identical tracer distributions. Shrinking tau restores
    fidelity but saturates the sigmoid and kills the gradient, so the two
    requirements are in direct conflict.

    Importance sampling removes the conflict. Because every mass-weighted map this
    package builds is of the form `sum_i w_i g(J_i)`, the DF never needs to be
    turned into a set of accepted/rejected samples: drawing from any proposal `q`
    and weighting by `w_i = f(J_i) / q(J_i)` estimates the same integrals without
    bias, with no temperature, and differentiably.

    The proposal is exponential in each action with mean `J_scales[k]`, drawn as
    `scale * Exp(1)` so it is reparameterized: gradients flow through `J_scales`
    (which the caller derives from the DF parameters) as well as through `f`.
    Matching the proposal to the DF's own scales is what makes this efficient --
    with a uniform proposal over the default box the effective sample size is
    ~55 of 40000 candidates, while an exponential proposal at 0.5*J0 gives ~6000.

    Parameters
    ----------
    df : Callable
        The DF, with signature `df(Jr, Jz, Jphi, Phi_xyz, theta, params)`.
    key : PRNGKey
    params : dict
        DF parameters.
    Phi_xyz : Callable
        Potential Phi(x, y, z).
    theta : tuple
        Extra potential parameters.
    n_candidates : int
        Number of candidates drawn.
    J_scales : tuple
        Exponential proposal means for (J_r, J_z, J_phi), in the same units as the
        actions. Should be of order the DF's own action scales; they need not be
        accurate, only free of gross mismatch (too small truncates the DF, too
        large wastes samples).

    Returns
    -------
    candidates, weights : tuple
        - candidates: sampled actions, shape (n_candidates, 3).
        - weights: importance weights f(J)/q(J), non-negative and unnormalized.
          Downstream code normalizes them to a component mass, so only their
          relative values matter.
    """
    # The proposal is only a sampling device: importance weighting is unbiased for
    # ANY proposal, so the scales carry no information the fit needs and are cut out
    # of the gradient. This matters in practice, not just for tidiness -- callers
    # derive them from the potential's epicyclic frequencies (kappa, nu, v_circ),
    # which are themselves second derivatives of Phi. Differentiating through that
    # produced NaN gradients for every potential and disk parameter partway into a
    # fit; the NaN guard in `pipeline.fit` then zeroed them, silently freezing those
    # 16 parameters at their initial values while the 5 bulge parameters (whose
    # proposal scale depends only on J0_spheroid) kept moving. stop_gradient removes
    # that failure mode without biasing the estimator.
    s_r, s_z, s_phi = (jax.lax.stop_gradient(jnp.maximum(jnp.asarray(s), 1e-6))
                       for s in J_scales)

    kr, kz, kphi = random.split(key, 3)
    # Reparameterized draws: scale * Exp(1) keeps d/d(scale) available to autodiff.
    Jr = random.exponential(kr, shape=(n_candidates,)) * s_r
    Jz = random.exponential(kz, shape=(n_candidates,)) * s_z
    Jphi = random.exponential(kphi, shape=(n_candidates,)) * s_phi
    candidates = jnp.stack([Jr, Jz, Jphi], axis=1)

    f_vals = vmap(lambda c: df(c[0], c[1], c[2], Phi_xyz, theta, params))(candidates)

    # log q of the (independent) exponential proposal, in logs so the ratio stays
    # finite when the DF spans many orders of magnitude.
    log_q = (-Jr / s_r - jnp.log(s_r)
             - Jz / s_z - jnp.log(s_z)
             - Jphi / s_phi - jnp.log(s_phi))

    # f legitimately underflows to exactly 0 far out in the tail, where log() is -inf
    # and its gradient 1/f is undefined. Guarding with `jnp.maximum(f, 1e-300)` does
    # NOT work: JAX defaults to float32, whose smallest denormal is ~1e-45, so 1e-300
    # rounds to 0.0 and the clamp is a silent no-op. The result was NaN gradients for
    # every parameter the DF depends on (the potential and disk parameters, but not
    # the bulge ones, whose DF has no potential dependence), which `pipeline.fit`
    # then zeroed -- silently freezing those parameters mid-fit.
    #
    # The two nested `where`s are both needed. The outer one selects a constant for
    # underflowed entries so their gradient is exactly 0; the inner one keeps log()
    # from ever *evaluating* at 0, because reverse-mode differentiates both branches
    # of a `where` and a -inf in the untaken branch still poisons the result with NaN.
    TINY = 1e-30  # representable in float32, and 1/TINY stays finite
    supported = f_vals > TINY
    log_f = jnp.where(supported,
                      jnp.log(jnp.where(supported, f_vals, 1.0)),
                      jnp.log(TINY))
    log_w = log_f - log_q
    # Subtract the max before exponentiating: the overall scale is irrelevant
    # because the weights are renormalized to a component mass downstream.
    weights = jnp.exp(log_w - jnp.max(log_w))
    return candidates, weights
