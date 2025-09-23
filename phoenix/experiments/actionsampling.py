# produces points with probability proportional to f(J)
# sampleactions.py
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped
from jax.typing import ArrayLike

# ---- imports from your other modules ----
from phoenix.distributionfunctions import (
    Sigma_exp, sigmaR_of_Rc, sigmaz_of_Rc,
    vcirc, kappa, nu, Rc_from_Lz
)

# ===========================================================
# Internal helpers (strict, JAX-only types)
# ===========================================================

@jaxtyped
def _sample_exponential_core(key: Array, scale: Float[Array, ""]) -> Float[Array, ""]:
    """Draw X ~ Exp(mean=scale) as a JAX scalar."""
    u = jax.random.uniform(key, minval=1e-12, maxval=1.0 - 1e-12)
    return -scale * jnp.log(u)

@jaxtyped
def _inverse_cdf_sample_Rc_core(
    key: Array,
    Phi_xyz: Callable, *theta: Float[Array, ""],
    Rmin: Float[Array, ""], Rmax: Float[Array, ""],
    R0:   Float[Array, ""], Rd:  Float[Array, ""], Sigma0: Float[Array, ""],
    n_grid: int = 4096,
) -> Float[Array, ""]:
    """
    Sample guiding radius Rc from p(R) ∝ Σ(R) R on [Rmin,Rmax].
    """
    R = jnp.linspace(Rmin, Rmax, n_grid)
    Sigma = Sigma_exp(R, R0, Rd, Sigma0)
    pdf = jnp.clip(Sigma * R, 0.0, jnp.inf)
    cdf = jnp.cumsum(pdf)
    cdf = cdf / jnp.clip(cdf[-1], 1e-20)

    u = jax.random.uniform(key)
    idx = jnp.searchsorted(cdf, u, side="left")
    idx = jnp.clip(idx, 1, n_grid - 1)
    t = (u - cdf[idx - 1]) / jnp.clip(cdf[idx] - cdf[idx - 1], 1e-12)
    Rc = R[idx - 1] + t * (R[idx] - R[idx - 1])
    return Rc

# ===========================================================
# Core samplers (strict, JAX-only types) — used under jit/vmap
# ===========================================================

@jaxtyped
def _sample_actions_one_core(
    key: Array,
    Phi_xyz: Callable, *theta: Float[Array, ""],
    R0: Float[Array, ""], Rd: Float[Array, ""], Sigma0: Float[Array, ""],
    RsigR: Float[Array, ""], RsigZ: Float[Array, ""],
    sigmaR0_R0: Float[Array, ""], sigmaz0_R0: Float[Array, ""],
    Rmin: Float[Array, ""], Rmax: Float[Array, ""],
    Rinit_for_Rc: Float[Array, ""],
):
    """
    Draw a single (J_R, J_z, J_phi) ~ quasi-isothermal DF (factorized sampler).
    Returns JR, Jz, Jphi, Rc, kappa, nu as JAX scalars.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # 1) sample guiding radius Rc
    Rc = _inverse_cdf_sample_Rc_core(
        k1, Phi_xyz, *theta,
        Rmin=Rmin, Rmax=Rmax, R0=R0, Rd=Rd, Sigma0=Sigma0
    )

    # 2) circular speed and angular momentum (prograde)
    vc = vcirc(Phi_xyz, Rc, *theta)
    Jphi = Rc * vc

    # 3) local frequencies
    kap = kappa(Phi_xyz, Rc, *theta)
    nv  = nu   (Phi_xyz, Rc, *theta)

    # 4) local dispersions
    sigR = sigmaR_of_Rc(Rc, R0, RsigR, sigmaR0_R0)
    sigZ = sigmaz_of_Rc(Rc, R0, RsigZ,  sigmaz0_R0)

    # 5) actions from exponentials implied by DF
    JR = _sample_exponential_core(k2, scale=jnp.clip(sigR**2 / jnp.clip(kap, 1e-12), 1e-12))
    Jz = _sample_exponential_core(k3, scale=jnp.clip(sigZ**2 / jnp.clip(nv , 1e-12), 1e-12))

    return JR, Jz, Jphi, Rc, kap, nv

# ===========================================================
# Public API (wrapper: accepts Python floats/ArrayLike, casts once)
# ===========================================================

@beartype
def sample_actions_one(
    key: ArrayLike,
    Phi_xyz: Callable, *theta: ArrayLike,
    R0: ArrayLike, Rd: ArrayLike, Sigma0: ArrayLike,
    RsigR: ArrayLike, RsigZ: ArrayLike,
    sigmaR0_R0: ArrayLike, sigmaz0_R0: ArrayLike,
    Rmin: ArrayLike = 0.2, Rmax: ArrayLike = 30.0,
    Rinit_for_Rc: ArrayLike = 8.0,
):
    key = jnp.asarray(key)
    theta = tuple(jnp.asarray(t) for t in theta)
    R0, Rd, Sigma0 = jnp.asarray(R0), jnp.asarray(Rd), jnp.asarray(Sigma0)
    RsigR, RsigZ = jnp.asarray(RsigR), jnp.asarray(RsigZ)
    sigmaR0_R0, sigmaz0_R0 = jnp.asarray(sigmaR0_R0), jnp.asarray(sigmaz0_R0)
    Rmin, Rmax, Rinit_for_Rc = jnp.asarray(Rmin), jnp.asarray(Rmax), jnp.asarray(Rinit_for_Rc)

    return _sample_actions_one_core(
        key, Phi_xyz, *theta,
        R0=R0, Rd=Rd, Sigma0=Sigma0,
        RsigR=RsigR, RsigZ=RsigZ,
        sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
        Rmin=Rmin, Rmax=Rmax, Rinit_for_Rc=Rinit_for_Rc,
    )

# Only the callable and N are static. Keep Rmin/Rmax dynamic to avoid unnecessary recompiles.
@partial(jax.jit, static_argnums=0, static_argnames=("N",))
def sample_actions_batch(
    Phi_xyz: Callable, *theta: ArrayLike,
    N: int = 10000, seed: int = 0,
    R0: ArrayLike = 8.0, Rd: ArrayLike = 2.6, Sigma0: ArrayLike = 1.0,
    RsigR: ArrayLike = 7.0, RsigZ: ArrayLike = 7.0,
    sigmaR0_R0: ArrayLike = 35.0, sigmaz0_R0: ArrayLike = 20.0,
    Rmin: ArrayLike = 0.2, Rmax: ArrayLike = 30.0,
    Rinit_for_Rc: ArrayLike = 8.0,
):
    """
    Vectorized sampler: JR (N,), Jz (N,), Jphi (N,), Rc (N,), kappa (N,), nu (N,).
    """
    # Cast once at the boundary.
    theta = tuple(jnp.asarray(t) for t in theta)
    R0, Rd, Sigma0 = jnp.asarray(R0), jnp.asarray(Rd), jnp.asarray(Sigma0)
    RsigR, RsigZ = jnp.asarray(RsigR), jnp.asarray(RsigZ)
    sigmaR0_R0, sigmaz0_R0 = jnp.asarray(sigmaR0_R0), jnp.asarray(sigmaz0_R0)
    Rmin, Rmax, Rinit_for_Rc = jnp.asarray(Rmin), jnp.asarray(Rmax), jnp.asarray(Rinit_for_Rc)

    keys = jax.random.split(jax.random.PRNGKey(seed), N)

    def one(k):
        return _sample_actions_one_core(
            k, Phi_xyz, *theta,
            R0=R0, Rd=Rd, Sigma0=Sigma0,
            RsigR=RsigR, RsigZ=RsigZ,
            sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
            Rmin=Rmin, Rmax=Rmax, Rinit_for_Rc=Rinit_for_Rc,
        )

    JR, Jz, Jphi, Rc, kap, nv = jax.vmap(one)(keys)
    return JR, Jz, Jphi, Rc, kap, nv
