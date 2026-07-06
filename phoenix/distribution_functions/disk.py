import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, jaxtyped
from typing import Dict, Callable
import functools

# Import frequencies from your Phoenix package
from phoenix.distribution_functions.frequencies import Omega, kappa, nu, Rc_from_Lz

# ==============================================================================
# DISK RADIAL PROFILES
# ==============================================================================
@jaxtyped
@jax.jit
def Sigma_exp(Rc: ArrayLike, R0: ArrayLike, Rd: ArrayLike, Sigma0: ArrayLike) -> Float[Array, "..."]:
    """Exponential surface density profile Sigma(Rc) = Sigma0 * exp(-(Rc-R0)/Rd)"""
    Rc, R0, Rd, Sigma0 = map(jnp.asarray, (Rc, R0, Rd, Sigma0))
    return Sigma0 * jnp.exp(-(Rc - R0) / jnp.clip(Rd, 1e-12))

@jaxtyped
@jax.jit
def sigmaR_of_Rc(Rc: ArrayLike, R0: ArrayLike, RsigR: ArrayLike, sigmaR0_at_R0: ArrayLike) -> Float[Array, "..."]:
    """Radial velocity dispersion profile sigmaR(Rc) = sigmaR0_at_R0 * exp((R0-Rc)/RsigR)"""
    Rc, R0, RsigR, sigmaR0_at_R0 = map(jnp.asarray, (Rc, R0, RsigR, sigmaR0_at_R0))
    sigmaR_val = sigmaR0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigR, 1e-3))
    return jnp.clip(sigmaR_val, 1e-3)

@jaxtyped
@jax.jit
def sigmaz_of_Rc(Rc: ArrayLike, R0: ArrayLike, RsigZ: ArrayLike, sigmaz0_at_R0: ArrayLike) -> Float[Array, "..."]:
    """Vertical velocity dispersion profile sigmaz(Rc) = sigmaz0_at_R0 * exp((R0-Rc)/RsigZ)"""
    Rc, R0, RsigZ, sigmaz0_at_R0 = map(jnp.asarray, (Rc, R0, RsigZ, sigmaz0_at_R0))
    sigmaz_val = sigmaz0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigZ, 1e-3))
    return jnp.clip(sigmaz_val, 1e-3)

@jaxtyped
@jax.jit
def sigma_age(sigma_ref: ArrayLike, tau: ArrayLike, tau1: ArrayLike, taum: ArrayLike, beta: ArrayLike) -> Float[Array, "..."]:
    """Age-velocity dispersion relation sigma(tau) = sigma_ref * ((tau + tau1) / (taum + tau1))**beta"""
    sigma_ref, tau, tau1, taum, beta = map(jnp.asarray, (sigma_ref, tau, tau1, taum, beta))
    return sigma_ref * ((tau + tau1) / jnp.clip(taum + tau1, 1e-5))**beta


# ==============================================================================
# DISK DISTRIBUTION FUNCTIONS (qDF)
# ==============================================================================
@functools.partial(jax.jit, static_argnums=(3,))
@jaxtyped
def quasi_isothermal_df(
    JR: ArrayLike, Jz: ArrayLike, Jphi: ArrayLike,
    Phi_xyz: Callable, *theta: ArrayLike,
    R0: ArrayLike, Rd: ArrayLike, Sigma0: ArrayLike,
    RsigR: ArrayLike, RsigZ: ArrayLike,
    sigmaR0_R0: ArrayLike, sigmaz0_R0: ArrayLike,
    L0: ArrayLike,
    Rinit_for_Rc: ArrayLike = 8.0,
) -> Float[Array, "..."]:
    """Quasi-isothermal distribution function in action space."""
    JR, Jz, Jphi = jnp.asarray(JR), jnp.asarray(Jz), jnp.asarray(Jphi)
    R0, Rd, Sigma0 = jnp.asarray(R0), jnp.asarray(Rd), jnp.asarray(Sigma0)
    RsigR, RsigZ = jnp.asarray(RsigR), jnp.asarray(RsigZ)
    sigmaR0_R0, sigmaz0_R0 = jnp.asarray(sigmaR0_R0), jnp.asarray(sigmaz0_R0)
    L0, Rinit_for_Rc = jnp.asarray(L0), jnp.asarray(Rinit_for_Rc)
    theta = tuple(jnp.asarray(t) for t in theta)

    Rc  = Rc_from_Lz(Phi_xyz, Jphi, Rinit_for_Rc, *theta)
    Om  = Omega(Phi_xyz, Rc, *theta)
    kap = kappa(Phi_xyz, Rc, *theta)
    nv  = nu(Phi_xyz, Rc, *theta)

    Sigma = Sigma_exp(Rc, R0, Rd, Sigma0)
    sigR  = sigmaR_of_Rc(Rc, R0, RsigR, sigmaR0_R0)
    sigZ  = sigmaz_of_Rc(Rc, R0, RsigZ, sigmaz0_R0)

    # CRITICAL JAX FIX: Safe denominators to prevent NaN gradients during autodiff
    kap_safe = kap #jnp.clip(kap, 1e-7, None)
    sigR_safe = sigR #jnp.clip(sigR, 1e-7, None)
    sigZ_safe = sigZ #jnp.clip(sigZ, 1e-7, None)
    L0_safe = L0 #jnp.clip(L0, 1e-7, None)

    pref = (Om * Sigma) / (2.0 * jnp.pi**2 * sigR_safe**2 * sigZ_safe**2 * kap_safe)
    eR   = jnp.exp(- kap_safe * JR / sigR_safe**2)
    eZ   = jnp.exp(- nv  * Jz / sigZ_safe**2)
    rot  = 0.5 * (1.0 + jnp.tanh(Jphi / L0_safe))
    
    return pref * eR * eZ * rot

@functools.partial(jax.jit, static_argnums=(3,))
def f_disc_from_params(Jr: ArrayLike, Jz: ArrayLike, Jphi: ArrayLike, Phi_xyz: Callable, theta: tuple, params: Dict):  
    """Wrapper function to compute the quasi-isothermal disk DF from a parameter dictionary."""
    R0 = params["R0"]
    Rd = params["Rd"]
    Sigma0 = params["Sigma0"]
    RsigR = params["RsigR"]
    RsigZ = params["RsigZ"]
    sigmaR0_R0 = params["sigmaR0_R0"]
    sigmaz0_R0 = params["sigmaz0_R0"]
    L0 = params["L0"]
    Rinit_for_Rc = params.get("Rinit_for_Rc", 8.0)

    return quasi_isothermal_df(
        Jr, Jz, Jphi,
        Phi_xyz, *theta,
        R0=R0, Rd=Rd, Sigma0=Sigma0,
        RsigR=RsigR, RsigZ=RsigZ,
        sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
        L0=L0, Rinit_for_Rc=Rinit_for_Rc,
    )
