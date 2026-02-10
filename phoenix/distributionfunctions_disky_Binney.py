from typing import Callable
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, jaxtyped
from typing import Dict, Callable, Tuple
import jax.numpy as jnp

from phoenix.frequencies import Omega, kappa, nu, Rc_from_Lz

# ================= Radial profiles (disk) =================

@jaxtyped
def Sigma_exp(Rc: ArrayLike, R0: ArrayLike, Rd: ArrayLike, Sigma0: ArrayLike) -> Float[Array, "..."]:
    """
    Exponential surface density profile Sigma(Rc) = Sigma0 * exp(-(Rc-R0)/Rd)
    
    Parameters
    ----------
    Rc : float or array
        Guiding center radius corresponding to the angular momentum Jphi
    R0 : float or array
        Reference radius (e.g., solar radius)
    Rd : float or array
        Scale length of the exponential disk
    Sigma0 : float or array
        Central surface density normalization

    Returns
    -------
    Sigma : float or array
        Surface density at Rc
    """
    Rc, R0, Rd, Sigma0 = map(jnp.asarray, (Rc, R0, Rd, Sigma0))
    return Sigma0 * jnp.exp(-(Rc - R0) / jnp.clip(Rd, 1e-12))

@jaxtyped
def sigmaR_of_Rc(Rc: ArrayLike, R0: ArrayLike, RsigR: ArrayLike, sigmaR0_at_R0: ArrayLike) -> Float[Array, "..."]:
    """
    Radial velocity dispersion profile sigmaR(Rc) = sigmaR0_at_R0 * exp((R0-Rc)/RsigR)

    Parameters
    ----------
    Rc : float or array
        Guiding center radius corresponding to the angular momentum Jphi
    R0 : float or array
        Reference radius (e.g., solar radius)
    RsigR : float or array
        Scale length of the radial velocity dispersion profile
    sigmaR0_at_R0 : float or array
        Radial velocity dispersion at reference radius R0

    Returns
    -------
    sigmaR : float or array
        Radial velocity dispersion at Rc
    """
    Rc, R0, RsigR, sigmaR0_at_R0 = map(jnp.asarray, (Rc, R0, RsigR, sigmaR0_at_R0))
    sigmaR_of_Rc = sigmaR0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigR, 1e-3))
    return jnp.clip(sigmaR_of_Rc, 1e-3)

@jaxtyped
def sigmaz_of_Rc(Rc: ArrayLike, R0: ArrayLike, RsigZ: ArrayLike, sigmaz0_at_R0: ArrayLike) -> Float[Array, "..."]:
    """
    Vertical velocity dispersion profile sigmaz(Rc) = sigmaz0_at_R0 * exp((R0-Rc)/RsigZ)

    Parameters
    ----------
    Rc : float or array
        Guiding center radius corresponding to the angular momentum Jphi
    R0 : float or array
        Reference radius (e.g., solar radius)
    RsigZ : float or array
        Scale length of the vertical velocity dispersion profile
    sigmaz0_at_R0 : float or array
        Vertical velocity dispersion at reference radius R0 

    Returns
    -------
    sigmaz : float or array
        Vertical velocity dispersion at Rc
    """
    Rc, R0, RsigZ, sigmaz0_at_R0 = map(jnp.asarray, (Rc, R0, RsigZ, sigmaz0_at_R0))
    sigmaz_of_Rc = sigmaz0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigZ, 1e-3))
    return jnp.clip(sigmaz_of_Rc, 1e-3)

@jaxtyped
def sigma_age(sigma_ref: ArrayLike, tau: ArrayLike, tau1: ArrayLike, taum: ArrayLike, beta: ArrayLike) -> Float[Array, "..."]:
    """
    Age-velocity dispersion relation sigma(tau) = sigma_ref * ((tau + tau1) / (taum + tau1))**beta
    
    Parameters
    ----------
    sigma_ref : float or array
        Reference velocity dispersion at age taum
    tau : float or array
        Age of the stellar population
    tau1 : float or array
        Age offset parameter to avoid singularity at tau=0
    taum : float or array
        Age at which sigma = sigma_ref
    beta : float or array
        Power-law index controlling how velocity dispersion increases with age
    
    Returns
    -------
    sigma : float or array
        Velocity dispersion at age tau
    """
    sigma_ref, tau, tau1, taum, beta = map(jnp.asarray, (sigma_ref, tau, tau1, taum, beta))
    return sigma_ref * ((tau + tau1) / jnp.clip(taum + tau1, 1e-5))**beta

# ================= Quasi-isothermal DF =================

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
    """
    Quasi-isothermal distribution function in action space.
    
    Parameters
    ----------
    JR : float or array
        Radial action
    Jz : float or array
        Vertical action
    Jphi : float or array
        Angular momentum about the z-axis
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    theta : tuple
        Additional parameters for the potential
    R0 : float or array
        Reference radius (e.g., solar radius)
    Rd : float or array
        Scale length of the exponential disk
    Sigma0 : float or array
        Central surface density normalization
    RsigR : float or array
        Scale length of the radial velocity dispersion profile
    RsigZ : float or array
        Scale length of the vertical velocity dispersion profile
    sigmaR0_R0 : float or array
        Radial velocity dispersion at reference radius R0
    sigmaz0_R0 : float or array
        Vertical velocity dispersion at reference radius R0
    L0 : float or array
        Angular momentum scale for the rotation term
    Rinit_for_Rc : float or array, optional
        Initial guess for Rc in the Newton solve (default: 8.0)
    
    Returns
    -------
    float or array
        The value of the distribution function at the given actions.
    """
    JR, Jz, Jphi = jnp.asarray(JR), jnp.asarray(Jz), jnp.asarray(Jphi)
    R0, Rd, Sigma0 = jnp.asarray(R0), jnp.asarray(Rd), jnp.asarray(Sigma0)
    RsigR, RsigZ = jnp.asarray(RsigR), jnp.asarray(RsigZ)
    sigmaR0_R0, sigmaz0_R0 = jnp.asarray(sigmaR0_R0), jnp.asarray(sigmaz0_R0)
    L0, Rinit_for_Rc = jnp.asarray(L0), jnp.asarray(Rinit_for_Rc)
    theta = tuple(jnp.asarray(t) for t in theta)

    Rc  = Rc_from_Lz(Phi_xyz, Jphi, Rinit_for_Rc, *theta)
    Om  = Omega(Phi_xyz, Rc, *theta)
    kap = kappa(Phi_xyz, Rc, *theta)
    nv  = nu   (Phi_xyz, Rc, *theta)

    Sigma = Sigma_exp(Rc, R0, Rd, Sigma0)
    sigR  = sigmaR_of_Rc(Rc, R0, RsigR, sigmaR0_R0)
    sigZ  = sigmaz_of_Rc(Rc, R0, RsigZ,  sigmaz0_R0)

    pref = (Om * Sigma) / (2.0 * jnp.pi**2 * sigR**2 * sigZ**2 * kap)
    eR   = jnp.exp(- kap * JR / sigR**2)
    eZ   = jnp.exp(- nv  * Jz / sigZ**2)
    rot  = 0.5 * (1.0 + jnp.tanh(Jphi / L0))
    return pref * eR * eZ * rot

#================== DF wrapper for parameters =================

def f_disc_from_params(Jr, Jz, Jphi, Phi_xyz, theta, params: Dict):  
    """
    Wrapper function to compute the quasi-isothermal disk DF from a parameter dictionary.

    Parameters
    ----------
    Jr : float or array
        Radial action
    Jz : float or array
        Vertical action
    Jphi : float or array
        Angular momentum about the z-axis
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    theta : tuple
        Additional parameters for the potential
    params : dict
        Dictionary containing DF parameters:
        - R0: Reference radius 
        - Rd: Scale length of the exponential disk
        - Sigma0: Central surface density normalization
        - RsigR: Scale length of the radial velocity dispersion profile
        - RsigZ: Scale length of the vertical velocity dispersion profile
        - sigmaR0_R0: Radial velocity dispersion at reference radius R0
        - sigmaz0_R0: Vertical velocity dispersion at reference radius R0
        - L0: Angular momentum scale for the rotation term
        - Rinit_for_Rc: Initial guess for Rc in the Newton solve (optional, default: 8.0)
    
    Returns
    -------
    float or array
        The value of the distribution function at the given actions.
    """
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