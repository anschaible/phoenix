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
    Rc, R0, Rd, Sigma0 = map(jnp.asarray, (Rc, R0, Rd, Sigma0))
    return Sigma0 * jnp.exp(-(Rc - R0) / jnp.clip(Rd, 1e-12))

@jaxtyped
def sigmaR_of_Rc(Rc: ArrayLike, R0: ArrayLike, RsigR: ArrayLike, sigmaR0_at_R0: ArrayLike) -> Float[Array, "..."]:
    Rc, R0, RsigR, sigmaR0_at_R0 = map(jnp.asarray, (Rc, R0, RsigR, sigmaR0_at_R0))
    sigmaR_of_Rc = sigmaR0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigR, 1e-3))
    return jnp.clip(sigmaR_of_Rc, 1e-3)

@jaxtyped
def sigmaz_of_Rc(Rc: ArrayLike, R0: ArrayLike, RsigZ: ArrayLike, sigmaz0_at_R0: ArrayLike) -> Float[Array, "..."]:
    Rc, R0, RsigZ, sigmaz0_at_R0 = map(jnp.asarray, (Rc, R0, RsigZ, sigmaz0_at_R0))
    sigmaz_of_Rc = sigmaz0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigZ, 1e-3))
    return jnp.clip(sigmaz_of_Rc, 1e-3)

@jaxtyped
def sigma_age(sigma_ref: ArrayLike, tau: ArrayLike, tau1: ArrayLike, taum: ArrayLike, beta: ArrayLike) -> Float[Array, "..."]:
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

# ================= Thin/thick wrappers =================

@jaxtyped
def f_thin_disc(JR: ArrayLike, Jz: ArrayLike, Jphi: ArrayLike,
                Phi_xyz: Callable, *theta: ArrayLike,
                R0: ArrayLike, Rd: ArrayLike, Sigma0: ArrayLike,
                RsigR: ArrayLike, RsigZ: ArrayLike,
                sigmaR0_R0: ArrayLike, sigmaz0_R0: ArrayLike,
                L0: ArrayLike, Rinit_for_Rc: ArrayLike = 8.0) -> Float[Array, "..."]:
    return quasi_isothermal_df(JR, Jz, Jphi, Phi_xyz, *theta,
                               R0=R0, Rd=Rd, Sigma0=Sigma0,
                               RsigR=RsigR, RsigZ=RsigZ,
                               sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
                               L0=L0, Rinit_for_Rc=Rinit_for_Rc)

@jaxtyped
def f_thick_disc(JR: ArrayLike, Jz: ArrayLike, Jphi: ArrayLike,
                 Phi_xyz: Callable, *theta: ArrayLike,
                 R0: ArrayLike, Rd: ArrayLike, Sigma0: ArrayLike,
                 RsigR: ArrayLike, RsigZ: ArrayLike,
                 sigmaR0_R0: ArrayLike, sigmaz0_R0: ArrayLike,
                 L0: ArrayLike, Rinit_for_Rc: ArrayLike = 8.0) -> Float[Array, "..."]:
    return quasi_isothermal_df(JR, Jz, Jphi, Phi_xyz, *theta,
                               R0=R0, Rd=Rd, Sigma0=Sigma0,
                               RsigR=RsigR, RsigZ=RsigZ,
                               sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
                               L0=L0, Rinit_for_Rc=Rinit_for_Rc)


def f_total_disc(JR: ArrayLike, Jz: ArrayLike, Jphi: ArrayLike,
                 Phi_xyz: Callable, *theta: ArrayLike,
                 R0_thin: ArrayLike, Rd_thin: ArrayLike, Sigma0_thin: ArrayLike,
                 RsigR_thin: ArrayLike, RsigZ_thin: ArrayLike,
                 sigmaR0_R0_thin: ArrayLike, sigmaz0_R0_thin: ArrayLike,
                 L0_thin: ArrayLike,
                 R0_thick: ArrayLike, Rd_thick: ArrayLike, Sigma0_thick: ArrayLike,
                 RsigR_thick: ArrayLike, RsigZ_thick: ArrayLike,
                 sigmaR0_R0_thick: ArrayLike, sigmaz0_R0_thick: ArrayLike,
                 L0_thick: ArrayLike,
                 Rinit_for_Rc_thin: ArrayLike = 8.0,
                 Rinit_for_Rc_thick: ArrayLike = 8.0,
                 f_thin: float = 0.7, f_thick: float = 0.3) -> Float[Array, "..."]:
    """
    Total DF = f_thin * f_thin_disc + f_thick * f_thick_disc
    where f_thin + f_thick = 1 (not enforced here).
    """
    JR, Jz, Jphi = jnp.asarray(JR), jnp.asarray(Jz), jnp.asarray(Jphi)
    theta = tuple(jnp.asarray(t) for t in theta)

    f_thin_disc_vals = f_thin_disc(
        JR, Jz, Jphi, Phi_xyz, *theta,
        R0=R0_thin, Rd=Rd_thin, Sigma0=Sigma0_thin,
        RsigR=RsigR_thin, RsigZ=RsigZ_thin,
        sigmaR0_R0=sigmaR0_R0_thin, sigmaz0_R0=sigmaz0_R0_thin,
        L0=L0_thin, Rinit_for_Rc=Rinit_for_Rc_thin,
    )
    f_thick_disc_vals = f_thick_disc(
        JR, Jz, Jphi, Phi_xyz, *theta,
        R0=R0_thick, Rd=Rd_thick, Sigma0=Sigma0_thick,
        RsigR=RsigR_thick, RsigZ=RsigZ_thick,
        sigmaR0_R0=sigmaR0_R0_thick, sigmaz0_R0=sigmaz0_R0_thick,
        L0=L0_thick, Rinit_for_Rc=Rinit_for_Rc_thick,
    )
    return f_thin * f_thin_disc_vals + f_thick * f_thick_disc_vals


def f_total_disc_from_params(Jr, Jz, Jphi, Phi_xyz, theta, params: Dict):
    #Phi_xyz: Callable = params["Phi_xyz"]
    #theta: Tuple = tuple(params.get("theta", ()))

    if "thin" in params and "thick" in params:
        thin  = params["thin"];  thick = params["thick"]
        R0_thin, Rd_thin, Sigma0_thin = thin["R0"], thin["Rd"], thin["Sigma0"]
        RsigR_thin, RsigZ_thin = thin["RsigR"], thin["RsigZ"]
        sigmaR0_R0_thin, sigmaz0_R0_thin = thin["sigmaR0_R0"], thin["sigmaz0_R0"]
        L0_thin = thin["L0"]; Rinit_for_Rc_thin = thin.get("Rinit_for_Rc", 8.0)

        R0_thick, Rd_thick, Sigma0_thick = thick["R0"], thick["Rd"], thick["Sigma0"]
        RsigR_thick, RsigZ_thick = thick["RsigR"], thick["RsigZ"]
        sigmaR0_R0_thick, sigmaz0_R0_thick = thick["sigmaR0_R0"], thick["sigmaz0_R0"]
        L0_thick = thick["L0"]; Rinit_for_Rc_thick = thick.get("Rinit_for_Rc", 8.0)
    else:
        # flat style also supported
        R0_thin, Rd_thin, Sigma0_thin = params["R0_thin"], params["Rd_thin"], params["Sigma0_thin"]
        RsigR_thin, RsigZ_thin = params["RsigR_thin"], params["RsigZ_thin"]
        sigmaR0_R0_thin, sigmaz0_R0_thin = params["sigmaR0_R0_thin"], params["sigmaz0_R0_thin"]
        L0_thin = params["L0_thin"]; Rinit_for_Rc_thin = params.get("Rinit_for_Rc_thin", 8.0)

        R0_thick, Rd_thick, Sigma0_thick = params["R0_thick"], params["Rd_thick"], params["Sigma0_thick"]
        RsigR_thick, RsigZ_thick = params["RsigR_thick"], params["RsigZ_thick"]
        sigmaR0_R0_thick, sigmaz0_R0_thick = params["sigmaR0_R0_thick"], params["sigmaz0_R0_thick"]
        L0_thick = params["L0_thick"]; Rinit_for_Rc_thick = params.get("Rinit_for_Rc_thick", 8.0)

    f_thin  = jnp.asarray(params.get("f_thin", 0.7),  dtype=jnp.float32)
    f_thick = jnp.asarray(params.get("f_thick", 0.3), dtype=jnp.float32)

    # optional: sicherstellen, dass f_thin+f_thick=1 (ohne Python-Logik)
    #norm = jnp.clip(f_thin + f_thick, 1e-12)
    #f_thin, f_thick = f_thin/norm, f_thick/norm


    return f_total_disc(
        Jr, Jz, Jphi,
        Phi_xyz, *theta,
        R0_thin=R0_thin, Rd_thin=Rd_thin, Sigma0_thin=Sigma0_thin,
        RsigR_thin=RsigR_thin, RsigZ_thin=RsigZ_thin,
        sigmaR0_R0_thin=sigmaR0_R0_thin, sigmaz0_R0_thin=sigmaz0_R0_thin,
        L0_thin=L0_thin,
        R0_thick=R0_thick, Rd_thick=Rd_thick, Sigma0_thick=Sigma0_thick,
        RsigR_thick=RsigR_thick, RsigZ_thick=RsigZ_thick,
        sigmaR0_R0_thick=sigmaR0_R0_thick, sigmaz0_R0_thick=sigmaz0_R0_thick,
        L0_thick=L0_thick,
        Rinit_for_Rc_thin=Rinit_for_Rc_thin,
        Rinit_for_Rc_thick=Rinit_for_Rc_thick,
        f_thin=f_thin, f_thick=f_thick,
    )

def f_thin_disc_from_params(Jr, Jz, Jphi, Phi_xyz, theta, params: Dict):
    R0 = params["R0_thin"]
    Rd = params["Rd_thin"]
    Sigma0 = params["Sigma0_thin"]
    RsigR = params["RsigR_thin"]
    RsigZ = params["RsigZ_thin"]
    sigmaR0_R0 = params["sigmaR0_R0_thin"]
    sigmaz0_R0 = params["sigmaz0_R0_thin"]
    L0 = params["L0_thin"]
    Rinit_for_Rc = params.get("Rinit_for_Rc_thin", 8.0)

    return f_thin_disc(
        Jr, Jz, Jphi,
        Phi_xyz, *theta,
        R0=R0, Rd=Rd, Sigma0=Sigma0,
        RsigR=RsigR, RsigZ=RsigZ,
        sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
        L0=L0, Rinit_for_Rc=Rinit_for_Rc,
    )

def f_thick_disc_from_params(Jr, Jz, Jphi, Phi_xyz, theta, params: Dict):  
    R0 = params["R0_thick"]
    Rd = params["Rd_thick"]
    Sigma0 = params["Sigma0_thick"]
    RsigR = params["RsigR_thick"]
    RsigZ = params["RsigZ_thick"]
    sigmaR0_R0 = params["sigmaR0_R0_thick"]
    sigmaz0_R0 = params["sigmaz0_R0_thick"]
    L0 = params["L0_thick"]
    Rinit_for_Rc = params.get("Rinit_for_Rc_thick", 8.0)

    return f_thick_disc(
        Jr, Jz, Jphi,
        Phi_xyz, *theta,
        R0=R0, Rd=Rd, Sigma0=Sigma0,
        RsigR=RsigR, RsigZ=RsigZ,
        sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
        L0=L0, Rinit_for_Rc=Rinit_for_Rc,
    )