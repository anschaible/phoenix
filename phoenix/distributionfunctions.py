import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

# =============== Axisymmetric wrappers ===============

@jaxtyped(typechecker=typechecker)
def Phi_Rz_from_xyz(Phi_xyz, R: float, z: float, *theta):
    """Axisymmetric wrapper: Phi(R,z) := Phi(x=R,y=0,z)."""
    return Phi_xyz(R, 0.0, z, *theta)

@jaxtyped(typechecker=typechecker)
def vcirc(Phi_xyz, R: float, *theta) -> float:
    """v_c(R) from Phi: v_c^2 = R dPhi/dR at z=0."""
    dPhi_dR = jax.grad(lambda Rp: Phi_Rz_from_xyz(Phi_xyz, Rp, 0.0, *theta))(R)
    vc2 = jnp.clip(R, 1e-12) * dPhi_dR
    return jnp.sqrt(jnp.clip(vc2, 1e-20))

@jaxtyped(typechecker=typechecker)
def Omega(Phi_xyz, R: float, *theta) -> float:
    return vcirc(Phi_xyz, R, *theta) / jnp.clip(R, 1e-12)

@jaxtyped(typechecker=typechecker)
def kappa(Phi_xyz, R: float, *theta) -> float:
    """κ^2 = R d(Ω^2)/dR + 4Ω^2."""
    def Omega2(Rp):
        om = Omega(Phi_xyz, Rp, *theta)
        return om * om
    dOm2_dR = jax.grad(Omega2)(R)
    om = Omega(Phi_xyz, R, *theta)
    kap2 = jnp.clip(R, 1e-12) * dOm2_dR + 4.0 * om * om
    return jnp.sqrt(jnp.clip(kap2, 1e-20))

@jaxtyped(typechecker=typechecker)
def nu(Phi_xyz, R: float, *theta) -> float:
    """ν^2 = ∂^2 Phi / ∂z^2 at z=0."""
    dPhi_dz  = jax.grad(lambda z: Phi_Rz_from_xyz(Phi_xyz, R, z, *theta))
    d2Phi_dz2 = jax.grad(dPhi_dz)(0.0)
    return jnp.sqrt(jnp.clip(d2Phi_dz2, 1e-20))

# =============== Find R_c from Lz ===============

@jaxtyped(typechecker=typechecker)
def Rc_from_Lz(Phi_xyz, Lz: float, R_init: float, *theta) -> float:
    """
    Solve g(R)=R*vcirc(R)-Lz=0 by Newton with clamped steps.
    R_init: good guess (e.g. max(Lz/vc_guess, 1e-2)).
    """
    def g(R):
        return R * vcirc(Phi_xyz, R, *theta) - Lz

    def dg(R):
        return jax.grad(lambda Rp: Rp * vcirc(Phi_xyz, Rp, *theta))(R)

    R = jnp.clip(R_init, 1e-2)
    def body(i, R):
        gR = g(R)
        d  = jnp.clip(dg(R), 1e-12)
        Rn = R - gR / d
        # keep R positive and avoid giant jumps
        Rn = jnp.clip(Rn, 1e-3, 1e3 * jnp.maximum(1.0, R_init))
        # simple damping
        Rn = 0.7 * Rn + 0.3 * R
        return Rn

    R = jax.lax.fori_loop(0, 30, body, R)
    return R

# =============== Radial profiles (disk) ===============

@jaxtyped(typechecker=typechecker)
def Sigma_exp(Rc: float, R0: float, Rd: float, Sigma0: float) -> float:
    return Sigma0 * jnp.exp(-(Rc - R0) / jnp.clip(Rd, 1e-12))

@jaxtyped(typechecker=typechecker)
def sigmaR_of_Rc(Rc: float, R0: float, RsigR: float, sigmaR0_at_R0: float) -> float:
    return sigmaR0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigR, 1e-12))

@jaxtyped(typechecker=typechecker)
def sigmaz_of_Rc(Rc: float, R0: float, RsigZ: float, sigmaz0_at_R0: float) -> float:
    return sigmaz0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigZ, 1e-12))

# Optional age-heating (single helper for both components)
@jaxtyped(typechecker=typechecker)
def sigma_age(sigma_ref: float, tau: float, tau1: float, taum: float, beta: float) -> float:
    return sigma_ref * ((tau + tau1) / jnp.clip(taum + tau1, 1e-12))**beta

# =============== Quasi-isothermal DF ===============

@jaxtyped(typechecker=typechecker)
def quasi_isothermal_df(
    JR: float, Jz: float, Jphi: float,
    # potential handle
    Phi_xyz,
    # potential parameters (unpacked as *theta)
    *theta,
    # structural & DF params
    R0: float, Rd: float, Sigma0: float,
    RsigR: float, RsigZ: float,
    sigmaR0_R0: float, sigmaz0_R0: float,
    L0: float,
    # numerical: initial guess for Rc
    Rinit_for_Rc: float = 8.0,
) -> float:
    # 1) find Rc from Lz
    Rc = Rc_from_Lz(Phi_xyz, Jphi, Rinit_for_Rc, *theta)

    # 2) local frequencies from Phi
    Om = Omega(Phi_xyz, Rc, *theta)
    kap = kappa(Phi_xyz, Rc, *theta)
    nv = nu(Phi_xyz, Rc, *theta)

    # 3) radial profiles
    Sigma = Sigma_exp(Rc, R0, Rd, Sigma0)
    sigR  = sigmaR_of_Rc(Rc, R0, RsigR, sigmaR0_R0)
    sigZ  = sigmaz_of_Rc(Rc, R0, RsigZ, sigmaz0_R0)

    # 4) DF
    pref = (Om * Sigma) / (2.0 * jnp.pi**2 * jnp.clip(sigR,1e-12)**2 * jnp.clip(sigZ,1e-12)**2 * jnp.clip(kap,1e-12))
    eR   = jnp.exp(- kap * JR / jnp.clip(sigR,1e-12)**2)
    eZ   = jnp.exp(- nv  * Jz / jnp.clip(sigZ,1e-12)**2)
    rot  = 0.5 * (1.0 + jnp.tanh(Jphi / jnp.clip(L0, 1e-12)))
    return pref * eR * eZ * rot

# Thin/thick wrappers: SAME DF, DIFFERENT PARAMS
@jaxtyped(typechecker=typechecker)
def f_thin_disc(JR: float, Jz: float, Jphi: float,
                Phi_xyz, *theta,
                R0: float, Rd: float, Sigma0: float,
                RsigR: float, RsigZ: float,
                sigmaR0_R0: float, sigmaz0_R0: float,
                L0: float, Rinit_for_Rc: float = 8.0) -> float:
    return quasi_isothermal_df(JR, Jz, Jphi, Phi_xyz, *theta,
                               R0=R0, Rd=Rd, Sigma0=Sigma0,
                               RsigR=RsigR, RsigZ=RsigZ,
                               sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
                               L0=L0, Rinit_for_Rc=Rinit_for_Rc)

@jaxtyped(typechecker=typechecker)
def f_thick_disc(JR: float, Jz: float, Jphi: float,
                 Phi_xyz, *theta,
                 R0: float, Rd: float, Sigma0: float,
                 RsigR: float, RsigZ: float,
                 sigmaR0_R0: float, sigmaz0_R0: float,
                 L0: float, Rinit_for_Rc: float = 8.0) -> float:
    return quasi_isothermal_df(JR, Jz, Jphi, Phi_xyz, *theta,
                               R0=R0, Rd=Rd, Sigma0=Sigma0,
                               RsigR=RsigR, RsigZ=RsigZ,
                               sigmaR0_R0=sigmaR0_R0, sigmaz0_R0=sigmaz0_R0,
                               L0=L0, Rinit_for_Rc=Rinit_for_Rc)
