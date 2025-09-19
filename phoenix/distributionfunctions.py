# distributionfunctions.py
from typing import Callable
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, jaxtyped

# ---------- axisymmetric Phi(R,z) wrapper ----------
@jaxtyped
def Phi_Rz_from_xyz(Phi_xyz: Callable, R: ArrayLike, z: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    R, z = jnp.asarray(R), jnp.asarray(z)
    theta = tuple(jnp.asarray(t) for t in theta)
    return Phi_xyz(R, 0.0, z, *theta)

# ---------- scalar cores (R is scalar) ----------
def _vcirc_scalar(Phi_xyz: Callable, R: Float[Array, ""], *theta: Array) -> Float[Array, ""]:
    dPhi_dR = jax.grad(lambda Rp: Phi_Rz_from_xyz(Phi_xyz, Rp, 0.0, *theta))(R)
    vc2 = jnp.clip(R, 1e-12) * dPhi_dR
    return jnp.sqrt(jnp.clip(vc2, 1e-20))

def _Omega_scalar(Phi_xyz: Callable, R: Float[Array, ""], *theta: Array) -> Float[Array, ""]:
    return _vcirc_scalar(Phi_xyz, R, *theta) / jnp.clip(R, 1e-12)

def _kappa_scalar(Phi_xyz: Callable, R: Float[Array, ""], *theta: Array) -> Float[Array, ""]:
    Omega2 = lambda Rp: _Omega_scalar(Phi_xyz, Rp, *theta) ** 2
    dOm2_dR = jax.grad(Omega2)(R)
    om = _Omega_scalar(Phi_xyz, R, *theta)
    kap2 = jnp.clip(R, 1e-12) * dOm2_dR + 4.0 * om * om
    return jnp.sqrt(jnp.clip(kap2, 1e-20))

def _nu_scalar(Phi_xyz: Callable, R: Float[Array, ""], *theta: Array) -> Float[Array, ""]:
    dPhi_dz  = jax.grad(lambda z: Phi_Rz_from_xyz(Phi_xyz, R, z, *theta))
    d2Phi_dz2 = jax.grad(dPhi_dz)(jnp.asarray(0.0))
    return jnp.sqrt(jnp.clip(d2Phi_dz2, 1e-20))

# ---------- public, vectorized wrappers ----------
@jaxtyped
def vcirc(Phi_xyz: Callable, R: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    R = jnp.asarray(R); theta = tuple(jnp.asarray(t) for t in theta)
    f = lambda r: _vcirc_scalar(Phi_xyz, r, *theta)
    return jax.vmap(f)(R) if R.ndim else f(R)

@jaxtyped
def Omega(Phi_xyz: Callable, R: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    R = jnp.asarray(R); theta = tuple(jnp.asarray(t) for t in theta)
    f = lambda r: _Omega_scalar(Phi_xyz, r, *theta)
    return jax.vmap(f)(R) if R.ndim else f(R)

@jaxtyped
def kappa(Phi_xyz: Callable, R: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    R = jnp.asarray(R); theta = tuple(jnp.asarray(t) for t in theta)
    f = lambda r: _kappa_scalar(Phi_xyz, r, *theta)
    return jax.vmap(f)(R) if R.ndim else f(R)

@jaxtyped
def nu(Phi_xyz: Callable, R: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    R = jnp.asarray(R); theta = tuple(jnp.asarray(t) for t in theta)
    f = lambda r: _nu_scalar(Phi_xyz, r, *theta)
    return jax.vmap(f)(R) if R.ndim else f(R)

# ---------- batched Newton solve for Rc(Lz) ----------
@jaxtyped
def Rc_from_Lz(Phi_xyz: Callable, Lz: ArrayLike, R_init: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    Lz = jnp.asarray(Lz); R_init = jnp.asarray(R_init)
    theta = tuple(jnp.asarray(t) for t in theta)

    shape = jnp.broadcast_shapes(Lz.shape, R_init.shape)
    Lz_b = jnp.broadcast_to(Lz, shape)
    R0   = jnp.clip(jnp.broadcast_to(R_init, shape), 1e-2)

    # scalar g(R, Lz) and its grad; then vmap over arrays
    def _g_scalar(Rs, Ls):
        return Rs * _vcirc_scalar(Phi_xyz, Rs, *theta) - Ls
    vg = jax.vmap(jax.value_and_grad(_g_scalar), in_axes=(0, 0))

    def body(_, R):
        gR, dgR = vg(R, Lz_b)
        Rn = R - gR / jnp.clip(dgR, 1e-12)
        Rn = jnp.clip(Rn, 1e-3, 1e3 * jnp.maximum(1.0, R0))
        return 0.7 * Rn + 0.3 * R

    return jax.lax.fori_loop(0, 30, body, R0)


# ================= Radial profiles (disk) =================

@jaxtyped
def Sigma_exp(Rc: ArrayLike, R0: ArrayLike, Rd: ArrayLike, Sigma0: ArrayLike) -> Float[Array, "..."]:
    Rc, R0, Rd, Sigma0 = map(jnp.asarray, (Rc, R0, Rd, Sigma0))
    return Sigma0 * jnp.exp(-(Rc - R0) / jnp.clip(Rd, 1e-12))

@jaxtyped
def sigmaR_of_Rc(Rc: ArrayLike, R0: ArrayLike, RsigR: ArrayLike, sigmaR0_at_R0: ArrayLike) -> Float[Array, "..."]:
    Rc, R0, RsigR, sigmaR0_at_R0 = map(jnp.asarray, (Rc, R0, RsigR, sigmaR0_at_R0))
    return sigmaR0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigR, 1e-12))

@jaxtyped
def sigmaz_of_Rc(Rc: ArrayLike, R0: ArrayLike, RsigZ: ArrayLike, sigmaz0_at_R0: ArrayLike) -> Float[Array, "..."]:
    Rc, R0, RsigZ, sigmaz0_at_R0 = map(jnp.asarray, (Rc, R0, RsigZ, sigmaz0_at_R0))
    return sigmaz0_at_R0 * jnp.exp((R0 - Rc) / jnp.clip(RsigZ, 1e-12))

@jaxtyped
def sigma_age(sigma_ref: ArrayLike, tau: ArrayLike, tau1: ArrayLike, taum: ArrayLike, beta: ArrayLike) -> Float[Array, "..."]:
    sigma_ref, tau, tau1, taum, beta = map(jnp.asarray, (sigma_ref, tau, tau1, taum, beta))
    return sigma_ref * ((tau + tau1) / jnp.clip(taum + tau1, 1e-12))**beta

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

    pref = (Om * Sigma) / (2.0 * jnp.pi**2 * jnp.clip(sigR, 1e-12)**2 * jnp.clip(sigZ, 1e-12)**2 * jnp.clip(kap, 1e-12))
    eR   = jnp.exp(- kap * JR / jnp.clip(sigR, 1e-12)**2)
    eZ   = jnp.exp(- nv  * Jz / jnp.clip(sigZ, 1e-12)**2)
    rot  = 0.5 * (1.0 + jnp.tanh(Jphi / jnp.clip(L0, 1e-12)))
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
