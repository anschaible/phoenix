from typing import Callable
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, jaxtyped
from typing import Dict, Callable, Tuple
import jax.numpy as jnp

# ---------- axisymmetric Phi(R,z) wrapper ----------
def Phi_Rz_from_xyz(Phi_xyz: Callable, R: ArrayLike, z: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    """
    Wrapper to evaluate an axisymmetric potential Phi(R,z) from a 3D potential Phi(x,y,z).

    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float or array
        Cylindrical radius  
    z : float or array
        Vertical height
    theta : tuple of float or array
        Additional parameters for the potential

    Returns
    -------
    Phi_Rz : float or array
        Potential evaluated at (R, z)
    """
    R, z = jnp.asarray(R), jnp.asarray(z)
    theta = tuple(jnp.asarray(t) for t in theta)
    return Phi_xyz(R, 0.0, z, *theta)

# ---------- scalar cores (R is scalar) ----------
def _vcirc_scalar(Phi_xyz: Callable, R: Float[Array, ""], *theta: Array) -> Float[Array, ""]:
    """
    Circular velocity vc at radius R (scalar core).
    
    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float
        Cylindrical radius
    theta : tuple of float
        Additional parameters for the potential
        
    Returns
    -------
    vcirc : float
        Circular velocity at radius R
    """
    dPhi_dR = jax.grad(lambda Rp: Phi_Rz_from_xyz(Phi_xyz, Rp, 0.0, *theta))(R)
    vc2 = jnp.clip(R, 1e-12) * dPhi_dR
    return jnp.sqrt(jnp.clip(vc2, 1e-20))

def _Omega_scalar(Phi_xyz: Callable, R: Float[Array, ""], *theta: Array) -> Float[Array, ""]:
    """
    Circular frequency Omega at radius R (scalar core).

    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float
        Cylindrical radius
    theta : tuple of float
        Additional parameters for the potential
        
    Returns
    -------
    Omega : float
        Circular frequency at radius R
    """
    return _vcirc_scalar(Phi_xyz, R, *theta) / jnp.clip(R, 1e-12)

def _kappa_scalar(Phi_xyz: Callable, R: Float[Array, ""], *theta: Array) -> Float[Array, ""]:
    """
    Radial epicyclic frequency kappa at radius R (scalar core).
    
    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float
        Cylindrical radius
    theta : tuple of float  
        Additional parameters for the potential
    
    Returns
    -------
    kappa : float
        Radial epicyclic frequency at radius R
    """
    Omega2 = lambda Rp: _Omega_scalar(Phi_xyz, Rp, *theta) ** 2
    dOm2_dR = jax.grad(Omega2)(R)
    om = _Omega_scalar(Phi_xyz, R, *theta)
    kap2 = jnp.clip(R, 1e-12) * dOm2_dR + 4.0 * om * om
    return jnp.sqrt(jnp.clip(kap2, 1e-20))

def _nu_scalar(Phi_xyz: Callable, R: Float[Array, ""], *theta: Array) -> Float[Array, ""]:
    """
    Vertical epicyclic frequency nu at radius R (scalar core).
    
    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float
        Cylindrical radius
    theta : tuple of float
        Additional parameters for the potential
    
    Returns
    -------
    nu : float
        Vertical epicyclic frequency at radius R
    """
    dPhi_dz  = jax.grad(lambda z: Phi_Rz_from_xyz(Phi_xyz, R, z, *theta))
    d2Phi_dz2 = jax.grad(dPhi_dz)(jnp.asarray(0.0))
    return jnp.sqrt(jnp.clip(d2Phi_dz2, 1e-20))

# ---------- public, vectorized wrappers ----------
def vcirc(Phi_xyz: Callable, R: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    """
    Circular velocity vc at radius R.

    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float or array
        Cylindrical radius
    theta : tuple of float or array
        Additional parameters for the potential 
    
    Returns
    -------
    vcirc : float or array
        Circular velocity at radius R
    """
    R = jnp.asarray(R); theta = tuple(jnp.asarray(t) for t in theta)
    f = lambda r: _vcirc_scalar(Phi_xyz, r, *theta)
    return jax.vmap(f)(R) if R.ndim else f(R)

def Omega(Phi_xyz: Callable, R: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    """
    Circular frequency Omega at radius R.

    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float or array
        Cylindrical radius 
    theta : tuple of float or array
        Additional parameters for the potential
        
    Returns
    -------
    Omega : float or array
        Circular frequency at radius R
    """
    R = jnp.asarray(R); theta = tuple(jnp.asarray(t) for t in theta)
    f = lambda r: _Omega_scalar(Phi_xyz, r, *theta)
    return jax.vmap(f)(R) if R.ndim else f(R)

def kappa(Phi_xyz: Callable, R: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    """
    Radial epicyclic frequency kappa at radius R.
    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float or array
        Cylindrical radius
    theta : tuple of float or array
        Additional parameters for the potential
        
    Returns
    -------
    kappa : float or array
        Radial epicyclic frequency at radius R
    """
    R = jnp.asarray(R); theta = tuple(jnp.asarray(t) for t in theta)
    f = lambda r: _kappa_scalar(Phi_xyz, r, *theta)
    return jax.vmap(f)(R) if R.ndim else f(R)

def nu(Phi_xyz: Callable, R: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    """
    Vertical epicyclic frequency nu at radius R.

    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    R : float or array
        Cylindrical radius
    theta : tuple of float or array
        Additional parameters for the potential 

    Returns
    -------
    nu : float or array
        Vertical epicyclic frequency at radius R
    """
    R = jnp.asarray(R); theta = tuple(jnp.asarray(t) for t in theta)
    f = lambda r: _nu_scalar(Phi_xyz, r, *theta)
    return jax.vmap(f)(R) if R.ndim else f(R)

# ---------- batched Newton solve for Rc(Lz) ----------
def Rc_from_Lz(Phi_xyz: Callable, Lz: ArrayLike, R_init: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    """
    The guiding center radius Rc corresponding to angular momentum Lz.

    Parameters
    ----------
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    Lz : float or array
        Angular momentum along z
    R_init : float or array
        Initial guess for Rc
    theta : tuple of float or array
        Additional parameters for the potential

    Returns
    -------
    Rc : float or array
        Guiding center radius corresponding to Lz  
    """
    Lz = jnp.asarray(Lz); R_init = jnp.asarray(R_init)
    theta = tuple(jnp.asarray(t) for t in theta)

    # Broadcast to common shape, then flatten to (N,) so vmap always sees axis 0
    shape = jnp.broadcast_shapes(Lz.shape, R_init.shape)
    Lz_b = jnp.broadcast_to(Lz, shape).ravel()
    R0   = jnp.clip(jnp.broadcast_to(R_init, shape), 1e-2).ravel()  # (N,)

    # Scalar g and grad; vmap over vectors
    def _g_scalar(Rs, Ls):
        return Rs * _vcirc_scalar(Phi_xyz, Rs, *theta) - Ls

    vg = jax.vmap(jax.value_and_grad(_g_scalar), in_axes=(0, 0))

    def body(_, Rvec):
        gR, dgR = vg(Rvec, Lz_b)                     # both (N,)
        Rn = Rvec - gR / jnp.clip(dgR, 1e-12)
        Rn = jnp.clip(Rn, 1e-3, 1e3 * jnp.maximum(1.0, R0))
        return 0.7 * Rn + 0.3 * Rvec

    R_sol = jax.lax.fori_loop(0, 30, body, R0)       # (N,)
    return R_sol.reshape(shape)                       # back to original broadcast shape
