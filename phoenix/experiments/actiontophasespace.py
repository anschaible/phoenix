from typing import Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, jaxtyped

from phoenix.distributionfunctions import (
    Rc_from_Lz, Omega, kappa, nu
)

_TWO_PI = 2.0 * jnp.pi

@jaxtyped
def actions_angles_to_phase_epicycle(
    Phi_xyz: Callable,
    JR: ArrayLike, Jz: ArrayLike, Jphi: ArrayLike,
    thetaR: ArrayLike, thetaz: ArrayLike, thetaphi: ArrayLike,
    *theta: ArrayLike,
    Rinit_for_Rc: ArrayLike = 8.0,
) -> Tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."],
           Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """
    Map (J, theta) -> (x, y, z, vx, vy, vz) via epicycle approximation.

    Inputs broadcast. Good for near-circular (disc) orbits.
    """
    # Cast
    JR, Jz, Jphi = jnp.asarray(JR), jnp.asarray(Jz), jnp.asarray(Jphi)
    thetaR, thetaz, thetaphi = jnp.asarray(thetaR), jnp.asarray(thetaz), jnp.asarray(thetaphi)
    Rinit_for_Rc = jnp.asarray(Rinit_for_Rc)
    theta = tuple(jnp.asarray(t) for t in theta)

    # Guiding radius and local frequencies
    Rc  = Rc_from_Lz(Phi_xyz, Jphi, Rinit_for_Rc, *theta)           # shape: ...
    Om  = Omega(Phi_xyz, Rc, *theta)                                 # ...
    kap = kappa(Phi_xyz, Rc, *theta)
    nv  = nu   (Phi_xyz, Rc, *theta)

    # Epicycle amplitudes
    aR = jnp.sqrt(jnp.clip(2.0 * JR / jnp.clip(kap, 1e-12), 0.0))
    az = jnp.sqrt(jnp.clip(2.0 * Jz / jnp.clip(nv , 1e-12), 0.0))

    # Cylindrical positions & velocities
    R   = Rc + aR * jnp.cos(thetaR)
    z   = az * jnp.cos(thetaz)
    vR  = - aR * kap * jnp.sin(thetaR)
    vz  = - az * nv  * jnp.sin(thetaz)

    # Azimuth angle with standard epicyclic correction
    phi = thetaphi + (2.0 * jnp.clip(Om, 1e-12) / jnp.clip(kap, 1e-12)) * (aR / jnp.clip(Rc, 1e-12)) * jnp.sin(thetaR)

    # Tangential velocity from exact Lz constraint
    vphi = Jphi / jnp.clip(R, 1e-12)

    # Cylindrical -> Cartesian
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    x = R * cphi
    y = R * sphi
    vx = vR * cphi - vphi * sphi
    vy = vR * sphi + vphi * cphi

    return x, y, z, vx, vy, vz


# Convenience: draw random angles and map to phase space
@jaxtyped
def sample_phase_from_actions(
    key: Array,
    Phi_xyz: Callable,
    JR: ArrayLike, Jz: ArrayLike, Jphi: ArrayLike,
    *theta: ArrayLike,
    Rinit_for_Rc: ArrayLike = 8.0,
) -> Tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."],
           Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """
    Sample θ_R, θ_z, θ_φ ~ Uniform(0, 2π) and return (x,y,z,vx,vy,vz).
    `key` can be batched or broadcast to JR/Jz/Jphi.
    """
    JR, Jz, Jphi = jnp.asarray(JR), jnp.asarray(Jz), jnp.asarray(Jphi)
    shape = jnp.broadcast_shapes(JR.shape, Jz.shape, Jphi.shape)

    kR, kz, kphi = jax.random.split(key, 3)
    thetaR  = _TWO_PI * jax.random.uniform(kR, shape=shape)
    thetaz  = _TWO_PI * jax.random.uniform(kz, shape=shape)
    thetaphi= _TWO_PI * jax.random.uniform(kphi, shape=shape)

    return actions_angles_to_phase_epicycle(
        Phi_xyz, JR, Jz, Jphi, thetaR, thetaz, thetaphi,
        *theta, Rinit_for_Rc=Rinit_for_Rc
    )
