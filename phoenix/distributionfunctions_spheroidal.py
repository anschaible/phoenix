from typing import Callable
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, jaxtyped
from typing import Dict, Callable, Tuple
import jax.numpy as jnp

# ---------- axisymmetric Phi(R,z) wrapper ----------
@jaxtyped
def Phi_Rz_from_xyz(Phi_xyz: Callable, R: ArrayLike, z: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    R, z = jnp.asarray(R), jnp.asarray(z)
    theta = tuple(jnp.asarray(t) for t in theta)
    return Phi_xyz(R, 0.0, z, *theta)


def f_double_power_law(Jr, Jz, Jphi, Phi_xyz, theta, params: Dict):
    """
    Double power-law distribution function in action space.

    Parameters:
      - Jr: Radial action
      - Jz: Vertical action
      - Jphi: Angular momentum along z
      - Phi_xyz: Callable potential function Phi(x, y, z, *theta)
      - theta: Tuple of parameters for the potential
      - params: Dictionary with DF parameters:
          * N0: Normalization constant
          * J0: Action scale, break between inner and outer slopes
          * Gamma: Inner slope, must be < 3
          * Beta: Outer slope, must be > 3
          * eta: steepness of the transition between two assymptotic regimes, default is 1.0

    Returns:
      - f: Value of the distribution function at given actions
    """

    N0 = params["N0_spheroid"]
    J0 = params["J0_spheroid"]
    Gamma = params["Gamma_spheroid"]
    Beta = params["Beta_spheroid"]
    eta = params.get("eta_spheroid", 1.0)

    Jtot = Jr + Jz + jnp.abs(Jphi)

    factor = N0/((2.0 * jnp.pi * J0)**3)
    inner = ((1+J0/Jtot)**eta)**(Gamma/eta)
    outer = ((1+Jtot/J0)**eta)**(-Beta/eta)

    return factor * inner * outer                       