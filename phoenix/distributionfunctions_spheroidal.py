from typing import Callable
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, jaxtyped
from typing import Dict, Callable, Tuple
import jax.numpy as jnp

#Following AGAMA
#https://github.com/GalacticDynamics-Oxford/Agama/tree/master/doc/reference.pdf

# ---------- axisymmetric Phi(R,z) wrapper ----------
def Phi_Rz_from_xyz(Phi_xyz: Callable, R: ArrayLike, z: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    R, z = jnp.asarray(R), jnp.asarray(z)
    theta = tuple(jnp.asarray(t) for t in theta)
    return Phi_xyz(R, 0.0, z, *theta)


def f_double_power_law(Jr, Jz, Jphi, Phi_xyz, theta, params: Dict):
    """
    Double power-law distribution function in action space. 

    Parameters
    ----------
    Jr : array_like
        Radial action
    Jz : array_like 
        Vertical action
    Jphi : array_like
        Angular momentum about the z-axis
    Phi_xyz : Callable
        Gravitational potential function Phi(x, y, z, *theta)
    theta : tuple
        Additional parameters for the potential
    params : dict
        Dictionary of parameters, must include 'N0_spheroid', 'J0_spheroid', 'Gamma_spheroid', 'Beta_spheroid', and optionally 'eta_spheroid'.  

    Returns
    -------
    array_like
        The value of the distribution function at the given actions.
    """

    N0 = params["N0_spheroid"]
    J0 = params["J0_spheroid"]
    Gamma = params["Gamma_spheroid"]
    Beta = params["Beta_spheroid"]
    eta = params.get("eta_spheroid", 1.0)

    Jtot = Jr + Jz + jnp.abs(Jphi)

    factor = N0/((2.0 * jnp.pi * J0)**3)
    inner = (1+(J0/Jtot)**eta)**(Gamma/eta)
    outer = (1+(Jtot/J0)**eta)**(-Beta/eta)

    return factor * inner * outer                       