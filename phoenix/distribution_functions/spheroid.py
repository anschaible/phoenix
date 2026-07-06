import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float
from typing import Dict, Callable

# ==============================================================================
# UTILITIES
# ==============================================================================
def Phi_Rz_from_xyz(Phi_xyz: Callable, R: ArrayLike, z: ArrayLike, *theta: ArrayLike) -> Float[Array, "..."]:
    """Wrapper to evaluate a 3D Cartesian potential in axisymmetric cylindrical coordinates."""
    R, z = jnp.asarray(R), jnp.asarray(z)
    theta = tuple(jnp.asarray(t) for t in theta)
    return Phi_xyz(R, 0.0, z, *theta)


# ==============================================================================
# SPHEROID / HALO DISTRIBUTION FUNCTIONS
# ==============================================================================
@jax.jit
def f_double_power_law(Jr: ArrayLike, Jz: ArrayLike, Jphi: ArrayLike, params: Dict) -> Float[Array, "..."]:
    """
    Double power-law distribution function in action space (Agama standard).
    Used for spheroidal components like dark matter halos or stellar bulges.

    Parameters
    ----------
    Jr, Jz, Jphi : ArrayLike
        The orbital actions (Radial, Vertical, Azimuthal).
    params : dict
        Dictionary of physical parameters. Must contain:
        - 'N0_spheroid': Normalization mass scale
        - 'J0_spheroid': Transition action scale
        - 'Gamma_spheroid': Inner logarithmic slope
        - 'Beta_spheroid': Outer logarithmic slope
        - 'eta_spheroid' (optional): Sharpness of the transition (default: 1.0)

    Returns
    -------
    f_J : ArrayLike
        The phase-space density at the given actions.
    """
    Jr, Jz, Jphi = jnp.asarray(Jr), jnp.asarray(Jz), jnp.asarray(Jphi)
    
    N0 = params["N0_spheroid"]
    J0 = params["J0_spheroid"]
    Gamma = params["Gamma_spheroid"]
    Beta = params["Beta_spheroid"]
    eta = params.get("eta_spheroid", 1.0)

    # Calculate total action proxy (trace of actions)
    # CRITICAL JAX FIX: Add 1e-7 to prevent Division-by-Zero NaNs in gradients
    Jtot = Jr + Jz + jnp.abs(Jphi) #+ 1e-7

    # Agama mathematical formulation
    factor = N0 / ((2.0 * jnp.pi * J0)**3)
    inner = (1.0 + (J0 / Jtot)**eta)**(Gamma / eta)
    outer = (1.0 + (Jtot / J0)**eta)**(-Beta / eta)

    return factor * inner * outer