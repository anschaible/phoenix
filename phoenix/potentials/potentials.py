import jax
import jax.numpy as jnp

from phoenix.constants import G

# follow Agama notes (https://github.com/GalacticDynamics-Oxford/Agama/blob/master/doc/reference.pdf)

@jax.jit
def plummer_potential(x, y, z, M, a):
    """
    Plummer potential in 3D.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    M : float
        Mass
    a : float
        Scale radius

    Returns
    -------
    Phi : float or array
        Gravitational potential at (x, y, z)
    """
    r2 = x**2 + y**2 + z**2
    return -G * M / jnp.sqrt(a**2 + r2)

@jax.jit
def isochrone_potential(x, y, z, M, a):
    """
    Isochrone potential in 3D.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    M : float
        Mass
    a : float
        Scale radius

    Returns
    -------
    Phi : float or array
        Gravitational potential at (x, y, z)
    """
    r2 = x**2 + y**2 + z**2
    return -G * M / (a + jnp.sqrt(r2 + a**2))

@jax.jit
def nfw_potential(x, y, z, M, a):
    """
    Navarro-Frenk-White potential in 3D.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    M : float
        Mass
    a : float
        Scale radius

    Returns
    -------
    Phi : float or array
        Gravitational potential at (x, y, z)
    """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    r = jnp.maximum(r, 1e-6)  # avoid division by zero
    return -G * M / r * jnp.log(1 + r / a)

@jax.jit
def miyamoto_nagai_potential(x, y, z, M, a, b):
    """
    Miyamoto-Nagai potential in 3D.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    M : float
        Mass
    a : float
        Radial scale length
    b : float
        Vertical scale height

    Returns 
    -------
    Phi : float or array
        Gravitational potential at (x, y, z)
    """
    R2 = x**2 + y**2
    B = jnp.sqrt(z**2 + b**2)
    denom = jnp.sqrt(R2 + (a + B)**2)
    return -G * M / denom

@jax.jit
def logarithmic_potential(x, y, z, v0, rcore, p, q):
    """
    Logarithmic potential in 3D with axis ratios.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    v0 : float
        Velocity
    rcore : float
        Scale radius
    p : float
        Axis ratio (y-axis)
    q : float
        Axis ratio (z-axis)
    
    Returns
    -------
    Phi : float or array
        Gravitational potential at (x, y, z)
    """
    rtilde2 = x**2 + (y / p)**2 + (z / q)**2
    return 0.5 * v0**2 * jnp.log(rcore**2 + rtilde2)


@jax.jit
def harmonic_potential(x, y, z, Omega, p, q):
    """
    Harmonic potential in 3D with axis ratios.
    
    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    Omega : float
        Frequency
    p : float
        Axis ratio (y-axis)
    q : float
        Axis ratio (z-axis)
    
    Returns
    -------
    Phi : float or array
        Gravitational potential at (x, y, z)
    """
    rtilde2 = x**2 + (y / p)**2 + (z / q)**2
    return 0.5 * Omega**2 * rtilde2