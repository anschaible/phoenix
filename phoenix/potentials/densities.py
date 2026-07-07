import jax
import jax.numpy as jnp

@jax.jit
def disk_density(x, y, z, Sigma0, Rd, h, Rcut, n):
    """
    Compute the density ρ(R, z) of a generalized disk model.

    Parameters
    ----------
    x, y : float or array
        Cartesian coordinates in the disk plane
    z : float or array
        Height above the disk
    Sigma0 : float
        Central surface density
    Rd : float
        Scale radius
    h : float
        Scale height: if h == 0, razor-thin disk (delta), 
                      if h > 0, exponential vertical profile,
                      if h < 0, sech² vertical profile
    Rcut : float
        Inner cutoff radius
    n : float
        Sersic-like index controlling radial taper

    Returns
    -------
    rho : float or array
        The 3D density at (R, z)
    """
    R = jnp.sqrt(x**2 + y**2)
    R = jnp.maximum(R, 1e-6)  # avoid division by zero at R = 0

    # Radial profile
    exp_term = jnp.exp(- (R / Rd)**(1/n) - Rcut / R)

    # Vertical profile
    def razor_thin(z):
        return jnp.where(jnp.abs(z) < 1e-6, 1e6, 0.0)  # δ(z) ~ spike at z=0

    def exponential(z, h):
        return 0.5 / h * jnp.exp(-jnp.abs(z) / h)

    def sech2(z, h):
        h = jnp.abs(h)
        arg = jnp.abs(z) / (2 * h)
        return 1 / (4 * h) * jnp.cosh(arg)**-2

    fz = jax.lax.cond(
        h == 0.0,
        lambda _: razor_thin(z),
        lambda _: jax.lax.cond(
            h > 0,
            lambda _: exponential(z, h),
            lambda _: sech2(z, h),
            operand=None
        ),
        operand=None
    )

    return Sigma0 * exp_term * fz

@jax.jit
def spheroid_density(x, y, z, rho0, a, alpha, beta, gamma, p, q, rcut, xi):
    """
    Generalized spheroidal density profile with axis ratios and cutoff.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    rho0 : float
        Central density normalization
    a : float
        Scale radius
    alpha, beta, gamma : float
        Shape parameters
    p : float
        Axis ratio Y
    q : float
        Axis ratio Z
    rcut : float
        Outer cutoff radius
    xi : float
        Cutoff strength (steepness)

    Returns
    -------
    rho : float or array
        Density at given coordinates
    """
    # Modified radius for axisymmetric ellipsoids
    rtilde = jnp.sqrt(x**2 + (y/p)**2 + (z/q)**2)
    ra = rtilde / a

    # Main profile
    core = ra**(-gamma)
    outer = (1 + ra**alpha)**((gamma - beta) / alpha)
    cutoff = jnp.exp(- (rtilde / rcut)**xi)

    return rho0 * core * outer * cutoff

@jax.jit
def nuker_cylindrical(x, y, Sigma0, a, alpha, beta, gamma, rcut, xi):
    """
    Nuker profile in cylindrical geometry: R = sqrt(x² + y²)

    Parameters
    ----------
    x, y : float or array
        Cartesian coordinates in the disk plane
    Sigma0 : float
        Central surface density normalization
    a : float
        Scale radius
    alpha, beta, gamma : float
        Nuker shape parameters
    rcut : float
        Outer cutoff radius
    xi : float
        Cutoff steepness

    Returns
    -------
    rho : float or array
        Surface density at (x, y)
    """
    # Cylindrical radius
    R = jnp.sqrt(x**2 + y**2)
    R = jnp.maximum(R, 1e-6)  # avoid divide-by-zero at R=0

    Ra = R / a
    core = Ra ** (-gamma)
    transition = (0.5 + 0.5 * Ra ** alpha) ** ((gamma - beta) / alpha)
    cutoff = jnp.exp(- (R / rcut) ** xi)

    return Sigma0 * core * transition * cutoff

@jax.jit
def sersic_surface_density(x, y, Sigma0, bn, a, n):
    """
    Cylindrical 2D Sersic surface density profile (deprojected).

    Parameters
    ----------
    x, y : float or array
        Cartesian coordinates in the galactic plane
    Sigma0 : float
        Central surface density
    bn : float
        Normalization constant for Sersic profile
    a : float
        Scale radius (half-light radius)
    n : float
        Sersic index (e.g. n=1 exponential, n=4 de Vaucouleurs)

    Returns
    -------
    Sigma : float or array
        Surface density Σ(x, y)
    """
    # Cylindrical radius
    R = jnp.sqrt(x**2 + y**2)
    R = jnp.maximum(R, 1e-6)

    return Sigma0 * jnp.exp(-bn * (R / a) ** (1 / n))

@jax.jit
def perfect_ellipsoid_density(x, y, z, M, a, q):
    """
    Perfect ellipsoid density profile in 3D.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    M : float
        Total mass
    a : float
        Scale radius
    q : float
        Axis ratio (z-axis)
    
    Returns
    -------
    rho : float or array
        Density at (x, y, z)
    """
    R2 = x**2 + y**2
    rtilde2 = R2 + (z / q)**2
    norm = M / (jnp.pi**2 * q * a**3)
    return norm * (1 + rtilde2 / a**2) ** -2

@jax.jit
def dehnen_density(x, y, z, M, a, gamma, p, q):
    """
    Dehnen density profile in 3D with axis ratios.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    M : float
        Total mass
    a : float
        Scale radius
    gamma : float
        Inner slope parameter
    p : float
        Axis ratio (y-axis)
    q : float
        Axis ratio (z-axis)
    
    Returns
    -------
    rho : float or array
        Density at (x, y, z)
    """
    rtilde = jnp.sqrt(x**2 + (y / p)**2 + (z / q)**2)
    norm = M * (3 - gamma) / (4 * jnp.pi * p * q * a**3)
    term1 = (rtilde / a) ** (-gamma)
    term2 = (1 + rtilde / a) ** (gamma - 4)
    return norm * term1 * term2

@jax.jit
def ferrers_density(x, y, z, M, a, p, q):
    """
    Ferrers density profile in 3D with axis ratios.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates
    M : float
        Mass
    a : float
        Scale radius
    p : float
        Axis ratio (y-axis)
    q : float
        Axis ratio (z-axis)
    
    Returns
    -------
    rho : float or array
        Density at (x, y, z)
    """
    rtilde2 = x**2 + (y / p)**2 + (z / q)**2
    rtilde = jnp.sqrt(rtilde2)
    norm = (105 * M) / (32 * jnp.pi * p * q * a**3)
    
    # Ensure profile falls to 0 outside the ellipsoid (rtilde > a)
    profile = jnp.where(rtilde < a, (1 - (rtilde / a)**2)**2, 0.0)
    
    return norm * profile