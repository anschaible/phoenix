import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

# mainly following Binney 2009 and 2015 to implement the distribution functions

@jaxtyped(typechecker=typechecker)
def Rc_from_Lz(Lz: float, v0: float) -> float:
    """
    Calculate the radius of the circular orbit Rc of an angular momentum Lz.
    
    Parameters:
    Lz (float): Angular momentum.
    v0 (float): Circular velocity.
    
    Returns:
    float: Radius of the circular orbit.
    """
    return Lz / v0

@jaxtyped(typechecker=typechecker)
def surface_density(r: float, R0: float, Rd: float, sigma0: float) -> float:
    """
    Calculate the surface density of a thick disc at radius r.
    
    Parameters:
    r (float): Radius at which to calculate the surface density.
    R0 (float): Suns distance from galactic center.
    Rd (float): Scale length of the disc.
    sigma0 (float): Central surface density.
    
    Returns:
    float: Surface density at radius r.
    """
    return sigma0 * jnp.exp(- (r - R0) / Rd)

@jaxtyped(typechecker=typechecker)
def sigma_z0_tau(tau: float, sigma_z0: float, tau1: float, tau_m: float, beta: float) -> float:
    """
    Calculate the vertical velocity dispersion at time tau. According to Binney 2010 we assume that the DF of stars of age tau is pseudo-isothermal with sigma_z increasing with tau.

    Parameters:
    tau (float): Time since the formation of the disc.
    sigma_z0 (float): Velocity dispersion of stars at age tau_m.
    tau1 (float): Velocity dispersion at birth.
    tau_m (float): Time up to which the disc evolves.
    beta (float): Index that determiones how omega_z grows with age.

    Returns:
    float: Vertical time dependent velocity dispersion.
    """
    return sigma_z0 * ((tau + tau1) / (tau_m + tau1))**beta

@jaxtyped(typechecker=typechecker)
def sigma_z(Lz: float, sigma_z0: float, q: float, v0: float, Rc: float, Rd: float):
    """
    Calculate the vertical velocity dispersion sigma_z at radius Rc for a given angular momentum Lz.

    Parameters:
    Lz (float): Angular momentum.
    sigma_z0 (float): Velocity dispersion of stars at age tau_m.
    q (float): Flattening of the disc.
    v0 (float): Circular velocity.
    Rc (float): Radius of the circular orbit.
    Rd (float): Scale length of the disc.

    Returns:
    float: Vertical velocity dispersion for angular momentum Lz.
    """
    R0 = Rc_from_Lz(Lz, v0)
    return sigma_z0 * jnp.exp(q * (R0 - Rc) / Rd)

@jaxtyped(typechecker=typechecker)
def sigma_r0_tau(tau: float, sigma_r0: float, tau1: float, tau_m: float, beta: float) -> float:
    """
    Calculate the radial velocity dispersion at time tau. According to Binney 2010 we assume that the DF of stars of age tau is pseudo-isothermal with sigma_r increasing with tau.

    Parameters:
    tau (float): Time since the formation of the disc.
    sigma_r0 (float): Velocity dispersion of stars at age tau_m.
    tau1 (float): Velocity dispersion at birth.
    tau_m (float): Time up to which the disc evolves.
    beta (float): Index that determiones how omega_r grows with age.

    Returns:
    float: Radial time dependent velocity dispersion.
    """
    return sigma_r0 * ((tau + tau1) / (tau_m + tau1))**beta

@jaxtyped(typechecker=typechecker)
def sigma_r(Lz: float, sigma_r0: float, q: float, v0: float, R0: float, Rd: float):
    """
    Calculate the radial velocity dispersion sigma_r at radius Rc for a given angular momentum Lz.

    Parameters:
    Lz (float): Angular momentum.
    sigma_r0 (float): Velocity dispersion of stars at age tau_m.
    q (float): Flattening of the disc.
    v0 (float): Circular velocity.
    R0 (float): Suns distance from galactic center.
    Rd (float): Scale length of the disc.

    Returns:
    float: Radial velocity dispersion for angular momentum Lz.
    """
    Rc = Rc_from_Lz(Lz, v0)
    return sigma_r0 * jnp.exp(q * (R0 - Rc) / Rd)

@jaxtyped(typechecker=typechecker)
def omega(Lz: float, r: float) -> float:
    """
    Derive omega from a identity: dLz/dR = R kappa^2/ 2 omega.

    Parameters:
    Lz (float): Angular momentum.
    r (float): Radius at which to calculate the angular velocity.
    kappa (float): Epicyclic frequency.

    Returns:
    float: omega at radius r for angular momentum Lz.
    """
    dLz_dR = Lz / r # this is obviously not right, have to think how to get the derivative
    kappa_value = kappa(Lz)  # Assuming kappa is a function of Lz
    return r * kappa_value**2 / (2 * dLz_dR)

@jaxtyped(typechecker=typechecker)
def kappa(L_z):
    """
    Calculate the epicyclic frequency kappa from angular momentum Lz.

    Parameters:
    L_z (float): Angular momentum.

    Returns:
    float: Epicyclic frequency.
    """
    # This is a placeholder, actual implementation depends on the specific potential used.
    return jnp.sqrt(L_z)

@jaxtyped(typechecker=typechecker)
def omega_z(Lz: float, Jr: float, Jz: float):
    """
    An orbit’s vertical frequency Ωz is a function of all three actions, Jr, Jz and Lz. 
    However, the Jeans theorem assures us that the df remains a solution of the collisionless 
    Boltzmann equation if in Ωz we set Jr = 0.

    Parameters:
    Lz (float): Angular momentum.
    Jr (float): Radial action.
    Jz (float): Vertical action.

    Returns:
    float: orbits vertical frequency.
    """
    # This is a placeholder, actual implementation depends on the specific potential used.
    return Lz / (Jr + Jz)

@jaxtyped(typechecker=typechecker)
def normalization_f_omega_z(Jz: jnp.ndarray, omega_z: float, Lz: float, q: float, v0: float, Rc: float, Rd: float,
                            tau: float, sigma_z0: float, tau1: float, tau_m: float, beta: float):
    """
    Calculate the normalization factor for the distribution function f_omega_z.

    Parameters:
    Jz (float): Vertical action.
    omega_z (float): Vertical frequency.
    Lz (float): Angular momentum.
    sigma_z0 (float): Velocity dispersion of stars at age tau_m.
    q (float): Flattening of the disc.
    v0 (float): Circular velocity.
    Rc (float): Radius of the circular orbit.
    Rd (float): Scale length of the disc.

    Returns:
    float: Normalization factor for f_omega_z.
    """
    sigma_z0_value = sigma_z0_tau(tau, sigma_z0, tau1, tau_m, beta)
    sigma_z_value = sigma_z(Lz, sigma_z0_value, q, v0, Rc, Rd)
    function = jnp.exp(- omega_z * Jz / sigma_z_value**2)
    return 2 * jnp.pi * jnp.integrate.trapezoid(function, Jz)

@jaxtyped(typechecker=typechecker)
def f_omega_z(Jz: float, omega_z: float, Lz: float, q: float, v0: float, Rc: float, Rd: float,
              tau: float, sigma_z0: float, tau1: float, tau_m: float, beta: float) -> float:
    """
    Calculate the distribution function f_omega_z for a given vertical action Jz.

    Parameters:
    Jz (float): Vertical action.
    omega_z (float): Vertical frequency.
    Lz (float): Angular momentum.
    sigma_z0 (float): Velocity dispersion of stars at age tau_m.
    q (float): Flattening of the disc.
    v0 (float): Circular velocity.
    Rc (float): Radius of the circular orbit.
    Rd (float): Scale length of the disc.

    Returns:
    float: Distribution function f_omega_z at vertical action Jz.
    """
    sigma_z0_value = sigma_z0_tau(tau, sigma_z0, tau1, tau_m, beta)
    sigma_z_value = sigma_z(Lz, sigma_z0_value, q, v0, Rc, Rd)
    normalization = normalization_f_omega_z(Jz, omega_z, Lz, sigma_z0, q, v0, Rc, Rd)
    return jnp.exp(- omega_z * Jz / sigma_z_value**2) / normalization

@jaxtyped(typechecker=typechecker)
def factor_cold_disc_f_sigma_r(Lz: float, R0: float, Rd: float, sigma0: float, 
                               q: float, v0: float, tau: float, sigma_r0: float, tau1: float, 
                               tau_m: float, beta: float) -> float:
    Rc_value = Rc_from_Lz(Lz, v0)
    kappa_value = kappa(Lz)
    omega_value = omega(Lz, Rc_value)
   
    surface_density_value = surface_density(Rc_value, R0, Rd, sigma0)
    sigma_r0_value = sigma_r0_tau(tau, sigma_r0, tau1, tau_m, beta)
    sigma_r_value = sigma_r(Lz, sigma_r0_value, q, v0, Rc_value, Rd)
    return omega_value * surface_density_value / (jnp.pi *sigma_r_value**2 * kappa_value)

@jaxtyped(typechecker=typechecker)
def factor_rotaton_f_sigma_r(Lz: float, L0: float) -> float:
    """
    Calculate the factor for the distribution function f_sigma_r in a rotating disc.

    Parameters:
    Lz (float): Angular momentum.
    L0 (float): Reference angular momentum.

    Returns:
    float: Factor for f_sigma_r.
    """
    return (1 + jnp.tanh(Lz / L0))

@jaxtyped(typechecker=typechecker)
def f_omega_r(Lz: float, R0: float, Rd: float, sigma0: float, 
                q: float, v0: float, tau: float, sigma_r0: float, tau1: float, 
                tau_m: float, beta: float, Jr) -> float:
    """
    Calculate the distribution function f_omega_r for a given angular momentum Lz.

    Parameters:
    Lz (float): Angular momentum.
    kappa (float): Epicyclic frequency.
    R0 (float): Reference radius.
    Rd (float): Scale length of the disc.
    sigma0 (float): Central surface density.
    q (float): Flattening of the disc.
    v0 (float): Circular velocity.
    tau (float): Age of the stellar population.
    sigma_r0 (float): Velocity dispersion in the radial direction.
    tau1 (float): Time scale for the transition.
    tau_m (float): Time scale for the maximum.
    beta (float): Slope of the density profile.

    Returns:
    float: Distribution function f_omega_r at angular momentum Lz.
    """
    kappa_value = kappa(Lz)
    sigma_r0_value = sigma_r0_tau(tau, sigma_r0, tau1, tau_m, beta)
    sigma_r_value = sigma_r(Lz, sigma_r0_value, q, v0, R0, Rd)
    factor_cold_disc_f_sigma_r_value = factor_cold_disc_f_sigma_r(Lz, kappa_value, R0, Rd, sigma0, q, v0, tau, sigma_r0_value, tau1, tau_m, beta)
    factor_rotaton_f_sigma_r_value = factor_rotaton_f_sigma_r(Lz, kappa_value, R0, Rd, sigma0, q, v0, tau, sigma_r0_value, tau1, tau_m, beta)
    return factor_cold_disc_f_sigma_r_value * factor_rotaton_f_sigma_r_value * jnp.exp(- kappa_value * Jr / sigma_r_value**2)