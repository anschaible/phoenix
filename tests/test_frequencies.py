import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial

from phoenix.frequencies import (
    Phi_Rz_from_xyz, 
    vcirc, Omega, kappa, nu, 
    Rc_from_Lz
)

# ==========================================
# 1. Define Analytic Potentials for Testing
# ==========================================

def phi_harmonic(x, y, z, omega):
    """
    Harmonic Oscillator: Phi = 0.5 * omega^2 * r^2
    Analytic predictions:
      v_c   = omega * R
      Omega = omega (constant)
      kappa = 2 * omega
      nu    = omega
    """
    return 0.5 * omega**2 * (x**2 + y**2 + z**2)

def phi_kepler(x, y, z, GM):
    """
    Kepler Potential: Phi = -GM / r
    Analytic predictions:
      v_c   = sqrt(GM / R)
      Omega = sqrt(GM / R^3)
      kappa = Omega (1:1 resonance)
      nu    = Omega (1:1 resonance)
    """
    r = jnp.sqrt(x**2 + y**2 + z**2 + 1e-30)
    return -GM / r

def phi_miyamoto_nagai(x, y, z, M, a, b):
    """
    Miyamoto-Nagai Disk (Test for vertical frequency nu vs kappa)
    Phi = -GM / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2)
    """
    R2 = x**2 + y**2
    return -M / jnp.sqrt(R2 + (a + jnp.sqrt(z**2 + b**2))**2)

# ==========================================
# 2. Test Wrapper Logic (Phi_Rz)
# ==========================================

def test_phi_rz_mapping():
    """Verify cylindrical mapping ensures y=0 and passes z correctly."""
    # Potential: 2x + 10y + 5z
    def linear_pot(x, y, z): return 2*x + 10*y + 5*z
    
    R, z = 3.0, 4.0
    # Expected: 2(3) + 10(0) + 5(4) = 6 + 0 + 20 = 26
    res = Phi_Rz_from_xyz(linear_pot, R, z)
    assert res == 26.0

# ==========================================
# 3. Test Frequencies (vcirc, Omega, kappa, nu)
# ==========================================

@pytest.mark.parametrize("R", [1.0, 5.0])
def test_frequencies_harmonic(R):
    """Check vcirc, Omega, kappa, nu for Harmonic Oscillator."""
    omega_val = 2.0
    
    # Run functions
    vc_calc = vcirc(phi_harmonic, R, omega_val)
    Om_calc = Omega(phi_harmonic, R, omega_val)
    ka_calc = kappa(phi_harmonic, R, omega_val)
    nu_calc = nu(phi_harmonic, R, omega_val)

    # Analytic Expectations
    vc_true = omega_val * R
    Om_true = omega_val
    ka_true = 2.0 * omega_val
    nu_true = omega_val

    # Assertions
    tol = 1e-5
    np.testing.assert_allclose(vc_calc, vc_true, rtol=tol, err_msg="vcirc mismatch")
    np.testing.assert_allclose(Om_calc, Om_true, rtol=tol, err_msg="Omega mismatch")
    np.testing.assert_allclose(ka_calc, ka_true, rtol=tol, err_msg="kappa mismatch")
    np.testing.assert_allclose(nu_calc, nu_true, rtol=tol, err_msg="nu mismatch")

@pytest.mark.parametrize("R", [0.5, 2.0])
def test_frequencies_kepler(R):
    """Check vcirc, Omega, kappa, nu for Kepler Potential."""
    GM = 1.5
    
    # Run functions
    vc_calc = vcirc(phi_kepler, R, GM)
    Om_calc = Omega(phi_kepler, R, GM)
    ka_calc = kappa(phi_kepler, R, GM)
    nu_calc = nu(phi_kepler, R, GM)
    
    # Analytic Expectations
    vc_true = jnp.sqrt(GM / R)
    Om_true = jnp.sqrt(GM / R**3)
    ka_true = Om_true  # For point mass, kappa = Omega
    nu_true = Om_true  # For point mass, nu = Omega

    # Assertions
    tol = 1e-5
    np.testing.assert_allclose(vc_calc, vc_true, rtol=tol)
    np.testing.assert_allclose(Om_calc, Om_true, rtol=tol)
    np.testing.assert_allclose(ka_calc, ka_true, rtol=tol)
    np.testing.assert_allclose(nu_calc, nu_true, rtol=tol)

def test_nu_vs_kappa_disk():
    """
    In a flattened disk, nu (vertical) should generally be larger than Omega, 
    and kappa (radial) should be distinct.
    Using Miyamoto-Nagai.
    """
    M, a, b = 1.0, 1.0, 0.5
    R = 1.0
    
    om_val = Omega(phi_miyamoto_nagai, R, M, a, b)
    ka_val = kappa(phi_miyamoto_nagai, R, M, a, b)
    nu_val = nu(phi_miyamoto_nagai, R, M, a, b)
    
    # Basic physical sanity checks for a disk
    assert jnp.isfinite(om_val)
    assert jnp.isfinite(ka_val)
    assert jnp.isfinite(nu_val)
    
    # Ensure they are not all identical (as they are in Kepler/Harmonic)
    assert abs(ka_val - nu_val) > 1e-3

# ==========================================
# 4. Test Vectorization (vmap behavior)
# ==========================================

def test_vectorization():
    """Ensure all frequency functions handle arrays vs scalars."""
    omega_val = 2.0
    R_scalar = 1.0
    R_array = jnp.array([1.0, 2.0, 3.0])
    
    # Test vcirc scalar
    res_s = vcirc(phi_harmonic, R_scalar, omega_val)
    assert res_s.ndim == 0
    
    # Test vcirc array
    res_a = vcirc(phi_harmonic, R_array, omega_val)
    assert res_a.shape == (3,)
    np.testing.assert_allclose(res_a, R_array * omega_val)

    # Test kappa array
    res_k = kappa(phi_harmonic, R_array, omega_val)
    assert res_k.shape == (3,)
    np.testing.assert_allclose(res_k, 2.0 * omega_val)

# ==========================================
# 5. Test Rc_from_Lz (Inverse problem)
# ==========================================

def test_Rc_recovery_harmonic():
    """
    Recover Radius from Angular Momentum in Harmonic Pot.
    Lz = R * vc = R * (omega * R) = omega * R^2
    => R = sqrt(Lz / omega)
    """
    omega = 2.0
    Lz = 8.0
    R_expected = jnp.sqrt(Lz / omega) # 2.0
    
    R_calc = Rc_from_Lz(phi_harmonic, Lz, 1.0, omega)
    np.testing.assert_allclose(R_calc, R_expected, rtol=1e-5)

def test_Rc_recovery_kepler():
    """
    Recover Radius from Angular Momentum in Kepler Pot.
    Lz = R * vc = R * sqrt(GM/R) = sqrt(GM*R)
    => R = Lz^2 / GM
    """
    GM = 1.0
    Lz = 2.0
    R_expected = (Lz**2) / GM # 4.0
    
    R_calc = Rc_from_Lz(phi_kepler, Lz, 1.0, GM)
    np.testing.assert_allclose(R_calc, R_expected, rtol=1e-5)

def test_Rc_broadcasting():
    """
    Test Rc_from_Lz with array inputs for Lz and R_init.
    """
    omega = 1.0
    # Lz = 1 -> R=1; Lz=4 -> R=2
    Lzs = jnp.array([1.0, 4.0])
    R_inits = jnp.array([0.5, 0.5]) 
    
    R_sol = Rc_from_Lz(phi_harmonic, Lzs, R_inits, omega)
    
    assert R_sol.shape == (2,)
    np.testing.assert_allclose(R_sol, jnp.array([1.0, 2.0]), rtol=1e-4)

# ==========================================
# 6. Test JIT Compilation
# ==========================================

def test_jit_compilation():
    """Ensure the public functions can be JIT compiled."""
    omega = 1.0
    R = 1.0
    
    # We must mark Phi_xyz as static because it's a callable
    jit_vcirc = jax.jit(vcirc, static_argnames=['Phi_xyz'])
    jit_kappa = jax.jit(kappa, static_argnames=['Phi_xyz'])
    jit_Rc    = jax.jit(Rc_from_Lz, static_argnames=['Phi_xyz'])
    
    assert jit_vcirc(phi_harmonic, R, omega) == 1.0
    assert jit_kappa(phi_harmonic, R, omega) == 2.0
    assert jit_Rc(phi_harmonic, 1.0, 1.0, omega) == 1.0