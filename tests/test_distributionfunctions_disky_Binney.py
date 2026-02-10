import pytest
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

# --- Import your module ---
# Ensure your file is named disk_df.py or adjust the import
from phoenix.distributionfunctions_disky_Binney import (
    Sigma_exp,
    sigmaR_of_Rc,
    sigmaz_of_Rc,
    sigma_age,
    f_disc_from_params
)

# --- Helpers for Real Physics Testing ---

def harmonic_potential(x, y, z, omega0, nu0):
    """
    Simple Harmonic Oscillator Potential.
    Phi = 0.5 * omega0^2 * R^2 + 0.5 * nu0^2 * z^2
    
    Analytic properties for Phoenix to find:
    - Omega = omega0
    - kappa = 2 * omega0
    - nu    = nu0
    - Rc    = sqrt(Lz / omega0)
    """
    R2 = x**2 + y**2
    return 0.5 * (omega0**2) * R2 + 0.5 * (nu0**2) * z**2

@pytest.fixture
def harmonic_params():
    """
    Parameters for the harmonic potential. 
    We set omega0=1, nu0=1 for simplicity.
    """
    return (1.0, 1.0) # omega0, nu0

@pytest.fixture
def disk_params():
    """Standard Milky Way-ish disk parameters."""
    return {
        "R0": 8.0,
        "Rd": 2.5,
        "Sigma0": 100.0,
        "RsigR": 4.0,
        "RsigZ": 4.0,
        "sigmaR0_R0": 30.0,
        "sigmaz0_R0": 20.0,
        "L0": 10.0,      
        "Rinit_for_Rc": 8.0
    }

# ==========================================
# 1. Tests for Radial Profiles (Unit Tests)
# ==========================================

def test_Sigma_exp():
    """Test exponential surface density."""
    R0, Rd, Sigma0 = 8.0, 2.5, 100.0
    
    # Check normalization at R0
    assert jnp.isclose(Sigma_exp(R0, R0, Rd, Sigma0), Sigma0)
    
    # Check scale length decay
    val_at_scale = Sigma_exp(R0 + Rd, R0, Rd, Sigma0)
    assert jnp.isclose(val_at_scale, Sigma0 * jnp.exp(-1.0))

def test_sigma_profiles():
    """Test velocity dispersion profiles."""
    R0, Rsig, sig0 = 8.0, 4.0, 30.0
    
    # Normalization
    assert jnp.isclose(sigmaR_of_Rc(R0, R0, Rsig, sig0), sig0)
    
    # Gradient check: inner galaxy (R < R0) should be hotter
    assert sigmaR_of_Rc(4.0, R0, Rsig, sig0) > sig0
    # Outer galaxy (R > R0) should be cooler
    assert sigmaR_of_Rc(12.0, R0, Rsig, sig0) < sig0

def test_sigma_age():
    """Test the age-velocity dispersion relation (AVR)."""
    # sigma ~ t^beta
    tau = jnp.array([1.0, 8.0]) # 1 Gyr and 8 Gyr
    beta = 0.3
    sig_ref = 20.0
    taum = 8.0 # Reference age
    tau1 = 0.0 # Simplify
    
    vals = sigma_age(sig_ref, tau, tau1, taum, beta)
    
    # Old stars (8 Gyr) should match reference
    assert jnp.isclose(vals[1], sig_ref)
    
    # Young stars (1 Gyr) should be dynamically colder
    assert vals[0] < vals[1]

# ==========================================
# 2. Tests for DF Logic (Integration Tests)
# ==========================================

def test_df_runnable(harmonic_params, disk_params):
    """
    Basic sanity check: does the function run with the REAL Phoenix library
    and return the correct shape?
    """
    Jr = jnp.array([1.0, 2.0, 3.0])
    Jz = jnp.array([1.0, 1.0, 1.0])
    Jphi = jnp.array([10.0, 10.0, 10.0]) # Lz
    
    val = f_disc_from_params(
        Jr, Jz, Jphi, 
        harmonic_potential, harmonic_params, 
        disk_params
    )
    
    assert val.shape == (3,)
    assert jnp.all(jnp.isfinite(val))
    assert jnp.all(val > 0.0)

def test_df_analytic_harmonic(harmonic_params, disk_params):
    """
    Strict mathematical check using the Harmonic Oscillator.
    We know exactly what Omega, kappa, and Rc should be.
    """
    omega0, nu0 = harmonic_params
    
    # Inputs
    Jphi = 16.0  # Lz
    Jr = 0.0     # Cold orbit
    Jz = 0.0     # Cold orbit
    
    # 1. Calculate EXPECTED Physics for Harmonic Potential
    # For Phi = 0.5*w^2*R^2, Vc = w*R. 
    # Lz = R*Vc = w*R^2 -> R_c = sqrt(Lz/w)
    Rc_expected = jnp.sqrt(Jphi / omega0) # sqrt(16/1) = 4.0
    
    Om_expected = omega0            # 1.0
    kappa_expected = 2.0 * omega0   # 2.0
    nu_expected = nu0               # 1.0
    
    # 2. Calculate EXPECTED Profiles at Rc=4.0
    # Params from fixture: R0=8.0, Rd=2.5, Rsig=4.0
    R0 = disk_params['R0']
    Sigma_expected = disk_params['Sigma0'] * jnp.exp(-(Rc_expected - R0)/disk_params['Rd'])
    sigR_expected  = disk_params['sigmaR0_R0'] * jnp.exp((R0 - Rc_expected)/disk_params['RsigR'])
    sigZ_expected  = disk_params['sigmaz0_R0'] * jnp.exp((R0 - Rc_expected)/disk_params['RsigZ'])
    
    # 3. Calculate EXPECTED DF
    # prefactor = (Omega * Sigma) / (2 pi^2 sigR^2 sigZ^2 kappa)
    # rot = 0.5 * (1 + tanh(Jphi/L0))
    # exp terms = 1.0 (since Jr=Jz=0)
    
    prefactor = (Om_expected * Sigma_expected) / (
        2 * jnp.pi**2 * sigR_expected**2 * sigZ_expected**2 * kappa_expected
    )
    rot_factor = 0.5 * (1.0 + jnp.tanh(Jphi / disk_params['L0']))
    
    expected_val = prefactor * rot_factor
    
    # 4. Run Actual Function
    calculated_val = f_disc_from_params(
        Jr, Jz, Jphi, 
        harmonic_potential, harmonic_params, 
        disk_params
    )
    
    # Check with reasonable tolerance
    # This verifies that Phoenix found the correct frequencies AND your DF combined them correctly
    assert jnp.isclose(calculated_val, expected_val, rtol=1e-4)

def test_df_jit_compatible(harmonic_params, disk_params):
    """Ensure the function can be JIT compiled (crucial for performance)."""
    
    # Wrap in JIT. 
    # Note: 'Phi_xyz' is a function, so it must be static or a PyTree. 
    # 'params' is a dict, which JAX handles if values are arrays/floats.
    
    # We define a partial to bake in the potential function 
    # so we don't have to mark it static manually in this test.
    import functools
    
    func_to_jit = functools.partial(f_disc_from_params, Phi_xyz=harmonic_potential)
    jitted_func = jax.jit(func_to_jit)
    
    Jr, Jz, Jphi = 1.0, 1.0, 10.0
    
    # First run (compilation)
    res1 = jitted_func(Jr, Jz, Jphi, theta=harmonic_params, params=disk_params)
    
    # Second run (cached)
    res2 = jitted_func(Jr, Jz, Jphi, theta=harmonic_params, params=disk_params)
    
    assert jnp.isclose(res1, res2)

def test_autodiff_gradients(harmonic_params, disk_params):
    """Ensure we can take gradients w.r.t actions (e.g. for Hamiltonian sampling)."""
    
    def df_wrapper(actions):
        Jr, Jz, Jphi = actions
        return f_disc_from_params(
            Jr, Jz, Jphi, 
            harmonic_potential, harmonic_params, 
            disk_params
        )
    
    # Actions: Jr, Jz, Jphi
    actions = jnp.array([1.0, 1.0, 10.0])
    
    # Gradient of DF w.r.t actions
    grad_fn = jax.grad(df_wrapper)
    grads = grad_fn(actions)
    
    # Gradients for Jr and Jz should be negative (DF decreases as random energy increases)
    assert grads[0] < 0  # dDF/dJr
    assert grads[1] < 0  # dDF/dJz
    assert jnp.all(jnp.isfinite(grads))