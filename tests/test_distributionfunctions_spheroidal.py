import pytest
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from phoenix.distribution_functions.spheroid import f_double_power_law

# --- Fixtures and Helpers ---

@pytest.fixture
def standard_params():
    """Standard parameters for the DF."""
    return {
        "N0_spheroid": 1.0e5,
        "J0_spheroid": 100.0,
        "Gamma_spheroid": 2.0,  # Inner slope
        "Beta_spheroid": 5.0,   # Outer slope
        "eta_spheroid": 1.0
    }

# --- Tests ---

def test_scalar_output_shape(standard_params):
    """Test that scalar inputs produce scalar outputs."""
    Jr, Jz, Jphi = 10.0, 10.0, 10.0

    val = f_double_power_law(Jr, Jz, Jphi, standard_params)
    
    assert jnp.ndim(val) == 0
    assert not jnp.isnan(val)
    assert val > 0

def test_vectorization_shape(standard_params):
    """Test that the function handles JAX arrays (batching) correctly."""
    n_particles = 100
    Jr = jnp.ones(n_particles) * 10.0
    Jz = jnp.ones(n_particles) * 5.0
    Jphi = jnp.ones(n_particles) * 2.0
    
    val = f_double_power_law(Jr, Jz, Jphi, standard_params)
    
    assert val.shape == (n_particles,)
    # All values should be identical since inputs are identical
    assert_allclose(val, val[0])

def test_manual_calculation():
    """
    Verify the math against a hand-calculated known value.
    We choose Jtot = J0 to simplify the algebra.
    """
    params = {
        "N0_spheroid": (2.0 * np.pi)**3, # chosen to cancel the prefactor
        "J0_spheroid": 1.0,
        "Gamma_spheroid": 2.0,
        "Beta_spheroid": 4.0,
        "eta_spheroid": 1.0
    }
    
    # Set inputs so Jr + Jz + |Jphi| = J0 = 1.0
    Jr, Jz, Jphi = 0.5, 0.0, 0.5 
    
    # Expected Logic:
    # Factor = N0 / (2pi J0)^3 = 1.0
    # Jtot = 1.0, J0 = 1.0
    # Inner = (1 + (1/1)^1)^(2/1) = 2^2 = 4
    # Outer = (1 + (1/1)^1)^(-4/1) = 2^-4 = 1/16
    # Result = 1 * 4 * (1/16) = 0.25
    
    val = f_double_power_law(Jr, Jz, Jphi, params)
    
    assert_allclose(val, 0.25, rtol=1e-5)

def test_asymptotic_inner_slope(standard_params):
    """
    Test behavior when J_tot << J0.
    The function should scale roughly as J_tot^(-Gamma).
    """
    J0 = standard_params["J0_spheroid"]
    Gamma = standard_params["Gamma_spheroid"]
    
    # Very small actions compared to J0
    J_small = J0 * 1e-4
    J_very_small = J_small / 2.0
    
    # Pass as Jr, set others to 0
    val1 = f_double_power_law(J_small, 0., 0., standard_params)
    val2 = f_double_power_law(J_very_small, 0., 0., standard_params)
    
    # Since J halved, and f ~ J^-Gamma:
    # ratio should be (1/2)^-Gamma = 2^Gamma
    expected_ratio = 2.0**Gamma
    calculated_ratio = val2 / val1
    
    # Allow some tolerance because it's an asymptotic approximation, not exact
    assert_allclose(calculated_ratio, expected_ratio, rtol=1e-3)

def test_asymptotic_outer_slope(standard_params):
    """
    Test behavior when J_tot >> J0.
    The function should scale roughly as J_tot^(-Beta).
    """
    J0 = standard_params["J0_spheroid"]
    Beta = standard_params["Beta_spheroid"]
    
    # Very large actions
    J_large = J0 * 1e4
    J_very_large = J_large * 2.0
    
    val1 = f_double_power_law(J_large, 0., 0., standard_params)
    val2 = f_double_power_law(J_very_large, 0., 0., standard_params)
    
    # Since J doubled, and f ~ J^-Beta:
    # ratio should be 2^-Beta
    expected_ratio = 2.0**(-Beta)
    calculated_ratio = val2 / val1
    
    assert_allclose(calculated_ratio, expected_ratio, rtol=1e-3)

def test_eta_default(standard_params):
    """Test that eta defaults to 1.0 if not provided."""
    params_no_eta = standard_params.copy()
    del params_no_eta["eta_spheroid"]
    
    params_eta_1 = standard_params.copy()
    params_eta_1["eta_spheroid"] = 1.0
    
    Jr, Jz, Jphi = 10.0, 10.0, 10.0
    
    val_default = f_double_power_law(Jr, Jz, Jphi, params_no_eta)
    val_explicit = f_double_power_law(Jr, Jz, Jphi, params_eta_1)
    
    assert_allclose(val_default, val_explicit)

def test_jit_compilation(standard_params):
    """Ensure the function can be JIT compiled."""
    # `params` is a plain dict, i.e. a pytree, so nothing needs to be static.
    jit_f = jax.jit(f_double_power_law)
    
    Jr, Jz, Jphi = jnp.array([10.0]), jnp.array([5.0]), jnp.array([2.0])
    
    # Run once to compile
    res1 = jit_f(Jr, Jz, Jphi, standard_params)
    # Run again
    res2 = jit_f(Jr, Jz, Jphi, standard_params)
    
    assert_allclose(res1, res2)

def test_gradient_wrt_actions(standard_params):
    """Test that the function is differentiable with respect to actions."""
    Jr = 10.0
    Jz = 10.0
    Jphi = 10.0
    
    # Define a wrapper to take grad w.r.t first arg (Jr)
    grad_func = jax.grad(f_double_power_law, argnums=0)
    
    d_dJr = grad_func(Jr, Jz, Jphi, standard_params)
    
    assert not jnp.isnan(d_dJr)
    # Since DF decreases as J increases, gradient should be negative
    assert d_dJr < 0

def test_absolute_value_Jphi(standard_params):
    """Test that the function uses abs(Jphi) (symmetric in angular momentum)."""
    Jr, Jz = 10.0, 5.0
    Jphi_pos = 5.0
    Jphi_neg = -5.0
    
    val_pos = f_double_power_law(Jr, Jz, Jphi_pos, standard_params)
    val_neg = f_double_power_law(Jr, Jz, Jphi_neg, standard_params)
    
    assert_allclose(val_pos, val_neg)