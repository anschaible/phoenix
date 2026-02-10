import pytest
import jax
import jax.numpy as jnp
from jax import random

# --- 1. JAX-COMPATIBLE STUBS (The Physics "Fakes") ---
# We define real mathematical functions so JAX can differentiation (grad) them.
# We simulate a "Flat Rotation Curve" where v_circ = v0 (constant).

def stub_vcirc(Phi, R, *theta):
    # Constant velocity curve (v0 = 200.0)
    # We use jnp.array to ensure JAX compatibility
    return jnp.array(200.0)

def stub_Rc_from_Lz(Phi, Lz, R0, *theta):
    # For flat rotation: Lz = R * v0  =>  R = Lz / v0
    v0 = 200.0
    return Lz / v0

def stub_kappa(Phi, R, *theta):
    # Epicyclic frequency for flat rotation: kappa = sqrt(2) * Omega = sqrt(2) * v0 / R
    v0 = 200.0
    return jnp.sqrt(2.0) * (v0 / R)

def stub_nu(Phi, R, *theta):
    # Vertical frequency. For simplicity, let's assume nu = v0 / R
    v0 = 200.0
    return v0 / R

# --- 2. IMPORT AND PATCH ---
from phoenix import actions_to_phase_space as dynamics

@pytest.fixture(autouse=True)
def patch_physics(monkeypatch):
    """
    Automatically patch the physics functions INSIDE the user's module 
    for every test in this file. This prevents 'pollution' of other test files.
    """
    # We patch the module where actions_to_phase_space is DEFINED.
    monkeypatch.setattr(dynamics, "vcirc", stub_vcirc)
    monkeypatch.setattr(dynamics, "Rc_from_Lz", stub_Rc_from_Lz)
    monkeypatch.setattr(dynamics, "kappa", stub_kappa)
    monkeypatch.setattr(dynamics, "nu", stub_nu)

# --- 3. FIXTURES ---

@pytest.fixture
def params():
    return {"R0": 8.0}

@pytest.fixture
def key():
    return random.PRNGKey(42)

@pytest.fixture
def phi_pot():
    # A dummy potential function (content doesn't matter as we stubbed the frequencies)
    return lambda x, y, z, *theta: 0.0

@pytest.fixture
def theta():
    return ()

# --- 4. TESTS ---

def test_circular_orbit_limit(params, phi_pot, theta, key):
    """
    Test the 'cold' limit: Jr -> 0, Jz -> 0.
    1. Planar (z ~ 0, vz ~ 0)
    2. Circular (R ~ Rc, v_R ~ 0)
    """
    # Lz corresponding to R=8kpc (8 * 200 = 1600)
    Jr, Jz, Lz = 0.0001, 0.0001, 1600.0
    
    x, y, z, vx, vy, vz = dynamics.actions_to_phase_space(
        Jr, Jz, Lz, params, key, phi_pot, theta
    )
    
    R = jnp.sqrt(x**2 + y**2)
    v_R = (x*vx + y*vy) / R
    
    # Expected Guiding Center Radius (Lz / v0 = 1600/200 = 8.0)
    expected_Rc = 8.0
    
    # Check R is close to Rc
    assert jnp.allclose(R, expected_Rc, atol=0.1)
    # Check Planarity
    assert jnp.allclose(z, 0.0, atol=0.05)
    assert jnp.allclose(vz, 0.0, atol=0.5)
    # Check Circularity
    assert jnp.allclose(v_R, 0.0, atol=1.0) 

def test_guiding_center_reconstruction(params, phi_pot, theta, key):
    """
    Check if the output Lz is consistent with the input Lz.
    """
    Jr, Jz, Lz = 10.0, 10.0, 1600.0 
    
    x, y, z, vx, vy, vz = dynamics.actions_to_phase_space(
        Jr, Jz, Lz, params, key, phi_pot, theta
    )
    
    R = jnp.sqrt(x**2 + y**2)
    v_phi_actual = (x*vy - y*vx) / R
    Lz_actual = R * v_phi_actual
    
    # Tolerance: Epicyclic approx is not exact Lz conservation, 
    # but should be close (within 10%)
    assert jnp.allclose(Lz, Lz_actual, rtol=0.1)

def test_vertical_oscillation_bounds(params, phi_pot, theta, key):
    """
    Ensure z stays within the estimated amplitude A_z.
    """
    Jr, Jz, Lz = 0.0, 100.0, 1600.0 
    
    # Calculated expected Amplitude manually based on our stubs:
    # nu = v0 / R = 200 / 8 = 25
    # Az = sqrt(2 * Jz / nu) = sqrt(200 / 25) = sqrt(8) ~ 2.82
    
    candidates = jnp.array([[Jr, Jz, Lz]] * 50)
    
    # Map batch
    results = dynamics.map_actions_to_phase_space(
        candidates, params, key, phi_pot, theta
    )
    
    zs = results[:, 2] # z column
    
    # Max Z should be around 2.82. 
    # We assert it is definitely below 4.0 and definitely implies motion (>0.1)
    assert jnp.max(jnp.abs(zs)) < 4.0 
    assert jnp.max(jnp.abs(zs)) > 1.0 

def test_output_shapes(params, phi_pot, theta, key):
    """Test vectorization shape."""
    n_samples = 10
    candidates = jnp.array([[10.0, 5.0, 1600.0]] * n_samples)
    
    result = dynamics.map_actions_to_phase_space(
        candidates, params, key, phi_pot, theta
    )
    
    assert result.shape == (n_samples, 6)
    assert not jnp.isnan(result).any()