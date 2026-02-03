import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial

# --- Import or Paste your function here ---
# Assuming the function is in a module named 'dynamics'
# from dynamics import Rc_from_Lz

# For this standalone example, I will define the missing dependency 
# and the function itself so this block is runnable as-is.

def _vcirc_scalar(Phi_xyz, R, *theta):
    """Calculates circular velocity v_c = sqrt(R * dPhi/dR)."""
    # Evaluate gradient at (x=R, y=0, z=0)
    pos = jnp.array([R, 0.0, 0.0])
    # argnums=0 takes grad w.r.t position vector
    grad_phi = jax.grad(Phi_xyz, argnums=0)(pos, *theta)
    dPhi_dR = grad_phi[0] 
    return jnp.sqrt(jnp.maximum(0.0, R * dPhi_dR))

def Rc_from_Lz(Phi_xyz, Lz, R_init, *theta):
    # [PASTE YOUR FUNCTION CODE HERE]
    # For the test to run immediately, I am replicating the core logic:
    Lz = jnp.asarray(Lz); R_init = jnp.asarray(R_init)
    theta = tuple(jnp.asarray(t) for t in theta)
    shape = jnp.broadcast_shapes(Lz.shape, R_init.shape)
    Lz_b = jnp.broadcast_to(Lz, shape).ravel()
    R0   = jnp.clip(jnp.broadcast_to(R_init, shape), 1e-2).ravel()

    def _g_scalar(Rs, Ls):
        return Rs * _vcirc_scalar(Phi_xyz, Rs, *theta) - Ls

    vg = jax.vmap(jax.value_and_grad(_g_scalar), in_axes=(0, 0))

    def body(_, Rvec):
        gR, dgR = vg(Rvec, Lz_b)
        Rn = Rvec - gR / jnp.clip(dgR, 1e-12)
        Rn = jnp.clip(Rn, 1e-3, 1e3 * jnp.maximum(1.0, R0))
        return 0.7 * Rn + 0.3 * Rvec

    R_sol = jax.lax.fori_loop(0, 30, body, R0)
    return R_sol.reshape(shape)

# --- Define Potentials for Testing ---

def phi_harmonic(r_vec, omega):
    """Harmonic Oscillator: Phi = 0.5 * omega^2 * r^2"""
    r2 = jnp.sum(r_vec**2)
    return 0.5 * omega**2 * r2

def phi_kepler(r_vec, GM):
    """Kepler Potential: Phi = -GM / r"""
    r = jnp.linalg.norm(r_vec)
    return -GM / r

# --- Tests ---

def test_harmonic_oscillator_recovery():
    """
    Case 1: Harmonic Oscillator
    v_c = omega * R
    L_z = R * v_c = omega * R^2  => R = sqrt(L_z / omega)
    """
    omega = 2.0
    Lz_target = 16.0
    expected_Rc = jnp.sqrt(Lz_target / omega) # sqrt(8) ~ 2.828
    
    # Initial guess can be anything reasonable
    R_guess = 1.0 
    
    rc = Rc_from_Lz(phi_harmonic, Lz_target, R_guess, omega)
    
    np.testing.assert_allclose(rc, expected_Rc, rtol=1e-5, 
        err_msg="Failed to recover Rc for Harmonic Oscillator")

def test_kepler_recovery():
    """
    Case 2: Kepler/Point Mass
    v_c = sqrt(GM / R)
    L_z = R * sqrt(GM/R) = sqrt(GM * R) => R = L_z^2 / GM
    """
    GM = 1.0
    Lz_target = 3.0
    expected_Rc = (Lz_target**2) / GM # 9.0
    
    # Start guess far away to test convergence
    R_guess = 0.5 
    
    rc = Rc_from_Lz(phi_kepler, Lz_target, R_guess, GM)
    
    np.testing.assert_allclose(rc, expected_Rc, rtol=1e-5,
        err_msg="Failed to recover Rc for Kepler potential")

def test_broadcasting_shapes():
    """
    Case 3: Ensure scalar Lz broadcasts against array R_init (and vice versa)
    and handles batch dimensions correctly.
    """
    omega = 1.0
    # Lz = [1, 4, 9], Omega=1 => Expected R = [1, 2, 3]
    Lz_array = jnp.array([1.0, 4.0, 9.0]) 
    R_init_scalar = 1.5 # Single guess for all
    
    rc = Rc_from_Lz(phi_harmonic, Lz_array, R_init_scalar, omega)
    
    assert rc.shape == (3,)
    np.testing.assert_allclose(rc, jnp.array([1.0, 2.0, 3.0]), rtol=1e-5)

def test_jit_compilation():
    """
    Case 4: Ensure the function can be JIT compiled.
    """
    omega = 1.0
    Lz = 4.0
    R_init = 1.0
    
    # Wrap in jit
    jit_Rc = jax.jit(Rc_from_Lz, static_argnames=['Phi_xyz'])
    
    # Run once to compile
    res = jit_Rc(phi_harmonic, Lz, R_init, omega)
    assert res.shape == () # Scalar check
    np.testing.assert_allclose(res, 2.0, rtol=1e-5)

def test_potential_parameter_passing():
    """
    Case 5: Ensure *theta args are passed through correctly to the potential.
    Using Harmonic oscillator but varying omega via args.
    """
    Lz = 1.0
    R_init = 1.0
    
    # Omega = 1.0 -> R = 1.0
    rc1 = Rc_from_Lz(phi_harmonic, Lz, R_init, 1.0)
    # Omega = 4.0 -> R = sqrt(1/4) = 0.5
    rc2 = Rc_from_Lz(phi_harmonic, Lz, R_init, 4.0)
    
    np.testing.assert_allclose(rc1, 1.0)
    np.testing.assert_allclose(rc2, 0.5)