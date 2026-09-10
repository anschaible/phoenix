"""Shared fixtures for the `phoenix.optimization` tests.

The optimization code is built around a neural-network surrogate (`PhoenixMapper`)
that maps actions+angles+potential parameters to phase-space coordinates. Loading
the trained weights in a unit test would be slow and would tie the tests to a
particular checkpoint, so the tests use `FakeMapper`: a small analytic stand-in
with the same `map_to_phase_space(actions, angles, potentials) -> (N, 6)`
interface. It is written entirely in `jnp` ops so it stays differentiable and
jittable, and it depends on the potential parameters (as the real surrogate does),
which is what makes the gradient tests meaningful.
"""
import jax.numpy as jnp
import pytest


class FakeMapper:
    """Analytic, differentiable stand-in for `PhoenixMapper`.

    Produces a rotating disk-like population: the guiding radius grows with
    J_phi, J_r drives an in-plane epicycle, J_z drives vertical motion, and the
    circular speed depends on the (network-scaled) potential parameters.
    """

    def map_to_phase_space(self, actions, angles, potentials):
        Jr, Jz, Jphi = actions[:, 0], actions[:, 1], actions[:, 2]
        th_r, th_z, th_phi = angles[:, 0], angles[:, 1], angles[:, 2]
        # Same column order as the real surrogate, masses in units of 1e11 Msun.
        M_halo, M_disk, a_disk = potentials[:, 0], potentials[:, 2], potentials[:, 3]

        # Guiding radius (strictly positive) with a bounded epicyclic wobble.
        R_g = 0.3 + 0.004 * Jphi * a_disk / 3.0
        R = R_g * (1.0 + 0.2 * jnp.cos(th_r) * jnp.tanh(Jr / 100.0))

        z = 0.01 * Jz * jnp.sin(th_z)

        # Circular speed rises with the enclosed mass and flattens beyond a_disk.
        v_circ = 220.0 * jnp.sqrt(M_disk + 0.1 * M_halo) * R / (R + a_disk)
        v_R = 0.3 * Jr * jnp.sin(th_r)
        v_z = 0.3 * Jz * jnp.cos(th_z)

        cos_phi, sin_phi = jnp.cos(th_phi), jnp.sin(th_phi)
        x = R * cos_phi
        y = R * sin_phi
        vx = v_R * cos_phi - v_circ * sin_phi
        vy = v_R * sin_phi + v_circ * cos_phi

        return jnp.stack([x, y, z, vx, vy, v_z], axis=1)


@pytest.fixture
def fake_mapper():
    return FakeMapper()


@pytest.fixture
def pot_params():
    return {
        'M_halo': 1e12, 'a_halo': 20.0,
        'M_disk': 5e10, 'a_disk': 3.0, 'b_disk': 0.3,
        'M_bulge': 1e10, 'a_bulge': 1.0,
    }


@pytest.fixture
def disk_df_params():
    return {
        'R0': 8.0, 'Rd': 3.0, 'Sigma0': 1000.0,
        'RsigR': 6.0, 'RsigZ': 6.0,
        'sigmaR0_R0': 35.0, 'sigmaz0_R0': 20.0,
        'L0': 10.0, 'Rinit_for_Rc': 8.0,
    }


@pytest.fixture
def bulge_df_params():
    return {
        'N0_spheroid': 1e10, 'J0_spheroid': 100.0,
        'Gamma_spheroid': 1.5, 'Beta_spheroid': 4.5, 'eta_spheroid': 1.0,
    }


@pytest.fixture
def all_params(pot_params, disk_df_params, bulge_df_params):
    """The three parameter dicts as one tuple, in the order the API takes them."""
    return pot_params, disk_df_params, bulge_df_params
