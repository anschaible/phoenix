import numpy as np
import jax.numpy as jnp

from phoenix.sampling import soft_acceptance, sample_df_potential
from jax import random, vmap


def _simple_df(jr, jz, lz, Phi_xyz, theta, params):
    # simple, deterministic DF: linear combination of actions
    return jr + 2.0 * jz + 0.1 * lz


def test_soft_acceptance_bounds():
    df = jnp.array([0.0, 0.5, 1.0])
    rand = jnp.array([0.2, 0.5, 0.8])
    env = 1.0
    out = soft_acceptance(df, rand, env)
    arr = np.asarray(out)
    assert np.all(arr >= 0.0) and np.all(arr <= 1.0)


def test_soft_acceptance_midpoint():
    env = 2.0
    rand = jnp.array([0.3])
    df = env * rand
    out = soft_acceptance(df, rand, env)
    val = float(np.asarray(out).item())
    assert np.isclose(val, 0.5)


def test_soft_acceptance_extremes():
    df = jnp.array([2.0])
    rand = jnp.array([0.0])
    env = 1.0
    out = soft_acceptance(df, rand, env, tau=1e-6)
    val = float(np.asarray(out).item())
    assert val > 0.999


def test_soft_acceptance_tau_effect():
    df = jnp.array([0.75])
    rand = jnp.array([0.25])
    env = 1.0
    out_small = float(np.asarray(soft_acceptance(df, rand, env, tau=0.01)).item())
    out_large = float(np.asarray(soft_acceptance(df, rand, env, tau=2.0)).item())
    assert out_small > out_large
    assert abs(out_large - 0.5) < abs(out_small - 0.5)


def test_sample_df_potential_shapes_and_ranges():
    key = random.PRNGKey(0)
    n = 8
    env = 1000.0
    candidates, weighted_candidates, soft = sample_df_potential(_simple_df, key, {}, None, {}, n, env, tau=0.1)
    assert candidates.shape == (n, 3)
    assert weighted_candidates.shape == (n, 3)
    assert soft.shape == (n,)
    arr_soft = np.asarray(soft)
    assert np.all(arr_soft >= 0.0) and np.all(arr_soft <= 1.0)


def test_sample_df_potential_reproducible_and_consistent():
    key = random.PRNGKey(42)
    n = 5
    env = 500.0
    tau = 0.05

    # call function under test
    candidates, weighted, soft = sample_df_potential(_simple_df, key, {}, None, {}, n, env, tau=tau)

    # reproduce internal random draws deterministically
    k = key
    k, sub = random.split(k)
    Jr = random.uniform(sub, shape=(n,), minval=0.0, maxval=400.0)
    k, sub = random.split(k)
    Jz = random.uniform(sub, shape=(n,), minval=0.0, maxval=400.0)
    k, sub = random.split(k)
    Lz = random.uniform(sub, shape=(n,), minval=0.0, maxval=2000.0)
    expected_candidates = jnp.stack([Jr, Jz, Lz], axis=1)

    k, sub = random.split(k)
    rand_vals = random.uniform(sub, shape=(n,))

    # compute df values and expected soft weights
    df_vals = vmap(lambda c: _simple_df(c[0], c[1], c[2], None, {}, {}))(expected_candidates)
    expected_soft = soft_acceptance(df_vals, rand_vals, env, tau)
    expected_weighted = expected_candidates * expected_soft[:, None]

    np.testing.assert_allclose(np.asarray(candidates), np.asarray(expected_candidates))
    np.testing.assert_allclose(np.asarray(soft), np.asarray(expected_soft))
    np.testing.assert_allclose(np.asarray(weighted), np.asarray(expected_weighted))
