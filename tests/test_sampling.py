import numpy as np
import jax.numpy as jnp

from phoenix.sampling import soft_acceptance


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
