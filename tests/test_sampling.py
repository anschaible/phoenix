import numpy as np
import jax
import jax.numpy as jnp

from phoenix.distribution_functions.sampling import (
    soft_acceptance,
    sample_df_potential,
    sample_df_importance,
)
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
    candidates, soft = sample_df_potential(_simple_df, key, {}, None, (), n, env, tau=0.1)
    assert candidates.shape == (n, 3)
    assert soft.shape == (n,)
    arr_soft = np.asarray(soft)
    assert np.all(arr_soft >= 0.0) and np.all(arr_soft <= 1.0)


def test_sample_df_potential_respects_the_action_bounds():
    key = random.PRNGKey(1)
    n = 64
    J_bounds = (50.0, 30.0, 1500.0)
    candidates, _ = sample_df_potential(_simple_df, key, {}, None, (), n, 1000.0,
                                        J_bounds=J_bounds)
    arr = np.asarray(candidates)
    for i, upper in enumerate(J_bounds):
        assert np.all(arr[:, i] >= 0.0)
        assert np.all(arr[:, i] <= upper)


def test_sample_df_potential_reproducible_and_consistent():
    key = random.PRNGKey(42)
    n = 5
    env = 500.0
    tau = 0.05

    # call function under test
    candidates, soft = sample_df_potential(_simple_df, key, {}, None, (), n, env, tau=tau)

    # reproduce the internal random draws deterministically
    _, kr, kz, kphi, krand = random.split(key, 5)
    Jr = random.uniform(kr, shape=(n,), minval=0.0, maxval=200.0)
    Jz = random.uniform(kz, shape=(n,), minval=0.0, maxval=200.0)
    Lz = random.uniform(kphi, shape=(n,), minval=0.0, maxval=6000.0)
    expected_candidates = jnp.stack([Jr, Jz, Lz], axis=1)

    rand_vals = random.uniform(krand, shape=(n,))

    # compute df values and expected soft weights
    df_vals = vmap(lambda c: _simple_df(c[0], c[1], c[2], None, (), {}))(expected_candidates)
    expected_soft = soft_acceptance(df_vals, rand_vals, env, tau)

    np.testing.assert_allclose(np.asarray(candidates), np.asarray(expected_candidates))
    np.testing.assert_allclose(np.asarray(soft), np.asarray(expected_soft))


def test_sample_df_potential_is_deterministic_in_the_key():
    n = 16
    args = (_simple_df, random.PRNGKey(3), {}, None, (), n, 1000.0)
    c1, s1 = sample_df_potential(*args)
    c2, s2 = sample_df_potential(*args)
    c3, s3 = sample_df_potential(_simple_df, random.PRNGKey(4), {}, None, (), n, 1000.0)
    np.testing.assert_array_equal(np.asarray(c1), np.asarray(c2))
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
    assert not np.allclose(np.asarray(c1), np.asarray(c3))


def test_sample_df_potential_auto_calibrates_the_envelope():
    """`envelope_max=None` uses the candidates' own maximum DF value, which is what
    keeps the sampler well-behaved when the DF parameters move during a fit."""
    key = random.PRNGKey(7)
    n = 32
    candidates, soft = sample_df_potential(_simple_df, key, {}, None, (), n,
                                           envelope_max=None, tau=0.05)
    df_vals = vmap(lambda c: _simple_df(c[0], c[1], c[2], None, (), {}))(candidates)
    expected_env = jnp.maximum(jnp.max(df_vals), 1e-30)
    rand_vals = random.uniform(random.split(key, 5)[4], shape=(n,))
    expected_soft = soft_acceptance(df_vals, rand_vals, expected_env, 0.05)
    np.testing.assert_allclose(np.asarray(soft), np.asarray(expected_soft), rtol=1e-6)
    arr = np.asarray(soft)
    assert np.all(arr >= 0.0) and np.all(arr <= 1.0)


def test_sample_df_potential_auto_envelope_survives_a_vanishing_df():
    """A DF that is identically zero must not produce NaNs (the 1e-30 floor)."""
    zero_df = lambda jr, jz, lz, Phi_xyz, theta, params: 0.0 * jr
    _, soft = sample_df_potential(zero_df, random.PRNGKey(0), {}, None, (), 8,
                                  envelope_max=None)
    assert np.all(np.isfinite(np.asarray(soft)))


# ==============================================================================
# sample_df_importance
# ==============================================================================
def _flat_potential(x, y, z):
    """Trivial potential; `sample_df_importance` only passes it through to the DF."""
    return jnp.zeros_like(jnp.asarray(x) + jnp.asarray(y) + jnp.asarray(z))


def test_sample_df_importance_weights_track_the_df():
    """The whole point of importance weighting: w must be proportional to f/q, so a
    DF whose scale changes must move the weighted action distribution. The
    soft-acceptance sampler fails this (it returned bit-identical populations for
    J0 25 vs 400), which is why this sampler exists."""
    def df(Jr, Jz, Jphi, Phi, theta, params):
        # Exponential in total action with scale params['J0'].
        return jnp.exp(-(Jr + Jz + Jphi) / params['J0'])

    means = []
    for J0 in (10.0, 50.0, 200.0):
        cand, w = sample_df_importance(
            df=df, key=random.PRNGKey(0), params={'J0': J0},
            Phi_xyz=_flat_potential, theta=(), n_candidates=20_000,
            J_scales=(J0, J0, J0),
        )
        Jtot = jnp.sum(cand, axis=1)
        means.append(float(jnp.sum(w * Jtot) / jnp.sum(w)))

    # For f ~ exp(-Jtot/J0) over three actions, <Jtot> = 3*J0.
    for J0, got in zip((10.0, 50.0, 200.0), means):
        assert 0.75 * 3 * J0 < got < 1.25 * 3 * J0, (J0, got)
    assert means[0] < means[1] < means[2]


def test_sample_df_importance_gradients_finite_when_df_underflows():
    """Regression test for a silent NaN-gradient bug.

    log(f) was guarded with `jnp.maximum(f, 1e-300)`, but JAX defaults to float32
    whose smallest denormal is ~1e-45, so the clamp was a no-op: tail candidates with
    f underflowing to exactly 0 produced log(0) = -inf and a NaN gradient. Because
    `pipeline.fit` zeroes non-finite gradients, this silently froze every parameter
    the DF depends on and still looked like a converged fit. The DF here underflows
    hard on purpose, so the weights must still be finite AND differentiable.
    """
    def df(Jr, Jz, Jphi, Phi, theta, params):
        # Underflows to exactly 0.0 in float32 well inside the sampled range.
        return jnp.exp(-(Jr + Jz + Jphi) * params['rate'])

    def total(rate):
        _, w = sample_df_importance(
            df=df, key=random.PRNGKey(1), params={'rate': rate},
            Phi_xyz=_flat_potential, theta=(), n_candidates=4_000,
            J_scales=(50.0, 50.0, 50.0),
        )
        return jnp.sum(w)

    # Confirm the premise: some candidates really do underflow to zero.
    cand, w = sample_df_importance(
        df=df, key=random.PRNGKey(1), params={'rate': 5.0},
        Phi_xyz=_flat_potential, theta=(), n_candidates=4_000,
        J_scales=(50.0, 50.0, 50.0))
    f = jnp.exp(-jnp.sum(cand, axis=1) * 5.0)
    assert jnp.any(f == 0.0), "premise: the DF must underflow for this to be a test"

    assert jnp.isfinite(total(5.0)), "forward pass must stay finite"
    g = jax.grad(total)(5.0)
    assert jnp.isfinite(g), f"gradient must be finite through DF underflow, got {g}"


def test_sample_df_importance_is_unbiased_against_known_mean():
    """Importance weighting must be unbiased, with no temperature to tune: for
    f ~ exp(-J/s) the weighted mean of J_r is the analytic s, whatever proposal
    scale is used (a deliberately mismatched one included)."""
    s_true = 30.0

    def df(Jr, Jz, Jphi, Phi, theta, params):
        return jnp.exp(-Jr / params['s']) * jnp.exp(-Jz) * jnp.exp(-Jphi)

    for proposal in (10.0, 30.0, 90.0):
        cand, w = sample_df_importance(
            df=df, key=random.PRNGKey(3), params={'s': s_true},
            Phi_xyz=_flat_potential, theta=(), n_candidates=60_000,
            J_scales=(proposal, 1.0, 1.0),
        )
        got = float(jnp.sum(w * cand[:, 0]) / jnp.sum(w))
        assert abs(got - s_true) / s_true < 0.15, (proposal, got)
