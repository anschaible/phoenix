"""Tests for `phoenix.optimization.poisson_penalty`.

The analytic-density helper is checked against closed-form Plummer and NFW
densities, and the penalty itself is checked on a tracer population drawn from a
real Plummer sphere -- a population that *is* the density sourcing the potential,
so the penalty must be small for it and clearly larger for a mis-shaped one.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phoenix.constants import G
from phoenix.optimization.poisson_penalty import (
    _single_point_kde,
    get_density_from_potential,
    compute_poisson_penalty,
)
from phoenix.potentials.potentials import (
    plummer_potential,
    nfw_potential,
    miyamoto_nagai_potential,
)

M_PLUMMER, A_PLUMMER = 1e10, 3.0


def _plummer_sphere(n=2000, M=M_PLUMMER, a=A_PLUMMER, seed=0):
    """Exact inverse-transform sampling of a Plummer sphere (equal-mass tracers)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, n)
    r = a / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    cos_t = rng.uniform(-1.0, 1.0, n)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    x = jnp.asarray(r * sin_t * np.cos(phi), dtype=jnp.float32)
    y = jnp.asarray(r * sin_t * np.sin(phi), dtype=jnp.float32)
    z = jnp.asarray(r * cos_t, dtype=jnp.float32)
    w = jnp.full((n,), M / n)
    return x, y, z, w


def _plummer_density(r, M=M_PLUMMER, a=A_PLUMMER):
    return 3.0 * M / (4.0 * np.pi * a ** 3) * (1.0 + r ** 2 / a ** 2) ** -2.5


# ==============================================================================
# _single_point_kde
# ==============================================================================
def test_single_point_kde_single_particle_at_its_own_position():
    """A lone particle contributes w / (h*sqrt(2pi))**3 at its own location."""
    h, w = 0.7, 2.5
    got = _single_point_kde(0.0, 0.0, 0.0,
                            jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]),
                            jnp.array([w]), h)
    expected = w / (h * np.sqrt(2.0 * np.pi)) ** 3
    np.testing.assert_allclose(float(got), expected, rtol=1e-5)


def test_single_point_kde_matches_the_explicit_gaussian_sum():
    rng = np.random.default_rng(1)
    n, h = 6, 1.3
    px, py, pz = (rng.normal(0.0, 2.0, n) for _ in range(3))
    w = rng.uniform(0.5, 2.0, n)
    for point in [(0.0, 0.0, 0.0), (1.5, -2.0, 0.5), (5.0, 5.0, 5.0)]:
        got = _single_point_kde(*point, jnp.asarray(px), jnp.asarray(py),
                                jnp.asarray(pz), jnp.asarray(w), h)
        d2 = ((px - point[0]) ** 2 + (py - point[1]) ** 2 + (pz - point[2]) ** 2)
        expected = np.sum(w * np.exp(-0.5 * d2 / h ** 2)) / (h * np.sqrt(2 * np.pi)) ** 3
        np.testing.assert_allclose(float(got), expected, rtol=1e-4)


def test_single_point_kde_decays_with_distance():
    args = (jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([1.0]), 1.0)
    values = [float(_single_point_kde(d, 0.0, 0.0, *args)) for d in (0.0, 1.0, 2.0, 4.0)]
    assert values == sorted(values, reverse=True)
    # Isotropic: the same distance along any axis gives the same density.
    np.testing.assert_allclose(float(_single_point_kde(0.0, 1.5, 0.0, *args)),
                               float(_single_point_kde(0.0, 0.0, 1.5, *args)), rtol=1e-6)


def test_single_point_kde_integrates_to_the_total_weight():
    """The kernel is normalized, so integrating the field recovers sum(weights)."""
    h = 1.0
    px = jnp.array([-1.0, 1.0])
    py = jnp.array([0.0, 0.5])
    pz = jnp.array([0.0, -0.5])
    w = jnp.array([1.0, 3.0])

    n_side, half = 25, 6.0
    grid = jnp.linspace(-half, half, n_side)
    dv = (2.0 * half / (n_side - 1)) ** 3
    X, Y, Z = jnp.meshgrid(grid, grid, grid, indexing='ij')
    field = jax.vmap(lambda a, b, c: _single_point_kde(a, b, c, px, py, pz, w, h))(
        X.ravel(), Y.ravel(), Z.ravel())
    np.testing.assert_allclose(float(jnp.sum(field) * dv), float(jnp.sum(w)), rtol=2e-2)


def test_single_point_kde_is_linear_in_the_weights():
    px, py, pz = jnp.array([0.0, 2.0]), jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0])
    base = float(_single_point_kde(0.5, 0.0, 0.0, px, py, pz, jnp.array([1.0, 1.0]), 1.0))
    scaled = float(_single_point_kde(0.5, 0.0, 0.0, px, py, pz, jnp.array([3.0, 3.0]), 1.0))
    np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-5)


# ==============================================================================
# get_density_from_potential
# ==============================================================================
def test_density_from_plummer_potential_matches_the_analytic_profile():
    rho_fn = get_density_from_potential(plummer_potential)
    for r in (0.3, 1.0, 3.0, 8.0):
        got = float(rho_fn(r, 0.0, 0.0, M_PLUMMER, A_PLUMMER))
        np.testing.assert_allclose(got, _plummer_density(r), rtol=1e-4)


def test_density_from_plummer_potential_is_isotropic():
    rho_fn = get_density_from_potential(plummer_potential)
    r = 2.5
    along_x = float(rho_fn(r, 0.0, 0.0, M_PLUMMER, A_PLUMMER))
    along_z = float(rho_fn(0.0, 0.0, r, M_PLUMMER, A_PLUMMER))
    diagonal = float(rho_fn(*(r / np.sqrt(3.0),) * 3, M_PLUMMER, A_PLUMMER))
    np.testing.assert_allclose(along_z, along_x, rtol=1e-4)
    np.testing.assert_allclose(diagonal, along_x, rtol=1e-4)


def test_density_from_nfw_potential_matches_the_analytic_profile():
    """For Phi = -G M ln(1 + r/a) / r the NFW density is M / (4 pi r (a + r)**2)."""
    rho_fn = get_density_from_potential(nfw_potential)
    M, a = 1e12, 20.0
    for r in (1.0, 5.0, 20.0):
        got = float(rho_fn(r, 0.0, 0.0, M, a))
        expected = M / (4.0 * np.pi * r * (a + r) ** 2)
        np.testing.assert_allclose(got, expected, rtol=1e-3)


def test_density_from_potential_is_linear_in_the_potential():
    """Poisson's equation is linear, so the density of a sum of potentials is the
    sum of their densities -- which is what lets `pipeline` split off the halo."""
    def combined(x, y, z, M_d, a_d, b_d, M_b, a_b):
        return (miyamoto_nagai_potential(x, y, z, M_d, a_d, b_d) +
                plummer_potential(x, y, z, M_b, a_b))

    params = (5e10, 3.0, 0.3, 1e10, 1.0)
    rho_sum = get_density_from_potential(combined)
    rho_mn = get_density_from_potential(miyamoto_nagai_potential)
    rho_pl = get_density_from_potential(plummer_potential)

    point = (2.0, 1.0, 0.4)
    got = float(rho_sum(*point, *params))
    expected = (float(rho_mn(*point, *params[:3])) +
                float(rho_pl(*point, *params[3:])))
    np.testing.assert_allclose(got, expected, rtol=1e-4)


def test_density_from_potential_scales_inversely_with_G():
    """rho = Laplace(Phi) / (4 pi G): doubling G halves the inferred density."""
    rho_default = get_density_from_potential(plummer_potential)
    rho_double = get_density_from_potential(plummer_potential, G=2.0 * G)
    point = (1.0, 0.5, 0.25)
    np.testing.assert_allclose(float(rho_double(*point, M_PLUMMER, A_PLUMMER)),
                               0.5 * float(rho_default(*point, M_PLUMMER, A_PLUMMER)),
                               rtol=1e-5)


def test_density_from_potential_is_vmappable_and_differentiable():
    rho_fn = get_density_from_potential(plummer_potential)
    rs = jnp.linspace(0.5, 5.0, 8)
    vals = jax.vmap(lambda r: rho_fn(r, 0.0, 0.0, M_PLUMMER, A_PLUMMER))(rs)
    assert vals.shape == (8,)
    assert np.all(np.diff(np.asarray(vals)) < 0.0)  # a Plummer profile decreases

    g = jax.grad(lambda a: rho_fn(1.0, 0.0, 0.0, M_PLUMMER, a))(A_PLUMMER)
    assert np.isfinite(float(g)) and float(g) != 0.0


# ==============================================================================
# compute_poisson_penalty
# ==============================================================================
def test_penalty_is_a_nonnegative_finite_scalar():
    x, y, z, w = _plummer_sphere(n=500)
    p = compute_poisson_penalty(x, y, z, w, plummer_potential,
                               (M_PLUMMER, A_PLUMMER), grid_size=12)
    assert np.shape(p) == ()
    assert np.isfinite(float(p)) and float(p) >= 0.0


def test_penalty_is_small_for_a_self_consistent_population():
    """Tracers drawn from the very density that sources the potential.

    The residual is not exactly zero (the KDE blurs the central cusp) but it must
    be far below what a mis-shaped population scores.
    """
    x, y, z, w = _plummer_sphere(n=3000)
    kw = dict(grid_size=16, h=0.8)
    good = float(compute_poisson_penalty(x, y, z, w, plummer_potential,
                                        (M_PLUMMER, A_PLUMMER), **kw))
    too_concentrated = float(compute_poisson_penalty(
        0.3 * x, 0.3 * y, 0.3 * z, w, plummer_potential, (M_PLUMMER, A_PLUMMER), **kw))
    too_extended = float(compute_poisson_penalty(
        3.0 * x, 3.0 * y, 3.0 * z, w, plummer_potential, (M_PLUMMER, A_PLUMMER), **kw))

    assert good < 0.05
    assert too_concentrated > 3.0 * good
    assert too_extended > 3.0 * good


def test_penalty_shape_only_ignores_a_global_density_rescaling():
    """With `shape_only=True` the penalty must be invariant under w -> c*w."""
    x, y, z, w = _plummer_sphere(n=1000)
    args = (plummer_potential, (M_PLUMMER, A_PLUMMER))
    kw = dict(grid_size=12, h=0.8)
    base = float(compute_poisson_penalty(x, y, z, w, *args, shape_only=True, **kw))
    scaled = float(compute_poisson_penalty(x, y, z, 100.0 * w, *args,
                                           shape_only=True, **kw))
    np.testing.assert_allclose(scaled, base, rtol=1e-4)


def test_penalty_without_shape_only_reacts_to_a_density_rescaling():
    x, y, z, w = _plummer_sphere(n=1000)
    args = (plummer_potential, (M_PLUMMER, A_PLUMMER))
    kw = dict(grid_size=12, h=0.8, shape_only=False)
    base = float(compute_poisson_penalty(x, y, z, w, *args, **kw))
    scaled = float(compute_poisson_penalty(x, y, z, 100.0 * w, *args, **kw))
    # log10(100) = 2 per pixel -> the penalty must grow by roughly 4
    assert scaled > base + 3.0


def test_penalty_shape_only_is_insensitive_to_the_potential_normalization():
    """The total mass only sets the density's amplitude, which `shape_only`
    divides out; the scale radius changes the shape and must matter."""
    x, y, z, w = _plummer_sphere(n=1000)

    def penalty(M, a):
        return compute_poisson_penalty(x, y, z, w, plummer_potential, (M, a),
                                       grid_size=12, h=0.8)

    d_lnM = float(jax.grad(penalty, argnums=0)(M_PLUMMER, A_PLUMMER)) * M_PLUMMER
    d_lna = float(jax.grad(penalty, argnums=1)(M_PLUMMER, A_PLUMMER)) * A_PLUMMER
    assert abs(d_lnM) < 1e-3 * abs(d_lna)


def test_penalty_density_floor_excludes_empty_pixels():
    """Most of the grid is empty; scoring it compares log10(~0) against a finite
    analytic density and manufactures a large artificial residual."""
    x, y, z, w = _plummer_sphere(n=2000)
    args = (plummer_potential, (M_PLUMMER, A_PLUMMER))
    kw = dict(grid_size=16, h=0.8)
    unmasked = float(compute_poisson_penalty(x, y, z, w, *args,
                                             density_floor_frac=0.0, **kw))
    default = float(compute_poisson_penalty(x, y, z, w, *args, **kw))
    strict = float(compute_poisson_penalty(x, y, z, w, *args,
                                           density_floor_frac=0.5, **kw))
    assert unmasked > default > strict
    assert all(np.isfinite(v) for v in (unmasked, default, strict))


def test_penalty_is_finite_when_no_pixel_passes_the_floor():
    """The n_pix guard must keep the result finite for a degenerate mask."""
    x = jnp.array([100.0, 120.0])       # far outside the grid
    y = jnp.array([0.0, 0.0])
    z = jnp.array([0.0, 0.0])
    w = jnp.array([1.0, 1.0])
    p = compute_poisson_penalty(x, y, z, w, plummer_potential,
                                (M_PLUMMER, A_PLUMMER), grid_size=8)
    assert np.isfinite(float(p))


def test_penalty_with_match_kernel_stays_close_to_the_plain_version():
    """Convolving the analytic side with the tracer kernel refines the comparison;
    for a self-consistent population it must not move the penalty wildly."""
    x, y, z, w = _plummer_sphere(n=2000)
    args = (plummer_potential, (M_PLUMMER, A_PLUMMER))
    plain = float(compute_poisson_penalty(x, y, z, w, *args, grid_size=16))
    matched = float(compute_poisson_penalty(x, y, z, w, *args, grid_size=16,
                                            match_kernel=True, n_quad=5))
    assert np.isfinite(matched)
    assert 0.5 * plain < matched < 2.0 * plain


@pytest.mark.parametrize("n_quad", [3, 5])
def test_penalty_match_kernel_runs_for_several_quadrature_orders(n_quad):
    x, y, z, w = _plummer_sphere(n=400)
    p = compute_poisson_penalty(x, y, z, w, plummer_potential,
                                (M_PLUMMER, A_PLUMMER), grid_size=8,
                                match_kernel=True, n_quad=n_quad)
    assert np.isfinite(float(p)) and float(p) >= 0.0


@pytest.mark.xfail(reason="match_kernel builds the Gauss-Hermite nodes with numpy, "
                          "so a traced `h` raises TracerArrayConversionError; only "
                          "the default `h` (never traced) works",
                   raises=jax.errors.TracerArrayConversionError, strict=False)
def test_penalty_match_kernel_accepts_an_explicit_bandwidth():
    x, y, z, w = _plummer_sphere(n=200)
    compute_poisson_penalty(x, y, z, w, plummer_potential,
                            (M_PLUMMER, A_PLUMMER), grid_size=8, h=0.8,
                            match_kernel=True, n_quad=3)


def test_penalty_gradients_are_finite_in_tracers_and_potential():
    x, y, z, w = _plummer_sphere(n=500)

    def penalty(x, y, z, w, a):
        return compute_poisson_penalty(x, y, z, w, plummer_potential,
                                       (M_PLUMMER, a), grid_size=12, h=0.8)

    gx, gw, ga = jax.grad(penalty, argnums=(0, 3, 4))(x, y, z, w, A_PLUMMER)
    assert np.all(np.isfinite(np.asarray(gx)))
    assert np.all(np.isfinite(np.asarray(gw)))
    assert np.isfinite(float(ga)) and float(ga) != 0.0


@pytest.mark.parametrize("grid_size", [8, 12])
def test_penalty_recompiles_for_a_new_static_grid_size(grid_size):
    """`grid_size` is a static argument: changing it must retrace, not error."""
    x, y, z, w = _plummer_sphere(n=300)
    p = compute_poisson_penalty(x, y, z, w, plummer_potential,
                                (M_PLUMMER, A_PLUMMER), grid_size=grid_size)
    assert np.isfinite(float(p))


def test_penalty_uses_the_grid_extents():
    """Shrinking the grid changes which pixels are compared, hence the penalty."""
    x, y, z, w = _plummer_sphere(n=1000)
    args = (plummer_potential, (M_PLUMMER, A_PLUMMER))
    wide = float(compute_poisson_penalty(x, y, z, w, *args, grid_size=12,
                                         extent_x=15.0, extent_z=10.0))
    narrow = float(compute_poisson_penalty(x, y, z, w, *args, grid_size=12,
                                           extent_x=4.0, extent_z=3.0))
    assert np.isfinite(wide) and np.isfinite(narrow)
    assert not np.isclose(wide, narrow)
