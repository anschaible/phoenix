"""Tests for `phoenix.optimization.observables`.

The sampling/mapping stage uses the `FakeMapper` from `conftest.py` instead of the
trained surrogate; the binning and blurring functions are pure array code and are
tested directly on hand-built particle sets, where the expected result can be
written down analytically.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phoenix.distribution_functions.spheroid import f_double_power_law
from phoenix.optimization.observables import (
    apply_mass_to_light,
    spheroid_df_wrapper,
    sample_and_map_particles,
    project_to_sky,
    bin_maps,
    render_maps_batched,
    blur_maps,
    generate_edge_on_maps,
)

MAP_KEYS = ('mass', 'v_rot', 'sigma', 'x_edges', 'z_edges')


def _v_phi(x, y, vx, vy):
    """Azimuthal velocity, reconstructed the same way `sample_and_map_particles` does."""
    R = np.maximum(np.sqrt(x**2 + y**2), 0.05)
    return (x * vy - y * vx) / R


# ==============================================================================
# spheroid_df_wrapper
# ==============================================================================
def test_spheroid_df_wrapper_matches_double_power_law(bulge_df_params):
    """The wrapper only exists to add the (unused) potential arguments."""
    Jr, Jz, Jphi = 10.0, 20.0, 300.0
    got = spheroid_df_wrapper(Jr, Jz, Jphi, None, (), bulge_df_params)
    expected = f_double_power_law(Jr, Jz, Jphi, bulge_df_params)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


def test_spheroid_df_wrapper_ignores_potential_and_theta(bulge_df_params):
    """The spheroid DF depends on the actions alone, so Phi/theta must not matter."""
    a = spheroid_df_wrapper(5.0, 5.0, 50.0, None, (), bulge_df_params)
    b = spheroid_df_wrapper(5.0, 5.0, 50.0, lambda x, y, z: 1.0, (1.0, 2.0), bulge_df_params)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))


# ==============================================================================
# sample_and_map_particles
# ==============================================================================
def test_sample_and_map_particles_shapes_and_finiteness(fake_mapper, all_params):
    pot, disk, bulge = all_params
    n_disk, n_bulge = 64, 32
    out = sample_and_map_particles(fake_mapper, pot, disk, bulge,
                                   N_disk=n_disk, N_bulge=n_bulge, prng_seed=0)
    assert len(out) == 7
    for arr in out:
        assert arr.shape == (n_disk + n_bulge,)
        assert np.all(np.isfinite(np.asarray(arr)))


def test_sample_and_map_particles_weights_sum_to_baryonic_mass(fake_mapper, all_params):
    """Weights are renormalized so the disk carries M_disk and the bulge M_bulge."""
    pot, disk, bulge = all_params
    n_disk, n_bulge = 64, 32
    *_, w = sample_and_map_particles(fake_mapper, pot, disk, bulge,
                                     N_disk=n_disk, N_bulge=n_bulge, prng_seed=1)
    w = np.asarray(w)
    assert np.all(w >= 0.0)
    np.testing.assert_allclose(w[:n_disk].sum(), pot['M_disk'], rtol=1e-5)
    np.testing.assert_allclose(w[n_disk:].sum(), pot['M_bulge'], rtol=1e-5)
    np.testing.assert_allclose(w.sum(), pot['M_disk'] + pot['M_bulge'], rtol=1e-5)


def test_sample_and_map_particles_is_deterministic_in_the_seed(fake_mapper, all_params):
    """Gradient descent needs the sampling to be a deterministic function of the seed."""
    pot, disk, bulge = all_params
    kw = dict(N_disk=32, N_bulge=16)
    a = sample_and_map_particles(fake_mapper, pot, disk, bulge, prng_seed=7, **kw)
    b = sample_and_map_particles(fake_mapper, pot, disk, bulge, prng_seed=7, **kw)
    c = sample_and_map_particles(fake_mapper, pot, disk, bulge, prng_seed=8, **kw)
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(np.asarray(arr_a), np.asarray(arr_b))
    assert not np.allclose(np.asarray(a[0]), np.asarray(c[0]))


def test_spheroid_corotation_flips_only_the_bulge_azimuthal_velocity(fake_mapper, all_params):
    """corotation=1 makes every bulge orbit prograde, 0 makes every one retrograde.

    Positions, vertical velocities and the whole disk population must be untouched.
    """
    pot, disk, bulge = all_params
    n_disk, n_bulge = 48, 32
    kw = dict(N_disk=n_disk, N_bulge=n_bulge, prng_seed=3)
    pro = [np.asarray(a) for a in sample_and_map_particles(
        fake_mapper, pot, disk, bulge, spheroid_corotation=1.0, **kw)]
    retro = [np.asarray(a) for a in sample_and_map_particles(
        fake_mapper, pot, disk, bulge, spheroid_corotation=0.0, **kw)]

    # positions, vz and weights are unaffected by the flip
    for i in (0, 1, 2, 5, 6):
        np.testing.assert_array_equal(pro[i], retro[i])

    vphi_pro = _v_phi(pro[0], pro[1], pro[3], pro[4])
    vphi_retro = _v_phi(retro[0], retro[1], retro[3], retro[4])

    np.testing.assert_allclose(vphi_pro[:n_disk], vphi_retro[:n_disk], rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(vphi_pro[n_disk:], -vphi_retro[n_disk:], rtol=1e-4, atol=1e-3)


def test_spheroid_corotation_half_gives_a_mix_of_both_signs(fake_mapper, all_params):
    """The default (0.5) must produce a pressure-supported, non-rotating spheroid."""
    pot, disk, bulge = all_params
    n_disk, n_bulge = 32, 200
    out = [np.asarray(a) for a in sample_and_map_particles(
        fake_mapper, pot, disk, bulge, N_disk=n_disk, N_bulge=n_bulge,
        prng_seed=5, spheroid_corotation=0.5)]
    vphi_bulge = _v_phi(*[a[n_disk:] for a in (out[0], out[1], out[3], out[4])])
    assert np.any(vphi_bulge > 0) and np.any(vphi_bulge < 0)


def test_sample_and_map_particles_is_differentiable(fake_mapper, all_params):
    """The whole sample->map chain must carry gradients into the parameters."""
    pot, disk, bulge = all_params

    def scalar(m_disk, rd):
        p = dict(pot, M_disk=m_disk)
        d = dict(disk, Rd=rd)
        x, y, z, vx, vy, vz, w = sample_and_map_particles(
            fake_mapper, p, d, bulge, N_disk=32, N_bulge=16, prng_seed=0)
        return jnp.sum(w * (x**2 + z**2 + vy**2))

    g_mass, g_rd = jax.grad(scalar, argnums=(0, 1))(pot['M_disk'], disk['Rd'])
    assert np.isfinite(float(g_mass)) and np.isfinite(float(g_rd))
    assert float(g_mass) != 0.0
    assert float(g_rd) != 0.0


# ==============================================================================
# bin_maps
# ==============================================================================
def test_bin_maps_shapes_and_edges():
    x = jnp.array([0.0, 1.0])
    z = jnp.array([0.0, -1.0])
    vy = jnp.array([10.0, -10.0])
    w = jnp.array([1.0, 1.0])
    maps = bin_maps(x, z, vy, w, grid_size=12, extent_x=15.0, extent_z=10.0)

    assert set(maps) == set(MAP_KEYS)
    for k in ('mass', 'v_rot', 'sigma'):
        assert maps[k].shape == (12, 12)
    np.testing.assert_allclose(np.asarray(maps['x_edges']),
                               np.linspace(-15.0, 15.0, 13), atol=1e-5)
    np.testing.assert_allclose(np.asarray(maps['z_edges']),
                               np.linspace(-10.0, 10.0, 13), atol=1e-5)


def test_bin_maps_conserves_total_mass():
    """The kernel is normalized by the pixel area, so the map integrates to sum(w)."""
    rng = np.random.default_rng(0)
    n = 200
    x = jnp.asarray(rng.normal(0.0, 3.0, n))
    z = jnp.asarray(rng.normal(0.0, 1.0, n))
    vy = jnp.asarray(rng.normal(0.0, 50.0, n))
    w = jnp.asarray(rng.uniform(0.5, 1.5, n))
    maps = bin_maps(x, z, vy, w, grid_size=32, soft_bin_h=1.0)
    np.testing.assert_allclose(float(jnp.sum(maps['mass'])), float(jnp.sum(w)), rtol=1e-3)


def test_bin_maps_single_particle_moments():
    """One particle: the peak sits on its pixel, v_rot is its velocity, sigma ~ 0."""
    x, z, v = 2.0, 1.0, 120.0
    maps = bin_maps(jnp.array([x]), jnp.array([z]), jnp.array([v]), jnp.array([3.0]),
                    grid_size=20, soft_bin_h=1.0)
    mass = np.asarray(maps['mass'])
    iz, ix = np.unravel_index(np.argmax(mass), mass.shape)

    # the maximum must land on the pixel whose center is closest to the particle
    x_centers = 0.5 * (np.asarray(maps['x_edges'])[:-1] + np.asarray(maps['x_edges'])[1:])
    z_centers = 0.5 * (np.asarray(maps['z_edges'])[:-1] + np.asarray(maps['z_edges'])[1:])
    assert ix == np.argmin(np.abs(x_centers - x))
    assert iz == np.argmin(np.abs(z_centers - z))

    np.testing.assert_allclose(float(maps['v_rot'][iz, ix]), v, rtol=1e-4)
    assert float(maps['sigma'][iz, ix]) < 1e-3


def test_bin_maps_dispersion_of_two_opposite_velocities():
    """Two co-located particles at +-v give v_rot = 0 and sigma = |v|."""
    v = 80.0
    maps = bin_maps(jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]),
                    jnp.array([v, -v]), jnp.array([1.0, 1.0]),
                    grid_size=21, soft_bin_h=1.0)
    mass = np.asarray(maps['mass'])
    iz, ix = np.unravel_index(np.argmax(mass), mass.shape)
    assert abs(float(maps['v_rot'][iz, ix])) < 1e-2
    np.testing.assert_allclose(float(maps['sigma'][iz, ix]), v, rtol=1e-4)


def test_bin_maps_kinematics_are_zeroed_in_empty_pixels():
    """Pixels below the mass threshold must not carry spurious kinematics."""
    maps = bin_maps(jnp.array([0.0]), jnp.array([0.0]), jnp.array([100.0]),
                    jnp.array([1.0]), grid_size=20, soft_bin_h=0.2)
    mass = np.asarray(maps['mass'])
    empty = mass <= 1e-5
    assert empty.any()
    np.testing.assert_array_equal(np.asarray(maps['v_rot'])[empty], 0.0)
    np.testing.assert_allclose(np.asarray(maps['sigma'])[empty], 1e-6, rtol=1e-6)


def test_bin_maps_default_bandwidth_is_quarter_pixel():
    """soft_bin_h=None must equal 0.25 * the larger pixel dimension."""
    grid_size, extent_x, extent_z = 20, 15.0, 10.0
    expected_h = 0.25 * max(2 * extent_x / grid_size, 2 * extent_z / grid_size)
    args = (jnp.array([1.0, -2.0]), jnp.array([0.5, -0.5]),
            jnp.array([30.0, -30.0]), jnp.array([1.0, 2.0]))
    kw = dict(grid_size=grid_size, extent_x=extent_x, extent_z=extent_z)
    auto = bin_maps(*args, soft_bin_h=None, **kw)
    explicit = bin_maps(*args, soft_bin_h=expected_h, **kw)
    for k in ('mass', 'v_rot', 'sigma'):
        np.testing.assert_allclose(np.asarray(auto[k]), np.asarray(explicit[k]))


def test_bin_maps_larger_bandwidth_smooths_the_map():
    """A wider kernel lowers the peak and spreads the same total mass further."""
    args = (jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([1.0]))
    sharp = bin_maps(*args, grid_size=32, soft_bin_h=0.8)
    broad = bin_maps(*args, grid_size=32, soft_bin_h=2.5)
    assert float(jnp.max(broad['mass'])) < float(jnp.max(sharp['mass']))
    np.testing.assert_allclose(float(jnp.sum(broad['mass'])),
                               float(jnp.sum(sharp['mass'])), rtol=1e-2)


def test_bin_maps_is_differentiable_in_positions_and_weights():
    def scalar(x, w):
        maps = bin_maps(x, jnp.zeros_like(x), jnp.array([10.0, -20.0]), w,
                        grid_size=12, soft_bin_h=1.0)
        return jnp.sum(maps['mass'] ** 2) + jnp.sum(maps['v_rot'] ** 2)

    gx, gw = jax.grad(scalar, argnums=(0, 1))(jnp.array([1.0, -1.0]), jnp.array([1.0, 2.0]))
    assert np.all(np.isfinite(np.asarray(gx)))
    assert np.all(np.isfinite(np.asarray(gw)))
    assert np.any(np.asarray(gw) != 0.0)


# ==============================================================================
# generate_edge_on_maps
# ==============================================================================
def test_generate_edge_on_maps_equals_sample_then_bin(fake_mapper, all_params):
    """The end-to-end helper must be exactly the composition of its three steps:
    sample -> project onto the sky -> bin."""
    pot, disk, bulge = all_params
    kw = dict(N_disk=48, N_bulge=32, prng_seed=11)
    maps = generate_edge_on_maps(fake_mapper, pot, disk, bulge,
                                 grid_size=10, extent_x=15.0, extent_z=10.0, **kw)
    x, y, z, vx, vy, vz, w = sample_and_map_particles(fake_mapper, pot, disk, bulge, **kw)
    x_sky, z_sky, v_los = project_to_sky(x, y, z, vx, vy, vz, 90.0)
    expected = bin_maps(x_sky, z_sky, v_los, w, grid_size=10, extent_x=15.0, extent_z=10.0)

    assert set(maps) == set(MAP_KEYS)
    for k in MAP_KEYS:
        np.testing.assert_array_equal(np.asarray(maps[k]), np.asarray(expected[k]))


def test_generate_edge_on_maps_default_matches_bin_on_raw_coords(fake_mapper, all_params):
    """The default (edge-on) path must still reproduce binning the raw (x, z, vy)
    directly. `project_to_sky` at i = 90 is the identity up to float32 round-off in
    cos(pi/2) (~4e-8), so this is a tight tolerance rather than exact equality."""
    pot, disk, bulge = all_params
    kw = dict(N_disk=48, N_bulge=32, prng_seed=11)
    maps = generate_edge_on_maps(fake_mapper, pot, disk, bulge,
                                 grid_size=10, extent_x=15.0, extent_z=10.0, **kw)
    x, y, z, vx, vy, vz, w = sample_and_map_particles(fake_mapper, pot, disk, bulge, **kw)
    raw = bin_maps(x, z, vy, w, grid_size=10, extent_x=15.0, extent_z=10.0)
    for k in ('mass', 'v_rot', 'sigma'):
        np.testing.assert_allclose(np.asarray(maps[k]), np.asarray(raw[k]),
                                   rtol=1e-4, atol=1e-2)


# ==============================================================================
# project_to_sky
# ==============================================================================
def test_project_to_sky_edge_on_is_identity():
    """At i = 90 the projection returns (x, z, vy) — the convention the rest of the
    module is written against."""
    x, y, z = jnp.array([1.0, -2.0]), jnp.array([3.0, 4.0]), jnp.array([-5.0, 6.0])
    vx, vy, vz = jnp.array([7.0, 8.0]), jnp.array([9.0, -1.0]), jnp.array([2.0, 3.0])
    xs, zs, vl = project_to_sky(x, y, z, vx, vy, vz, 90.0)
    np.testing.assert_allclose(np.asarray(xs), np.asarray(x), atol=1e-6)
    np.testing.assert_allclose(np.asarray(zs), np.asarray(z), atol=1e-5)
    np.testing.assert_allclose(np.asarray(vl), np.asarray(vy), atol=1e-5)


def test_project_to_sky_face_on_kills_rotation():
    """At i = 0 the line of sight is the symmetry axis: v_los = vz, and the sky
    plane is the disk plane, so in-plane rotation contributes nothing."""
    x, y, z = jnp.array([1.0, -2.0]), jnp.array([3.0, 4.0]), jnp.array([-5.0, 6.0])
    vx, vy, vz = jnp.array([7.0, 8.0]), jnp.array([9.0, -1.0]), jnp.array([2.0, 3.0])
    xs, zs, vl = project_to_sky(x, y, z, vx, vy, vz, 0.0)
    np.testing.assert_allclose(np.asarray(vl), np.asarray(vz), atol=1e-6)
    np.testing.assert_allclose(np.asarray(zs), -np.asarray(y), atol=1e-6)


def test_project_to_sky_los_amplitude_scales_with_sin_i():
    """A purely in-plane (vz = 0) motion projects as vy sin i — the degeneracy that
    lets a kinematic-only fit trade inclination against mass."""
    zero = jnp.zeros(3)
    vy = jnp.array([100.0, -50.0, 220.0])
    for inc in (90.0, 70.0, 45.0, 30.0):
        _, _, vl = project_to_sky(zero, zero, zero, zero, vy, zero, inc)
        np.testing.assert_allclose(np.asarray(vl),
                                   np.asarray(vy) * np.sin(np.deg2rad(inc)), rtol=1e-5)


def test_generate_edge_on_maps_projects_edge_on(fake_mapper, all_params):
    """The line-of-sight velocity is vy, so the map must be antisymmetric in x
    for a rotating disk: opposite sides of the galaxy recede and approach."""
    pot, disk, bulge = all_params
    maps = generate_edge_on_maps(fake_mapper, pot, disk, bulge, N_disk=2000,
                                 N_bulge=10, grid_size=16, soft_bin_h=1.5,
                                 prng_seed=0, spheroid_corotation=1.0)
    v_rot = np.asarray(maps['v_rot'])
    mass = np.asarray(maps['mass'])
    bright = mass > 1e-3 * mass.max()
    left = v_rot[:, :8][bright[:, :8]]
    right = v_rot[:, 8:][bright[:, 8:]]
    assert left.mean() * right.mean() < 0.0


def test_generate_edge_on_maps_all_values_finite(fake_mapper, all_params):
    pot, disk, bulge = all_params
    maps = generate_edge_on_maps(fake_mapper, pot, disk, bulge,
                                 N_disk=64, N_bulge=64, grid_size=12)
    for k in MAP_KEYS:
        assert np.all(np.isfinite(np.asarray(maps[k]))), k


# ==============================================================================
# render_maps_batched
# ==============================================================================
@pytest.mark.parametrize("chunk", [7, 31, 1000])
def test_render_maps_batched_matches_the_unchunked_render(fake_mapper, all_params, chunk):
    """Chunking only changes the order of a sum, so the maps must agree."""
    pot, disk, bulge = all_params
    kw = dict(N_disk=48, N_bulge=32, grid_size=12, prng_seed=4, soft_bin_h=1.2)
    expected = generate_edge_on_maps(fake_mapper, pot, disk, bulge, **kw)
    got = render_maps_batched(fake_mapper, pot, disk, bulge, chunk=chunk, **kw)

    assert set(got) == set(MAP_KEYS)
    np.testing.assert_allclose(np.asarray(got['mass']), np.asarray(expected['mass']),
                               rtol=1e-4)
    for k in ('v_rot', 'sigma'):
        scale = np.abs(np.asarray(expected[k])).max()
        np.testing.assert_allclose(np.asarray(got[k]), np.asarray(expected[k]),
                                   atol=1e-3 * max(scale, 1.0), rtol=1e-3)
    for k in ('x_edges', 'z_edges'):
        np.testing.assert_allclose(np.asarray(got[k]), np.asarray(expected[k]))


def test_render_maps_batched_handles_a_ragged_final_chunk(fake_mapper, all_params):
    """N is not a multiple of `chunk`: every particle must still be binned once."""
    pot, disk, bulge = all_params
    maps = render_maps_batched(fake_mapper, pot, disk, bulge, N_disk=50, N_bulge=25,
                               grid_size=16, soft_bin_h=1.0, chunk=20, prng_seed=2)
    total = pot['M_disk'] + pot['M_bulge']
    np.testing.assert_allclose(float(jnp.sum(maps['mass'])), total, rtol=1e-2)


# ==============================================================================
# blur_maps
# ==============================================================================
def _blob_maps(grid_size=32, soft_bin_h=0.7, seed=0):
    rng = np.random.default_rng(seed)
    n = 300
    x = jnp.asarray(rng.normal(0.0, 3.0, n))
    z = jnp.asarray(rng.normal(0.0, 1.0, n))
    vy = jnp.asarray(rng.normal(50.0, 30.0, n))
    w = jnp.asarray(rng.uniform(0.5, 1.5, n))
    return bin_maps(x, z, vy, w, grid_size=grid_size, soft_bin_h=soft_bin_h)


def test_blur_maps_zero_width_is_the_identity():
    """blur_h <= 0 means 'do not sharpen the observation' -> maps unchanged."""
    maps = _blob_maps()
    out = blur_maps(maps, 0.0, grid_size=32)
    for k in ('mass', 'v_rot', 'sigma'):
        np.testing.assert_allclose(np.asarray(out[k]), np.asarray(maps[k]), atol=1e-5)


def test_blur_maps_composes_gaussian_bandwidths():
    """The documented exactness claim: a KDE at h_obs blurred by
    sqrt(h**2 - h_obs**2) equals the KDE at bandwidth h."""
    grid_size, h_obs, h = 32, 0.7, 1.5
    sharp = _blob_maps(grid_size=grid_size, soft_bin_h=h_obs)
    direct = _blob_maps(grid_size=grid_size, soft_bin_h=h)
    blurred = blur_maps(sharp, float(np.sqrt(h**2 - h_obs**2)), grid_size=grid_size)

    mass = np.asarray(direct['mass'])
    support = mass > 1e-5 * mass.max()
    for k, tol in (('mass', 1e-3), ('v_rot', 5e-3), ('sigma', 2e-2)):
        got, exp = np.asarray(blurred[k])[support], np.asarray(direct[k])[support]
        assert np.abs(got - exp).max() <= tol * max(np.abs(exp).max(), 1e-12), k


def test_blur_maps_leaves_a_uniform_field_unchanged():
    """The 1D kernels are row-normalized, so a constant map is a fixed point."""
    grid_size = 24
    maps = {
        'mass': jnp.ones((grid_size, grid_size)),
        'v_rot': jnp.full((grid_size, grid_size), 30.0),
        'sigma': jnp.full((grid_size, grid_size), 10.0),
    }
    out = blur_maps(maps, 1.5, grid_size=grid_size)
    np.testing.assert_allclose(np.asarray(out['mass']), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out['v_rot']), 30.0, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out['sigma']), 10.0, rtol=1e-4)


def test_blur_maps_conserves_mass_and_lowers_the_peak():
    maps = _blob_maps()
    out = blur_maps(maps, 2.0, grid_size=32)
    np.testing.assert_allclose(float(jnp.sum(out['mass'])),
                               float(jnp.sum(maps['mass'])), rtol=1e-2)
    assert float(jnp.max(out['mass'])) < float(jnp.max(maps['mass']))


def test_blur_maps_passes_through_extra_keys_and_does_not_mutate_input():
    maps = _blob_maps()
    original_mass = np.array(maps['mass'])
    out = blur_maps(maps, 1.0, grid_size=32)
    assert set(out) == set(maps)
    np.testing.assert_allclose(np.asarray(out['x_edges']), np.asarray(maps['x_edges']))
    np.testing.assert_allclose(np.asarray(out['z_edges']), np.asarray(maps['z_edges']))
    np.testing.assert_array_equal(np.asarray(maps['mass']), original_mass)


def test_blur_maps_is_differentiable_in_the_width():
    maps = _blob_maps(grid_size=16)

    def scalar(blur_h):
        return jnp.sum(blur_maps(maps, blur_h, grid_size=16)['mass'] ** 2)

    g = jax.grad(scalar)(1.0)
    assert np.isfinite(float(g))


# ==============================================================================
# MASS-TO-LIGHT CONVERSION
# ==============================================================================
def test_apply_mass_to_light_none_is_the_identity():
    w = jnp.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(np.asarray(apply_mass_to_light(w, 2, None)), np.asarray(w))


def test_apply_mass_to_light_divides_each_component_by_its_own_ratio():
    w = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = np.asarray(apply_mass_to_light(w, 2, (2.0, 5.0)))
    np.testing.assert_allclose(out, [0.5, 1.0, 0.6, 0.8, 1.0], rtol=1e-6)


def test_apply_mass_to_light_is_differentiable_in_the_ratios():
    w = jnp.arange(1.0, 7.0)

    def scalar(ml_disk):
        return jnp.sum(apply_mass_to_light(w, 3, (ml_disk, 4.0)) ** 2)

    g = float(jax.grad(scalar)(2.0))
    assert np.isfinite(g) and g != 0.0


def test_constant_ml_only_rescales_the_first_map(fake_mapper, all_params):
    """A spatially constant Upsilon carries no information: the first map is the mass
    map divided by it, and the kinematics -- being ratios of weighted sums -- are
    untouched. This is why fitting light instead of mass is only a real change when
    the two components have different Upsilon.

    The kinematics are compared only where the weight map clears `bin_maps`' hard
    1e-5 cutoff by a wide margin. That cutoff is ABSOLUTE, so a pixel sitting just
    above it in mass can fall just below it once the weights are divided by Upsilon,
    and its velocities are then zeroed in one map but not the other. Such pixels are
    orders of magnitude below any realistic detection floor and never enter a fit,
    but they do break an unguarded elementwise comparison.
    """
    pot, disk, bulge = all_params
    cfg = dict(N_disk=200, N_bulge=100, grid_size=8, prng_seed=3)
    ml = 2.5
    mass = generate_edge_on_maps(fake_mapper, pot, disk, bulge, **cfg)
    light = generate_edge_on_maps(fake_mapper, pot, disk, bulge, ml_ratios=(ml, ml), **cfg)

    np.testing.assert_allclose(np.asarray(light['mass']) * ml,
                               np.asarray(mass['mass']), rtol=1e-4)

    well_above = np.asarray(light['mass']) > 1e-3
    assert well_above.sum() > 10          # the comparison is not vacuous
    for key in ('v_rot', 'sigma'):
        scale = max(float(np.abs(np.asarray(mass[key])).max()), 1.0)
        np.testing.assert_allclose(np.asarray(light[key])[well_above],
                                   np.asarray(mass[key])[well_above],
                                   rtol=1e-3, atol=1e-3 * scale)


def test_component_ml_changes_the_kinematics(fake_mapper, all_params):
    """Two different Upsilon values re-weight disk against bulge, so the light-weighted
    velocity moments genuinely differ from the mass-weighted ones."""
    pot, disk, bulge = all_params
    cfg = dict(N_disk=200, N_bulge=100, grid_size=8, prng_seed=3)
    mass = generate_edge_on_maps(fake_mapper, pot, disk, bulge, **cfg)
    light = generate_edge_on_maps(fake_mapper, pot, disk, bulge, ml_ratios=(1.5, 4.5), **cfg)

    assert not np.allclose(np.asarray(light['sigma']), np.asarray(mass['sigma']),
                           rtol=1e-3, atol=1e-2)


def test_render_maps_batched_honours_ml_ratios(fake_mapper, all_params):
    """The chunked renderer must apply the conversion the same way the single-shot
    path does, or high-tracer renders would not be comparable to the fitted maps."""
    pot, disk, bulge = all_params
    cfg = dict(N_disk=200, N_bulge=100, grid_size=8, prng_seed=3)
    ref = generate_edge_on_maps(fake_mapper, pot, disk, bulge, ml_ratios=(1.5, 4.5), **cfg)
    got = render_maps_batched(fake_mapper, pot, disk, bulge, ml_ratios=(1.5, 4.5),
                              chunk=64, **cfg)
    np.testing.assert_allclose(np.asarray(got['mass']), np.asarray(ref['mass']), rtol=1e-4)
    for key in ('v_rot', 'sigma'):
        scale = max(float(np.abs(np.asarray(ref[key])).max()), 1.0)
        np.testing.assert_allclose(np.asarray(got[key]), np.asarray(ref[key]),
                                   rtol=1e-3, atol=1e-3 * scale)
