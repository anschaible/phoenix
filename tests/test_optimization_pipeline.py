"""Tests for `phoenix.optimization.pipeline`.

The optimizer is exercised with a handful of steps and a few hundred tracers, so
the tests check *properties* (invariants, structure, bookkeeping, monotone loss
decrease) rather than convergence to a physical answer. The trained surrogate is
replaced by the analytic `FakeMapper` from `conftest.py`.

A key trick used repeatedly: `make_loss_fn` samples with a fixed seed, so if the
"observation" is generated with the very same seed/tracer count/grid, then the
data-fit terms are *exactly* zero at the truth. That makes several otherwise
statistical statements into exact ones.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from phoenix.optimization.observables import generate_edge_on_maps
from phoenix.optimization.pipeline import (
    DEFAULT_PARAM_BOUNDS,
    baryonic_potential_raw,
    data_fit_loss,
    fit,
    fit_multistart,
    log_bounds_tree,
    log_to_params,
    make_loss_fn,
    make_observation,
    make_self_consistent_truth,
    params_to_log,
    pot_params_to_tuple,
    total_potential_raw,
    _scatter_params,
)
from phoenix.potentials.potentials import (
    miyamoto_nagai_potential,
    nfw_potential,
    plummer_potential,
)

# Small, fast configuration shared by the loss/fit tests.
CFG = dict(N_disk=200, N_bulge=100, grid_size=8, prng_seed=5)
POISSON_KW = dict(grid_size=8)


@pytest.fixture
def obs_maps(fake_mapper, all_params):
    """A mock observation generated with exactly the configuration the loss uses."""
    pot, disk, bulge = all_params
    return make_observation(fake_mapper, pot, disk, bulge, **CFG)


# ==============================================================================
# POTENTIAL HELPERS
# ==============================================================================
def test_total_potential_raw_is_the_sum_of_its_components(pot_params):
    args = pot_params_to_tuple(pot_params)
    point = (3.0, 1.0, 0.5)
    expected = (nfw_potential(*point, pot_params['M_halo'], pot_params['a_halo']) +
                miyamoto_nagai_potential(*point, pot_params['M_disk'],
                                         pot_params['a_disk'], pot_params['b_disk']) +
                plummer_potential(*point, pot_params['M_bulge'], pot_params['a_bulge']))
    np.testing.assert_allclose(float(total_potential_raw(*point, *args)),
                               float(expected), rtol=1e-6)


def test_baryonic_potential_raw_is_the_total_minus_the_halo(pot_params):
    args = pot_params_to_tuple(pot_params)
    point = (3.0, 1.0, 0.5)
    halo = float(nfw_potential(*point, pot_params['M_halo'], pot_params['a_halo']))
    np.testing.assert_allclose(float(baryonic_potential_raw(*point, *args)),
                               float(total_potential_raw(*point, *args)) - halo,
                               rtol=1e-5)


def test_baryonic_potential_raw_ignores_the_halo_parameters(pot_params):
    """The halo arguments exist only to keep the signature interchangeable."""
    point = (3.0, 1.0, 0.5)
    base = float(baryonic_potential_raw(*point, *pot_params_to_tuple(pot_params)))
    other = float(baryonic_potential_raw(
        *point, *pot_params_to_tuple(dict(pot_params, M_halo=5e13, a_halo=2.0))))
    assert base == other


def test_pot_params_to_tuple_orders_parameters_for_the_potential_signature(pot_params):
    got = pot_params_to_tuple(pot_params)
    assert got == (pot_params['M_halo'], pot_params['a_halo'],
                   pot_params['M_disk'], pot_params['a_disk'], pot_params['b_disk'],
                   pot_params['M_bulge'], pot_params['a_bulge'])
    # the tuple must be directly splattable into the raw potential functions
    assert np.isfinite(float(total_potential_raw(1.0, 0.0, 0.0, *got)))


# ==============================================================================
# LOG-SPACE PARAMETER TRANSFORM
# ==============================================================================
def test_params_to_log_takes_logs_and_preserves_structure(all_params):
    pot, disk, bulge = all_params
    out = params_to_log(pot, disk, bulge)
    assert set(out) == {'pot', 'disk', 'bulge'}
    for group, src in (('pot', pot), ('disk', disk), ('bulge', bulge)):
        assert set(out[group]) == set(src)
        for k, v in src.items():
            np.testing.assert_allclose(float(out[group][k]), np.log(v), rtol=1e-6)


def test_log_to_params_inverts_params_to_log(all_params):
    pot, disk, bulge = all_params
    got_pot, got_disk, got_bulge = log_to_params(params_to_log(pot, disk, bulge))
    for got, src in ((got_pot, pot), (got_disk, disk), (got_bulge, bulge)):
        assert set(got) == set(src)
        for k, v in src.items():
            np.testing.assert_allclose(float(got[k]), v, rtol=1e-5)


def test_log_to_params_always_returns_positive_parameters():
    """Positivity is the whole point of optimizing in log-space."""
    params_log = {
        'pot': {'M_halo': jnp.asarray(-30.0)},
        'disk': {'Rd': jnp.asarray(50.0)},
        'bulge': {'eta_spheroid': jnp.asarray(0.0)},
    }
    for group in log_to_params(params_log):
        for v in group.values():
            assert float(v) > 0.0


def test_log_bounds_tree_matches_the_default_bounds(all_params):
    pot, disk, bulge = all_params
    params_log = params_to_log(pot, disk, bulge)
    lo, hi = log_bounds_tree(params_log)

    assert set(lo) == set(hi) == set(params_log)
    for group in params_log:
        assert set(lo[group]) == set(params_log[group])
        for k in params_log[group]:
            lo_v, hi_v = DEFAULT_PARAM_BOUNDS[group][k]
            # `Gamma_spheroid` has a lower bound of 0, i.e. -inf in log-space
            expected_lo = -np.inf if lo_v == 0.0 else np.log(lo_v)
            np.testing.assert_allclose(float(lo[group][k]), expected_lo, rtol=1e-6)
            np.testing.assert_allclose(float(hi[group][k]), np.log(hi_v), rtol=1e-6)
            assert float(lo[group][k]) < float(hi[group][k])


def test_log_bounds_tree_accepts_custom_bounds():
    params_log = {'pot': {'M_disk': jnp.log(5e10)}, 'disk': {}, 'bulge': {}}
    bounds = {'pot': {'M_disk': (1e10, 1e11)}, 'disk': {}, 'bulge': {}}
    lo, hi = log_bounds_tree(params_log, bounds)
    np.testing.assert_allclose(float(lo['pot']['M_disk']), np.log(1e10), rtol=1e-6)
    np.testing.assert_allclose(float(hi['pot']['M_disk']), np.log(1e11), rtol=1e-6)


def test_log_bounds_tree_reports_an_unbounded_parameter():
    """A parameter with no entry in the bounds table is a configuration error."""
    params_log = {'pot': {'M_mystery': jnp.log(1.0)}}
    with pytest.raises(KeyError):
        log_bounds_tree(params_log, {'pot': {}})


def test_default_param_bounds_are_ordered_and_positive():
    for group, entries in DEFAULT_PARAM_BOUNDS.items():
        for k, (lo, hi) in entries.items():
            assert lo < hi, f'{group}.{k}'
            assert lo >= 0.0, f'{group}.{k}'


# ==============================================================================
# make_observation
# ==============================================================================
def test_make_observation_matches_generate_edge_on_maps(fake_mapper, all_params):
    pot, disk, bulge = all_params
    obs = make_observation(fake_mapper, pot, disk, bulge, **CFG)
    expected = generate_edge_on_maps(fake_mapper, pot, disk, bulge, **CFG)
    assert set(obs) == set(expected)
    for k in expected:
        assert isinstance(obs[k], jax.Array)
        np.testing.assert_array_equal(np.asarray(obs[k]), np.asarray(expected[k]))


def test_make_observation_has_the_requested_resolution(fake_mapper, all_params):
    pot, disk, bulge = all_params
    obs = make_observation(fake_mapper, pot, disk, bulge, N_disk=64, N_bulge=32,
                           grid_size=10)
    for k in ('mass', 'v_rot', 'sigma'):
        assert obs[k].shape == (10, 10)
    assert obs['x_edges'].shape == (11,)
    assert np.all(np.asarray(obs['mass']) >= 0.0)


# ==============================================================================
# data_fit_loss
# ==============================================================================
def _flat_maps(mass=1.0, v_rot=100.0, sigma=50.0, n=6):
    return {
        'mass': jnp.full((n, n), mass),
        'v_rot': jnp.full((n, n), v_rot),
        'sigma': jnp.full((n, n), sigma),
    }


def test_data_fit_loss_is_zero_for_identical_maps():
    maps = _flat_maps()
    losses = data_fit_loss(maps, maps)
    assert len(losses) == 3
    for term in losses:
        np.testing.assert_allclose(float(term), 0.0, atol=1e-12)


def test_data_fit_loss_normalizes_velocities_by_their_reference_scales():
    """v_rot is divided by 200 km/s and sigma by 100 km/s, so those offsets give 1."""
    obs = _flat_maps()
    model = _flat_maps(v_rot=100.0 + 200.0, sigma=50.0 + 100.0)
    _, vrot_loss, sigma_loss = data_fit_loss(model, obs)
    np.testing.assert_allclose(float(vrot_loss), 1.0, rtol=1e-5)
    np.testing.assert_allclose(float(sigma_loss), 1.0, rtol=1e-5)


def test_data_fit_loss_compares_mass_in_log_space():
    """A factor of 10 in mass is one dex, hence a squared residual of 1."""
    obs = _flat_maps(mass=10.0)
    model = _flat_maps(mass=100.0)
    mass_loss, _, _ = data_fit_loss(model, obs, mass_floor=1e-6)
    np.testing.assert_allclose(float(mass_loss), 1.0, rtol=1e-3)


def test_data_fit_loss_only_scores_pixels_above_the_mass_floor():
    """Half the observation is blank; only the detected half must contribute."""
    n = 6
    mass = np.zeros((n, n))
    mass[:, :3] = 1.0
    obs = {'mass': jnp.asarray(mass), 'v_rot': jnp.zeros((n, n)),
           'sigma': jnp.zeros((n, n))}
    model_vrot = np.zeros((n, n))
    model_vrot[:, :3] = 200.0    # residual 1 on the detected half
    model_vrot[:, 3:] = 4000.0   # gross error where nothing was observed
    model = {'mass': obs['mass'], 'v_rot': jnp.asarray(model_vrot),
             'sigma': jnp.zeros((n, n))}
    _, vrot_loss, _ = data_fit_loss(model, obs, mass_floor=1e-3)
    np.testing.assert_allclose(float(vrot_loss), 1.0, rtol=1e-5)


def test_data_fit_loss_honours_an_explicit_mask():
    n = 4
    obs = _flat_maps(n=n)
    model = _flat_maps(v_rot=100.0 + 200.0, n=n)
    mask = np.zeros((n, n), dtype=bool)
    mask[0, 0] = True
    _, vrot_loss, _ = data_fit_loss(model, obs, mask=jnp.asarray(mask))
    # one pixel with residual 1, averaged over the one masked pixel
    np.testing.assert_allclose(float(vrot_loss), 1.0, rtol=1e-5)


def test_data_fit_loss_is_finite_for_an_empty_mask():
    """The `max(sum(mask), 1)` guard must prevent a 0/0 NaN."""
    maps = _flat_maps()
    losses = data_fit_loss(maps, _flat_maps(v_rot=1e4),
                           mask=jnp.zeros((6, 6), dtype=bool))
    for term in losses:
        assert np.isfinite(float(term))
        np.testing.assert_allclose(float(term), 0.0, atol=1e-12)


def test_data_fit_loss_is_differentiable():
    obs = _flat_maps()

    def scalar(v):
        model = {'mass': obs['mass'], 'v_rot': jnp.full((6, 6), v),
                 'sigma': obs['sigma']}
        return sum(data_fit_loss(model, obs))

    g = jax.grad(scalar)(150.0)
    assert np.isfinite(float(g)) and float(g) > 0.0


# ==============================================================================
# make_loss_fn
# ==============================================================================
def test_loss_is_zero_at_the_truth_when_the_observation_shares_the_sampling(
        fake_mapper, all_params, obs_maps):
    """Same seed, tracer count and grid => the model maps *are* the observation."""
    pot, disk, bulge = all_params
    loss_fn = make_loss_fn(fake_mapper, obs_maps, loss_weights=(1.0, 1.0, 1.0, 0.0),
                           poisson_kwargs=POISSON_KW, **CFG)
    loss, aux = loss_fn(params_to_log(pot, disk, bulge))
    assert float(loss) < 1e-8
    for key in ('mass_loss', 'vrot_loss', 'sigma_loss'):
        assert float(aux[key]) < 1e-8


def test_loss_aux_contains_all_telemetry(fake_mapper, all_params, obs_maps):
    pot, disk, bulge = all_params
    loss_fn = make_loss_fn(fake_mapper, obs_maps, poisson_kwargs=POISSON_KW, **CFG)
    loss, aux = loss_fn(params_to_log(pot, disk, bulge))
    assert np.shape(loss) == ()
    assert set(aux) == {'mass_loss', 'vrot_loss', 'sigma_loss', 'poisson_penalty',
                        'reg', 'model_maps'}
    assert aux['model_maps']['mass'].shape == (CFG['grid_size'], CFG['grid_size'])
    assert float(aux['poisson_penalty']) > 0.0


def test_loss_combines_the_terms_with_the_given_weights(fake_mapper, all_params,
                                                        obs_maps):
    pot, disk, bulge = all_params
    weights = (2.0, 3.0, 4.0, 0.5)
    loss_fn = make_loss_fn(fake_mapper, obs_maps, loss_weights=weights,
                           poisson_kwargs=POISSON_KW, **CFG)
    params_log = params_to_log(dict(pot, M_disk=8e10), disk, bulge)
    loss, aux = loss_fn(params_log)
    expected = (weights[0] * float(aux['mass_loss']) +
                weights[1] * float(aux['vrot_loss']) +
                weights[2] * float(aux['sigma_loss']) +
                weights[3] * float(aux['poisson_penalty']))
    np.testing.assert_allclose(float(loss), expected, rtol=1e-5)


def test_loss_is_deterministic_and_grows_away_from_the_truth(fake_mapper, all_params,
                                                             obs_maps):
    pot, disk, bulge = all_params
    loss_fn = make_loss_fn(fake_mapper, obs_maps, loss_weights=(1.0, 1.0, 1.0, 0.0),
                           poisson_kwargs=POISSON_KW, **CFG)
    truth = params_to_log(pot, disk, bulge)
    wrong = params_to_log(dict(pot, M_disk=2.0 * pot['M_disk']), disk, bulge)

    l1 = float(loss_fn(truth)[0])
    l2 = float(loss_fn(truth)[0])
    assert l1 == l2
    assert float(loss_fn(wrong)[0]) > l1 + 1e-3


def test_loss_gradient_is_finite_and_informative(fake_mapper, all_params, obs_maps):
    pot, disk, bulge = all_params
    loss_fn = make_loss_fn(fake_mapper, obs_maps, poisson_kwargs=POISSON_KW, **CFG)
    params_log = params_to_log(dict(pot, M_disk=8e10), disk, bulge)
    grads = jax.grad(loss_fn, has_aux=True)(params_log)[0]
    flat, _ = ravel_pytree(grads)
    assert np.all(np.isfinite(np.asarray(flat)))
    assert float(jnp.max(jnp.abs(flat))) > 0.0


def test_loss_regularization_vanishes_at_its_own_center(fake_mapper, all_params,
                                                        obs_maps):
    pot, disk, bulge = all_params
    center = params_to_log(pot, disk, bulge)
    loss_fn = make_loss_fn(fake_mapper, obs_maps, reg_weight=1.0,
                           reg_center_log=center, poisson_kwargs=POISSON_KW, **CFG)
    _, aux = loss_fn(center)
    np.testing.assert_allclose(float(aux['reg']), 0.0, atol=1e-10)


def test_loss_regularization_equals_the_ridge_penalty(fake_mapper, all_params,
                                                      obs_maps):
    pot, disk, bulge = all_params
    center = params_to_log(pot, disk, bulge)
    shifted = params_to_log(dict(pot, M_disk=3.0 * pot['M_disk']), disk, bulge)
    reg_weight = 0.25
    loss_fn = make_loss_fn(fake_mapper, obs_maps, reg_weight=reg_weight,
                           reg_center_log=center, poisson_kwargs=POISSON_KW, **CFG)
    _, aux = loss_fn(shifted)
    u, _ = ravel_pytree(shifted)
    c, _ = ravel_pytree(center)
    expected = reg_weight * float(jnp.mean((u - c) ** 2))
    np.testing.assert_allclose(float(aux['reg']), expected, rtol=1e-4)


def test_loss_regularization_is_off_without_a_center(fake_mapper, all_params,
                                                     obs_maps):
    pot, disk, bulge = all_params
    loss_fn = make_loss_fn(fake_mapper, obs_maps, reg_weight=10.0,
                           reg_center_log=None, poisson_kwargs=POISSON_KW, **CFG)
    _, aux = loss_fn(params_to_log(dict(pot, M_disk=9e10), disk, bulge))
    assert float(aux['reg']) == 0.0


def test_obs_bandwidth_does_nothing_when_the_model_matches_the_observation(
        fake_mapper, all_params, obs_maps):
    """At soft_bin_h == obs_bandwidth the required blur is zero, so the blurred
    objective must coincide with the unblurred one."""
    pot, disk, bulge = all_params
    h_obs = 0.25 * max(2 * 15.0 / CFG['grid_size'], 2 * 10.0 / CFG['grid_size'])
    params_log = params_to_log(dict(pot, M_disk=7e10), disk, bulge)

    plain = make_loss_fn(fake_mapper, obs_maps, poisson_kwargs=POISSON_KW, **CFG)
    blurred = make_loss_fn(fake_mapper, obs_maps, obs_bandwidth=h_obs,
                           poisson_kwargs=POISSON_KW, **CFG)
    np.testing.assert_allclose(float(blurred(params_log, h_obs)[0]),
                               float(plain(params_log, h_obs)[0]), rtol=1e-5)


def test_obs_bandwidth_blurs_the_observation_at_a_wider_bandwidth(
        fake_mapper, all_params, obs_maps):
    """At the truth, a coarse bandwidth only matches if the data is blurred too."""
    pot, disk, bulge = all_params
    h_obs = 0.25 * max(2 * 15.0 / CFG['grid_size'], 2 * 10.0 / CFG['grid_size'])
    truth = params_to_log(pot, disk, bulge)
    coarse_h = 4.0 * h_obs
    kw = dict(loss_weights=(1.0, 1.0, 1.0, 0.0), poisson_kwargs=POISSON_KW, **CFG)

    uncorrected = make_loss_fn(fake_mapper, obs_maps, **kw)(truth, coarse_h)[0]
    corrected = make_loss_fn(fake_mapper, obs_maps, obs_bandwidth=h_obs,
                             **kw)(truth, coarse_h)[0]
    assert float(corrected) < float(uncorrected)


def test_loss_mass_floor_controls_the_data_mask(fake_mapper, all_params, obs_maps):
    """A floor above every observed pixel empties the mask, zeroing the data terms."""
    pot, disk, bulge = all_params
    huge = float(jnp.max(obs_maps['mass'])) * 10.0
    loss_fn = make_loss_fn(fake_mapper, obs_maps, mass_floor=huge,
                           loss_weights=(1.0, 1.0, 1.0, 0.0),
                           poisson_kwargs=POISSON_KW, **CFG)
    _, aux = loss_fn(params_to_log(dict(pot, M_disk=2e11), disk, bulge))
    for key in ('mass_loss', 'vrot_loss', 'sigma_loss'):
        np.testing.assert_allclose(float(aux[key]), 0.0, atol=1e-12)


# ==============================================================================
# fit
# ==============================================================================
def _fit_kwargs(**overrides):
    kw = dict(n_steps=4, poisson_kwargs=POISSON_KW, **CFG)
    kw.update(overrides)
    return kw


def test_fit_returns_the_documented_schema(fake_mapper, all_params, obs_maps):
    pot, disk, bulge = all_params
    res = fit(fake_mapper, obs_maps, pot, disk, bulge, **_fit_kwargs())

    assert set(res) == {'pot_params', 'disk_df_params', 'bulge_df_params', 'history',
                        'nan_steps', 'nan_grads_total'}
    assert set(res['pot_params']) == set(pot)
    assert set(res['disk_df_params']) == set(disk)
    assert set(res['bulge_df_params']) == set(bulge)
    for group in ('pot_params', 'disk_df_params', 'bulge_df_params'):
        for k, v in res[group].items():
            assert np.isfinite(float(v)), f'{group}.{k}'
            assert float(v) > 0.0, f'{group}.{k}'

    history = res['history']
    assert set(history) == {'loss', 'mass_loss', 'vrot_loss', 'sigma_loss',
                            'poisson_penalty', 'reg', 'pot_params',
                            'disk_df_params', 'bulge_df_params'}
    for key, values in history.items():
        assert len(values) == 4, key
    assert all(np.isfinite(v) for v in history['loss'])
    assert history['pot_params'][0].keys() == pot.keys()


def test_fit_decreases_the_loss_from_a_perturbed_start(fake_mapper, all_params,
                                                       obs_maps):
    """Pure data fit (no physics term) from a displaced disk mass must improve."""
    pot, disk, bulge = all_params
    start = dict(pot, M_disk=1.6 * pot['M_disk'])
    res = fit(fake_mapper, obs_maps, start, disk, bulge,
              **_fit_kwargs(n_steps=15, learning_rate=0.05,
                            loss_weights=(1.0, 1.0, 1.0, 0.0)))
    loss = res['history']['loss']
    assert min(loss) < loss[0]
    assert loss[-1] < loss[0]


def test_fit_holds_frozen_parameters_fixed(fake_mapper, all_params, obs_maps):
    """Frozen parameters have their gradient zeroed, so they must not move."""
    pot, disk, bulge = all_params
    frozen = ('Sigma0', 'N0_spheroid', 'L0', 'M_halo')
    res = fit(fake_mapper, obs_maps, pot, disk, bulge,
              **_fit_kwargs(n_steps=6, frozen_params=frozen))

    np.testing.assert_allclose(float(res['pot_params']['M_halo']), pot['M_halo'],
                               rtol=1e-5)
    np.testing.assert_allclose(float(res['disk_df_params']['Sigma0']), disk['Sigma0'],
                               rtol=1e-5)
    np.testing.assert_allclose(float(res['disk_df_params']['L0']), disk['L0'],
                               rtol=1e-5)
    np.testing.assert_allclose(float(res['bulge_df_params']['N0_spheroid']),
                               bulge['N0_spheroid'], rtol=1e-5)
    # something must still have moved, otherwise the test proves nothing
    assert float(res['pot_params']['M_disk']) != pot['M_disk']


def test_fit_respects_custom_parameter_bounds(fake_mapper, all_params, obs_maps):
    """Parameters are clipped in log-space after every update."""
    pot, disk, bulge = all_params
    bounds = {group: {k: (0.999 * v, 1.001 * v) for k, v in src.items()}
              for group, src in (('pot', pot), ('disk', disk), ('bulge', bulge))}
    res = fit(fake_mapper, obs_maps, pot, disk, bulge,
              **_fit_kwargs(n_steps=6, learning_rate=0.5, param_bounds=bounds))
    for group, key in (('pot_params', 'pot'), ('disk_df_params', 'disk'),
                       ('bulge_df_params', 'bulge')):
        for k, v in res[group].items():
            lo, hi = bounds[key][k]
            assert lo * (1 - 1e-4) <= float(v) <= hi * (1 + 1e-4), f'{group}.{k}'


def test_fit_runs_with_bandwidth_annealing(fake_mapper, all_params, obs_maps):
    pot, disk, bulge = all_params
    res = fit(fake_mapper, obs_maps, pot, disk, bulge,
              **_fit_kwargs(n_steps=5, anneal_bandwidth=(4.0, 0.6)))
    assert len(res['history']['loss']) == 5
    assert all(np.isfinite(v) for v in res['history']['loss'])


def test_fit_runs_with_annealing_and_a_blurred_observation(fake_mapper, all_params,
                                                           obs_maps):
    """The hold-then-jump schedule (anneal_hold_frac) path."""
    pot, disk, bulge = all_params
    h_obs = 0.25 * max(2 * 15.0 / CFG['grid_size'], 2 * 10.0 / CFG['grid_size'])
    res = fit(fake_mapper, obs_maps, pot, disk, bulge,
              **_fit_kwargs(n_steps=6, anneal_bandwidth=(5.0, h_obs),
                            obs_bandwidth=h_obs, anneal_hold_frac=0.5))
    assert all(np.isfinite(v) for v in res['history']['loss'])


def test_fit_records_a_regularization_term(fake_mapper, all_params, obs_maps):
    """With no explicit center the prior is anchored at the initial guess, so the
    first recorded `reg` (evaluated before any update) is zero."""
    pot, disk, bulge = all_params
    res = fit(fake_mapper, obs_maps, pot, disk, bulge,
              **_fit_kwargs(n_steps=4, reg_weight=1.0))
    reg = res['history']['reg']
    np.testing.assert_allclose(reg[0], 0.0, atol=1e-10)
    assert all(v >= 0.0 for v in reg)


def test_fit_without_regularization_reports_zero_reg(fake_mapper, all_params,
                                                     obs_maps):
    pot, disk, bulge = all_params
    res = fit(fake_mapper, obs_maps, pot, disk, bulge, **_fit_kwargs(n_steps=3))
    assert res['history']['reg'] == [0.0, 0.0, 0.0]


# ==============================================================================
# make_self_consistent_truth
# ==============================================================================
def test_make_self_consistent_truth_reduces_the_poisson_residual(fake_mapper,
                                                                 all_params):
    pot, disk, bulge = all_params
    res = make_self_consistent_truth(fake_mapper, pot, disk, bulge, N_disk=200,
                                     N_bulge=100, n_steps=10, prng_seed=0,
                                     poisson_kwargs=POISSON_KW)

    assert set(res) == {'pot_params', 'disk_df_params', 'bulge_df_params',
                        'penalty_initial', 'penalty_final', 'history'}
    assert len(res['history']) == 10
    assert all(np.isfinite(v) for v in res['history'])
    assert res['penalty_final'] < res['penalty_initial']
    for group in ('pot_params', 'disk_df_params', 'bulge_df_params'):
        for v in res[group].values():
            assert isinstance(v, float) and np.isfinite(v) and v > 0.0


def test_make_self_consistent_truth_only_moves_the_tuned_parameters(fake_mapper,
                                                                    all_params):
    pot, disk, bulge = all_params
    tune = ('M_disk', 'a_disk')
    res = make_self_consistent_truth(fake_mapper, pot, disk, bulge,
                                     tune_params=tune, N_disk=200, N_bulge=100,
                                     n_steps=6, prng_seed=0,
                                     poisson_kwargs=POISSON_KW)

    fixed = {**{k: v for k, v in pot.items() if k not in tune}, **disk, **bulge}
    got = {**res['pot_params'], **res['disk_df_params'], **res['bulge_df_params']}
    for k, v in fixed.items():
        # only float32 log/exp round-trip error is allowed
        np.testing.assert_allclose(got[k], v, rtol=1e-5, err_msg=k)
    for k in tune:
        assert got[k] != pytest.approx(pot[k], rel=1e-5)


def test_make_self_consistent_truth_with_nothing_to_tune_is_a_no_op(fake_mapper,
                                                                    all_params):
    pot, disk, bulge = all_params
    res = make_self_consistent_truth(fake_mapper, pot, disk, bulge, tune_params=(),
                                     N_disk=100, N_bulge=50, n_steps=3, prng_seed=0,
                                     poisson_kwargs=POISSON_KW)
    np.testing.assert_allclose(res['penalty_final'], res['penalty_initial'], rtol=1e-5)
    for k, v in pot.items():
        np.testing.assert_allclose(res['pot_params'][k], v, rtol=1e-5, err_msg=k)


# ==============================================================================
# _scatter_params
# ==============================================================================
def test_scatter_params_keeps_keys_and_positivity(pot_params):
    rng = np.random.default_rng(0)
    out = _scatter_params(pot_params, 0.4, rng)
    assert set(out) == set(pot_params)
    assert all(isinstance(v, float) and v > 0.0 for v in out.values())
    assert any(out[k] != pot_params[k] for k in pot_params)


def test_scatter_params_with_zero_sigma_is_the_identity(pot_params):
    out = _scatter_params(pot_params, 0.0, np.random.default_rng(0))
    for k, v in pot_params.items():
        np.testing.assert_allclose(out[k], v, rtol=1e-12)


def test_scatter_params_is_reproducible_from_the_generator_seed(pot_params):
    a = _scatter_params(pot_params, 0.3, np.random.default_rng(7))
    b = _scatter_params(pot_params, 0.3, np.random.default_rng(7))
    c = _scatter_params(pot_params, 0.3, np.random.default_rng(8))
    assert a == b
    assert a != c


def test_scatter_params_is_multiplicative_in_log_space(pot_params):
    """The perturbation is lognormal, so log-ratios have the requested sigma
    regardless of a parameter's magnitude."""
    rng = np.random.default_rng(0)
    log_sigma = 0.4
    ratios = []
    for _ in range(200):
        out = _scatter_params(pot_params, log_sigma, rng)
        ratios += [np.log(out[k] / pot_params[k]) for k in pot_params]
    assert abs(np.std(ratios) - log_sigma) < 0.1 * log_sigma
    assert abs(np.mean(ratios)) < 0.1


# ==============================================================================
# fit_multistart
# ==============================================================================
def _fake_run(final_loss, tag):
    return {'pot_params': {'M_disk': tag}, 'disk_df_params': {}, 'bulge_df_params': {},
            'history': {'loss': [10.0, final_loss]}}


def test_fit_multistart_returns_the_run_with_the_lowest_final_loss(monkeypatch,
                                                                  all_params):
    """Selection logic in isolation: `fit` is stubbed with canned outcomes."""
    import phoenix.optimization.pipeline as pipeline

    losses = iter([1.0, 0.25, 3.0])
    calls = []

    def fake_fit(mapper, obs_maps, p, d, b, **kwargs):
        calls.append(p)
        loss = next(losses)
        return _fake_run(loss, tag=loss)

    monkeypatch.setattr(pipeline, 'fit', fake_fit)
    pot, disk, bulge = all_params
    res = pipeline.fit_multistart(None, {}, pot, disk, bulge, n_restarts=3)

    assert len(calls) == 3
    assert res['final_losses'] == [1.0, 0.25, 3.0]
    assert res['best_index'] == 1
    assert res['pot_params']['M_disk'] == 0.25
    assert len(res['all_runs']) == 3
    assert res['all_runs'][1]['pot_params']['M_disk'] == 0.25


def test_fit_multistart_never_selects_a_diverged_run(monkeypatch, all_params):
    """A NaN final loss must be treated as infinitely bad, not as a minimum."""
    import phoenix.optimization.pipeline as pipeline

    losses = iter([float('nan'), 2.0])

    def fake_fit(mapper, obs_maps, p, d, b, **kwargs):
        loss = next(losses)
        return _fake_run(loss, tag=loss)

    monkeypatch.setattr(pipeline, 'fit', fake_fit)
    pot, disk, bulge = all_params
    res = pipeline.fit_multistart(None, {}, pot, disk, bulge, n_restarts=2)

    assert res['best_index'] == 1
    assert res['final_losses'][0] == float('inf')
    assert np.isnan(res['all_runs'][0]['history']['loss'][-1])


def test_fit_multistart_uses_the_given_start_first_then_scatters(monkeypatch,
                                                                all_params):
    import phoenix.optimization.pipeline as pipeline

    calls = []

    def fake_fit(mapper, obs_maps, p, d, b, **kwargs):
        calls.append((p, d, b))
        return _fake_run(1.0, tag=len(calls))

    monkeypatch.setattr(pipeline, 'fit', fake_fit)
    pot, disk, bulge = all_params
    pipeline.fit_multistart(None, {}, pot, disk, bulge, n_restarts=3,
                            include_given_init=True, restart_log_sigma=0.4)

    assert calls[0] == (pot, disk, bulge)
    for p, d, b in calls[1:]:
        assert p != pot and d != disk and b != bulge
        assert set(p) == set(pot)


def test_fit_multistart_can_scatter_every_start(monkeypatch, all_params):
    import phoenix.optimization.pipeline as pipeline

    calls = []

    def fake_fit(mapper, obs_maps, p, d, b, **kwargs):
        calls.append(p)
        return _fake_run(1.0, tag=len(calls))

    monkeypatch.setattr(pipeline, 'fit', fake_fit)
    pot, disk, bulge = all_params
    pipeline.fit_multistart(None, {}, pot, disk, bulge, n_restarts=2,
                            include_given_init=False)
    assert all(p != pot for p in calls)


def test_fit_multistart_is_reproducible_for_a_given_restart_seed(monkeypatch,
                                                                 all_params):
    import phoenix.optimization.pipeline as pipeline

    def make_fake(store):
        def fake_fit(mapper, obs_maps, p, d, b, **kwargs):
            store.append(p)
            return _fake_run(1.0, tag=len(store))
        return fake_fit

    pot, disk, bulge = all_params
    first, second = [], []
    monkeypatch.setattr(pipeline, 'fit', make_fake(first))
    pipeline.fit_multistart(None, {}, pot, disk, bulge, n_restarts=3, restart_seed=3)
    monkeypatch.setattr(pipeline, 'fit', make_fake(second))
    pipeline.fit_multistart(None, {}, pot, disk, bulge, n_restarts=3, restart_seed=3)
    assert first == second


def test_fit_multistart_end_to_end(fake_mapper, all_params, obs_maps):
    """A real (tiny) two-restart run: the returned dict is a `fit` result plus the
    multistart bookkeeping, and the reported best run is the best one."""
    pot, disk, bulge = all_params
    res = fit_multistart(fake_mapper, obs_maps, pot, disk, bulge, n_restarts=2,
                         restart_log_sigma=0.2, restart_seed=0,
                         **_fit_kwargs(n_steps=3))

    assert {'pot_params', 'disk_df_params', 'bulge_df_params', 'history',
            'nan_steps', 'nan_grads_total',
            'all_runs', 'final_losses', 'best_index'} == set(res)
    assert len(res['all_runs']) == len(res['final_losses']) == 2
    assert res['best_index'] == int(np.argmin(res['final_losses']))
    best = res['all_runs'][res['best_index']]
    assert res['history']['loss'] == best['history']['loss']
    assert set(res['pot_params']) == set(pot)


# ==============================================================================
# MASS-TO-LIGHT IN THE LOSS
# ==============================================================================
def _terms(aux):
    return tuple(float(aux[k]) for k in ('mass_loss', 'vrot_loss', 'sigma_loss'))


def test_constant_ml_gives_exactly_the_mass_loss(fake_mapper, all_params):
    """With one Upsilon for both components the objective is unchanged: the first map
    enters as a log-space residual, so a common factor cancels once the detection floor
    is scaled with the map, and the kinematic maps are ratios and cancel outright.

    The check is made at a PERTURBED point -- at the truth every data term is exactly
    zero and the agreement would be vacuous.
    """
    pot, disk, bulge = all_params
    ml = (3.0, 3.0)
    obs_mass = make_observation(fake_mapper, pot, disk, bulge, **CFG)
    obs_light = make_observation(fake_mapper, pot, disk, bulge, ml_ratios=ml, **CFG)

    floor_mass = 1e-4 * float(jnp.max(obs_mass['mass']))
    floor_light = 1e-4 * float(jnp.max(obs_light['mass']))

    off = params_to_log({k: v * 1.3 for k, v in pot.items()}, disk, bulge)
    a = _terms(make_loss_fn(fake_mapper, obs_mass, mass_floor=floor_mass,
                            poisson_kwargs=POISSON_KW, **CFG)(off, None)[1])
    b = _terms(make_loss_fn(fake_mapper, obs_light, mass_floor=floor_light, ml_ratios=ml,
                            poisson_kwargs=POISSON_KW, **CFG)(off, None)[1])
    assert a[0] > 0.0            # the perturbation really does move the objective
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-8)


def test_component_ml_is_a_different_objective(fake_mapper, all_params):
    pot, disk, bulge = all_params
    obs_mass = make_observation(fake_mapper, pot, disk, bulge, **CFG)
    obs_light = make_observation(fake_mapper, pot, disk, bulge, ml_ratios=(1.5, 4.5), **CFG)

    off = params_to_log({k: v * 1.3 for k, v in pot.items()}, disk, bulge)
    a = _terms(make_loss_fn(fake_mapper, obs_mass, poisson_kwargs=POISSON_KW,
                            **CFG)(off, None)[1])
    b = _terms(make_loss_fn(fake_mapper, obs_light, ml_ratios=(1.5, 4.5),
                            poisson_kwargs=POISSON_KW, **CFG)(off, None)[1])
    assert not np.allclose(a, b, rtol=1e-3)


def test_ml_ratios_do_not_touch_the_poisson_penalty(fake_mapper, all_params):
    """Self-consistency is a statement about MASS sourcing the potential, so the penalty
    must keep the mass weights however the observables are photometrically weighted.
    If light ever leaked into it, the fit would be asking the light to gravitate."""
    pot, disk, bulge = all_params
    obs_mass = make_observation(fake_mapper, pot, disk, bulge, **CFG)
    obs_light = make_observation(fake_mapper, pot, disk, bulge, ml_ratios=(1.5, 4.5), **CFG)

    p_log = params_to_log(pot, disk, bulge)
    _, aux_m = make_loss_fn(fake_mapper, obs_mass, poisson_kwargs=POISSON_KW,
                            **CFG)(p_log, None)
    _, aux_l = make_loss_fn(fake_mapper, obs_light, ml_ratios=(1.5, 4.5),
                            poisson_kwargs=POISSON_KW, **CFG)(p_log, None)
    np.testing.assert_allclose(float(aux_l['poisson_penalty']),
                               float(aux_m['poisson_penalty']), rtol=1e-6)


def test_light_loss_is_exactly_zero_at_the_truth(fake_mapper, all_params):
    """The same seed/tracer trick as the mass case: a correctly specified Upsilon leaves
    the truth an exact minimum of the data term."""
    pot, disk, bulge = all_params
    ml = (1.5, 4.5)
    obs = make_observation(fake_mapper, pot, disk, bulge, ml_ratios=ml, **CFG)
    _, aux = make_loss_fn(fake_mapper, obs, ml_ratios=ml,
                          poisson_kwargs=POISSON_KW, **CFG)(params_to_log(pot, disk, bulge),
                                                            None)
    for value in _terms(aux):
        assert value == pytest.approx(0.0, abs=1e-10)


def test_mis_specified_ml_leaves_a_residual_at_the_truth(fake_mapper, all_params):
    """Assuming one global Upsilon for a galaxy that has two is a systematic no choice
    of parameters can absorb: the data term no longer vanishes at the true parameters."""
    pot, disk, bulge = all_params
    obs = make_observation(fake_mapper, pot, disk, bulge, ml_ratios=(1.5, 4.5), **CFG)
    _, aux = make_loss_fn(fake_mapper, obs, ml_ratios=(2.0, 2.0),
                          poisson_kwargs=POISSON_KW, **CFG)(params_to_log(pot, disk, bulge),
                                                            None)
    assert sum(_terms(aux)) > 1e-6


def test_fit_forwards_ml_ratios_and_mass_floor(fake_mapper, all_params):
    """`fit` must reach the same objective `make_loss_fn` builds, or the polish stage of
    a notebook chain would silently optimize something else."""
    pot, disk, bulge = all_params
    ml = (1.5, 4.5)
    obs = make_observation(fake_mapper, pot, disk, bulge, ml_ratios=ml, **CFG)
    floor = 1e-4 * float(jnp.max(obs['mass']))
    init = {k: v * 1.2 for k, v in pot.items()}

    res = fit(fake_mapper, obs, init, disk, bulge, n_steps=3, learning_rate=0.01,
              ml_ratios=ml, mass_floor=floor, poisson_kwargs=POISSON_KW, **CFG)
    expected = make_loss_fn(fake_mapper, obs, ml_ratios=ml, mass_floor=floor,
                            poisson_kwargs=POISSON_KW, **CFG)(
        params_to_log(init, disk, bulge), None)[0]
    np.testing.assert_allclose(res['history']['loss'][0], float(expected), rtol=1e-5)


# ==============================================================================
# SAMPLER PLUMBING
# ==============================================================================
def test_make_observation_honours_the_sampler(fake_mapper, all_params):
    """The two samplers draw genuinely different populations from the same DF, so the
    mock must change with `sampler` -- otherwise the argument would be silently ignored
    and a mock could never be built to match an importance-sampled fit."""
    pot, disk, bulge = all_params
    soft = make_observation(fake_mapper, pot, disk, bulge, sampler='soft', **CFG)
    imp = make_observation(fake_mapper, pot, disk, bulge, sampler='importance', **CFG)
    assert not np.allclose(np.asarray(soft['mass']), np.asarray(imp['mass']), rtol=1e-3)


def test_make_self_consistent_truth_honours_the_sampler(fake_mapper, all_params):
    """Self-consistency is a property of the sampled population, so solving for it must
    depend on how that population is drawn."""
    pot, disk, bulge = all_params
    kw = dict(tune_params=('M_disk',), N_disk=200, N_bulge=100, prng_seed=5,
              n_steps=3, poisson_kwargs=POISSON_KW)
    soft = make_self_consistent_truth(fake_mapper, pot, disk, bulge, sampler='soft', **kw)
    imp = make_self_consistent_truth(fake_mapper, pot, disk, bulge,
                                     sampler='importance', **kw)
    assert soft['penalty_initial'] != pytest.approx(imp['penalty_initial'], rel=1e-6)


def test_importance_sampled_loss_is_zero_at_the_truth(fake_mapper, all_params):
    """The same seed/tracer trick as for the default sampler: when the mock and the loss
    agree on `sampler`, the truth is an exact minimum of the data terms.

    The bar is 1e-8 rather than the 1e-10 the soft sampler manages: importance weights
    span a far wider dynamic range, so the float32 sums carry more round-off. It is
    still seven orders below the O(0.1) these terms reach at a perturbed point, which
    is the distinction the test is making.
    """
    pot, disk, bulge = all_params
    obs = make_observation(fake_mapper, pot, disk, bulge, sampler='importance', **CFG)
    _, aux = make_loss_fn(fake_mapper, obs, sampler='importance',
                          poisson_kwargs=POISSON_KW, **CFG)(params_to_log(pot, disk, bulge),
                                                            None)
    for value in _terms(aux):
        assert value == pytest.approx(0.0, abs=1e-8)


def test_mismatched_samplers_leave_a_residual_at_the_truth(fake_mapper, all_params):
    """Building the mock one way and fitting the other is the trap `sampler` exists to
    let you avoid: it puts an irreducible residual at the true parameters, exactly the
    bias `make_self_consistent_truth` is otherwise there to remove."""
    pot, disk, bulge = all_params
    obs = make_observation(fake_mapper, pot, disk, bulge, sampler='importance', **CFG)
    _, aux = make_loss_fn(fake_mapper, obs, sampler='soft',
                          poisson_kwargs=POISSON_KW, **CFG)(params_to_log(pot, disk, bulge),
                                                            None)
    assert sum(_terms(aux)) > 1e-6


def test_sampler_and_ml_ratios_compose(fake_mapper, all_params):
    """The photometric conversion and the sampler are independent knobs; used together
    the truth must still be an exact zero of the data term."""
    pot, disk, bulge = all_params
    ml = (1.5, 4.5)
    obs = make_observation(fake_mapper, pot, disk, bulge, sampler='importance',
                           ml_ratios=ml, **CFG)
    _, aux = make_loss_fn(fake_mapper, obs, sampler='importance', ml_ratios=ml,
                          poisson_kwargs=POISSON_KW, **CFG)(params_to_log(pot, disk, bulge),
                                                            None)
    for value in _terms(aux):
        assert value == pytest.approx(0.0, abs=1e-8)
