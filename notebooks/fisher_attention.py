"""
Fisher information and parameter "attention" maps for the edge-on twin experiment.

Which parameter of the potential or of the distribution function shapes which part
of the observable maps -- and how much does each map tell us about each parameter?
This script answers both at the self-consistent ground truth of the twin experiment
(`self_consistent_truth.json`, written by `self_consistent_optimization.py`), by
forward-mode differentiation of the whole model (DF sampling -> surrogate torus
mapping -> projection -> KDE binning) with respect to every parameter.

Run with:
    taskset -c 0-15 python notebooks/fisher_attention.py            # full run (~2 min, CPU)
    QUICK=1 taskset -c 0-15 python notebooks/fisher_attention.py    # smoke test (~20 s)

`taskset` keeps XLA's CPU backend from sizing its thread pool to all 256 cores of
this host, which is what kills long JAX runs here. Environment overrides: SAMPLER
(importance | soft), GRID_SIZE, N_PARTICLES, N_SEEDS, MASS_FLOOR_REL, CENTRE_FREE=0,
and RECOMPUTE=1 to ignore the cached Jacobians.

What is computed
----------------
The model maps mu -- log10 surface density, v_los and sigma_los on the fit grid --
are differentiated with respect to the log-parameters u = ln(theta), the coordinates
`fit` and `fit_vi` work in:

    J[map, pixel, i] = d mu[map, pixel] / d u_i     (response per unit ln theta)

With independent Gaussian pixel errors -- the noise model of `fit_vi` (NOISE_MASS_DEX
on the mass map, NOISE_V and NOISE_SIGMA on the kinematics) -- the Fisher matrix is

    F_ij = sum over maps and pixels of  J_i J_j / noise^2

and its inverse bounds the posterior covariance (Cramer-Rao). Reported are the
conditional error 1/sqrt([F + P]_ii) (every other parameter known), the marginal error
sqrt([F + P]^-1_ii) with the same broad Gaussian prior P as `fit_vi`
(PRIOR_LOG_STD), the marginal correlation matrix, and the eigen-directions of F --
the stiff and the sloppy parameter combinations.

The attention maps distribute that information over the sky:
  * response map        d mu / d ln theta_i in physical units (dex or km/s). Red:
                        the map value rises when theta_i grows.
  * information density (J_i / noise)^2 summed over the three maps, with the response
                        to a rigid shift of the galaxy projected out (see "The galaxy
                        centre"): where the information on theta_i comes from if
                        every other parameter were known. Its pixel sum is F_ii.
  * marginal attention  W_i^2, where W = (F + P)^-1 J^T / noise are the weights of the
                        linearised (Gauss-Newton) estimate theta_i = sum W_i * data.
                        A pixel whose response another parameter can mimic gets ~zero
                        weight, so this shows where the information that
                        *distinguishes* theta_i from the other parameters lives.
  * dominance map       per pixel, the parameter with the largest information density,
                        i.e. the one whose fractional change moves that pixel by the
                        most noise-sigmas. "Fractional" matters: the comparison is per
                        unit ln theta, the scale on which the fits and the prior work.

Why the Jacobian is averaged over seeds
---------------------------------------
The maps are KDE estimates from a finite tracer sample, so a Jacobian taken from one
sampling seed carries that sample's shot noise. The scalar Fisher errors move by tens
of percent between seeds (up to a factor ~2 for single seeds of 6k tracers), and the
per-pixel attention maps far more: between two seeds of 24k tracers the cosine
similarity of a parameter's information map went down to 0.3 (0.1 at 6k tracers),
i.e. a single-seed map largely shows where that particular sample happened to be
sparse. Averaging the Jacobian over N_SEEDS seeds gives the response of the
*expected* maps, which is the physical question. The remaining Monte Carlo error is
measured, not assumed: responses below MC_SIGNIFICANCE standard errors are shown as
zero, the information densities are debiased by the squared standard error, the
marginal errors carry jackknife-over-seeds error bars, and the summary lists the
fraction of each F_ii that is still shot noise.

Why a relative detection floor
------------------------------
The fits keep every pixel above an ABSOLUTE floor of 1e-3 M_sun, i.e. every pixel the
tail of any tracer's kernel reaches -- 12 orders of magnitude below the peak. With a
constant 0.05 dex error, those essentially empty pixels carry a large share of the
log-mass information (in a single-seed Jacobian of the twin setup, the most
informative pixel for a_halo held 0.05 M_sun, against 7e9 M_sun in the peak pixel, and
8-18% of a_halo's information), which no observation could deliver. MASS_FLOOR_REL
instead keeps pixels above a fixed fraction of the peak, like a surface-brightness
limit (1e-3 = 7.5 mag below the peak). MASS_FLOOR_REL=0 restores the fits' floor.

The sampler
-----------
The default is SAMPLER=importance. Under the soft-acceptance sampler the DF parameters
barely reach the tracers (`observables.sample_and_map_particles`), so their attention
maps would show an artefact of the sampler rather than the physics. Forward mode is
also what makes the importance sampler usable here: its reverse-mode gradient goes
non-finite for the potential and disk parameters, but the forward-mode Jacobian is
finite.

The galaxy centre
-----------------
By symmetry the model galaxy's centre never moves. The surrogate's does: it places
the tracers ~1 kpc off the origin, and that offset changes with the potential --
measured at the truth, the centroid moves by +2.9 kpc in x per e-fold of M_disk and
by (+2.0, +2.3, -1.1) kpc in (x, y, z) per e-fold of a_disk. 85% of the noise-weighted
a_disk response is nothing but this rigid shift. A twin experiment, where mock and
model come from the same surrogate, can use it as information; a fit to real data,
whose centre is set by the data, cannot. So the centre is a nuisance parameter here
(CENTRE_FREE): the Jacobian also carries the exact response to a rigid sky shift, and
every information-based quantity is computed with that shift marginalised. The signed
response maps still show what the model does, and the tables list the centroid drift,
each response's rigid-shift share, and the marginal errors with the centre fixed.

Cross-check against the VI posterior
------------------------------------
If `vi_self_consistent.py` has been run, the Fisher bound of exactly its setup (soft
sampler, 6000 tracers, seed 0, the fits' absolute floor, centre fixed) is compared with
its posterior widths. Where the parameters are constrained by the data the two should
agree; where the Fisher matrix has a (near-)null direction they cannot, because the
posterior there is set by the prior, which a finite VI run does not explore. A
full-rank VI whose correlations have not finished converging also lands between the
conditional and the marginal bound: a mean-field Gaussian recovers exactly the
conditional widths.

Outputs (notebooks/figures/):
  - fisher_summary_<sampler>.png              errors, correlations, eigen-directions
  - fisher_attention_potential_<sampler>.png  attention maps, potential parameters
  - fisher_attention_df_<sampler>.png         attention maps, DF parameters
  - fisher_dominant_<sampler>.png             dominant parameter per pixel
  - fisher_vs_vi.png                          Fisher bound vs the VI posterior
  - fisher_<sampler>.npz, fisher_<sampler>_summary.txt
  - fisher_jacobians_*.npz                    cached per-seed Jacobians
"""

import hashlib
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.patches import Patch
import numpy as np

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from phoenix.actions_to_phasespace.actions_to_phasespace_nn import PhoenixMapper
from phoenix.optimization.observables import sample_and_map_particles, project_to_sky, bin_maps
from phoenix.optimization.pipeline import params_to_log, log_to_params, _flat_param_names

# ==============================================================================
# CONFIGURATION  (geometry, noise and prior kept identical to vi_self_consistent.py)
# ==============================================================================
QUICK = bool(os.environ.get("QUICK"))
RECOMPUTE = bool(os.environ.get("RECOMPUTE"))

SAMPLER = os.environ.get("SAMPLER", "importance")
GRID_SIZE = int(os.environ.get("GRID_SIZE", 14))
EXTENT_X, EXTENT_Z = 15.0, 10.0
# The bandwidth the mock and the fits' final annealing stage use.
BANDWIDTH = 0.25 * max(2 * EXTENT_X / GRID_SIZE, 2 * EXTENT_Z / GRID_SIZE)
INCLINATION_DEG = 90.0
SPHEROID_COROTATION = 0.5

# Tracers per component per seed, and the number of seeds the Jacobian is averaged
# over (see module docstring). The importance sampler's effective sample size is
# only ~15% of N, hence the large N.
N_PARTICLES = int(os.environ.get("N_PARTICLES", 6_000 if QUICK else 24_000))
N_SEEDS = int(os.environ.get("N_SEEDS", 4 if QUICK else 24))
# Tangent directions per forward-mode pass. Each pass evaluates the model once and
# carries JVP_CHUNK tangents; larger is faster but holds JVP_CHUNK copies of the
# (grid x grid x tracers) KDE arrays in memory. 21 parameters + 2 centre shifts
# = 3 passes per seed.
JVP_CHUNK = 8

# Detection floor relative to the peak of the expected mass map; 0 = the fits'
# absolute floor (see module docstring).
MASS_FLOOR_REL = float(os.environ.get("MASS_FLOOR_REL", 1e-3))
MASS_SOFTENING = 1e-3      # `mass_floor` of data_fit_loss / fit_vi: log10(mass + 1e-3)

# Per-pixel measurement errors and the log-space prior, as in vi_self_consistent.py.
NOISE_MASS_DEX, NOISE_V, NOISE_SIGMA = 0.05, 10.0, 10.0
NOISE = np.array([NOISE_MASS_DEX, NOISE_V, NOISE_SIGMA])
PRIOR_LOG_STD = 2.0

# Parameters with exactly zero response (the DF amplitudes cancel in the normalised
# tracer weights; L0 and the root-finder seed do not reach these maps). Their
# response is still computed and reported, but they are excluded from F, as from
# the fits.
FROZEN = ("Sigma0", "N0_spheroid", "Rinit_for_Rc", "L0")

# Treat the galaxy centre as a free nuisance parameter (see module docstring): the
# surrogate shifts the whole galaxy when the potential parameters change, and only
# information that survives re-fitting the centre is usable on real data.
CENTRE_FREE = os.environ.get("CENTRE_FREE", "1") != "0"

# Responses smaller than this many seed-to-seed standard errors are drawn as zero.
MC_SIGNIFICANCE = 2.0

# A parameter is prior-limited when its marginal error is set by the prior rather than
# by the maps: it grows by more than this factor when the prior is made 10x wider.
# (A flat error threshold misses e.g. R0, which sits on an exactly flat direction but
# shares it with two other parameters, so the prior holds its marginal below 1.)
PRIOR_LIMITED_GROWTH = 1.2
# An eigen-direction of F is unconstrained when the error along it exceeds this
# fraction of the prior width.
PRIOR_LIMITED_FRAC = 0.5

# VI cross-check: the exact setup of vi_self_consistent.py.
VI_N_PARTICLES = 6_000
VI_SAMPLER = "soft"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIGDIR = os.path.join(HERE, "figures")
TRUTH_JSON = os.path.join(HERE, "self_consistent_truth.json")
VI_NPZ = os.path.join(FIGDIR, "VI_theory_posterior.npz")
NN_DIR = os.path.join(REPO, "phoenix", "torus_mapping_neural_network")

# Used only if self_consistent_truth.json is missing.
POT_NOMINAL = {
    "M_halo": 1e12, "a_halo": 20.0,
    "M_disk": 5e10, "a_disk": 3.0, "b_disk": 0.3,
    "M_bulge": 1e10, "a_bulge": 1.0,
}
DISK_NOMINAL = {
    "R0": 8.0, "Rd": 3.0, "Sigma0": 1000.0,
    "RsigR": 6.0, "RsigZ": 6.0,
    "sigmaR0_R0": 35.0, "sigmaz0_R0": 20.0,
    "L0": 10.0, "Rinit_for_Rc": 8.0,
}
BULGE_NOMINAL = {
    "N0_spheroid": 1e10, "J0_spheroid": 100.0,
    "Gamma_spheroid": 1.5, "Beta_spheroid": 4.5, "eta_spheroid": 1.0,
}

# Display order: potential, disk DF, bulge DF.
DISPLAY = ["M_halo", "a_halo", "M_disk", "a_disk", "b_disk", "M_bulge", "a_bulge",
           "R0", "Rd", "RsigR", "RsigZ", "sigmaR0_R0", "sigmaz0_R0",
           "J0_spheroid", "Gamma_spheroid", "Beta_spheroid", "eta_spheroid"]
GROUP = {k: "pot" for k in POT_NOMINAL}
GROUP.update({k: "disk" for k in DISK_NOMINAL})
GROUP.update({k: "bulge" for k in BULGE_NOMINAL})
GROUP_LABEL = {"pot": "potential", "disk": "disk DF", "bulge": "bulge DF"}
MAP_LABEL = [r"$\log_{10}\Sigma$", r"$v_{\rm los}$", r"$\sigma_{\rm los}$"]
MAP_UNIT = ["dex", "km/s", "km/s"]

# ==============================================================================
# STYLE  (same palette as vi_self_consistent.py)
# ==============================================================================
# Sequential single-hue blue ramp for magnitudes (information, attention); a
# blue <-> red diverging scale with a neutral grey midpoint for signed responses,
# the red arm matched step for step to the blue one in OKLab lightness and chroma;
# and the first three categorical slots for the three parameter groups.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("phoenix_seq", BLUE_RAMP)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "phoenix_div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f1aea8", "#e34948", "#892c2a"])
GROUP_COLOR = {"pot": "#2a78d6", "disk": "#eb6834", "bulge": "#1baf7a"}
C_FISHER, C_VI = "#2a78d6", "#eb6834"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a887f"
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"
for _cmap in (CMAP_SEQ, CMAP_DIV):
    _cmap.set_bad(SURFACE)          # pixels outside the footprint

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK3, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.labelcolor": INK, "text.color": INK,
    "font.size": 9,
})

PRETTY = {
    "M_halo": r"$M_{\rm halo}$", "a_halo": r"$a_{\rm halo}$",
    "M_disk": r"$M_{\rm disk}$", "a_disk": r"$a_{\rm disk}$", "b_disk": r"$b_{\rm disk}$",
    "M_bulge": r"$M_{\rm bulge}$", "a_bulge": r"$a_{\rm bulge}$",
    "R0": r"$R_0$", "Rd": r"$R_d$", "RsigR": r"$R_{\sigma R}$", "RsigZ": r"$R_{\sigma z}$",
    "sigmaR0_R0": r"$\sigma_{R0}$", "sigmaz0_R0": r"$\sigma_{z0}$",
    "J0_spheroid": r"$J_{0,\rm sph}$", "Gamma_spheroid": r"$\Gamma_{\rm sph}$",
    "Beta_spheroid": r"$\beta_{\rm sph}$", "eta_spheroid": r"$\eta_{\rm sph}$",
}
# Compact labels for the per-pixel dominance maps.
SHORT = {
    "M_halo": r"$M_h$", "a_halo": r"$a_h$", "M_disk": r"$M_d$", "a_disk": r"$a_d$",
    "b_disk": r"$b_d$", "M_bulge": r"$M_b$", "a_bulge": r"$a_b$",
    "R0": r"$R_0$", "Rd": r"$R_d$", "RsigR": r"$R_{\sigma R}$", "RsigZ": r"$R_{\sigma z}$",
    "sigmaR0_R0": r"$\sigma_R$", "sigmaz0_R0": r"$\sigma_z$",
    "J0_spheroid": r"$J_0$", "Gamma_spheroid": r"$\Gamma$", "Beta_spheroid": r"$\beta$",
    "eta_spheroid": r"$\eta$",
}
UNIT = {"M_halo": r"$M_\odot$", "M_disk": r"$M_\odot$", "M_bulge": r"$M_\odot$",
        "a_halo": "kpc", "a_disk": "kpc", "b_disk": "kpc", "a_bulge": "kpc",
        "R0": "kpc", "Rd": "kpc", "RsigR": "kpc", "RsigZ": "kpc",
        "sigmaR0_R0": "km/s", "sigmaz0_R0": "km/s", "J0_spheroid": "kpc km/s"}

MAP_EXTENT = [-EXTENT_X, EXTENT_X, -EXTENT_Z, EXTENT_Z]


# ==============================================================================
# MODEL AND JACOBIANS
# ==============================================================================
def load_mapper():
    """The surrogate, from PHOENIX_WEIGHTS / PHOENIX_STATS if set, else the package
    default location, else `old_delete/` -- where the current weights sit while the
    network is being retrained. Returns the mapper and a hash of its weights, which
    keys the Jacobian cache."""
    weights = os.environ.get("PHOENIX_WEIGHTS")
    stats = os.environ.get("PHOENIX_STATS")
    if weights is None:
        for d in (NN_DIR, os.path.join(NN_DIR, "old_delete")):
            if os.path.exists(os.path.join(d, "phoenix_weights.msgpack")):
                weights = os.path.join(d, "phoenix_weights.msgpack")
                stats = stats or os.path.join(d, "phoenix_norm_stats.npz")
                break
    if weights is None:
        raise FileNotFoundError(f"no phoenix_weights.msgpack in {NN_DIR} or its old_delete/; "
                                f"set PHOENIX_WEIGHTS and PHOENIX_STATS")
    with open(weights, "rb") as fh:
        sha = hashlib.sha1(fh.read()).hexdigest()[:12]
    return PhoenixMapper(weights_path=weights, stats_path=stats), sha


def load_truth():
    if os.path.exists(TRUTH_JSON):
        saved = json.load(open(TRUTH_JSON))
        return (saved["pot_params"], saved["disk_df_params"], saved["bulge_df_params"],
                os.path.basename(TRUTH_JSON))
    return POT_NOMINAL, DISK_NOMINAL, BULGE_NOMINAL, "nominal parameters (no truth file)"


def seed_jacobians(mapper, params_log, sampler, n_particles, n_seeds):
    """Per-seed Jacobians of the maps with respect to every log-parameter and to a
    rigid shift of the galaxy on the sky.

    Returns J (seeds, params + 2, 3, grid, grid) -- per unit ln theta, the last two
    columns per kpc of shift along x and z --, the Jacobian of the tracers' centroid
    (seeds, params + 2, 3) in kpc, the maps (seeds, 3, grid, grid) and the raw mass
    maps (seeds, grid, grid). Forward mode: one model evaluation carries JVP_CHUNK
    tangent directions, so a seed costs ~(params / JVP_CHUNK) evaluations, independent
    of the number of pixels.
    """
    vec0, unravel = ravel_pytree(params_log)
    n_par = int(vec0.size)
    ext0 = jnp.concatenate([vec0, jnp.zeros(2)])        # + sky shift (x0, z0) in kpc
    n_tan = n_par + 2

    def model(ext, seed):
        pot, disk, bulge = log_to_params(unravel(ext[:n_par]))
        x, y, z, vx, vy, vz, w = sample_and_map_particles(
            mapper, pot, disk, bulge, N_disk=n_particles, N_bulge=n_particles,
            prng_seed=seed, spheroid_corotation=SPHEROID_COROTATION, sampler=sampler,
        )
        centroid = jnp.stack([x @ w, y @ w, z @ w]) / jnp.sum(w)
        x_sky, z_sky, v_los = project_to_sky(x, y, z, vx, vy, vz, INCLINATION_DEG)
        m = bin_maps(x_sky + ext[n_par], z_sky + ext[n_par + 1], v_los, w,
                     grid_size=GRID_SIZE, extent_x=EXTENT_X, extent_z=EXTENT_Z,
                     soft_bin_h=BANDWIDTH)
        # The residuals of fit_vi: log10(mass + mass_floor), v_rot, sigma.
        obs = jnp.stack([jnp.log10(m["mass"] + MASS_SOFTENING), m["v_rot"], m["sigma"]])
        return obs, m["mass"], centroid

    # The seed is a traced argument, so this compiles once for all seeds. Under vmap
    # the primal does not depend on the tangent and is evaluated once per chunk.
    @jax.jit
    def jvp_chunk(tangents, seed):
        primal, tangent = jax.vmap(
            lambda t: jax.jvp(lambda v: model(v, seed), (ext0,), (t,)))(tangents)
        return primal[0][0], primal[1][0], tangent[0], tangent[2]

    n_pad = -(-n_tan // JVP_CHUNK) * JVP_CHUNK
    basis = jnp.eye(n_pad, n_tan)
    J = np.empty((n_seeds, n_tan, 3, GRID_SIZE, GRID_SIZE), np.float32)
    J_cen = np.empty((n_seeds, n_tan, 3), np.float32)
    maps = np.empty((n_seeds, 3, GRID_SIZE, GRID_SIZE), np.float32)
    masses = np.empty((n_seeds, GRID_SIZE, GRID_SIZE), np.float32)
    t0 = time.time()
    for s in range(n_seeds):
        for c in range(0, n_pad, JVP_CHUNK):
            obs, mass, d_obs, d_cen = jvp_chunk(basis[c:c + JVP_CHUNK], s)
            take = min(JVP_CHUNK, n_tan - c)
            J[s, c:c + take] = np.asarray(d_obs)[:take]
            J_cen[s, c:c + take] = np.asarray(d_cen)[:take]
        maps[s], masses[s] = np.asarray(obs), np.asarray(mass)
        if s == 0 or (s + 1) % max(n_seeds // 4, 1) == 0:
            print(f"    seed {s + 1:3d}/{n_seeds}   {time.time() - t0:6.1f} s", flush=True)
    n_bad = int((~np.isfinite(J)).sum())
    if n_bad:
        raise FloatingPointError(f"{n_bad} non-finite Jacobian entries ({sampler} sampler)")
    return {"J": J, "J_cen": J_cen, "maps": maps, "mass": masses}


def cached_jacobians(mapper, params_log, sampler, n_particles, n_seeds, weights_sha):
    """`seed_jacobians`, cached on disk against everything that determines them."""
    key = json.dumps({
        "layout": 2, "sampler": sampler, "grid": GRID_SIZE, "extent": [EXTENT_X, EXTENT_Z],
        "bandwidth": BANDWIDTH, "inclination": INCLINATION_DEG,
        "corotation": SPHEROID_COROTATION, "n_particles": n_particles, "n_seeds": n_seeds,
        "params": {g: {k: float(np.exp(v)) for k, v in sorted(p.items())}
                   for g, p in sorted(params_log.items())},
        "weights": weights_sha,
    }, sort_keys=True)
    path = os.path.join(FIGDIR, f"fisher_jacobians_{sampler}_G{GRID_SIZE}"
                                f"_N{n_particles}_K{n_seeds}.npz")
    if os.path.exists(path) and not RECOMPUTE:
        d = np.load(path)
        if str(d["key"]) == key:
            print(f"    reusing {os.path.basename(path)}  (RECOMPUTE=1 to redo it)")
            return {k: d[k] for k in ("J", "J_cen", "maps", "mass")}
    jac = seed_jacobians(mapper, params_log, sampler, n_particles, n_seeds)
    np.savez(path, key=key, **jac)
    return jac


# ==============================================================================
# FISHER ANALYSIS
# ==============================================================================
def analyse(jac, names, mass_floor_rel, centre_free=None):
    """Fisher matrix, Cramer-Rao errors and attention maps from per-seed Jacobians.

    With `centre_free` (default CENTRE_FREE) the galaxy centre is a nuisance parameter
    and everything information-based -- F, the errors, correlations, eigen-directions,
    information densities, marginal attention -- refers to the physical parameters
    with the centre marginalised. Everything is returned in DISPLAY order, frozen
    parameters dropped.
    """
    centre_free = CENTRE_FREE if centre_free is None else centre_free
    n_par = len(names)
    order = [names.index(k) for k in DISPLAY if k in names and k not in FROZEN]
    labels = [names[i] for i in order]
    P = len(labels)
    J = np.moveaxis(jac["J"].astype(np.float64), 1, -1)     # (K, 3, G, G, params + 2)
    K = J.shape[0]

    mass = jac["mass"].mean(0).astype(np.float64)
    floor = mass_floor_rel * mass.max() if mass_floor_rel > 0 else MASS_SOFTENING
    mask = mass > floor
    n = int(mask.sum())

    # Whitened, footprint-only design matrices, rows = (map, pixel): the physical
    # parameters, and the two rigid shifts of the galaxy on the sky.
    D = (J[:, :, mask, :] / NOISE[None, :, None, None]).reshape(K, 3 * n, -1)
    Ak = D[:, :, order]                                       # (K, 3n, P)
    T = D[:, :, n_par:].mean(0)                               # (3n, 2)
    shift_proj = T @ np.linalg.solve(T.T @ T, T.T)

    # How much of each response a rigid shift of the galaxy reproduces. By symmetry the
    # true model never moves its centre, so this is the surrogate's doing.
    A_raw = Ak.mean(0)
    shift_share = 1.0 - ((A_raw - shift_proj @ A_raw)**2).sum(0) / (A_raw**2).sum(0)
    prior = np.eye(P) / PRIOR_LOG_STD**2
    sig_marg_fixed = np.sqrt(np.diag(np.linalg.inv(A_raw.T @ A_raw + prior)))
    if centre_free:
        # Flat prior on the centre: marginalising it projects the two shift templates
        # out of every seed's design matrix. The per-pixel decompositions below then
        # sum exactly to the centre-marginalised F.
        Ak = Ak - shift_proj @ Ak
    A = Ak.mean(0)
    F = A.T @ A
    cov = np.linalg.inv(F + prior)
    sig_marg = np.sqrt(np.diag(cov))
    # Conditional on all other parameters, with the same prior as the marginal, so that
    # sig_cond <= sig_marg always holds (without it a barely constrained parameter
    # would show a conditional error far above its prior-bounded marginal one).
    sig_cond = 1.0 / np.sqrt(np.diag(F + prior))
    corr = cov / np.outer(sig_marg, sig_marg)
    share = (A.reshape(3, n, P)**2).sum(1) / np.diag(F)               # (3, P)
    lam, evec = np.linalg.eigh(F)
    sig_marg_wide = np.sqrt(np.diag(np.linalg.inv(F + prior / 100.0)))
    prior_limited = sig_marg_wide > PRIOR_LIMITED_GROWTH * sig_marg

    # Monte Carlo diagnostics: shot-noise fraction of F_ii (the average of K noisy
    # Jacobians still carries 1/K of their variance) and jackknife-over-seeds errors
    # of the marginal errors.
    noise_frac = np.full(P, np.nan)
    sig_marg_err = np.full(P, np.nan)
    se = np.zeros_like(A)
    if K > 1:
        se = Ak.std(0, ddof=1) / np.sqrt(K)
        noise_frac = (se**2).sum(0) / np.diag(F)
        total = Ak.sum(0)
        jk = []
        for k in range(K):
            Aj = (total - Ak[k]) / (K - 1)
            jk.append(np.sqrt(np.diag(np.linalg.inv(Aj.T @ Aj + prior))))
        jk = np.array(jk)
        sig_marg_err = np.sqrt((K - 1) / K * ((jk - jk.mean(0))**2).sum(0))

    # Attention maps. Information densities are debiased by the squared standard error
    # (an unbiased estimate of the squared expected response), so pure shot noise does
    # not show up as information.
    dens = np.zeros((3, GRID_SIZE, GRID_SIZE, P))
    dens[:, mask, :] = np.maximum(A**2 - se**2, 0.0).reshape(3, n, P)
    W = (cov @ A.T).reshape(P, 3, n)
    marg = np.zeros((GRID_SIZE, GRID_SIZE, P))
    marg[mask] = (W**2).sum(1).T

    # The signed responses are shown as the model produces them (centre not refitted).
    Jp = J[..., order]
    resp = Jp.mean(0)                                                  # (3, G, G, P)
    resp_se = Jp.std(0, ddof=1) / np.sqrt(K) if K > 1 else np.zeros_like(resp)
    significant = (np.abs(resp) >= MC_SIGNIFICANCE * resp_se) if K > 1 else np.ones(resp.shape, bool)
    resp_shown = np.where(significant & mask[None, :, :, None], resp, 0.0)

    cen = jac["J_cen"][:, order, :].astype(np.float64)                 # (K, P, 3)
    drift = cen.mean(0)
    drift_se = cen.std(0, ddof=1) / np.sqrt(K) if K > 1 else np.zeros_like(drift)

    maps = jac["maps"].mean(0)
    return {
        "labels": labels, "K": K, "mask": mask, "floor": floor, "n_pix": n,
        "centre_free": centre_free,
        "F": F, "cov": cov, "corr": corr, "sig_marg": sig_marg, "sig_cond": sig_cond,
        "sig_marg_err": sig_marg_err, "sig_marg_fixed": sig_marg_fixed,
        "noise_frac": noise_frac, "share": share, "shift_share": shift_share,
        "drift": drift, "drift_se": drift_se, "lam": lam, "evec": evec,
        "prior_limited": prior_limited,
        "resp": resp, "resp_se": resp_se, "resp_shown": resp_shown,
        "dens": dens, "marg": marg,
        "model_maps": np.stack([np.log10(np.maximum(mass, 1e-30)), maps[1], maps[2]]),
    }


def describe_direction(v, labels, cut=0.2):
    """'+0.63 sigma_R0 - 0.63 sigma_z0 + 0.47 R0' for an eigenvector (sign-fixed)."""
    v = v * np.sign(v[np.argmax(np.abs(v))])
    idx = [i for i in np.argsort(-np.abs(v)) if abs(v[i]) >= cut]
    return " ".join(f"{'+' if v[i] > 0 else '-'}{abs(v[i]):.2f} {labels[i]}" for i in idx)


def sloppy_directions(res):
    """Eigen-directions of F along which the data constrain ln theta worse than
    PRIOR_LIMITED_FRAC of the prior width (at least the three sloppiest)."""
    lam, evec = res["lam"], res["evec"]
    sd = 1.0 / np.sqrt(np.maximum(lam, 1e-300))
    n = max(3, int((sd > PRIOR_LIMITED_FRAC * PRIOR_LOG_STD).sum()))
    return [(sd[j], evec[:, j]) for j in range(n)]


# ==============================================================================
# FIGURES
# ==============================================================================
def crop_height(mask, pad=0.5):
    """Half-height (kpc) of the detected footprint plus `pad` pixels. The edge-on disk
    fills only the middle third of the +-EXTENT_Z field, so the maps are cropped to
    it -- symmetric about the midplane."""
    dz = 2 * EXTENT_Z / mask.shape[0]
    rows = np.where(mask.any(1))[0]
    lo = -EXTENT_Z + (rows.min() - pad) * dz
    hi = -EXTENT_Z + (rows.max() + 1 + pad) * dz
    return min(max(-lo, hi), EXTENT_Z)


def footprint_outline(ax, mask, color=INK3, lw=0.6):
    """Draws the boundary of the detected footprint along pixel edges."""
    G = mask.shape[0]
    dx, dz = 2 * EXTENT_X / G, 2 * EXTENT_Z / G
    segs = []
    for r in range(G):
        for c in range(G):
            if not mask[r, c]:
                continue
            x0, z0 = -EXTENT_X + c * dx, -EXTENT_Z + r * dz
            if r == 0 or not mask[r - 1, c]:
                segs.append([(x0, z0), (x0 + dx, z0)])
            if r == G - 1 or not mask[r + 1, c]:
                segs.append([(x0, z0 + dz), (x0 + dx, z0 + dz)])
            if c == 0 or not mask[r, c - 1]:
                segs.append([(x0, z0), (x0, z0 + dz)])
            if c == G - 1 or not mask[r, c + 1]:
                segs.append([(x0 + dx, z0), (x0 + dx, z0 + dz)])
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))


def draw_map(ax, img, mask, cmap, norm, zmax, alpha=1.0):
    ax.imshow(np.where(mask, img, np.nan), origin="lower", extent=MAP_EXTENT, cmap=cmap,
              norm=norm, interpolation="nearest", alpha=alpha, aspect="auto")
    footprint_outline(ax, mask)
    ax.set_xlim(-EXTENT_X, EXTENT_X)
    ax.set_ylim(-zmax, zmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def label_above(ax, s, where="left", color=INK2, size=6.8, weight="normal"):
    x, ha = (0.0, "left") if where == "left" else (1.0, "right")
    ax.text(x, 1.04, s, transform=ax.transAxes, ha=ha, va="bottom", fontsize=size,
            color=color, fontweight=weight)


def fmt_sig(v):
    return f"{v:.2g}" if v < 0.995 else f"{v:.3g}"


def fmt_frac(s):
    """A log-space error as a fractional error, e.g. 0.0123 -> '1.2%'."""
    return f"{100 * s:.2g}%" if s < 0.1 else f"{s:.2f} in ln"


def fmt_value(k, v):
    if abs(v) >= 1e4:
        e = int(np.floor(np.log10(abs(v))))
        num = rf"${v / 10**e:.2f}\times10^{{{e}}}$"
    else:
        num = f"{v:.3g}"
    return f"{num} {UNIT.get(k, '')}".strip()


def ink_on(rgb):
    """Ink or white text, whichever contrasts better with the fill."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055)**2.4 for c in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if lum < 0.179 else INK


def figure_attention(res, truth, keys, fname, title):
    """Rows = parameters; columns = signed response of the three maps, information
    density, marginal attention. A header row shows the model maps themselves."""
    labels, mask = res["labels"], res["mask"]
    rows = [labels.index(k) for k in keys]
    n = len(rows)
    zmax = crop_height(mask)

    # Manual layout in inches: label column, 3 response panels, wider gap, 2 attention
    # panels. Panels keep the true x:z aspect of the cropped field.
    pw = 2.3
    ph = pw * zmax / EXTENT_X
    left, gap, family_gap, right = 1.75, 0.08, 0.32, 0.3
    top, head_title, head_gap, row_gap, bottom = 0.95, 0.32, 0.95, 0.32, 1.15
    W = left + 5 * pw + 3 * gap + family_gap + right
    H = top + head_title + ph + head_gap + n * ph + (n - 1) * row_gap + bottom
    fig = plt.figure(figsize=(W, H))

    def col_x(c):
        return left + c * (pw + gap) + (family_gap - gap if c >= 3 else 0.0)

    def rect(c, y_top, width=pw):
        return [col_x(c) / W, 1 - (y_top + ph) / H, width / W, ph / H]

    seq_norm = PowerNorm(0.5, vmin=0.0, vmax=1.0)
    div_norm = plt.Normalize(-1.0, 1.0)

    # --- header row: the model maps ------------------------------------------------
    y0 = top + head_title
    mm = res["model_maps"]
    inside = [mm[c][mask] for c in range(3)]
    vmax_v = np.abs(inside[1]).max()
    lims = [(inside[0].min(), inside[0].max()), (-vmax_v, vmax_v),
            (inside[2].min(), inside[2].max())]
    head = [r"model $\log_{10}\Sigma$", r"model $v_{\rm los}$", r"model $\sigma_{\rm los}$"]
    head_range = [f"{lims[0][0]:.3g} to {lims[0][1]:.3g}", f"$\\pm${vmax_v:.3g} km/s",
                  f"{lims[2][0]:.3g} to {lims[2][1]:.3g} km/s"]
    for c in range(3):
        ax = fig.add_axes(rect(c, y0))
        draw_map(ax, mm[c], mask, CMAP_DIV if c == 1 else CMAP_SEQ,
                 plt.Normalize(*lims[c]), zmax)
        label_above(ax, head[c], size=8.5, color=INK)
        label_above(ax, head_range[c], where="right", size=6.8)
    fig.text(col_x(3) / W, 1 - y0 / H,
             "How to read a row", fontsize=8.5, color=INK, va="top", fontweight="bold")
    fig.text(col_x(3) / W, 1 - (y0 + 0.2) / H,
             "Left: change of each map per e-fold increase of the parameter, each panel on\n"
             "its own scale ($\\pm$max above it); the % is that map's share of the parameter's\n"
             "Fisher information. Right: where that information sits on the sky -- with\n"
             "every other parameter known, and with all of them free. Under each name:\n"
             "the share of its response that is a mere shift of the whole galaxy, which\n"
             "the surrogate produces and the true model cannot. ($\\Sigma$: $M_\\odot$/pixel.)",
             fontsize=7.2, color=INK2, va="top", linespacing=1.4)

    # --- parameter rows ------------------------------------------------------------
    col_title = [r"$\partial\log_{10}\Sigma\,/\,\partial\ln\theta$",
                 r"$\partial v_{\rm los}\,/\,\partial\ln\theta$",
                 r"$\partial\sigma_{\rm los}\,/\,\partial\ln\theta$",
                 "information density", "marginal attention"]
    centre = res["centre_free"]
    col_sub = ["dex", "km/s", "km/s",
               "others known" + (", centre refitted" if centre else ""),
               "all parameters" + (" and centre" if centre else "") + " free"]
    y1 = top + head_title + ph + head_gap
    for c in range(5):
        x = (col_x(c) + pw / 2) / W
        fig.text(x, 1 - (y1 - 0.42) / H, col_title[c], ha="center", va="bottom",
                 fontsize=9.5, color=INK)
        fig.text(x, 1 - (y1 - 0.25) / H, col_sub[c], ha="center", va="bottom",
                 fontsize=7.5, color=INK2)

    for r, i in enumerate(rows):
        k = labels[i]
        y = y1 + r * (ph + row_gap)
        for c in range(3):
            ax = fig.add_axes(rect(c, y))
            img = res["resp_shown"][c, :, :, i]
            vmax = np.abs(img[mask]).max()
            draw_map(ax, img / vmax if vmax > 0 else img, mask, CMAP_DIV, div_norm, zmax)
            label_above(ax, f"$\\pm${fmt_sig(vmax)} {MAP_UNIT[c]}" if vmax > 0
                        else "no significant response")
            label_above(ax, f"{100 * res['share'][c, i]:.0f}%", where="right")
        limited = res["prior_limited"][i]
        for c, img, txt in ((3, res["dens"][..., i].sum(0),
                             r"$\sigma_{\rm cond}$ = " + fmt_frac(res["sig_cond"][i])),
                            (4, res["marg"][..., i],
                             r"$\sigma_{\rm marg}$ = " + fmt_frac(res["sig_marg"][i]))):
            ax = fig.add_axes(rect(c, y))
            vmax = img[mask].max()
            faded = c == 4 and limited
            draw_map(ax, img / vmax if vmax > 0 else img, mask, CMAP_SEQ, seq_norm, zmax,
                     alpha=0.35 if faded else 1.0)
            label_above(ax, txt)
            if faded:
                label_above(ax, "prior-limited", where="right", color=INK, weight="bold")
        yc = 1 - (y + 0.5 * ph) / H
        fig.text((left - 0.15) / W, yc + 0.11 / H, PRETTY[k], ha="right", va="bottom",
                 fontsize=13)
        fig.text((left - 0.15) / W, yc + 0.03 / H,
                 f"{GROUP_LABEL[GROUP[k]]},  {fmt_value(k, truth[k])}",
                 ha="right", va="top", fontsize=7.3, color=INK2)
        fig.text((left - 0.15) / W, yc - 0.13 / H,
                 f"{100 * res['shift_share'][i]:.0f}% rigid shift",
                 ha="right", va="top", fontsize=7.3,
                 color=INK if res["shift_share"][i] >= 0.5 else INK2,
                 fontweight="bold" if res["shift_share"][i] >= 0.5 else "normal")

    # --- colour keys ----------------------------------------------------------------
    yb = 1 - (H - bottom + 0.32) / H
    cax = fig.add_axes([col_x(0) / W, yb, (3 * pw + 2 * gap) / W, 0.1 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_DIV, norm=div_norm), cax=cax,
                      orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["$-$max", "0", "+max"], fontsize=7.5)
    cb.set_label("change of the map per e-fold of the parameter  "
                 "(red: map value rises, blue: falls)", fontsize=7.8, color=INK2, labelpad=3)
    cb.outline.set_visible(False)
    cax2 = fig.add_axes([col_x(3) / W, yb, (2 * pw + gap) / W, 0.1 / H])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_SEQ, norm=seq_norm), cax=cax2,
                       orientation="horizontal", ticks=[0, 0.25, 1])
    cb2.ax.set_xticklabels(["0", "max/4", "max"], fontsize=7.5)
    cb2.set_label("share of the parameter's information, per-panel scale "
                  "(square-root stretch)", fontsize=7.8, color=INK2, labelpad=3)
    cb2.outline.set_visible(False)
    fig.text(col_x(0) / W, 1 - (H - 0.22) / H,
             f"Responses below {MC_SIGNIFICANCE:g} Monte Carlo standard errors are drawn as 0.  "
             f"Grey outline: detected footprint ({res['n_pix']} pixels).  "
             f"Field: x $\\pm${EXTENT_X:g} kpc along the major axis, z cropped to "
             f"$\\pm${zmax:.1f} kpc.",
             fontsize=7.3, color=INK2)

    fig.text(col_x(0) / W, 1 - 0.38 / H, title, fontsize=15, color=INK, va="baseline")
    fig.text(col_x(0) / W, 1 - 0.64 / H,
             f"{SAMPLER} sampler, Jacobian averaged over {res['K']} seeds x {N_PARTICLES} "
             f"tracers/component;  noise {NOISE_MASS_DEX:g} dex / {NOISE_V:g} km/s / "
             f"{NOISE_SIGMA:g} km/s per pixel;  prior $\\sigma_{{\\ln}}$ = {PRIOR_LOG_STD:g}",
             fontsize=9, color=INK2, va="baseline")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname


def figure_dominance(res, fname):
    """Per pixel: the parameter whose fractional change moves that pixel by the most
    noise-sigmas. Colour = its group; paler = influence shared with others."""
    labels, mask = res["labels"], res["mask"]
    G = GRID_SIZE
    dx, dz = 2 * EXTENT_X / G, 2 * EXTENT_Z / G
    xc = -EXTENT_X + dx * (np.arange(G) + 0.5)
    zc = -EXTENT_Z + dz * (np.arange(G) + 0.5)
    surf = np.array(to_rgb(SURFACE))
    zmax = crop_height(mask)

    pw = 6.4
    ph = pw * zmax / EXTENT_X
    left, cgap, right = 0.6, 0.5, 0.25
    top, rgap, bottom = 1.75, 0.85, 1.05
    W = left + 2 * pw + cgap + right
    H = top + 2 * ph + rgap + bottom
    fig = plt.figure(figsize=(W, H))

    def shade(group, share):
        t = 0.2 + 0.8 * share
        return t * np.array(to_rgb(GROUP_COLOR[group])) + (1 - t) * surf

    panels = [(0, "mass map  " + MAP_LABEL[0]), (1, "velocity map  " + MAP_LABEL[1]),
              (2, "dispersion map  " + MAP_LABEL[2]), (None, "all three maps, noise-weighted")]
    winners = set()
    for p, (k, title) in enumerate(panels):
        row, col = divmod(p, 2)
        x0 = left + col * (pw + cgap)
        y0 = top + row * (ph + rgap)
        ax = fig.add_axes([x0 / W, 1 - (y0 + ph) / H, pw / W, ph / H])
        dens = res["dens"].sum(0) if k is None else res["dens"][k]      # (G, G, P)
        total = dens.sum(-1)
        win = dens.argmax(-1)
        share = dens.max(-1) / np.maximum(total, 1e-300)
        rgb = np.broadcast_to(surf, (G, G, 3)).copy()
        for r in range(G):
            for c in range(G):
                if not mask[r, c]:
                    continue
                if total[r, c] <= 0:
                    rgb[r, c] = to_rgb(NEUTRAL)
                    continue
                name = labels[win[r, c]]
                winners.add(name)
                rgb[r, c] = shade(GROUP[name], share[r, c])
                ax.text(xc[c], zc[r], SHORT[name], ha="center", va="center", fontsize=8,
                        color=ink_on(rgb[r, c]))
        ax.imshow(rgb, origin="lower", extent=MAP_EXTENT, interpolation="nearest",
                  aspect="auto")
        footprint_outline(ax, mask)
        ax.set_xlim(-EXTENT_X, EXTENT_X)
        ax.set_ylim(-zmax, zmax)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=4)
        ax.tick_params(length=2, pad=1.5, labelsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("x [kpc]", fontsize=7.5, color=INK2, labelpad=1)
        ax.set_ylabel("z [kpc]", fontsize=7.5, color=INK2, labelpad=1)

    # Legend: the three groups, then a lightness key for the share.
    handles = [Patch(facecolor=GROUP_COLOR[g], label=GROUP_LABEL[g])
               for g in ("pot", "disk", "bulge")]
    handles.append(Patch(facecolor=NEUTRAL, label="no significant response"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left / W, 1 - 1.02 / H),
               ncol=4, frameon=False, fontsize=9, handlelength=1.4, borderaxespad=0)
    kx, kw = left + pw + cgap, 2.6
    kax = fig.add_axes([kx / W, 1 - 1.22 / H, kw / W, 0.13 / H])
    shares = np.linspace(0, 1, 256)
    kax.imshow(np.array([[shade("pot", s) for s in shares]]), extent=[0, 1, 0, 1],
               aspect="auto")
    kax.set_yticks([])
    kax.set_xticks([0.25, 0.5, 1.0])
    kax.set_xticklabels(["25%", "50%", "100%"], fontsize=7)
    kax.tick_params(length=2, pad=1)
    for s in kax.spines.values():
        s.set_visible(False)
    fig.text((kx + kw + 0.12) / W, 1 - 1.16 / H,
             "lightness: the labelled parameter's share\nof the pixel's total information",
             fontsize=7.8, color=INK2, va="center")

    key = "   ".join(f"{SHORT[k]} = {PRETTY[k]}" for k in DISPLAY if k in winners)
    fig.text(left / W, 1 - (H - 0.55) / H, "labels:  " + key, fontsize=9, color=INK2)
    fig.text(left / W, 1 - (H - 0.27) / H,
             "Dominant = largest information density in the pixel: response per unit "
             "ln$\\theta$ divided by the pixel noise, squared (summed over the maps in the "
             f"last panel).  {res['n_pix']} detected pixels; z cropped to $\\pm${zmax:.1f} kpc.",
             fontsize=8, color=INK2)
    fig.text(left / W, 1 - 0.45 / H, "Which parameter moves each pixel most", fontsize=15,
             color=INK, va="baseline")
    fig.text(left / W, 1 - 0.72 / H,
             f"per e-fold change of the parameter, at the twin experiment's truth"
             + (", galaxy centre refitted" if res["centre_free"] else "")
             + f";  {SAMPLER} sampler, {res['K']} seeds x {N_PARTICLES} tracers/component",
             fontsize=9.5, color=INK2, va="baseline")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname


def figure_summary(res, fname):
    """Marginal correlations, Cramer-Rao errors, eigen-directions."""
    labels = res["labels"]
    P = len(labels)
    ticks = [PRETTY[k] for k in labels]
    bounds = [j for j in range(1, P) if GROUP[labels[j]] != GROUP[labels[j - 1]]]
    fig = plt.figure(figsize=(18.5, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.34,
                          left=0.06, right=0.985, top=0.78, bottom=0.12)
    title_y = 0.865

    def panel_title(spec, s):
        fig.text(spec.get_position(fig).x0, title_y, s, fontsize=10.5, color=INK)

    # (a) marginal correlation matrix ---------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.set_anchor("NW")
    corr = res["corr"].copy()
    np.fill_diagonal(corr, np.nan)
    cmap = CMAP_DIV.copy()
    cmap.set_bad(SURFACE)
    ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(P))
    ax.set_xticklabels(ticks, rotation=90, fontsize=8)
    ax.set_yticks(range(P))
    ax.set_yticklabels(ticks, fontsize=8)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    for b in bounds:
        ax.axhline(b - 0.5, color=SURFACE, lw=2.2)
        ax.axvline(b - 0.5, color=SURFACE, lw=2.2)
    for a in range(P):
        for b in range(P):
            v = corr[a, b]
            if np.isfinite(v) and abs(v) >= 0.9:
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                        color=ink_on(CMAP_DIV((v + 1) / 2)[:3]))
    cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_DIV, norm=plt.Normalize(-1, 1)), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    panel_title(gs[0], "Marginal correlations  (labelled where |r| $\\geq$ 0.9)")

    # (b) Cramer-Rao errors ---------------------------------------------------------
    ax = fig.add_subplot(gs[1])
    y = np.arange(P)
    xmin = 0.5 * res["sig_cond"].min()
    xmax = 6.0 * PRIOR_LOG_STD
    band = PRIOR_LIMITED_FRAC * PRIOR_LOG_STD
    for j in np.where(res["prior_limited"])[0]:
        ax.axhspan(j - 0.5, j + 0.5, color=NEUTRAL, lw=0, zorder=0)
    ax.axvline(PRIOR_LOG_STD, color=INK3, lw=0.8, zorder=0)
    ax.text(PRIOR_LOG_STD * 1.08, -0.75, "prior width", fontsize=7.5, color=INK2,
            va="center")
    for j in y:
        ax.plot([res["sig_cond"][j], res["sig_marg"][j]], [j, j], color=BLUE_RAMP[1], lw=1.2,
                zorder=1)
    if np.all(np.isfinite(res["sig_marg_err"])):
        ax.errorbar(res["sig_marg"], y, xerr=res["sig_marg_err"], fmt="none", ecolor=INK2,
                    elinewidth=0.9, capsize=2, zorder=2)
    ax.plot(res["sig_cond"], y, "o", ms=6.5, mfc=SURFACE, mec=C_FISHER, mew=1.6, zorder=3,
            label="conditional (all other parameters known)")
    ax.plot(res["sig_marg"], y, "o", ms=7, color=C_FISHER, mec=SURFACE, mew=1.0, zorder=4,
            label="marginal (all parameters free)" + (", jackknife error bar"
                                                      if res["K"] > 1 else ""))
    if res["centre_free"]:
        ax.plot(res["sig_marg_fixed"], y, "|", ms=12, mew=1.8, color=INK, zorder=5,
                label="marginal with the galaxy centre fixed (as in the twin fits)")
    for j in y:
        ax.text(xmax / 1.06, j, "prior-limited" if res["prior_limited"][j]
                else fmt_frac(res["sig_marg"][j]), fontsize=7, color=INK2, va="center",
                ha="right")
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(P - 0.5, -1.3)
    ax.set_yticks(y)
    ax.set_yticklabels(ticks, fontsize=8)
    for b in bounds:
        ax.axhline(b - 0.5, color=INK3, lw=0.5)
    ax.set_xlabel(r"error on $\ln\theta$  ($\approx$ fractional error)", fontsize=8.5)
    ax.grid(axis="x", which="major", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=7.5,
              borderaxespad=0.3)
    panel_title(gs[1], "Cramer-Rao bound per parameter")

    # (c) eigen-directions ------------------------------------------------------------
    ax = fig.add_subplot(gs[2])
    sd = 1.0 / np.sqrt(np.maximum(res["lam"][::-1], 1e-300))     # stiff -> sloppy
    rank = np.arange(1, P + 1)
    top_clip = 300.0
    shown = np.minimum(sd, top_clip)
    constrained = sd <= band
    ax.axhspan(band, top_clip * 3, color=NEUTRAL, lw=0, zorder=0)
    ax.axhline(PRIOR_LOG_STD, color=INK3, lw=0.8)
    ax.text(P + 0.5, PRIOR_LOG_STD * 1.12, "prior width", fontsize=7.5, color=INK2,
            va="bottom", ha="right")
    ax.plot(rank, shown, color=BLUE_RAMP[1], lw=1.2, zorder=1)
    ax.plot(rank[constrained], shown[constrained], "o", ms=7, color=C_FISHER, mec=SURFACE,
            zorder=3, label="constrained by the data")
    ax.plot(rank[~constrained], shown[~constrained], "o", ms=7, mfc=SURFACE, mec=C_FISHER,
            mew=1.6, zorder=3, label="not constrained (prior-limited)")
    for j in np.where(sd > top_clip)[0]:
        ax.annotate(f"{sd[j]:.0e}", (rank[j], top_clip), xytext=(-9, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=6.5,
                    color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.5 * sd.min(), top_clip * 3)
    ax.set_xlim(0.3, P + 0.7)
    ax.set_xticks(rank)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("eigen-direction of F, stiffest to sloppiest", fontsize=8.5)
    ax.set_ylabel(r"error along it  $1/\sqrt{\lambda}$  [ln units]", fontsize=8.5)
    ax.grid(axis="y", which="major", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=7.5,
              borderaxespad=0.3)
    panel_title(gs[2], "Stiff and sloppy parameter combinations")
    lines = ["sloppiest directions (error along it: unit eigenvector)"]
    for s, v in sloppy_directions(res)[:4]:
        lines.append(f"{s:.2g}:  " + describe_direction(v, ticks, cut=0.25))
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=7.2, color=INK2,
            va="top", linespacing=1.55)

    fig.text(0.06, 0.95, "Fisher information of the edge-on maps", fontsize=15, color=INK)
    fig.text(0.06, 0.918,
             f"twin experiment truth;  {SAMPLER} sampler, {res['K']} seeds x {N_PARTICLES} "
             f"tracers/component;  "
             + ("galaxy centre marginalised;  " if res["centre_free"] else "")
             + f"{res['n_pix']} pixels above "
             + (f"{MASS_FLOOR_REL:g} x peak surface density" if MASS_FLOOR_REL > 0
                else f"{MASS_SOFTENING:g} $M_\\odot$")
             + f";  noise {NOISE_MASS_DEX:g} dex / {NOISE_V:g} km/s / {NOISE_SIGMA:g} km/s;  "
             f"prior $\\sigma_{{\\ln}}$ = {PRIOR_LOG_STD:g}",
             fontsize=9, color=INK2)
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname


def figure_vi_check(labels, sig_marg, sig_cond, limited, vi_std, fname):
    """Fisher bound of the VI run's exact setup against the VI posterior widths."""
    P = len(labels)
    y = np.arange(P)
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    fig.subplots_adjust(left=0.15, right=0.76, top=0.80, bottom=0.15)
    for j in y:
        ax.plot([min(sig_marg[j], vi_std[j]), max(sig_marg[j], vi_std[j])], [j, j],
                color=NEUTRAL, lw=3, zorder=0, solid_capstyle="round")
    ax.plot(sig_cond, y, "o", ms=6, mfc=SURFACE, mec=C_FISHER, mew=1.4, zorder=2,
            label="Fisher, conditional")
    ax.plot(sig_marg, y, "o", ms=7, color=C_FISHER, mec=SURFACE, zorder=3,
            label="Fisher, marginal")
    ax.plot(vi_std, y, "D", ms=6, color=C_VI, mec=SURFACE, zorder=4,
            label="VI posterior")
    ax.set_xscale("log")
    ax.set_ylim(P - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[k] for k in labels], fontsize=8.5)
    ax.set_xlabel(r"width on $\ln\theta$", fontsize=8.5)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(1.03, 1.01, "VI / Fisher", transform=ax.transAxes, fontsize=7.5, color=INK,
            va="bottom")
    for j in y:
        note = f"{vi_std[j] / sig_marg[j]:.2f}" + ("   prior-limited" if limited[j] else "")
        ax.text(1.03, j, note, transform=ax.get_yaxis_transform(), fontsize=7.5,
                color=INK2, va="center")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
              fontsize=7.5, handletextpad=0.3, columnspacing=1.0)
    fig.text(0.15, 0.945, "Fisher bound vs the VI posterior", fontsize=14, color=INK)
    fig.text(0.15, 0.915,
             f"the VI run's own setup: {VI_SAMPLER} sampler, {VI_N_PARTICLES} tracers, seed 0, "
             f"absolute floor {MASS_SOFTENING:g} $M_\\odot$, same noise and prior",
             fontsize=8.5, color=INK2)
    fig.text(0.15, 0.03,
             "prior-limited: the maps constrain the parameter only in combination with others, "
             "so its true posterior\nwidth is set by the prior -- along directions a finite VI "
             "run does not explore. Those VI widths are unreliable.",
             fontsize=7.5, color=INK2, linespacing=1.4)
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    mapper, weights_sha = load_mapper()
    pot, disk, bulge, source = load_truth()
    truth = {**pot, **disk, **bulge}
    params_log = params_to_log(pot, disk, bulge)
    names = [k for _, k in _flat_param_names(params_log)]

    # --------------------------------------------------------------------------
    # 1. Seed-averaged Jacobian at the truth
    # --------------------------------------------------------------------------
    print(f"\n=== 1. Jacobian of the maps at the truth ({source}) ===")
    print(f"    {SAMPLER} sampler, {N_SEEDS} seeds x {N_PARTICLES} tracers/component, "
          f"grid {GRID_SIZE}x{GRID_SIZE}, bandwidth {BANDWIDTH:.3f} kpc")
    jac = cached_jacobians(mapper, params_log, SAMPLER, N_PARTICLES, N_SEEDS, weights_sha)

    # --------------------------------------------------------------------------
    # 2. Fisher matrix and attention maps
    # --------------------------------------------------------------------------
    res = analyse(jac, names, MASS_FLOOR_REL)
    labels = res["labels"]

    # The frozen parameters should have no response at all.
    frozen_check = {}
    for k in FROZEN:
        Jk = np.abs(jac["J"][:, names.index(k)].mean(0))
        frozen_check[k] = [float(Jk[c][res["mask"]].max()) for c in range(3)]
    print(f"\n=== 2. Fisher information ({res['n_pix']}/{GRID_SIZE**2} pixels above "
          + (f"{MASS_FLOOR_REL:g} x peak" if MASS_FLOOR_REL > 0 else f"{MASS_SOFTENING:g} M_sun")
          + (", galaxy centre marginalised" if res["centre_free"] else "") + ") ===")
    lines = [f"{'parameter':16s} {'group':10s} {'value':>11s} {'sig_cond':>9s} "
             f"{'sig_marg':>9s} {'+-(jk)':>8s} {'ctr fixed':>9s} {'MC noise':>8s}  "
             f"{'info share mass/v/sigma':>23s} {'shift':>6s}  flag"]
    for j, k in enumerate(labels):
        sh = res["share"][:, j]
        lines.append(
            f"{k:16s} {GROUP_LABEL[GROUP[k]]:10s} {truth[k]:11.4g} {res['sig_cond'][j]:9.2e} "
            f"{res['sig_marg'][j]:9.2e} {res['sig_marg_err'][j]:8.1e} "
            f"{res['sig_marg_fixed'][j]:9.2e} {100 * res['noise_frac'][j]:7.1f}%  "
            f"{100 * sh[0]:7.1f} / {100 * sh[1]:5.1f} / {100 * sh[2]:5.1f} "
            f"{100 * res['shift_share'][j]:5.0f}%  "
            f"{'PRIOR-LIMITED' if res['prior_limited'][j] else ''}")
    table = "\n".join(lines)
    print(table)

    drift_lines = [f"      {'parameter':16s} {'d<x>':>15s} {'d<y>':>15s} {'d<z>':>15s}"]
    for j, k in enumerate(labels):
        d, e = res["drift"][j], res["drift_se"][j]
        drift_lines.append(f"      {k:16s} " + " ".join(f"{d[a]:+7.3f} +-{e[a]:5.3f}"
                                                        for a in range(3)))
    print("\n    surrogate centroid drift per e-fold [kpc] (zero for the true model):")
    print("\n".join(drift_lines))

    slop = sloppy_directions(res)
    print("\n    sloppiest eigen-directions of F (error along it, unit eigenvector):")
    slop_lines = [f"      {s:9.3g}   {describe_direction(v, labels)}" for s, v in slop]
    print("\n".join(slop_lines))

    corr = res["corr"]
    iu = np.triu_indices(len(labels), 1)
    strongest = np.argsort(-np.abs(corr[iu]))[:10]
    corr_lines = [f"      {labels[iu[0][o]]:16s} {labels[iu[1][o]]:16s} r = {corr[iu][o]:+.3f}"
                  for o in strongest]
    print("\n    strongest marginal correlations:")
    print("\n".join(corr_lines))
    print("\n    frozen parameters, max |response| over the footprint (dex, km/s, km/s):")
    for k, v in frozen_check.items():
        print(f"      {k:16s} {v[0]:.1e}  {v[1]:.1e}  {v[2]:.1e}")

    # --------------------------------------------------------------------------
    # 3. Cross-check against the VI posterior of vi_self_consistent.py
    # --------------------------------------------------------------------------
    vi_lines = []
    vi_fig = None
    if os.path.exists(VI_NPZ):
        print(f"\n=== 3. Cross-check: the VI run's setup ({VI_SAMPLER}, {VI_N_PARTICLES} "
              f"tracers, seed 0, absolute floor) ===")
        jac_tw = cached_jacobians(mapper, params_log, VI_SAMPLER, VI_N_PARTICLES, 1, weights_sha)
        tw = analyse(jac_tw, names, 0.0, centre_free=False)
        vi = np.load(VI_NPZ)
        vi_std = dict(zip([str(s) for s in vi["labels"]], vi["std_log"]))
        common = [k for k in tw["labels"] if k in vi_std]
        idx = [tw["labels"].index(k) for k in common]
        vi_arr = np.array([vi_std[k] for k in common])
        vi_lines.append(f"{'parameter':16s} {'Fisher marg':>11s} {'VI std':>9s} {'VI/Fisher':>9s}")
        for j, k in zip(idx, common):
            vi_lines.append(f"{k:16s} {tw['sig_marg'][j]:11.3e} {vi_std[k]:9.3e} "
                            f"{vi_std[k] / tw['sig_marg'][j]:9.2f}"
                            f"{'   prior-limited' if tw['prior_limited'][j] else ''}")
        print(f"    {tw['n_pix']} pixels (VI run: {int(vi['n_pix'])})")
        print("\n".join("    " + s for s in vi_lines))
        vi_fig = figure_vi_check(common, tw["sig_marg"][idx], tw["sig_cond"][idx],
                                 tw["prior_limited"][idx], vi_arr,
                                 os.path.join(FIGDIR, "fisher_vs_vi.png"))
    else:
        print(f"\n=== 3. (skipped: {os.path.basename(VI_NPZ)} not found -- run "
              f"vi_self_consistent.py for the VI cross-check) ===")

    # --------------------------------------------------------------------------
    # 4. Persist and plot
    # --------------------------------------------------------------------------
    tag = SAMPLER
    np.savez(
        os.path.join(FIGDIR, f"fisher_{tag}.npz"),
        labels=np.array(labels), groups=np.array([GROUP[k] for k in labels]),
        truth=np.array([truth[k] for k in labels]),
        F=res["F"], cov=res["cov"], corr=res["corr"], sig_marg=res["sig_marg"],
        sig_cond=res["sig_cond"], sig_marg_err=res["sig_marg_err"],
        sig_marg_centre_fixed=res["sig_marg_fixed"], shift_share=res["shift_share"],
        centroid_drift=res["drift"], centroid_drift_se=res["drift_se"],
        centre_free=res["centre_free"],
        noise_frac=res["noise_frac"], share=res["share"], eigval=res["lam"],
        eigvec=res["evec"], mask=res["mask"], response=res["resp"],
        response_se=res["resp_se"], info_density=res["dens"], marginal_attention=res["marg"],
        model_maps=res["model_maps"], noise=NOISE, prior_log_std=PRIOR_LOG_STD,
        mass_floor_rel=MASS_FLOOR_REL, n_seeds=res["K"], n_particles=N_PARTICLES,
    )
    with open(os.path.join(FIGDIR, f"fisher_{tag}_summary.txt"), "w") as fh:
        fh.write("Fisher information of the edge-on maps at the twin-experiment truth\n")
        fh.write(f"truth: {source};  sampler: {SAMPLER};  {res['K']} seeds x {N_PARTICLES} "
                 f"tracers/component;  grid {GRID_SIZE}x{GRID_SIZE}, bandwidth "
                 f"{BANDWIDTH:.3f} kpc\n")
        fh.write(f"footprint: {res['n_pix']} pixels above {res['floor']:.3g} M_sun "
                 f"({MASS_FLOOR_REL:g} x peak)\n" if MASS_FLOOR_REL > 0 else
                 f"footprint: {res['n_pix']} pixels above {MASS_SOFTENING:g} M_sun\n")
        fh.write(f"noise: {NOISE_MASS_DEX} dex, {NOISE_V} km/s, {NOISE_SIGMA} km/s per pixel; "
                 f"prior sigma_ln = {PRIOR_LOG_STD}\n\n")
        fh.write("sig_cond = 1/sqrt((F+P)_ii); sig_marg = sqrt((F+P)^-1_ii), +-(jk) its jackknife "
                 "error over seeds;\nctr fixed = sig_marg with the galaxy centre fixed instead "
                 "of marginalised; MC noise = shot-noise\nfraction of F_ii; info share = "
                 "fraction of F_ii from each map; shift = share of the response that is a\n"
                 "rigid shift of the galaxy\n\n")
        fh.write(f"galaxy centre: {'marginalised' if res['centre_free'] else 'fixed'}\n\n")
        fh.write(table + "\n\nsurrogate centroid drift per e-fold [kpc] (zero for the true "
                 "model):\n" + "\n".join(drift_lines) + "\n")
        fh.write("\nsloppiest eigen-directions (error along it, unit eigenvector):\n")
        fh.write("\n".join(slop_lines) + "\n\nstrongest marginal correlations:\n")
        fh.write("\n".join(corr_lines) + "\n\nfrozen parameters, max |response| "
                 "(dex, km/s, km/s):\n")
        for k, v in frozen_check.items():
            fh.write(f"  {k:16s} {v[0]:.1e}  {v[1]:.1e}  {v[2]:.1e}\n")
        if vi_lines:
            fh.write(f"\nVI cross-check ({VI_SAMPLER} sampler, {VI_N_PARTICLES} tracers, "
                     "seed 0, absolute floor):\n" + "\n".join(vi_lines) + "\n")

    figs = [figure_summary(res, os.path.join(FIGDIR, f"fisher_summary_{tag}.png"))]
    figs.append(figure_attention(
        res, truth, [k for k in labels if GROUP[k] == "pot"],
        os.path.join(FIGDIR, f"fisher_attention_potential_{tag}.png"),
        "Where the potential parameters act on the maps"))
    figs.append(figure_attention(
        res, truth, [k for k in labels if GROUP[k] != "pot"],
        os.path.join(FIGDIR, f"fisher_attention_df_{tag}.png"),
        "Where the distribution-function parameters act on the maps"))
    figs.append(figure_dominance(res, os.path.join(FIGDIR, f"fisher_dominant_{tag}.png")))
    if vi_fig:
        figs.append(vi_fig)
    print("\n    written to:")
    for f in figs + [os.path.join(FIGDIR, f"fisher_{tag}_summary.txt"),
                     os.path.join(FIGDIR, f"fisher_{tag}.npz")]:
        print(f"      {f}")
    return res


if __name__ == "__main__":
    main()
