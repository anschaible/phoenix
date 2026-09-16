"""
Variational inference on the self-consistent twin experiment.

This is the Bayesian companion to `self_consistent_optimization.py`. That script
produces a single best-fit point by gradient descent with KDE-bandwidth annealing;
this one takes that point estimate and asks the question a point estimate cannot
answer -- *how well is each parameter actually determined, and which parameters are
only determined in combination*. The answer is a posterior, and the figure that shows
it is a corner plot.

Run with:
    python notebooks/vi_self_consistent.py            # full run  (~20 min, CPU)
    QUICK=1 python notebooks/vi_self_consistent.py    # short run, for smoke-testing

Pipeline
--------
  1. Self-consistent ground truth. Reused from `self_consistent_truth.json` if it is
     present (that is what `self_consistent_optimization.py` writes), otherwise solved
     for here. Doing this first is what makes the physics term legitimate -- see that
     script's docstring.
  2. Mock observation from the self-consistent model.
  3. Point estimate: the SAME annealed fit as the theory example, from the same far-off
     start. Annealing the KDE bandwidth from 8 kpc down to the data resolution is what
     lets the fit cross the flat-gradient region between the far start and the truth.
     That stage ALONE does not reach the optimum, so it is followed by a refinement
     stage at fixed bandwidth and a smaller step -- without it the masses are biased
     by ~1-2% and VI then reports tight posteriors centred on the wrong point. See
     REFINE_STEPS.
  4. Variational inference, initialised at that MAP and evaluated at the FINAL annealed
     bandwidth. `fit_vi` maximizes the ELBO of a full-covariance Gaussian posterior over
     the log-parameters.
  5. Corner plot.

The annealed fit (step 3) is cached to `figures/VI_theory_map.json`, so re-running to
iterate on the posterior or the figures does not repeat it. `REFIT=1` forces a fresh
fit.

Why full-covariance and not mean-field
--------------------------------------
A mean-field (diagonal) posterior has no correlations by construction, so its corner
plot is a grid of axis-aligned blobs no matter what the real posterior looks like --
it cannot show a degeneracy, which is the main thing worth seeing here. `fit_vi`
defaults to a dense Cholesky factor, which can.

What sets the posterior width
-----------------------------
The mock is noiseless: the model reproduces it to chi2 ~ 2e-6 at the truth. The width
therefore comes entirely from the assumed per-pixel measurement errors below
(`NOISE_V`, `NOISE_SIGMA`, `NOISE_MASS_DEX`) folded through the local curvature.
Change those and every width scales with them; the *correlation structure* -- the part
the corner plot is for -- is essentially unaffected.

Note on the physics term: the fit in step 3 keeps the Poisson self-consistency
penalty exactly as the theory example has it, but the VI likelihood does not -- see
`VI_POISSON_TOL` below for why (a host memory limit, not a physical argument).

Outputs (written to notebooks/figures/):
  - VI_theory_corner.png            : corner plot, all 17 inferred parameters
  - VI_theory_corner_potential.png  : the 7 potential parameters, physical units
  - VI_theory_diagnostics.png       : ELBO trace + posterior widths
  - VI_theory_posterior.txt         : the posterior as a table
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

from phoenix.actions_to_phasespace.actions_to_phasespace_nn import PhoenixMapper
from phoenix.optimization.pipeline import (
    make_observation,
    make_self_consistent_truth,
    fit,
    fit_vi,
    DEFAULT_PARAM_BOUNDS,
)

# ==============================================================================
# CONFIGURATION  (kept identical to self_consistent_optimization.py where shared)
# ==============================================================================
QUICK = bool(os.environ.get("QUICK"))

SEED = 0
N_PARTICLES = 6_000
GRID_SIZE = 14
EXTENT_X, EXTENT_Z = 15.0, 10.0
OBS_BANDWIDTH = 0.25 * max(2 * EXTENT_X / GRID_SIZE, 2 * EXTENT_Z / GRID_SIZE)

POISSON_KWARGS = dict(grid_size=20, match_kernel=True, n_quad=3)
W_POISSON = 0.1
FROZEN = ("Sigma0", "N0_spheroid", "Rinit_for_Rc", "L0")

SC_STEPS = 60 if QUICK else 250
FIT_STEPS = 120 if QUICK else 700

# --- refinement stage --------------------------------------------------------
# The annealed fit alone does NOT reach the optimum, and the masses pay for it.
# Measured on this mock: the annealed fit stops at total loss 0.015102 while the
# TRUTH scores 0.004569, i.e. it leaves a factor ~3 on the table, and its masses are
# off by +1.93% (M_halo), +0.79% (M_disk), -0.98% (M_bulge). Running these extra
# steps with no annealing and a smaller step size reaches 0.004579 -- essentially the
# truth's own score -- and the mass errors collapse to -0.06%, +0.07%, +0.40%.
#
# The reason is a step-size mismatch, not a bad objective: annealing spends its budget
# travelling from the far-off start and arrives at the final bandwidth with
# `learning_rate` still 0.05, which is far larger than the ~0.001-0.003 log-space
# posterior width of the masses. Adam cannot resolve a basin that much narrower than
# its own step, so it rattles around inside it. Refining at 0.01 with the bandwidth
# fixed at the data resolution settles into it.
#
# This matters for the posterior too: VI characterises the region around wherever it
# is started, so starting it at the unrefined point produced tight (0.2-0.7%) mass
# posteriors centred 2-4 sigma away from the truth.
REFINE_STEPS = 100 if QUICK else 600
REFINE_LR = 0.01

# --- likelihood: assumed per-pixel measurement errors --------------------------
NOISE_V = 10.0          # km/s on v_rot
NOISE_SIGMA = 10.0      # km/s on sigma
NOISE_MASS_DEX = 0.05   # dex on the mass map

# --- VI -----------------------------------------------------------------------
# Physics term in the VI likelihood. `None` = data only.
#
# The point-estimate fit in step 3 keeps the Poisson term exactly as the theory
# example has it; only the VI stage drops it, and for a machine reason rather than a
# physical one. The penalty evaluates the density via the Laplacian of the potential
# under AD, which allocates a ~1.5 GB intermediate that does not shrink usefully with
# `grid_size`. This host runs with strict memory overcommit
# (`vm.overcommit_memory=2`) and its commit limit is nearly exhausted by other users,
# so any single allocation above ~1 GB is refused no matter how much RAM is free, and
# the ELBO step dies with a JAX OOM. Set this to `penalty_truth` on a machine with
# commit headroom to put self-consistency back in the posterior; expect it to tighten
# the potential parameters somewhat and to leave the DF parameters much as they are.
VI_POISSON_TOL = None

# n_mc is 2 rather than 4 deliberately. The ELBO step unrolls one model evaluation
# per Monte Carlo draw, and this host runs strict memory overcommit
# (`vm.overcommit_memory=2`) with only ~1 GB of commit headroom left by other users,
# so 4 unrolled draws intermittently fail to allocate no matter how much RAM is free.
# Twice as many steps at half the draws costs the same compute and converges to the
# same posterior, just with a slightly noisier trajectory.
VI_STEPS = 150 if QUICK else 1200
VI_MC = 2 if QUICK else 2
VI_LR = 0.05
VI_SEED = 0
N_POST = 40_000

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
TRUTH_JSON = os.path.join(HERE, "self_consistent_truth.json")
MAP_JSON = os.path.join(FIGDIR, "VI_theory_map.json")

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

# ==============================================================================
# STYLE
# ==============================================================================
# Sequential single-hue blue ramp for the posterior density (magnitude), and two
# categorical hues for the two point-valued overlays. Truth and MAP additionally
# differ in line style, so identity never rests on colour alone.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_POST = LinearSegmentedColormap.from_list("phoenix_post", BLUE_RAMP)
C_POST = "#2a78d6"
C_TRUTH = "#eb6834"
C_MAP = "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a887f"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK3, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.labelcolor": INK, "text.color": INK,
    "font.size": 9,
})

# Nicer axis names for the figures.
PRETTY = {
    "M_halo": r"$M_{\rm halo}$", "a_halo": r"$a_{\rm halo}$",
    "M_disk": r"$M_{\rm disk}$", "a_disk": r"$a_{\rm disk}$", "b_disk": r"$b_{\rm disk}$",
    "M_bulge": r"$M_{\rm bulge}$", "a_bulge": r"$a_{\rm bulge}$",
    "R0": r"$R_0$", "Rd": r"$R_d$", "RsigR": r"$R_{\sigma R}$", "RsigZ": r"$R_{\sigma z}$",
    "sigmaR0_R0": r"$\sigma_{R0}$", "sigmaz0_R0": r"$\sigma_{z0}$",
    "J0_spheroid": r"$J_{0,\rm sph}$", "Gamma_spheroid": r"$\Gamma_{\rm sph}$",
    "Beta_spheroid": r"$\beta_{\rm sph}$", "eta_spheroid": r"$\eta_{\rm sph}$",
}
GROUP_LABEL = {"pot": "potential", "disk": "disk DF", "bulge": "bulge DF"}


# ==============================================================================
# CORNER PLOT
# ==============================================================================
def corner_plot(fname, mu, cov, names, truth_log=None, map_log=None,
                units="log", title="", subtitle="", at_bound=None):
    """Corner plot of the Gaussian posterior N(mu, cov).

    The posterior is exactly Gaussian, so the 1D and 2D marginals are evaluated
    analytically rather than histogrammed from draws: no sampling noise, and the
    contours are the true 1/2/3-sigma credible regions instead of a binning artefact.

    `units="log"`   -- axes are the log-offset from the truth, ln(p) - ln(p_true).
                       Every panel is then exactly Gaussian and the panels are
                       comparable, which matters because the posterior widths here
                       span four orders of magnitude.
    `units="phys"`  -- axes are physical parameter values.
    """
    D = len(names)
    sd = np.sqrt(np.diag(cov))
    ref = np.asarray(truth_log) if truth_log is not None else mu

    # Panel ranges: +/-3.8 sigma about the mean, wide enough to always contain the
    # truth marker unless the fit is genuinely biased (which is worth seeing).
    if units == "log":
        centre = mu - ref
        t_mark = np.zeros(D)
        m_mark = None if map_log is None else np.asarray(map_log) - ref
    else:
        centre, t_mark = mu, ref
        m_mark = None if map_log is None else np.asarray(map_log)

    lo, hi = centre - 3.8 * sd, centre + 3.8 * sd
    if map_log is not None:                       # never clip an overlay out of frame
        lo, hi = np.minimum(lo, m_mark - 0.3 * sd), np.maximum(hi, m_mark + 0.3 * sd)
    lo, hi = np.minimum(lo, t_mark - 0.3 * sd), np.maximum(hi, t_mark + 0.3 * sd)

    # In physical units the parameters span 1e-2 to 1e12 while the posteriors are
    # fractions of a percent wide, so matplotlib falls back to a shared offset like
    # "1e10" tucked in the panel corner, which collides with neighbouring panels and
    # is unreadable at this grid size. Factor a power of ten out of each parameter
    # explicitly and put it in the axis label instead.
    if units == "phys":
        decade = np.floor(np.log10(np.abs(np.exp(mu))))
        decade = np.where(np.abs(decade) < 2, 0.0, decade)   # leave O(1) values alone
    else:
        decade = np.zeros(D)
    pow10 = 10.0 ** decade

    def show(vals, i):
        return np.exp(vals) / pow10[i] if units == "phys" else vals

    def axis_label(i):
        nm = PRETTY.get(names[i], names[i])
        if units == "phys" and decade[i] != 0:
            return nm + f"  [$10^{{{int(decade[i])}}}$]"
        return nm

    size = max(9.0, 1.15 * D)
    fig, axes = plt.subplots(D, D, figsize=(size, size))
    fig.subplots_adjust(wspace=0.06, hspace=0.06, left=0.075, right=0.985,
                        bottom=0.075, top=0.93 if title else 0.99)

    # 1 / 2 / 3-sigma levels of a 2D Gaussian, as fractions of the peak density.
    lv = np.exp(-0.5 * np.array([3.0, 2.0, 1.0]) ** 2)

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            g = np.linspace(lo[j], hi[j], 220)
            if i == j:
                # --- 1D marginal -------------------------------------------------
                pdf = np.exp(-0.5 * ((g - centre[i]) / sd[i]) ** 2)
                gx = show(g, i)
                ax.plot(gx, pdf, color=C_POST, lw=1.6)
                band = np.abs(g - centre[i]) <= sd[i]
                ax.fill_between(gx[band], 0, pdf[band], color=BLUE_RAMP[1], alpha=0.85,
                                lw=0)
                ax.set_ylim(0, 1.55)
                ax.set_yticks([])
                # median +/- 1 sigma, in the panel, as the direct label
                # Annotate INSIDE the panel. As a title it overflows into the
                # neighbouring column -- the grid is 17 panels wide and the strings
                # are longer than one panel.
                pinned = at_bound is not None and at_bound[i]
                ax.text(0.04, 0.93, PRETTY.get(names[i], names[i]),
                        transform=ax.transAxes, fontsize=8, color=INK,
                        ha="left", va="top")
                # top-RIGHT, so it cannot collide with the density peak
                ax.text(0.96, 0.93,
                        f"$\\sigma_{{\\ln}}$={sd[i]:.2g}" + ("\nbound!" if pinned else ""),
                        transform=ax.transAxes, fontsize=6.5,
                        color=C_TRUTH if pinned else INK2, ha="right", va="top")
            else:
                # --- 2D marginal -------------------------------------------------
                gy = np.linspace(lo[i], hi[i], 220)
                s2 = cov[np.ix_([j, i], [j, i])]
                P = np.linalg.inv(s2)
                dx = g[None, :] - centre[j]
                dy = gy[:, None] - centre[i]
                q = (P[0, 0] * dx**2 + 2 * P[0, 1] * dx * dy + P[1, 1] * dy**2)
                dens = np.exp(-0.5 * q)
                ax.contourf(show(g, j), show(gy, i), dens,
                            levels=np.concatenate([lv, [1.0]]),
                            colors=[BLUE_RAMP[1], BLUE_RAMP[3], BLUE_RAMP[5]])
                ax.contour(show(g, j), show(gy, i), dens, levels=lv,
                           colors=SURFACE, linewidths=0.5)
                ax.set_ylim(show(np.array([lo[i], hi[i]]), i))

            ax.set_xlim(show(np.array([lo[j], hi[j]]), j))

            # --- overlays: truth (solid) and MAP (dashed) ------------------------
            for mark, col, ls in ((t_mark, C_TRUTH, "-"), (m_mark, C_MAP, "--")):
                if mark is None:
                    continue
                ax.axvline(show(np.array([mark[j]]), j)[0], color=col, lw=1.1, ls=ls)
                if i != j:
                    ax.axhline(show(np.array([mark[i]]), i)[0], color=col, lw=1.1, ls=ls)
                    ax.plot(show(np.array([mark[j]]), j), show(np.array([mark[i]]), i),
                            "o", color=col, ms=4.5, mec=SURFACE, mew=0.8, zorder=5)

            ax.tick_params(length=2.2, width=0.5, pad=1.5)
            if i < D - 1:
                ax.set_xticklabels([])
            else:
                ax.ticklabel_format(useOffset=False, style="plain", axis="x")
                ax.set_xlabel(axis_label(j), fontsize=8)
                for lab in ax.get_xticklabels():
                    lab.set_rotation(45)
                    lab.set_ha("right")
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.ticklabel_format(useOffset=False, style="plain", axis="y")
                ax.set_ylabel(axis_label(i), fontsize=8)
            ax.locator_params(nbins=3)

    handles = [
        Line2D([], [], color=BLUE_RAMP[3], lw=7,
               label="VI posterior (1/2/3$\\sigma$)"),
        Line2D([], [], color=C_TRUTH, lw=1.6, ls="-", marker="o", ms=5,
               label="ground truth"),
    ]
    if map_log is not None:
        handles.append(Line2D([], [], color=C_MAP, lw=1.6, ls="--", marker="o", ms=5,
                              label="MAP (annealed + refined)"))
    if at_bound is not None and np.any(at_bound):
        handles.append(Line2D([], [], color=SURFACE, lw=0,
                              label="'bound!' = pinned on a physical bound;\n"
                                    "          width not trustworthy"))
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=10,
               bbox_to_anchor=(0.985, 0.985 if title else 1.0))
    if title:
        fig.text(0.075, 0.968, title, fontsize=15, color=INK, ha="left", va="bottom")
        fig.text(0.075, 0.952, subtitle, fontsize=9.5, color=INK2, ha="left", va="bottom")
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname


def _as_floats(r):
    return {
        "pot_params": {k: float(v) for k, v in r["pot_params"].items()},
        "disk_df_params": {k: float(v) for k, v in r["disk_df_params"].items()},
        "bulge_df_params": {k: float(v) for k, v in r["bulge_df_params"].items()},
        "history": {"loss": [float(v) for v in r["history"]["loss"]]},
    }


def _run_fit(mapper, obs_maps, pot_init, disk_init, bulge_init, common):
    """The theory example's annealed fit, then the refinement stage (see REFINE_STEPS).

    Returns (annealed, refined) so the cost of the annealed stage alone stays visible
    -- that difference is the whole point of the refinement.
    """
    common_fit = dict(
        **common, poisson_kwargs=POISSON_KWARGS,
        loss_weights=(1.0, 1.0, 1.0, W_POISSON),
        obs_bandwidth=OBS_BANDWIDTH, frozen_params=FROZEN,
        param_bounds=DEFAULT_PARAM_BOUNDS,
    )
    r = fit(mapper, obs_maps, pot_init, disk_init, bulge_init,
            learning_rate=0.05, n_steps=FIT_STEPS,
            anneal_bandwidth=(8.0, OBS_BANDWIDTH), **common_fit)
    annealed = _as_floats(r)

    # Hold the bandwidth at the annealed endpoint: this is pure refinement of the
    # minimum the annealed run stopped short of, against the SAME objective.
    #
    # `anneal_bandwidth=(h, h)` rather than `None`. Passing None does NOT mean "keep
    # the final bandwidth" -- it makes `fit` pass `soft_bin_h=None` to `bin_maps`,
    # which falls back to its own default of 0.25*pixel AND disables the
    # `obs_bandwidth` blur matching. Here those happen to coincide (OBS_BANDWIDTH is
    # *defined* as 0.25*pixel), but they do not in general: for the GECKOS fits
    # FINAL_H is 0.15 while 0.25*pixel is 0.075, so `None` would silently refine
    # against a twice-sharper objective than the one being refined.
    r2 = fit(mapper, obs_maps, r["pot_params"], r["disk_df_params"], r["bulge_df_params"],
             learning_rate=REFINE_LR, n_steps=REFINE_STEPS,
             anneal_bandwidth=(OBS_BANDWIDTH, OBS_BANDWIDTH), **common_fit)
    return annealed, _as_floats(r2)




# ==============================================================================
# MAIN
# ==============================================================================
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    mapper = PhoenixMapper()

    common = dict(
        N_disk=N_PARTICLES, N_bulge=N_PARTICLES,
        grid_size=GRID_SIZE, extent_x=EXTENT_X, extent_z=EXTENT_Z,
        prng_seed=SEED,
    )

    # --------------------------------------------------------------------------
    # 1. Self-consistent ground truth
    # --------------------------------------------------------------------------
    if os.path.exists(TRUTH_JSON):
        print(f"\n=== 1. Reusing the self-consistent truth from {os.path.basename(TRUTH_JSON)} ===")
        saved = json.load(open(TRUTH_JSON))
        pot_true = saved["pot_params"]
        disk_true = saved["disk_df_params"]
        bulge_true = saved["bulge_df_params"]
        penalty_truth = saved["penalty_final"]
    else:
        print("\n=== 1. Making the ground truth self-consistent ===")
        sc = make_self_consistent_truth(
            mapper, POT_NOMINAL, DISK_NOMINAL, BULGE_NOMINAL,
            tune_params=("M_disk", "a_disk", "b_disk", "M_bulge", "a_bulge"),
            N_disk=N_PARTICLES, N_bulge=N_PARTICLES, prng_seed=SEED,
            learning_rate=0.02, n_steps=SC_STEPS, poisson_kwargs=POISSON_KWARGS,
        )
        pot_true, disk_true, bulge_true = (sc["pot_params"], sc["disk_df_params"],
                                           sc["bulge_df_params"])
        penalty_truth = sc["penalty_final"]
    truth = {**pot_true, **disk_true, **bulge_true}
    print(f"    Poisson residual at truth: {penalty_truth:.4f}")

    # --------------------------------------------------------------------------
    # 2. Mock observation
    # --------------------------------------------------------------------------
    print("\n=== 2. Mock observation from the self-consistent model ===")
    obs_maps = make_observation(mapper, pot_true, disk_true, bulge_true, **common)
    n_pix = int((np.array(obs_maps["mass"]) > 1e-3).sum())
    print(f"    {n_pix}/{GRID_SIZE**2} detected pixels at bandwidth {OBS_BANDWIDTH:.3f} kpc")

    # --------------------------------------------------------------------------
    # 3. Point estimate: the annealed fit of the theory example
    # --------------------------------------------------------------------------
    pot_init = {k: v * 5.3 for k, v in pot_true.items()}
    disk_init = {k: v * 2.5 for k, v in disk_true.items()}
    bulge_init = {k: v * 0.1 for k, v in bulge_true.items()}

    print(f"\n=== 3. Annealed fit ({FIT_STEPS} steps) + refinement ({REFINE_STEPS} steps) ===")
    # The fit is the expensive half of this script and does not depend on any VI
    # setting, so cache it: iterating on the posterior or the figures should not pay
    # for 700 annealed gradient steps again. REFIT=1 forces a fresh fit.
    key = {"fit_steps": FIT_STEPS, "refine_steps": REFINE_STEPS, "refine_lr": REFINE_LR}
    cached = None
    if os.path.exists(MAP_JSON) and not os.environ.get("REFIT"):
        c = json.load(open(MAP_JSON))
        if c.get("key") == key:
            print(f"    reusing the cached MAP from {os.path.basename(MAP_JSON)}"
                  f"  (REFIT=1 to redo it)")
            cached = c

    if cached is None:
        annealed, res = _run_fit(mapper, obs_maps, pot_init, disk_init, bulge_init, common)
        with open(MAP_JSON, "w") as fh:
            json.dump({"key": key, "annealed": annealed, "res": res}, fh)
    else:
        annealed, res = cached["annealed"], cached["res"]

    def _errs(r):
        p = {**r["pot_params"], **r["disk_df_params"], **r["bulge_df_params"]}
        return {k: abs(float(p[k]) - truth[k]) / abs(truth[k]) * 100
                for k in truth if k not in FROZEN}

    map_params = {**res["pot_params"], **res["disk_df_params"], **res["bulge_df_params"]}
    map_params = {k: float(v) for k, v in map_params.items()}
    e_a, e_r = _errs(annealed), _errs(res)
    print(f"    annealed  ({FIT_STEPS:4d} steps)  loss {annealed['history']['loss'][0]:8.3f}"
          f" -> {annealed['history']['loss'][-1]:.6f}   median err {np.median(list(e_a.values())):5.1f}%")
    print(f"    refined   ({REFINE_STEPS:4d} steps)  loss {res['history']['loss'][0]:8.6f}"
          f" -> {res['history']['loss'][-1]:.6f}   median err {np.median(list(e_r.values())):5.1f}%")
    print(f"    {'mass':10s} {'truth':>12s} {'annealed':>12s} {'refined':>12s}   "
          f"{'annealed':>9s} {'refined':>9s}")
    for k in ("M_halo", "M_disk", "M_bulge"):
        a = float(annealed["pot_params"][k]); r_ = map_params[k]
        print(f"    {k:10s} {truth[k]:12.4e} {a:12.4e} {r_:12.4e}   "
              f"{(a/truth[k]-1)*100:+8.2f}% {(r_/truth[k]-1)*100:+8.2f}%")

    # --------------------------------------------------------------------------
    # 4. Variational inference around the MAP
    # --------------------------------------------------------------------------
    print(f"\n=== 4. Variational inference ({VI_STEPS} steps, {VI_MC} MC draws/step) ===")
    print(f"    full-covariance Gaussian posterior; assumed noise: "
          f"v {NOISE_V:g} km/s, sigma {NOISE_SIGMA:g} km/s, mass {NOISE_MASS_DEX:g} dex")
    print(f"    likelihood: mass + v_rot + sigma"
          + (f" + Poisson (tol {VI_POISSON_TOL:.4g})" if VI_POISSON_TOL
             else "  (Poisson term OFF -- see VI_POISSON_TOL)"))
    vi = fit_vi(
        mapper, obs_maps,
        res["pot_params"], res["disk_df_params"], res["bulge_df_params"],
        **common,
        soft_bin_h=OBS_BANDWIDTH,          # the fit's FINAL annealed bandwidth
        noise_v=NOISE_V, noise_sigma=NOISE_SIGMA, noise_mass_dex=NOISE_MASS_DEX,
        poisson_tol=VI_POISSON_TOL,
        poisson_kwargs=POISSON_KWARGS,
        frozen_params=FROZEN,
        full_rank=True, precondition=True,
        n_mc=VI_MC, n_steps=VI_STEPS, learning_rate=VI_LR, vi_seed=VI_SEED,
        param_bounds=DEFAULT_PARAM_BOUNDS,
        n_posterior_samples=N_POST,
        progress_every=max(VI_STEPS // 8, 1),
    )

    labels = vi["labels"]
    mu, cov, sd = vi["mu_log"], vi["cov_log"], vi["std_log"]
    truth_log = np.array([np.log(truth[k]) for k in labels])
    map_log = np.array([np.log(map_params[k]) for k in labels])
    pull = (mu - truth_log) / sd

    # --------------------------------------------------------------------------
    # 5. Report
    # --------------------------------------------------------------------------
    at_bound = np.asarray(vi["at_bound"])
    lines = []
    lines.append(f"{'parameter':16s} {'group':10s} {'truth':>12s} {'MAP':>12s} "
                 f"{'post.median':>12s} {'-1sig':>12s} {'+1sig':>12s} "
                 f"{'sig_ln':>8s} {'pull':>7s}  flag")
    for j, k in enumerate(labels):
        grp = GROUP_LABEL[vi["names"][j][0]]
        lines.append(
            f"{k:16s} {grp:10s} {truth[k]:12.4g} {map_params[k]:12.4g} "
            f"{np.exp(mu[j]):12.4g} {np.exp(mu[j]-sd[j]):12.4g} "
            f"{np.exp(mu[j]+sd[j]):12.4g} {sd[j]:8.3g} {pull[j]:7.2f}"
            f"  {'AT BOUND' if at_bound[j] else ''}")
    table = "\n".join(lines)
    print("\n=== 5. Posterior ===")
    print(table)
    ok = ~at_bound
    print(f"\n    truth inside 1 sigma: {int((np.abs(pull[ok]) < 1).sum())}/{int(ok.sum())}"
          f"   inside 2 sigma: {int((np.abs(pull[ok]) < 2).sum())}/{int(ok.sum())}"
          f"   (excluding parameters pinned on a bound)")
    if at_bound.any():
        print("    on a physical bound, posterior width NOT trustworthy (the true"
              " posterior there is truncated): "
              + ", ".join(np.array(labels)[at_bound]))
    print(f"    frozen (not inferred): {', '.join(FROZEN)}")

    corr = vi["corr_log"]
    iu = np.triu_indices(len(labels), 1)
    order = np.argsort(-np.abs(corr[iu]))[:8]
    print("\n    strongest posterior correlations:")
    for o in order:
        a, b = iu[0][o], iu[1][o]
        print(f"      {labels[a]:16s} {labels[b]:16s} r = {corr[a, b]:+.2f}")

    with open(os.path.join(FIGDIR, "VI_theory_posterior.txt"), "w") as fh:
        fh.write("Variational posterior, self-consistent twin experiment\n")
        fh.write(f"assumed noise: v_rot {NOISE_V} km/s, sigma {NOISE_SIGMA} km/s, "
                 f"mass {NOISE_MASS_DEX} dex; {n_pix} pixels\n")
        fh.write("posterior family: full-covariance Gaussian in log-parameters\n")
        fh.write("likelihood: mass + v_rot + sigma"
                 + (f" + Poisson (tol {VI_POISSON_TOL:.4g})\n\n" if VI_POISSON_TOL
                    else "  (Poisson self-consistency term OFF)\n\n"))
        fh.write(table + "\n\n")
        fh.write("frozen (zero gradient, not inferred): " + ", ".join(FROZEN) + "\n")
        if at_bound.any():
            fh.write("AT BOUND: sits on a physical bound, so the true posterior is "
                     "truncated there and\n          the reported (untruncated) "
                     "Gaussian width is not trustworthy.\n")

    # --------------------------------------------------------------------------
    # 6. Persist the posterior, so the figures can be redrawn (or new ones made)
    #    without paying for the fit and the ELBO optimization again.
    # --------------------------------------------------------------------------
    np.savez(
        os.path.join(FIGDIR, "VI_theory_posterior.npz"),
        labels=np.array(labels), groups=np.array([g for g, _ in vi["names"]]),
        mu_log=mu, cov_log=cov, std_log=sd, corr_log=vi["corr_log"],
        truth_log=truth_log, map_log=map_log, at_bound=at_bound,
        precond_scale=vi["precond_scale"], elbo_hist=np.array(vi["elbo_hist"]),
        samples_log=vi["samples_log"],
        noise=np.array([NOISE_V, NOISE_SIGMA, NOISE_MASS_DEX]), n_pix=n_pix,
    )

    # --------------------------------------------------------------------------
    # 7. Figures
    # --------------------------------------------------------------------------
    f1 = corner_plot(
        os.path.join(FIGDIR, "VI_theory_corner.png"),
        mu, cov, labels, truth_log=truth_log, map_log=map_log, units="log",
        at_bound=at_bound,
        title="Variational posterior of the self-consistent twin experiment",
        subtitle="axes: log-offset from the true value, $\\ln p - \\ln p_{\\rm true}$ "
                 "(the posterior is exactly Gaussian in these coordinates); "
                 f"{len(labels)} inferred parameters, {len(FROZEN)} frozen",
    )

    pot_keys = [k for k in labels if k in pot_true]
    pj = [labels.index(k) for k in pot_keys]
    f2 = corner_plot(
        os.path.join(FIGDIR, "VI_theory_corner_potential.png"),
        mu[pj], cov[np.ix_(pj, pj)], pot_keys,
        truth_log=truth_log[pj], map_log=map_log[pj], units="phys",
        at_bound=at_bound[pj],
        title="Potential parameters, physical units",
        subtitle="the same posterior, restricted to the 7 potential parameters",
    )

    # --- diagnostics ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    e = np.array(vi["elbo_hist"])
    ax[0].plot(e, color=C_POST, lw=1.0)
    ax[0].set_xlabel("VI iteration")
    ax[0].set_ylabel("ELBO")
    # A single unlucky MC draw can score orders of magnitude below the rest and then
    # sets the whole y-range, hiding the convergence it is supposed to show. Clip to
    # a robust range and say how many points fall outside it.
    lo_e = np.percentile(e, 2)
    span = np.max(e) - lo_e
    ax[0].set_ylim(lo_e - 0.15 * span, np.max(e) + 0.15 * span)
    n_out = int((e < lo_e - 0.15 * span).sum())
    ax[0].set_title("ELBO convergence"
                    + (f"  ({n_out} outlying draws below axis)" if n_out else ""),
                    color=INK2, fontsize=10)
    ax[0].axvspan(len(e) * 0.7, len(e), color=BLUE_RAMP[0], alpha=0.7, lw=0)
    ax[0].text(len(e) * 0.85, ax[0].get_ylim()[0], " averaged", fontsize=8,
               color=INK2, va="bottom", ha="center")
    ax[0].grid(alpha=0.25, lw=0.5, color=INK3)
    ax[0].set_axisbelow(True)

    o = np.argsort(sd)
    y = np.arange(len(labels))
    ax[1].barh(y, sd[o], color=C_POST, height=0.62)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([PRETTY.get(labels[i], labels[i]) for i in o], fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"posterior width  $\sigma_{\ln p}$  (~ fractional error)")
    ax[1].set_title("How well each parameter is determined", color=INK2, fontsize=10)
    ax[1].grid(alpha=0.25, lw=0.5, color=INK3, axis="x")
    ax[1].set_axisbelow(True)
    for k, i in enumerate(o):
        ax[1].text(sd[i] * 1.15, k, f"{sd[i]:.1e}", va="center", fontsize=7, color=INK2)
    ax[1].set_xlim(right=sd.max() * 4)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    f3 = os.path.join(FIGDIR, "VI_theory_diagnostics.png")
    fig.savefig(f3, dpi=140)
    plt.close(fig)

    print(f"\n    figures written to:\n      {f1}\n      {f2}\n      {f3}")
    print(f"      {os.path.join(FIGDIR, 'VI_theory_posterior.txt')}")
    print(f"      {os.path.join(FIGDIR, 'VI_theory_posterior.npz')}   "
          f"(posterior, for redrawing without re-running)")
    return vi, res


if __name__ == "__main__":
    main()
