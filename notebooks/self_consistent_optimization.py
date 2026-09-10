"""
Self-consistent twin experiment: fit a galaxy WITH the Poisson physics term switched on.

Run with:
    python notebooks/self_consistent_optimization.py            # full run
    QUICK=1 python notebooks/self_consistent_optimization.py    # short run, for smoke-testing

Background
----------
Picking `pot_params` and the DF parameters independently does not give a dynamically
self-consistent galaxy: the density of the DF-sampled tracers does not reproduce the
density that sources the potential. Measured on the original notebook setup, the
Poisson residual at the "true" parameters was large, so the truth was NOT a minimum of
the objective and the self-consistency term actively pulled the fit away from it (the
fit could reach a *more* self-consistent state than the true galaxy itself).

This script therefore does things in the order that makes the physics term legitimate:

  1. Solve for a self-consistent ground truth (adjust the baryonic potential until the
     DF's density reproduces the density sourcing it).
  2. Generate the mock observation from THAT self-consistent model.
  3. Fit from a far-off start with the Poisson term switched ON.

Three further points matter for the fit itself and are all applied below:

  * `obs_bandwidth` -- during bandwidth annealing the observation is blurred to the
    model's current resolution. Without it only the model is smoothed, the objective is
    no longer minimized at the truth, and the wide-bandwidth stages drag the fit away.
  * `frozen_params` -- `Sigma0` and `N0_spheroid` cancel exactly in the mass-normalized
    tracer weights, and `L0`/`Rinit_for_Rc` do not affect these maps. All four have
    (numerically) zero gradient, so fitting them only lets them drift.
  * The same `POISSON_KWARGS` are used for the self-consistency solve and for the fit.
    Self-consistency is only defined relative to a particular penalty definition, so
    tuning against one and fitting against another would reintroduce the bias.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # write figures to disk; no interactive display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from phoenix.actions_to_phasespace.actions_to_phasespace_nn import PhoenixMapper
from phoenix.optimization.observables import generate_edge_on_maps
from phoenix.optimization.pipeline import (
    make_observation,
    make_self_consistent_truth,
    fit,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
QUICK = bool(os.environ.get("QUICK"))

SEED = 0
N_PARTICLES = 6_000
GRID_SIZE = 14
EXTENT_X, EXTENT_Z = 15.0, 10.0

# The bandwidth `make_observation`/`bin_maps` use by default. The annealing must not go
# below this: the data carries no information on finer scales.
OBS_BANDWIDTH = 0.25 * max(2 * EXTENT_X / GRID_SIZE, 2 * EXTENT_Z / GRID_SIZE)

# One shared definition of the self-consistency penalty (see module docstring).
# `match_kernel=True` convolves the analytic density with the same kernel used for the
# tracer KDE, so both sides are compared at matched resolution -- it costs almost
# nothing here because the particle KDE dominates the runtime.
POISSON_KWARGS = dict(grid_size=20, match_kernel=True, n_quad=3)

# Weight of the physics term in the fit. Now that the mock is self-consistent this can
# be non-zero without biasing the solution.
W_POISSON = 0.1

# Structurally unconstrained parameters (zero gradient) -- held fixed.
FROZEN = ("Sigma0", "N0_spheroid", "Rinit_for_Rc", "L0")

SC_STEPS = 60 if QUICK else 250        # self-consistency solve
FIT_STEPS = 120 if QUICK else 700      # main fit
RUN_REFERENCE = True                   # also fit with the physics term off, to compare

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Nominal (not yet self-consistent) parameters used as the starting point for step 1.
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
# HELPERS
# ==============================================================================
def recovery(res, truth):
    """Relative parameter errors, split into the parameters the data can and cannot
    constrain (the frozen ones are structurally unrecoverable, so mixing them into a
    single median makes any result look worse than it is)."""
    fitted = {**res["pot_params"], **res["disk_df_params"], **res["bulge_df_params"]}
    errs = {k: abs(float(fitted[k]) - truth[k]) / abs(truth[k]) * 100 for k in truth}
    constrained = {k: v for k, v in errs.items() if k not in FROZEN}
    return errs, constrained


def summarize(tag, res, truth):
    errs, con = recovery(res, truth)
    loss = np.array(res["history"]["loss"])
    print(
        f"  {tag:32s} loss {loss[0]:8.3f} -> {loss[-1]:7.4f}   "
        f"poisson_end={res['history']['poisson_penalty'][-1]:.4f}   "
        f"median err {np.median(list(con.values())):5.1f}%   "
        f"within 20%: {sum(v < 20 for v in con.values())}/{len(con)}"
        + ("   [NaN present!]" if np.isnan(loss).any() else "")
    )
    return errs, con


def plot_row(axes, maps, mask, title, vmax_mass):
    extent = [-EXTENT_X, EXTENT_X, -EXTENT_Z, EXTENT_Z]
    m = np.where(mask, np.array(maps["mass"]), np.nan)
    v = np.where(mask, np.array(maps["v_rot"]), np.nan)
    s = np.where(mask, np.array(maps["sigma"]), np.nan)
    im0 = axes[0].imshow(m, origin="lower", extent=extent, cmap="magma",
                         norm=LogNorm(vmin=max(vmax_mass * 1e-4, 1e3), vmax=vmax_mass),
                         aspect="auto")
    axes[0].set_title(f"{title}: mass")
    im1 = axes[1].imshow(v, origin="lower", extent=extent, cmap="seismic",
                         vmin=-220, vmax=220, aspect="auto")
    axes[1].set_title(f"{title}: v_rot")
    im2 = axes[2].imshow(s, origin="lower", extent=extent, cmap="viridis",
                         vmin=0, vmax=150, aspect="auto")
    axes[2].set_title(f"{title}: sigma")
    return im0, im1, im2


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
    # 1. Solve for a self-consistent ground truth
    # --------------------------------------------------------------------------
    print("\n=== 1. Making the ground truth self-consistent ===")
    print(f"    tuning the baryonic potential; penalty definition: {POISSON_KWARGS}")
    sc = make_self_consistent_truth(
        mapper, POT_NOMINAL, DISK_NOMINAL, BULGE_NOMINAL,
        tune_params=("M_disk", "a_disk", "b_disk", "M_bulge", "a_bulge"),
        N_disk=N_PARTICLES, N_bulge=N_PARTICLES, prng_seed=SEED,
        learning_rate=0.02, n_steps=SC_STEPS,
        poisson_kwargs=POISSON_KWARGS,
    )
    pot_true = sc["pot_params"]
    disk_true = sc["disk_df_params"]
    bulge_true = sc["bulge_df_params"]
    print(f"    Poisson residual at truth: {sc['penalty_initial']:.4f} "
          f"-> {sc['penalty_final']:.4f}")
    print("    self-consistent baryonic parameters (vs nominal):")
    for k in ("M_disk", "a_disk", "b_disk", "M_bulge", "a_bulge"):
        print(f"      {k:9s} {POT_NOMINAL[k]:12.4g}  ->  {pot_true[k]:12.4g}")

    truth = {**pot_true, **disk_true, **bulge_true}

    # --------------------------------------------------------------------------
    # 2. Mock observation from the self-consistent model
    # --------------------------------------------------------------------------
    print("\n=== 2. Generating the mock observation from the self-consistent model ===")
    obs_maps = make_observation(mapper, pot_true, disk_true, bulge_true, **common)
    mass_mask = np.array(obs_maps["mass"]) > 1e-3
    vmax_mass = float(np.nanmax(np.array(obs_maps["mass"])))
    print(f"    observation binned at bandwidth {OBS_BANDWIDTH:.3f} kpc, "
          f"{int(mass_mask.sum())}/{GRID_SIZE**2} detected pixels")

    # --------------------------------------------------------------------------
    # 3. Fit from a far-off start, with the physics term ON
    # --------------------------------------------------------------------------
    pot_init = {k: v * 5.3 for k, v in pot_true.items()}
    disk_init = {k: v * 2.5 for k, v in disk_true.items()}
    bulge_init = {k: v * 0.1 for k, v in bulge_true.items()}
    init_maps = generate_edge_on_maps(mapper, pot_init, disk_init, bulge_init, **common)

    fit_kwargs = dict(
        **common,
        poisson_kwargs=POISSON_KWARGS,
        learning_rate=0.05,
        n_steps=FIT_STEPS,
        anneal_bandwidth=(8.0, OBS_BANDWIDTH),   # never anneal below the data resolution
        obs_bandwidth=OBS_BANDWIDTH,             # blur the data to match the model
        frozen_params=FROZEN,
    )

    print(f"\n=== 3. Fitting from a far-off start ({FIT_STEPS} steps) ===")
    res = fit(mapper, obs_maps, pot_init, disk_init, bulge_init,
              loss_weights=(1.0, 1.0, 1.0, W_POISSON), **fit_kwargs)
    _, con = summarize(f"physics ON (w_poisson={W_POISSON})", res, truth)

    res_ref = None
    if RUN_REFERENCE:
        res_ref = fit(mapper, obs_maps, pot_init, disk_init, bulge_init,
                      loss_weights=(1.0, 1.0, 1.0, 0.0), **fit_kwargs)
        summarize("physics OFF (reference)", res_ref, truth)

    # --------------------------------------------------------------------------
    # 4. Report + figures
    # --------------------------------------------------------------------------
    final_maps = generate_edge_on_maps(mapper, res["pot_params"], res["disk_df_params"],
                                       res["bulge_df_params"], **common)

    print("\n=== 4. Parameter recovery (physics ON) ===")
    print(f"    {'parameter':16s} {'true':>12s} {'fitted':>12s} {'init err':>9s} {'fit err':>8s}")
    all_init = {**pot_init, **disk_init, **bulge_init}
    fitted = {**res["pot_params"], **res["disk_df_params"], **res["bulge_df_params"]}
    for k in truth:
        ie = abs(all_init[k] - truth[k]) / abs(truth[k]) * 100
        fe = abs(float(fitted[k]) - truth[k]) / abs(truth[k]) * 100
        flag = "  (frozen: unconstrained)" if k in FROZEN else ""
        print(f"    {k:16s} {truth[k]:12.4g} {float(fitted[k]):12.4g} "
              f"{ie:8.0f}% {fe:7.0f}%{flag}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for col, (name, m) in enumerate([("Observed", obs_maps), ("Far init", init_maps),
                                     ("Fit (physics ON)", final_maps)]):
        plot_row(axes[:, col], m, mass_mask, name, vmax_mass)
    fig.tight_layout()
    f1 = os.path.join(FIGDIR, "self_consistent_maps.png")
    fig.savefig(f1, dpi=130)

    fig2, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(sc["history"], color="tab:purple")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("Poisson residual")
    ax[0].set_title("1. Making the truth self-consistent")
    h = res["history"]
    ax[1].plot(h["loss"], label="total", color="k")
    ax[1].plot(h["mass_loss"], label="mass")
    ax[1].plot(h["vrot_loss"], label="v_rot")
    ax[1].plot(h["sigma_loss"], label="sigma")
    ax[1].plot(h["poisson_penalty"], label="poisson", color="tab:red")
    if res_ref is not None:
        ax[1].plot(res_ref["history"]["loss"], ls=":", color="gray",
                   label="total (physics off)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("loss component")
    ax[1].set_title("3. Fit with the physics term on")
    ax[1].legend(fontsize=8)
    fig2.tight_layout()
    f2 = os.path.join(FIGDIR, "self_consistent_losses.png")
    fig2.savefig(f2, dpi=130)
    print(f"\n    figures written to:\n      {f1}\n      {f2}")

    out = os.path.join(OUTDIR, "self_consistent_truth.json")
    with open(out, "w") as fh:
        json.dump({
            "pot_params": pot_true, "disk_df_params": disk_true,
            "bulge_df_params": bulge_true,
            "penalty_initial": sc["penalty_initial"],
            "penalty_final": sc["penalty_final"],
            "config": {
                "SEED": SEED, "N_PARTICLES": N_PARTICLES, "GRID_SIZE": GRID_SIZE,
                "EXTENT_X": EXTENT_X, "EXTENT_Z": EXTENT_Z,
                "OBS_BANDWIDTH": OBS_BANDWIDTH, "POISSON_KWARGS": POISSON_KWARGS,
                "W_POISSON": W_POISSON, "FROZEN": list(FROZEN),
            },
        }, fh, indent=2)
    print(f"      {out}   (self-consistent truth, reusable from the notebook)")

    return res, res_ref


if __name__ == "__main__":
    main()
