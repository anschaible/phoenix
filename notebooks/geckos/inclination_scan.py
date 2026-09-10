"""
Is NGC 5010 really exactly edge-on, and would the fit improve at lower inclination?

The Phoenix observable pipeline used to hard-code an edge-on projection
(x = major axis, z = vertical, v_y = line of sight). `observables.project_to_sky`
now generalizes this to any inclination, so the question can be answered by
refitting the same galaxy at a range of inclinations and comparing the converged
kinematic loss.

Two things have to be kept apart, and this script reports both:

  * The AMPLITUDE of v_los scales as sin(i). In a kinematic-only fit (the GECKOS
    setup uses loss weights (0, 2, 3, 0), i.e. mass is not fitted) the optimizer can
    absorb almost any inclination by rescaling the potential mass: dropping i from
    90 to 70 deg costs only 6% in v_los, recovered by ~13% more mass. So a lower
    loss at lower i is NOT by itself evidence for a lower inclination -- the fitted
    masses have to be inspected alongside it.
  * The SHAPE of the projected light is what actually constrains i, and it is not
    in the loss at all here. `photometric_inclination` therefore measures the
    observed isophotal axis ratio directly from the MUSE flux map.

The converged loss also has a real seed-to-seed scatter, because the model maps are
Monte-Carlo estimates from a finite tracer sample. `SEEDS` runs each inclination at
several seeds so that scatter is measured rather than assumed -- without it a
sub-percent difference between two inclinations is not interpretable. Measured for
NGC 5010 at this budget (3 seeds each): i=90 gave 0.0557/0.0620/0.0527, i=80 gave
0.0554/0.0551/0.0566, i=70 gave 0.0679/0.0688/0.0707. So 80 and 90 deg are
indistinguishable, while 70 deg is worse by several times the scatter.

Run from the repo root:
    python notebooks/geckos/inclination_scan.py
    QUICK=1 python notebooks/geckos/inclination_scan.py     # timing probe / smoke test
"""
import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geckos_data import load_geckos_maps
from fit_geckos import GALAXIES, ARCSEC_PER_RAD, LOSS_WEIGHTS, LEARNING_RATE, REG_WEIGHT
from phoenix.actions_to_phasespace.actions_to_phasespace_nn import PhoenixMapper
from phoenix.optimization.pipeline import fit, DEFAULT_PARAM_BOUNDS
from phoenix.optimization.observables import render_maps_batched

QUICK = bool(os.environ.get("QUICK"))

NAME = "NGC5010"
INCLINATIONS = (90.0, 80.0, 70.0) if QUICK else (90.0, 85.0, 80.0, 75.0, 70.0, 60.0)

# Same grid as the production fit (the data are binned on it). Tracer count and
# step count are reduced so six fits are affordable on CPU -- but every
# inclination gets EXACTLY the same budget, seed and schedule, so the comparison
# between them is fair even though each individual fit is shorter than a
# production run.
GRID_SIZE = 40
N_PARTICLES = 4_000 if QUICK else 12_000
N_STEPS = 30 if QUICK else 400
N_RENDER = 20_000 if QUICK else 150_000
# Several seeds per inclination, so the Monte-Carlo scatter of the converged loss is
# measured alongside the inclination trend (see module docstring). SEEDS[0] is the
# seed whose fitted parameters are kept for the maps/JSON.
SEEDS = (0,) if QUICK else (0, 1, 2)
ANNEAL_BANDWIDTH = (4.0, 0.4)


# ==============================================================================
# PHOTOMETRIC INCLINATION (independent of the kinematic fit)
# ==============================================================================
def photometric_inclination(cfg, q0_values=(0.13, 0.20, 0.25)):
    """Isophotal axis ratio of the MUSE light map, and the inclination it implies.

    Uses the standard Hubble relation cos^2 i = (q^2 - q0^2) / (1 - q0^2), where q
    is the apparent and q0 the intrinsic (edge-on) axis ratio. q0 is NOT known
    independently, which is the whole difficulty: a rounder image can mean either a
    lower inclination or a thicker disk. Several q0 are therefore reported.
    """
    stem = cfg["stem"]
    d = os.path.join(HERE, "data", cfg["subdir"], "maps", stem)
    kpc = cfg["d_mpc"] * 1e3 / ARCSEC_PER_RAD
    with fits.open(os.path.join(d, stem + "_spatial_binning_maps.fits")) as h:
        F, X, Y = (np.array(h[k].data) for k in ("FLUX", "XBIN", "YBIN"))
    g = np.isfinite(F) & np.isfinite(X) & np.isfinite(Y) & (F > 0)
    x, y, f = X[g] * kpc, Y[g] * kpc, F[g]
    x -= (f * x).sum() / f.sum()
    y -= (f * y).sum() / f.sum()
    Ixx, Iyy, Ixy = ((f * a * b).sum() / f.sum() for a, b in ((x, x), (y, y), (x, y)))
    pa = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    c, s = np.cos(-pa), np.sin(-pa)
    xr, zr = c * x - s * y, s * x + c * y

    rows = []
    for pct in (97, 90, 80, 60):
        m = f >= np.percentile(f, pct)
        A = np.sqrt((f[m] * xr[m] ** 2).sum() / f[m].sum())
        B = np.sqrt((f[m] * zr[m] ** 2).sum() / f[m].sum())
        q = B / A
        incs = []
        for q0 in q0_values:
            cos2 = (q ** 2 - q0 ** 2) / (1 - q0 ** 2)
            incs.append(90.0 if cos2 <= 0 else float(np.degrees(np.arccos(np.sqrt(cos2)))))
        rows.append((100 - pct, float(q), incs))
    return rows, float(np.degrees(pa))


# ==============================================================================
# THE SCAN
# ==============================================================================
def main():
    cfg = GALAXIES[NAME]
    outdir = os.path.join(HERE, "plots", NAME)
    os.makedirs(outdir, exist_ok=True)
    mapper = PhoenixMapper()

    kpc_per_arcsec = cfg["d_mpc"] * 1e3 / ARCSEC_PER_RAD
    Mstar = 10.0 ** cfg["lmstar"]
    EX, EZ = cfg["extent_x"], cfg["extent_z"]
    datadir = os.path.join(HERE, "data", cfg["subdir"], "maps", cfg["stem"])
    obs_maps, info = load_geckos_maps(
        os.path.join(datadir, f"{cfg['stem']}_kin_maps.fits"),
        os.path.join(datadir, f"{cfg['stem']}_spatial_binning_maps.fits"),
        kpc_per_arcsec=kpc_per_arcsec, extent_x=EX, extent_z=EZ,
        grid_size=GRID_SIZE, Mstar_fiducial=Mstar,
        center_percentile=cfg["center_percentile"],
    )

    print(f"===== {NAME} inclination scan =====")
    print(f"  D = {cfg['d_mpc']} Mpc, PA = {info['position_angle_deg']:.1f} deg, "
          f"filled = {info['filled_fraction']*100:.0f}%")

    print("\n--- 1. Photometric constraint (light shape, NOT part of the loss) ---")
    rows, pa = photometric_inclination(cfg)
    print(f"    {'isophote':>10s} {'b/a':>7s}   implied i for q0 = 0.13 / 0.20 / 0.25")
    for top, q, incs in rows:
        print(f"    top {top:3d}%    {q:.3f}      "
              + " / ".join(f"{v:5.1f}" for v in incs) + "  deg")

    # bounds: same per-galaxy choices as the production fit
    param_bounds = copy.deepcopy(DEFAULT_PARAM_BOUNDS)
    param_bounds["disk"]["L0"] = cfg["L0_bounds"]
    param_bounds["disk"]["sigmaR0_R0"] = (5.0, 120.0)
    param_bounds["disk"]["sigmaz0_R0"] = (5.0, 90.0)

    print(f"\n--- 2. Refitting at each inclination "
          f"({N_STEPS} steps, {N_PARTICLES} tracers/component, seeds {SEEDS}) ---")
    results, per_seed = {}, {}
    for inc in INCLINATIONS:
        per_seed[inc] = []
        for seed in SEEDS:
            t0 = time.time()
            res = fit(
                mapper, obs_maps, cfg["init_pot"], cfg["init_disk"], cfg["init_bulge"],
                N_disk=N_PARTICLES, N_bulge=N_PARTICLES, grid_size=GRID_SIZE,
                extent_x=EX, extent_z=EZ, prng_seed=seed,
                loss_weights=LOSS_WEIGHTS, poisson_kwargs={},
                learning_rate=LEARNING_RATE, n_steps=N_STEPS,
                anneal_bandwidth=ANNEAL_BANDWIDTH,
                spheroid_corotation=cfg["spheroid_corotation"],
                inclination_deg=inc,
                param_bounds=param_bounds, reg_weight=REG_WEIGHT,
            )
            h = res["history"]
            per_seed[inc].append(float(h["loss"][-1]))
            if seed == SEEDS[0]:
                results[inc] = res
            print(f"  i = {inc:4.0f} deg  seed {seed}   loss {h['loss'][0]:7.3f} -> "
                  f"{h['loss'][-1]:.4f}   "
                  f"(vrot {h['vrot_loss'][-1]:.4f}, sigma {h['sigma_loss'][-1]:.4f}, "
                  f"reg {h['reg'][-1]:.4f})   "
                  f"M_disk={float(res['pot_params']['M_disk']):.3g}  "
                  f"M_halo={float(res['pot_params']['M_halo']):.3g}   "
                  f"[{time.time()-t0:.0f} s]")

    # ------------------------------------------------------------------
    # Compare inclinations on the SEED MEAN, and report the scatter, so a
    # difference is only called real if it exceeds the Monte-Carlo noise.
    mean = {i: float(np.mean(v)) for i, v in per_seed.items()}
    std = {i: float(np.std(v)) for i, v in per_seed.items()}
    best = min(mean, key=lambda i: mean[i])
    print(f"\n--- 3. Verdict ---")
    print(f"  {'i [deg]':>8s} {'mean loss':>10s} {'scatter':>9s}   vs edge-on")
    for inc in INCLINATIONS:
        print(f"  {inc:8.0f} {mean[inc]:10.4f} {std[inc]:9.4f}   "
              f"{100*(mean[inc]-mean[90.0])/mean[90.0]:+6.1f}%")
    print(f"  lowest mean loss at i = {best:.0f} deg; "
          f"edge-on scatter is {std[90.0]:.4f}, so any gap below that is not resolved.")
    print("  fitted total baryonic mass vs inclination "
          "(the sin(i) degeneracy -- should rise as i falls):")
    for inc in INCLINATIONS:
        p = results[inc]["pot_params"]
        mb = float(p["M_disk"]) + float(p["M_bulge"])
        print(f"    i = {inc:4.0f} deg   M_disk+M_bulge = {mb:.4g}   "
              f"(x sin i = {mb*np.sin(np.deg2rad(inc)):.4g})")

    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    incs = np.array(INCLINATIONS)
    ax[0].plot(incs, [results[i]["history"]["loss"][-1] for i in incs], "o-", color="k",
               label="total")
    ax[0].plot(incs, [results[i]["history"]["vrot_loss"][-1] for i in incs], "s--",
               label="v_rot")
    ax[0].plot(incs, [results[i]["history"]["sigma_loss"][-1] for i in incs], "^--",
               label="sigma")
    ax[0].set_xlabel("inclination [deg]"); ax[0].set_ylabel("converged loss")
    ax[0].set_title("Kinematic loss vs assumed inclination"); ax[0].legend()

    ax[1].plot(incs, [float(results[i]["pot_params"]["M_disk"]) +
                      float(results[i]["pot_params"]["M_bulge"]) for i in incs],
               "o-", color="tab:orange", label="fitted $M_{disk}+M_{bulge}$")
    ax[1].plot(incs, [float(results[i]["pot_params"]["M_halo"]) for i in incs],
               "s-", color="tab:blue", label="fitted $M_{halo}$")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("inclination [deg]"); ax[1].set_ylabel("fitted mass [M$_\\odot$]")
    ax[1].set_title("The sin(i)-mass degeneracy"); ax[1].legend(fontsize=8)

    for inc in INCLINATIONS:
        ax[2].plot(results[inc]["history"]["loss"], label=f"i = {inc:.0f}$^\\circ$")
    ax[2].set_yscale("log"); ax[2].set_xlabel("iteration"); ax[2].set_ylabel("loss")
    ax[2].set_title("Convergence"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    f1 = os.path.join(outdir, f"{NAME}_inclination_scan.png")
    fig.savefig(f1, dpi=130)
    print(f"\n  wrote {f1}")

    # Map comparison: edge-on vs the best-fitting inclination
    show = sorted({90.0, best, 70.0})
    render_h = ANNEAL_BANDWIDTH[1]
    fig, axes = plt.subplots(2, 1 + len(show), figsize=(4.2 * (1 + len(show)), 7.6))
    mask = np.array(obs_maps["mass"]) > 0
    extent = [-EX, EX, -EZ, EZ]
    specs = [("v_rot", "seismic", dict(vmin=-200, vmax=200)),
             ("sigma", "viridis", dict(vmin=0, vmax=180))]
    for row, (key, cmap, kw) in enumerate(specs):
        a = axes[row, 0]
        im = a.imshow(np.where(mask, np.array(obs_maps[key]), np.nan), origin="lower",
                      extent=extent, cmap=cmap, aspect="auto", **kw)
        a.set_title(f"Observed: {key}"); plt.colorbar(im, ax=a)
        for col, inc in enumerate(show, start=1):
            r = results[inc]
            mm = render_maps_batched(
                mapper, r["pot_params"], r["disk_df_params"], r["bulge_df_params"],
                N_disk=N_RENDER, N_bulge=N_RENDER, grid_size=GRID_SIZE,
                extent_x=EX, extent_z=EZ, prng_seed=SEEDS[0], soft_bin_h=render_h,
                spheroid_corotation=cfg["spheroid_corotation"],
                inclination_deg=inc, chunk=15_000,
            )
            a = axes[row, col]
            im = a.imshow(np.where(mask, np.array(mm[key]), np.nan), origin="lower",
                          extent=extent, cmap=cmap, aspect="auto", **kw)
            a.set_title(f"Model $i={inc:.0f}^\\circ$: {key}"); plt.colorbar(im, ax=a)
        for a in axes[row]:
            a.set_xlabel("x [kpc]"); a.set_ylabel("z [kpc]")
    fig.suptitle(f"{NAME}: fitted kinematics at different assumed inclinations")
    fig.tight_layout()
    f2 = os.path.join(outdir, f"{NAME}_inclination_maps.png")
    fig.savefig(f2, dpi=120)
    print(f"  wrote {f2}")

    out = {
        "galaxy": NAME,
        "config": {"GRID_SIZE": GRID_SIZE, "N_PARTICLES": N_PARTICLES,
                   "N_STEPS": N_STEPS, "SEEDS": list(SEEDS),
                   "LOSS_WEIGHTS": list(LOSS_WEIGHTS),
                   "ANNEAL_BANDWIDTH": list(ANNEAL_BANDWIDTH)},
        "photometric": [{"isophote_top_pct": t, "b_over_a": q,
                         "implied_i_q0_0.13_0.20_0.25": inc} for t, q, inc in rows],
        "loss_per_seed": {str(i): per_seed[i] for i in INCLINATIONS},
        "loss_mean": {str(i): mean[i] for i in INCLINATIONS},
        "loss_std": {str(i): std[i] for i in INCLINATIONS},
        "fits": {str(i): {"final_loss": float(results[i]["history"]["loss"][-1]),
                          "vrot_loss": float(results[i]["history"]["vrot_loss"][-1]),
                          "sigma_loss": float(results[i]["history"]["sigma_loss"][-1]),
                          "pot_params": {k: float(v) for k, v in results[i]["pot_params"].items()}}
                 for i in INCLINATIONS},
    }
    fj = os.path.join(HERE, f"{NAME}_inclination_scan.json")
    with open(fj, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {fj}")


if __name__ == "__main__":
    main()
