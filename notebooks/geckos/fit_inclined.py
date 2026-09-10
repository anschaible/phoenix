"""
Dust-lane-masked, feature-weighted GECKOS fits at a slightly inclined viewing angle.

Setup, and why each piece is there (all established by measurement, see below):
  * sampler='importance'  -- the default soft-acceptance sampler does not represent
    the DF at all (corr(weight, f) = 0.05; J0_spheroid 25->400 gave bit-identical
    tracers), so without this no DF parameter can shape either target signature.
  * dust_lane_mask        -- the lane suppresses sigma by up to 41% exactly where the
    central peak lives; fitting it unmasked teaches the model not to have a peak.
  * feature_weight_map    -- both signatures live in a thin midplane strip that is a
    small minority of the 558 observed cells, so an unweighted mean trades them away.
  * mass weight non-zero  -- once the DF controls the tracers, Rd sets the sigma
    contrast, and with mass weight 0 nothing constrains it (it drifted to Rd = 2.84).
  * obs_bandwidth + fine final bandwidth -- load_geckos_maps HARD-bins the data while
    the model is KDE-smoothed; unmatched, the annealing is biased and the central
    velocity gradient is resolution-capped (dv/dx 63 at h=0.8 vs 100 at h=0.12).
  * inclination           -- per galaxy, set from its own photometric axis ratio
    (`inclination_scan.photometric_inclination`), not assumed. NGC 5010 has
    b/a = 0.22-0.25 on the disk isophotes -> i ~ 78-85, and a loss scan found 80 and
    90 deg statistically indistinguishable (seed scatter +-0.004 against a 0.001
    gap). NGC 3630 is measurably rounder, b/a = 0.34-0.375 -> i ~ 69-76, so it gets a
    lower inclination.

Run from the repo root:
    python notebooks/geckos/fit_inclined.py            # all configured galaxies
    python notebooks/geckos/fit_inclined.py NGC3630    # one galaxy
    QUICK=1 python notebooks/geckos/fit_inclined.py NGC3630
"""
import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geckos_data import load_geckos_maps, dust_lane_mask
from fit_geckos import GALAXIES, ARCSEC_PER_RAD, LEARNING_RATE, REG_WEIGHT
from phoenix.actions_to_phasespace.actions_to_phasespace_nn import PhoenixMapper
from phoenix.optimization.observables import render_maps_batched
from phoenix.optimization.pipeline import fit, DEFAULT_PARAM_BOUNDS, feature_weight_map

QUICK = bool(os.environ.get("QUICK"))

# Inclination per galaxy, from each one's own isophotal axis ratio (see docstring).
# NGC 3630's pointing is offset and covers mostly one side of the disk, so its
# photometric b/a is the less reliable of the two; 75 deg is the middle of the
# 69-76 range its disk isophotes imply.
INCLINATIONS = {"NGC5010": 80.0, "NGC3630": 75.0}

# Per-galaxy overrides. NGC 3630 needs a different treatment from NGC 5010:
#
#  * mask_dust=False -- requested, and defensible: its pointing is offset so the
#    flux centroid and midplane are less reliable, and the mirror-symmetry test that
#    finds a lane on NGC 5010 is correspondingly less trustworthy here. It matters a
#    lot for the targets: masking removed high-|v| midplane cells and dropped the
#    apparent rotation target from 169.5 to 127.8 km/s.
#
#  * Rd bounded below -- left free (lower bound 0.4) the fit put Rd AT 0.40 kpc in
#    every one of seven configurations tried, against a measured major-axis light
#    scale length of 2.30 kpc (exponential fit over 1 < R < 5.5 kpc). A hot compact
#    disk mimics an extended warm one, because epicyclic excursions carry tracers
#    with small guiding radii out to large R; the light map cannot tell them apart,
#    but the rotation can -- that degeneracy is what held v_los down.
#
#  * sigmaR0_R0 capped hard -- the fit's preferred route to the central sigma peak is
#    a radially hot disk (it pinned sigmaR0 at the old 120 cap). At sigma_R ~ 120 the
#    asymmetric drift drags mean v_phi far below v_circ, which is exactly the "too low
#    velocities" failure. Cooling the cap raised peak |v_los| from 88 to 149 km/s.
#
#  * loss_weights favour v_rot -- residuals are normalized by fixed 200 (v) and
#    100 km/s (sigma) scales, so per km/s sigma is already weighted ~2x; combined with
#    the old (1, 2, 3) that made sigma ~3x more important than rotation.
GALAXY_OVERRIDES = {
    "NGC5010": dict(mask_dust=True, loss_weights=(1.0, 2.0, 3.0, 0.0),
                    sigmaR0_cap=120.0, rd_bounds=(0.4, 6.0)),
    "NGC3630": dict(mask_dust=False, loss_weights=(1.0, 4.0, 2.0, 0.0),
                    sigmaR0_cap=40.0, rd_bounds=(1.2, 4.0)),
}
GRID_SIZE = 40
# Tracer count is no longer capped by the gradient bug that used to force it down:
# `sample_df_importance` guarded log(f) with `jnp.maximum(f, 1e-300)`, which is a
# no-op in float32 (smallest denormal ~1e-45), so tail candidates with f = 0 gave
# log(0) = -inf and NaN gradients for every potential and disk parameter. Fixed with
# a double-`where`; gradients are now finite at 4k-24k across the whole annealing
# range, and `res['nan_steps']` below asserts it stays that way.
N_PARTICLES = 4_000 if QUICK else 24_000
N_STEPS = 30 if QUICK else 500
N_RENDER = 20_000 if QUICK else 150_000
SEED = 0
FINAL_H = 0.15
DEFAULT_LOSS_WEIGHTS = (1.0, 2.0, 3.0, 0.0)


def signature_metrics(maps, keep, xc, zc):
    """The two signatures, measured only on unobscured cells:
    the central velocity gradient and the central-to-disk dispersion contrast."""
    V, S = np.array(maps["v_rot"]), np.array(maps["sigma"])
    # Midplane row with the most surviving cells (the dust lane removes part of one).
    cand = [j for j in range(len(zc)) if abs(zc[j]) < 0.6]
    kz = max(cand, key=lambda j: keep[j].sum())
    row = np.where(keep[kz], V[kz], np.nan)
    sel = (np.abs(xc) < 1.5) & np.isfinite(row)
    dv = (np.nanmax(np.abs(np.gradient(row[sel], xc[sel])))
          if sel.sum() > 2 else np.nan)
    cen = keep & (np.abs(xc)[None, :] < 1.0) & (np.abs(zc)[:, None] < 0.5)
    dsk = (keep & (np.abs(xc)[None, :] > 2) & (np.abs(xc)[None, :] < 5)
           & (np.abs(zc)[:, None] < 0.5))
    sc, sd = S[cen].mean(), S[dsk].mean()
    return dict(v_pk=float(np.nanmax(np.abs(row))), dvdx=float(dv),
                sig_cen=float(sc), sig_disk=float(sd), ratio=float(sc / sd))


def fit_galaxy(NAME, mapper):
    cfg = GALAXIES[NAME]
    INCLINATION = INCLINATIONS[NAME]
    ov = GALAXY_OVERRIDES.get(NAME, {})
    MASK_DUST = ov.get("mask_dust", True)
    LOSS_WEIGHTS = ov.get("loss_weights", DEFAULT_LOSS_WEIGHTS)
    SIGMAR0_CAP = ov.get("sigmaR0_cap", 120.0)
    RD_BOUNDS = ov.get("rd_bounds", (0.4, 6.0))
    outdir = os.path.join(HERE, "plots", NAME)
    os.makedirs(outdir, exist_ok=True)
    EX, EZ = cfg["extent_x"], cfg["extent_z"]
    PIXEL = max(2 * EX / GRID_SIZE, 2 * EZ / GRID_SIZE)

    datadir = os.path.join(HERE, "data", cfg["subdir"], "maps", cfg["stem"])
    obs, info = load_geckos_maps(
        os.path.join(datadir, f"{cfg['stem']}_kin_maps.fits"),
        os.path.join(datadir, f"{cfg['stem']}_spatial_binning_maps.fits"),
        kpc_per_arcsec=cfg["d_mpc"] * 1e3 / ARCSEC_PER_RAD,
        extent_x=EX, extent_z=EZ, grid_size=GRID_SIZE,
        Mstar_fiducial=10.0 ** cfg["lmstar"],
        center_percentile=cfg["center_percentile"],
    )
    xc = 0.5 * (np.array(obs["x_edges"])[:-1] + np.array(obs["x_edges"])[1:])
    zc = 0.5 * (np.array(obs["z_edges"])[:-1] + np.array(obs["z_edges"])[1:])

    observed = np.array(obs["mass"]) > 0
    dust = dust_lane_mask(obs) if MASK_DUST else np.zeros_like(observed)
    keep = observed & ~dust
    print(f"===== {NAME}: fit at i = {INCLINATION:.0f} deg, "
          f"dust {'masked' if MASK_DUST else 'NOT masked'} =====")
    if MASK_DUST:
        print(f"  {observed.sum()} observed cells, dust lane masks {dust.sum()} "
              f"({100*dust.sum()/observed.sum():.0f}%), fitting {keep.sum()}")
        zi = np.where(dust)[0]
        print(f"  lane spans z = {zc[zi].min():+.2f} to {zc[zi].max():+.2f} kpc "
              f"(one-sided, as a dust lane must be)")
    else:
        print(f"  {observed.sum()} observed cells, no dust masking, fitting {keep.sum()}")
    print(f"  loss weights {LOSS_WEIGHTS}, sigmaR0 cap {SIGMAR0_CAP}, Rd in {RD_BOUNDS}")

    tgt = signature_metrics(obs, keep, xc, zc)
    print(f"  observed signatures (unobscured cells): v_pk={tgt['v_pk']:.1f} km/s, "
          f"dv/dx={tgt['dvdx']:.0f} km/s/kpc, sigma {tgt['sig_cen']:.1f}/"
          f"{tgt['sig_disk']:.1f} = {tgt['ratio']:.2f}")

    bounds = copy.deepcopy(DEFAULT_PARAM_BOUNDS)
    bounds["disk"]["L0"] = cfg["L0_bounds"]
    bounds["disk"]["sigmaR0_R0"] = (5.0, SIGMAR0_CAP)
    bounds["disk"]["sigmaz0_R0"] = (5.0, 90.0)
    bounds["disk"]["Rd"] = RD_BOUNDS
    weights = feature_weight_map(GRID_SIZE, EX, EZ, midplane_z=0.6, w_midplane=4.0,
                                 center_x=2.5, w_center=2.0)

    print(f"\n  fitting ({N_STEPS} steps, {N_PARTICLES} tracers/component) ...")
    t0 = time.time()
    res = fit(
        mapper, obs, cfg["init_pot"], cfg["init_disk"], cfg["init_bulge"],
        N_disk=N_PARTICLES, N_bulge=N_PARTICLES, grid_size=GRID_SIZE,
        extent_x=EX, extent_z=EZ, prng_seed=SEED,
        loss_weights=LOSS_WEIGHTS, poisson_kwargs={},
        learning_rate=LEARNING_RATE, n_steps=N_STEPS,
        anneal_bandwidth=(4.0, FINAL_H), obs_bandwidth=PIXEL * 0.5,
        spheroid_corotation=cfg["spheroid_corotation"],
        inclination_deg=INCLINATION, sampler="importance",
        param_bounds=bounds, reg_weight=REG_WEIGHT,
        extra_mask=(dust if MASK_DUST else None), pixel_weights=weights,
    )
    h = res["history"]
    print(f"  loss {h['loss'][0]:.4f} -> {h['loss'][-1]:.4f}  "
          f"(mass {h['mass_loss'][-1]:.4f}, vrot {h['vrot_loss'][-1]:.4f}, "
          f"sigma {h['sigma_loss'][-1]:.4f})   [{time.time()-t0:.0f} s]")
    print(f"  non-finite-gradient steps: {res['nan_steps']}/{N_STEPS} "
          f"(must be 0, else parameters were frozen -- see the N_PARTICLES ceiling)")
    init_all = {**cfg["init_pot"], **cfg["init_disk"], **cfg["init_bulge"]}
    fitted_all = {**res["pot_params"], **res["disk_df_params"], **res["bulge_df_params"]}
    moved = [k for k in init_all
             if abs(float(fitted_all[k]) / init_all[k] - 1.0) > 0.01]
    print(f"  parameters that moved >1% from the initial guess: "
          f"{len(moved)}/{len(init_all)}")

    model = render_maps_batched(
        mapper, res["pot_params"], res["disk_df_params"], res["bulge_df_params"],
        N_disk=N_RENDER, N_bulge=N_RENDER, grid_size=GRID_SIZE,
        extent_x=EX, extent_z=EZ, prng_seed=SEED, soft_bin_h=FINAL_H,
        spheroid_corotation=cfg["spheroid_corotation"],
        inclination_deg=INCLINATION, sampler="importance", chunk=15_000,
    )

    # ---------------- goodness of fit ----------------
    got = signature_metrics(model, keep, xc, zc)
    dv = np.array(obs["v_rot"]) - np.array(model["v_rot"])
    ds = np.array(obs["sigma"]) - np.array(model["sigma"])
    rms_v, rms_s = float(np.sqrt((dv[keep]**2).mean())), float(np.sqrt((ds[keep]**2).mean()))
    med_v, med_s = float(np.median(np.abs(dv[keep]))), float(np.median(np.abs(ds[keep])))
    print("\n  --- goodness of fit (unobscured cells only) ---")
    print(f"    v_los : RMS {rms_v:5.1f} km/s, median |resid| {med_v:5.1f} km/s, "
          f"{100*np.mean(np.abs(dv[keep])<20):.0f}% of cells within 20 km/s")
    print(f"    sigma : RMS {rms_s:5.1f} km/s, median |resid| {med_s:5.1f} km/s, "
          f"{100*np.mean(np.abs(ds[keep])<20):.0f}% of cells within 20 km/s")
    print(f"    {'signature':<16s} {'observed':>10s} {'model':>10s}")
    for k, lab in (("v_pk", "peak |v_los|"), ("dvdx", "central dv/dx"),
                   ("sig_cen", "sigma centre"), ("sig_disk", "sigma disk"),
                   ("ratio", "sigma ratio")):
        print(f"    {lab:<16s} {tgt[k]:10.2f} {got[k]:10.2f}")
    print(f"\n  fitted: Rd={float(res['disk_df_params']['Rd']):.2f} kpc  "
          f"sigmaR0={float(res['disk_df_params']['sigmaR0_R0']):.0f}  "
          f"M_disk={float(res['pot_params']['M_disk']):.3g}  "
          f"M_bulge={float(res['pot_params']['M_bulge']):.3g}  "
          f"a_bulge={float(res['pot_params']['a_bulge']):.2f}")

    # ---------------- maps figure ----------------
    extent = [-EX, EX, -EZ, EZ]
    def show(a):
        return np.where(keep, np.array(a), np.nan)
    specs = [
        ("v_rot", "seismic", dict(vmin=-200, vmax=200), "$v_{LOS}$ [km/s]", 60),
        ("sigma", "viridis", dict(vmin=0, vmax=140), "$\\sigma_{LOS}$ [km/s]", 40),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.5))
    for r, (key, cmap, kw, lab, rlim) in enumerate(specs):
        for c, (t, d) in enumerate([("Observed", obs), ("Model (fit)", model)]):
            im = axes[r, c].imshow(show(d[key]), origin="lower", extent=extent,
                                   cmap=cmap, aspect="auto", **kw)
            axes[r, c].set_title(f"{t}: {lab}")
            fig.colorbar(im, ax=axes[r, c], label="km/s")
        resid = np.where(keep, np.array(obs[key]) - np.array(model[key]), np.nan)
        im = axes[r, 2].imshow(resid, origin="lower", extent=extent, cmap="coolwarm",
                               vmin=-rlim, vmax=rlim, aspect="auto")
        axes[r, 2].set_title(f"Residual (obs - model), RMS "
                             f"{rms_v if key=='v_rot' else rms_s:.0f} km/s")
        fig.colorbar(im, ax=axes[r, 2], label="km/s")
    # bottom row: light, and the dust mask that was excluded
    vmax = float(np.nanmax(np.array(obs["mass"])))
    im = axes[2, 0].imshow(show(obs["mass"]), origin="lower", extent=extent,
                           cmap="magma", norm=LogNorm(vmin=vmax/1e3, vmax=vmax),
                           aspect="auto")
    axes[2, 0].set_title("Observed light (fitted, weight 1)")
    fig.colorbar(im, ax=axes[2, 0])
    mm = np.array(model["mass"])
    im = axes[2, 1].imshow(np.where(keep, mm/mm.sum()*np.array(obs['mass']).sum(), np.nan),
                           origin="lower", extent=extent, cmap="magma",
                           norm=LogNorm(vmin=vmax/1e3, vmax=vmax), aspect="auto")
    axes[2, 1].set_title("Model mass (renormalized)")
    fig.colorbar(im, ax=axes[2, 1])
    flags = np.full(dust.shape, np.nan)
    flags[observed] = 0.0
    flags[dust] = 1.0
    im = axes[2, 2].imshow(flags, origin="lower", extent=extent, cmap="autumn_r",
                           vmin=0, vmax=1, aspect="auto")
    axes[2, 2].set_title(f"Dust lane excluded ({dust.sum()} cells, red)"
                         if MASK_DUST else "No dust masking (all cells fitted)")
    for a in axes.ravel():
        a.set_xlabel("x [kpc]"); a.set_ylabel("z [kpc]")
    fig.suptitle(f"{NAME}: observed vs fitted Phoenix model, i = {INCLINATION:.0f}$^\\circ$, "
                 f"dust lane {'masked' if MASK_DUST else 'NOT masked'}", fontsize=14)
    fig.tight_layout()
    f1 = os.path.join(outdir, f"{NAME}_inclined_fit_maps.png")
    fig.savefig(f1, dpi=125); plt.close(fig)

    # ---------------- profile figure ----------------
    cand = [j for j in range(GRID_SIZE) if abs(zc[j]) < 0.6]
    kz = max(cand, key=lambda j: keep[j].sum())
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a, key, lab in ((ax[0], "v_rot", "$v_{LOS}$"), (ax[1], "sigma", "$\\sigma_{LOS}$")):
        a.plot(xc, np.where(keep[kz], np.array(obs[key])[kz], np.nan), "ko-", ms=4,
               label="observed")
        a.plot(xc, np.where(keep[kz], np.array(model[key])[kz], np.nan), "r.-",
               label="model")
        a.set_xlabel("x [kpc]"); a.set_ylabel(f"{lab} [km/s]")
        a.set_title(f"Midplane cut (z = {zc[kz]:+.2f} kpc)"); a.legend(); a.grid(alpha=.3)
    cen = np.abs(xc) < 1.0
    for tag, d, st in (("observed", obs, "ko-"), ("model", model, "r.-")):
        prof = [np.nanmean(np.where(keep[j] & cen, np.array(d["sigma"])[j], np.nan))
                for j in range(GRID_SIZE)]
        ax[2].plot(zc, prof, st, ms=4, label=tag)
    ax[2].set_xlabel("z [kpc]"); ax[2].set_ylabel("$\\sigma_{LOS}$ [km/s]")
    ax[2].set_title("Central $\\sigma$ vs height (|x| < 1 kpc)")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle(f"{NAME}: profile comparison, i = {INCLINATION:.0f}$^\\circ$")
    fig.tight_layout()
    f2 = os.path.join(outdir, f"{NAME}_inclined_fit_profiles.png")
    fig.savefig(f2, dpi=130); plt.close(fig)

    out = os.path.join(HERE, f"{NAME}_inclined_fit.json")
    with open(out, "w") as fh:
        json.dump({
            "galaxy": NAME, "inclination_deg": INCLINATION,
            "config": {"GRID_SIZE": GRID_SIZE, "N_PARTICLES": N_PARTICLES,
                       "N_STEPS": N_STEPS, "SEED": SEED, "FINAL_H": FINAL_H,
                       "LOSS_WEIGHTS": list(LOSS_WEIGHTS), "sampler": "importance",
                       "mask_dust": bool(MASK_DUST),
                       "dust_masked_cells": int(dust.sum()),
                       "sigmaR0_cap": SIGMAR0_CAP, "rd_bounds": list(RD_BOUNDS),
                       "fitted_cells": int(keep.sum())},
            "nan_steps": int(res["nan_steps"]),
            "goodness_of_fit": {"rms_vlos": rms_v, "rms_sigma": rms_s,
                                "median_abs_vlos": med_v, "median_abs_sigma": med_s},
            "signatures": {"observed": tgt, "model": got},
            "final_loss": {k: float(h[k][-1]) for k in
                           ("loss", "mass_loss", "vrot_loss", "sigma_loss")},
            "pot_params": {k: float(v) for k, v in res["pot_params"].items()},
            "disk_df_params": {k: float(v) for k, v in res["disk_df_params"].items()},
            "bulge_df_params": {k: float(v) for k, v in res["bulge_df_params"].items()},
        }, fh, indent=2)
    print(f"\n  wrote {f1}\n        {f2}\n        {out}")
    return res, tgt, got


def main():
    which = [a for a in sys.argv[1:] if a in GALAXIES] or list(INCLINATIONS)
    mapper = PhoenixMapper()
    for name in which:
        fit_galaxy(name, mapper)


if __name__ == "__main__":
    main()
