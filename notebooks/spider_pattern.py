"""
Does the Phoenix DF model produce a disc rotation ("spider") pattern?

The classic spider diagram is the iso-velocity contour pattern of the
line-of-sight velocity field of a rotating disc seen at intermediate
inclination. The general inclined projection lives in
`observables.project_to_sky` (the module used to only ever project edge-on:
x-z plane, v_y as line of sight); this script uses it to render the sky-plane
velocity field for several inclinations.

Geometry
--------
The model galaxy has its symmetry axis along z, the disc in the x-y plane.
For an inclination i (i = 0 face-on, i = 90 deg edge-on) the line of sight is

    n_hat = (0, sin i, cos i)

so the observables are

    v_los = v_y sin i + v_z cos i
    x_sky = x                        (kinematic major axis)
    y_sky = -y cos i + z sin i       (kinematic minor axis)

At i = 90 deg this reduces exactly to the edge-on convention already used in
`observables.bin_maps` (x_sky = x, y_sky = z, v_los = v_y).

Run from the repo root (the installed copy of phoenix is not editable):
    python notebooks/spider_pattern.py
    QUICK=1 python notebooks/spider_pattern.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from phoenix.actions_to_phasespace.actions_to_phasespace_nn import PhoenixMapper
from phoenix.optimization.observables import sample_and_map_particles, project_to_sky

QUICK = bool(os.environ.get("QUICK"))

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
TRUTH = os.path.join(HERE, "self_consistent_truth.json")

SEED = 0
N_DISK = 40_000 if QUICK else 300_000
N_BULGE = 10_000 if QUICK else 75_000
GRID = 24 if QUICK else 64
EXTENT = 14.0                 # sky half-size, kpc
BANDWIDTH = 0.55              # KDE bandwidth, kpc
INCLINATIONS = (90.0, 70.0, 50.0, 30.0)   # edge-on first: that is the target orientation

# The bulge DF is even in J_phi, so its sense of rotation is imposed by hand.
# 0.5 = pressure-supported, non-rotating: any rotation in the map then has to
# come from the disc DF, which is what is being tested here.
SPHEROID_COROTATION = 0.5


# ==============================================================================
# PROJECTION + BINNING
# ==============================================================================
def bin_sky(x_sky, y_sky, v_los, w, grid_size=GRID, extent=EXTENT,
            bandwidth=BANDWIDTH, chunk=8_000):
    """Gaussian soft-binned mass / velocity / dispersion maps on the sky plane.

    Same kernel and moment definitions as `observables.bin_maps`, accumulated in
    chunks so a few hundred thousand tracers do not need a dense
    (grid, grid, N) array.
    """
    d = 2.0 * extent / grid_size
    centers = jnp.linspace(-extent + d / 2, extent - d / 2, grid_size)
    norm = d * d / (2.0 * jnp.pi * bandwidth**2)

    @jax.jit
    def sums(xc, yc, vc, wc):
        dy = centers[:, None, None] - yc[None, None, :]
        dx = centers[None, :, None] - xc[None, None, :]
        k = norm * jnp.exp(-0.5 * (dx**2 + dy**2) / bandwidth**2)
        wk = wc[None, None, :] * k
        return (jnp.sum(wk, -1),
                jnp.sum(wk * vc[None, None, :], -1),
                jnp.sum(wk * (vc**2)[None, None, :], -1))

    m = p = q = jnp.zeros((grid_size, grid_size))
    for s in range(0, x_sky.shape[0], chunk):
        e = min(s + chunk, x_sky.shape[0])
        a, b, c = sums(x_sky[s:e], y_sky[s:e], v_los[s:e], w[s:e])
        m, p, q = m + a, p + b, q + c

    m_safe = jnp.maximum(m, 1e-12)
    v = jnp.where(m > 1e-5, p / m_safe, jnp.nan)
    v2 = jnp.where(m > 1e-5, q / m_safe, jnp.nan)
    sig = jnp.sqrt(jnp.maximum(v2 - v**2, 1e-12))
    return {"mass": m, "v_los": v, "sigma": sig, "centers": centers}


# ==============================================================================
# DIAGNOSTICS
# ==============================================================================
def diagnostics(maps, inc_deg):
    """Quantitative checks that the map really is a rotating-disc velocity field."""
    v = np.array(maps["v_los"])
    mass = np.array(maps["mass"])
    c = np.array(maps["centers"])
    # Only trust pixels with real signal; faint outskirt pixels carry almost no
    # mass and their velocity moment is pure sampling noise, which would
    # dominate any max-based statistic.
    good = np.isfinite(v) & (mass > np.nanmax(mass) * 1e-3)

    # 1. Point antisymmetry: v(x, y) = -v(-x, -y) for a rotating disc.
    v_flip = v[::-1, ::-1]
    both = good & good[::-1, ::-1]
    asym = np.nanmean(np.abs(v[both] + v_flip[both])) / np.nanmax(np.abs(v[good]))

    # 2. Rotation amplitude on the major axis, as half the peak-to-peak of the
    #    central row. Robust percentiles rather than min/max, so one noisy pixel
    #    cannot inflate it.
    i0 = np.argmin(np.abs(c))
    row = v[i0, good[i0, :]]
    v_amp = 0.5 * (np.percentile(row, 97) - np.percentile(row, 3)) if row.size else np.nan

    # 3. Zero-velocity line: the minor-axis column (x_sky = 0) should be near
    #    zero, since that sightline is symmetric about the rotation axis.
    j0 = np.argmin(np.abs(c))
    col = v[good[:, j0], j0]
    minor = np.nanmax(np.abs(col)) / v_amp if col.size else np.nan

    return {
        "inc": inc_deg,
        "v_amp": float(v_amp),
        "minor_over_major": float(minor),
        "antisymmetry": float(asym),
    }


def rotation_curve_vs_height(maps, heights=(0.0, 1.0, 2.0, 4.0)):
    """Edge-on diagnostic: the projected rotation profile v_los(x) at several
    heights |z|. A disc supported by rotation shows a rising-then-flat profile
    that lags at larger |z| (asymmetric drift), which is the edge-on signature
    of the same velocity field that makes the spider off edge-on."""
    v = np.array(maps["v_los"])
    mass = np.array(maps["mass"])
    c = np.array(maps["centers"])
    good = np.isfinite(v) & (mass > np.nanmax(mass) * 1e-3)
    out = {}
    for h in heights:
        k = int(np.argmin(np.abs(c - h)))
        prof = np.where(good[k], v[k], np.nan)
        out[h] = (c, prof)
    return out


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    os.makedirs(FIGDIR, exist_ok=True)

    with open(TRUTH) as fh:
        t = json.load(fh)
    pot, disk, bulge = t["pot_params"], t["disk_df_params"], t["bulge_df_params"]
    print(f"Using the self-consistent truth from {os.path.basename(TRUTH)}:")
    print(f"  M_disk={pot['M_disk']:.3g}  M_bulge={pot['M_bulge']:.3g}  "
          f"M_halo={pot['M_halo']:.3g}")
    print(f"  tracers: {N_DISK} disk + {N_BULGE} bulge candidates, "
          f"grid {GRID}x{GRID} over +/-{EXTENT} kpc, h={BANDWIDTH} kpc")

    print("\nSampling the DFs and mapping actions -> phase space ...")
    x, y, z, vx, vy, vz, w = sample_and_map_particles(
        mapper, pot, disk, bulge, N_disk=N_DISK, N_bulge=N_BULGE,
        prng_seed=SEED, spheroid_corotation=SPHEROID_COROTATION,
    )
    x, y, z, vx, vy, vz, w = map(jax.block_until_ready, (x, y, z, vx, vy, vz, w))
    R = np.sqrt(np.array(x)**2 + np.array(y)**2)
    print(f"  effective tracer weight: {float(jnp.sum(w)):.3g} Msun, "
          f"median R = {np.median(R):.2f} kpc")

    results, all_maps = [], {}
    for inc in INCLINATIONS:
        xs, ys, vl = project_to_sky(x, y, z, vx, vy, vz, inc)
        m = bin_sky(xs, ys, vl, w)
        all_maps[inc] = m
        d = diagnostics(m, inc)
        results.append(d)
        print(f"  i={inc:4.0f} deg   v_max(major) = {d['v_amp']:6.1f} km/s   "
              f"|v|_minor / v_major = {d['minor_over_major']:.3f}   "
              f"antisymmetry = {d['antisymmetry']:.3f}")

    # sin(i) scaling of the rotation amplitude
    print("\n  sin(i) scaling of the major-axis amplitude "
          "(v_max / sin i should be ~constant):")
    for d in results:
        print(f"    i={d['inc']:4.0f} deg   v_max/sin i = "
              f"{d['v_amp'] / np.sin(np.deg2rad(d['inc'])):7.1f} km/s")

    # --------------------------------------------------------------------------
    # Figure 1: the edge-on galaxy (the orientation of interest)
    # --------------------------------------------------------------------------
    ext = [-EXTENT, EXTENT, -EXTENT, EXTENT]
    m90 = all_maps[90.0]
    c = np.array(m90["centers"])
    mass90 = np.array(m90["mass"])
    mask90 = mass90 > np.nanmax(mass90) * 1e-4
    v90 = np.where(mask90, np.array(m90["v_los"]), np.nan)
    vlim90 = np.nanmax(np.abs(v90)) * 1.05

    fig0, ax0 = plt.subplots(1, 3, figsize=(16, 4.6))
    im = ax0[0].imshow(np.where(mask90, np.log10(np.maximum(mass90, 1e-12)), np.nan),
                       origin="lower", extent=ext, cmap="magma")
    ax0[0].set_title("edge-on: log$_{10}$ projected mass")
    plt.colorbar(im, ax=ax0[0], fraction=0.046)

    im = ax0[1].imshow(v90, origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=-vlim90, vmax=vlim90)
    levels = np.arange(-250, 251, 25)
    cs = ax0[1].contour(c, c, v90, levels=levels, colors="k",
                        linewidths=0.7, alpha=0.85)
    ax0[1].contour(c, c, v90, levels=[0.0], colors="k", linewidths=1.8)
    ax0[1].clabel(cs, levels=levels[::2], fmt="%.0f", fontsize=7)
    ax0[1].set_title("edge-on: $v_{los}$ with iso-velocity contours")
    plt.colorbar(im, ax=ax0[1], fraction=0.046, label="km/s")

    im = ax0[2].imshow(np.where(mask90, np.array(m90["sigma"]), np.nan),
                       origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=180)
    ax0[2].set_title("edge-on: $\\sigma_{los}$")
    plt.colorbar(im, ax=ax0[2], fraction=0.046, label="km/s")
    for a in ax0:
        a.set_xlabel("x$_{sky}$ [kpc]")
    ax0[0].set_ylabel("z$_{sky}$ [kpc]")
    fig0.suptitle("Phoenix DF model, edge-on ($i=90^\\circ$)", fontsize=13)
    fig0.tight_layout()
    out0 = os.path.join(FIGDIR, "spider_pattern_edge_on.png")
    fig0.savefig(out0, dpi=140)
    print(f"\nEdge-on figure written to {out0}")

    # Rotation profile vs height: the edge-on rotation signature.
    prof = rotation_curve_vs_height(m90)
    figp, axp = plt.subplots(figsize=(6.2, 4.4))
    print("\n  edge-on projected rotation profile, peak |v_los| per height:")
    for h, (cc, pp) in prof.items():
        axp.plot(cc, pp, marker="o", ms=3, label=f"|z| = {h:.0f} kpc")
        if np.isfinite(pp).any():
            print(f"    z = {h:4.1f} kpc   max |v_los| = {np.nanmax(np.abs(pp)):6.1f} km/s")
    axp.axhline(0, color="k", lw=0.6)
    axp.set_xlabel("x$_{sky}$ [kpc]")
    axp.set_ylabel("$v_{los}$ [km/s]")
    axp.set_title("Edge-on projected rotation profile vs height")
    axp.legend(fontsize=8)
    figp.tight_layout()
    outp = os.path.join(FIGDIR, "spider_pattern_rotation_profile.png")
    figp.savefig(outp, dpi=140)
    print(f"  rotation-profile figure written to {outp}")

    # --------------------------------------------------------------------------
    # Figure 2: inclination sequence -- the spider only opens up off edge-on
    # --------------------------------------------------------------------------
    vlim = max(d["v_amp"] for d in results) * 1.05
    fig, axes = plt.subplots(2, len(INCLINATIONS),
                             figsize=(4.1 * len(INCLINATIONS), 8.2))
    for col, inc in enumerate(INCLINATIONS):
        m = all_maps[inc]
        v = np.array(m["v_los"])
        mass = np.array(m["mass"])
        mask = mass > np.nanmax(mass) * 1e-4
        vv = np.where(mask, v, np.nan)
        c = np.array(m["centers"])

        ax = axes[0, col]
        im = ax.imshow(vv, origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=-vlim, vmax=vlim)
        # The spider: iso-velocity contours of the line-of-sight field.
        levels = np.arange(-250, 251, 25)
        cs = ax.contour(c, c, vv, levels=levels, colors="k",
                        linewidths=0.7, alpha=0.8)
        ax.contour(c, c, vv, levels=[0.0], colors="k", linewidths=1.8)
        ax.clabel(cs, levels=levels[::2], fmt="%.0f", fontsize=6)
        ax.set_title(f"$v_{{los}}$,  $i={inc:.0f}^\\circ$")
        ax.set_xlabel("x$_{sky}$ [kpc]")
        if col == 0:
            ax.set_ylabel("y$_{sky}$ [kpc]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="km/s")

        ax = axes[1, col]
        im = ax.imshow(np.where(mask, np.array(m["sigma"]), np.nan),
                       origin="lower", extent=ext, cmap="viridis",
                       vmin=0, vmax=180)
        ax.contour(c, c, np.where(mask, np.log10(np.maximum(mass, 1e-12)), np.nan),
                   levels=8, colors="w", linewidths=0.6, alpha=0.7)
        ax.set_title(f"$\\sigma_{{los}}$ + mass contours,  $i={inc:.0f}^\\circ$")
        ax.set_xlabel("x$_{sky}$ [kpc]")
        if col == 0:
            ax.set_ylabel("y$_{sky}$ [kpc]")
        plt.colorbar(im, ax=ax, fraction=0.046, label="km/s")

    fig.suptitle("Phoenix DF model: line-of-sight velocity field vs inclination "
                 "(spider diagram)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "spider_pattern.png")
    fig.savefig(out, dpi=140)
    print(f"\nFigure written to {out}")
    return results, all_maps


if __name__ == "__main__":
    mapper = PhoenixMapper()
    main()
