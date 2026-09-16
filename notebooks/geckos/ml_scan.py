"""
Profile scan over the bulge-to-disk mass-to-light ratio for the GECKOS galaxies.

Why this scan and not a fixed choice. The first map `load_geckos_maps` returns is
FLUX, rescaled to a fiducial stellar mass -- i.e. the data are light, converted to
"mass" with ONE mass-to-light ratio for the whole galaxy. Fitting a model mass map
against that is the same assumption, because a spatially constant Upsilon cancels
exactly in the log-space map residual (and in the kinematic maps, which are ratios of
weighted sums). Real galaxies do not have one Upsilon: an old bulge is several times
dimmer per unit mass than a younger disk, so the light map is a different combination
of the two components than the mass map is.

`ml_ratios=(1.0, r)` in the fit makes the model produce LIGHT with the bulge dimmed by
r. Only the ratio r = Upsilon_bulge/Upsilon_disk is identifiable: a common factor on
Upsilon is degenerate with the light map's arbitrary normalization and cancels in the
same log-space residual, so Upsilon_disk is pinned at 1. r = 1 reproduces the previous
single-M/L behaviour exactly, which makes it the control point of the scan.

Unlike the mock experiment in `notebooks/optimization_pipeline_luminosity.ipynb`, the
losses here ARE comparable across the scan: the observation, the mask, the pixel
weights and the loss weights are identical at every r, and only the model's forward
map changes. So the curve is a genuine profile likelihood over r and the data get to
choose, rather than r having to be assumed.

The scan runs at a reduced tracer count (it is many fits), so its absolute losses are
not comparable with the production `fit_inclined.py` numbers -- only with each other.
Outputs are tagged `_scan_ml<r>` and never overwrite the production files.

Each ratio is fitted in a SEPARATE PROCESS. That is not tidiness: running the six fits
in one process reliably dies partway through with `YNNPACK operation failed` or a
spurious `RESOURCE_EXHAUSTED` on this host, because the JAX CPU backend sizes its
thread pool from all 256 cores and exhausts its resources once many jitted executables
are live (the failures are cumulative -- the first four fits always succeed). A fresh
process per ratio also makes the scan RESUMABLE: a ratio whose JSON already exists is
skipped, so an interrupted scan can simply be re-run. Pass FORCE=1 to refit anyway.

Run from the repo root:
    python notebooks/geckos/ml_scan.py                 # all configured galaxies
    python notebooks/geckos/ml_scan.py NGC5010         # one galaxy
    ML_SCAN_N=8000 ML_SCAN_STEPS=250 python notebooks/geckos/ml_scan.py NGC5010
    FORCE=1 python notebooks/geckos/ml_scan.py NGC5010 # ignore cached results
"""
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_inclined as FI
from fit_geckos import GALAXIES

# Upsilon_bulge / Upsilon_disk. 1.0 is the control (one global M/L). The upper end is
# generous: population-synthesis M/L for an old bulge over a star-forming disk is
# typically a factor 2-4 in the optical.
RATIOS = [float(r) for r in os.environ.get("ML_SCAN_RATIOS", "1,1.5,2,3,4,6").split(",")]

# The scan is many fits, so it runs smaller than the production configuration.
SCAN_N = int(os.environ.get("ML_SCAN_N", 12_000))
SCAN_STEPS = int(os.environ.get("ML_SCAN_STEPS", 400))
SCAN_RENDER = int(os.environ.get("ML_SCAN_RENDER", 60_000))

FORCE = bool(os.environ.get("FORCE"))


def _json_path(NAME, r):
    return os.path.join(HERE, f"{NAME}_inclined_fit_scan_ml{r:g}.json")


def run_one(NAME, r):
    """Fit a single ratio. Called in a fresh subprocess by `scan_galaxy` -- see the
    module docstring for why that isolation is required rather than merely tidy."""
    from phoenix.actions_to_phasespace.actions_to_phasespace_nn import PhoenixMapper
    FI.N_PARTICLES, FI.N_STEPS, FI.N_RENDER = SCAN_N, SCAN_STEPS, SCAN_RENDER
    outdir = os.path.join(HERE, "plots", NAME, "ml_scan")
    os.makedirs(outdir, exist_ok=True)
    FI.fit_galaxy(NAME, PhoenixMapper(), ml_ratio=r, outdir=outdir,
                  out_tag=f"_scan_ml{r:g}")


def scan_galaxy(NAME):
    outdir = os.path.join(HERE, "plots", NAME, "ml_scan")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for r in RATIOS:
        path = _json_path(NAME, r)
        if os.path.exists(path) and not FORCE:
            print(f"\n########## {NAME}: ratio {r:g} -- cached, skipping ##########")
        else:
            print(f"\n########## {NAME}: Upsilon_bulge/Upsilon_disk = {r:g} ##########",
                  flush=True)
            t0 = time.time()
            subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__),
                            "--one", NAME, repr(r)], check=True)
            print(f"  [{time.time() - t0:.0f} s]")
        with open(path) as fh:
            rows.append(json.load(fh))

    best = min(rows, key=lambda d: d["final_loss"]["loss"])
    base = next(d for d in rows if d["ml_ratio"] == 1.0)

    print(f"\n===== {NAME}: mass-to-light profile scan =====")
    print(f"{'Y_b/Y_d':>8s} {'loss':>9s} {'map':>8s} {'vrot':>8s} {'sigma':>8s}"
          f" {'RMS v':>7s} {'RMS s':>7s} {'M_bulge':>10s} {'M_disk':>10s} {'B/T':>6s}")
    for d in rows:
        fl, g, p = d["final_loss"], d["goodness_of_fit"], d["pot_params"]
        bt = p["M_bulge"] / (p["M_bulge"] + p["M_disk"])
        mark = "  <-- best" if d is best else ("  (control)" if d is base else "")
        print(f"{d['ml_ratio']:8.2g} {fl['loss']:9.4f} {fl['mass_loss']:8.4f}"
              f" {fl['vrot_loss']:8.4f} {fl['sigma_loss']:8.4f}"
              f" {g['rms_vlos']:7.1f} {g['rms_sigma']:7.1f}"
              f" {p['M_bulge']:10.3g} {p['M_disk']:10.3g} {bt:6.3f}{mark}")

    d_loss = base["final_loss"]["loss"] - best["final_loss"]["loss"]
    print(f"\n  best ratio {best['ml_ratio']:g}: loss {best['final_loss']['loss']:.4f}"
          f" vs {base['final_loss']['loss']:.4f} at the single-M/L control"
          f"  (improvement {d_loss:+.4f}, {100 * d_loss / base['final_loss']['loss']:+.1f}%)")
    print(f"  M_bulge moves {base['pot_params']['M_bulge']:.3g} ->"
          f" {best['pot_params']['M_bulge']:.3g}"
          f"  (x{best['pot_params']['M_bulge'] / base['pot_params']['M_bulge']:.2f})")
    print("  NOTE: the losses above are comparable because every row fits the SAME data;")
    print("        only the model's forward map changes with the ratio.")

    # ---------------- figure ----------------
    x = [d["ml_ratio"] for d in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(x, [d["final_loss"]["loss"] for d in rows], "ko-")
    ax[0].axvline(best["ml_ratio"], color="tab:red", ls="--", lw=1,
                  label=f"best {best['ml_ratio']:g}")
    ax[0].set_xlabel(r"$\Upsilon_{\rm bulge}/\Upsilon_{\rm disk}$")
    ax[0].set_ylabel("total loss")
    ax[0].set_title("Profile likelihood (same data at every point)")
    ax[0].legend(); ax[0].grid(alpha=.3)

    for key, lab in (("mass_loss", "map"), ("vrot_loss", "v_rot"),
                     ("sigma_loss", "sigma")):
        ax[1].plot(x, [d["final_loss"][key] for d in rows], "o-", label=lab)
    ax[1].set_xlabel(r"$\Upsilon_{\rm bulge}/\Upsilon_{\rm disk}$")
    ax[1].set_ylabel("loss component"); ax[1].set_yscale("log")
    ax[1].set_title("Which term drives it"); ax[1].legend(); ax[1].grid(alpha=.3)

    bt = [d["pot_params"]["M_bulge"] / (d["pot_params"]["M_bulge"]
                                        + d["pot_params"]["M_disk"]) for d in rows]
    ax[2].plot(x, bt, "s-", color="tab:purple")
    ax[2].set_xlabel(r"$\Upsilon_{\rm bulge}/\Upsilon_{\rm disk}$")
    ax[2].set_ylabel(r"$M_{\rm bulge}/(M_{\rm bulge}+M_{\rm disk})$")
    ax[2].set_title("Inferred bulge-to-total MASS fraction"); ax[2].grid(alpha=.3)

    fig.suptitle(f"{NAME}: mass-to-light ratio scan "
                 f"({SCAN_N} tracers/component, {SCAN_STEPS} steps)")
    fig.tight_layout()
    fpng = os.path.join(outdir, f"{NAME}_ml_scan.png")
    fig.savefig(fpng, dpi=130)
    plt.close(fig)

    out = os.path.join(HERE, f"{NAME}_ml_scan.json")
    with open(out, "w") as fh:
        json.dump({"galaxy": NAME, "ratios": x,
                   "scan_config": {"N_PARTICLES": SCAN_N, "N_STEPS": SCAN_STEPS,
                                   "N_RENDER": SCAN_RENDER},
                   "best_ratio": best["ml_ratio"],
                   "control_ratio": 1.0,
                   "rows": rows}, fh, indent=2)
    print(f"\n  wrote {fpng}\n        {out}")
    return rows


def main():
    if len(sys.argv) > 3 and sys.argv[1] == "--one":
        run_one(sys.argv[2], float(sys.argv[3]))
        return
    which = [a for a in sys.argv[1:] if a in GALAXIES] or list(FI.INCLINATIONS)
    for name in which:
        scan_galaxy(name)


if __name__ == "__main__":
    main()
