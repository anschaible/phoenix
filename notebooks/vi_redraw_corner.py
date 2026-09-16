"""
Redraw the VI corner plots from the saved posterior, without re-running inference.

`vi_self_consistent.py` writes `figures/VI_theory_posterior.npz`; the fit and the ELBO
optimization behind it cost ~20 minutes, while the plots cost a second. Use this to
iterate on the figure, or to plot a subset of parameters.

    python notebooks/vi_redraw_corner.py                 # all inferred parameters
    python notebooks/vi_redraw_corner.py M_halo a_halo M_disk a_disk b_disk
    python notebooks/vi_redraw_corner.py --group pot     # one group: pot / disk / bulge
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vi_self_consistent import corner_plot, FIGDIR   # reuse the plotting code

NPZ = os.path.join(FIGDIR, "VI_theory_posterior.npz")


def main(argv):
    if not os.path.exists(NPZ):
        raise SystemExit(f"{NPZ} not found -- run notebooks/vi_self_consistent.py first.")
    d = np.load(NPZ, allow_pickle=True)
    labels = [str(x) for x in d["labels"]]
    groups = [str(x) for x in d["groups"]]

    if len(argv) >= 2 and argv[0] == "--group":
        keep = [i for i, g in enumerate(groups) if g == argv[1]]
        tag, units = argv[1], "phys"
    elif argv:
        missing = [a for a in argv if a not in labels]
        if missing:
            raise SystemExit(f"unknown parameter(s): {', '.join(missing)}\n"
                             f"available: {', '.join(labels)}")
        keep = [labels.index(a) for a in argv]
        tag, units = "subset", "phys"
    else:
        keep = list(range(len(labels)))
        tag, units = "all", "log"

    k = np.array(keep)
    out = os.path.join(FIGDIR, f"VI_theory_corner_{tag}.png")
    corner_plot(
        out, d["mu_log"][k], d["cov_log"][np.ix_(k, k)], [labels[i] for i in k],
        truth_log=d["truth_log"][k], map_log=d["map_log"][k],
        at_bound=d["at_bound"][k], units=units,
        title="Variational posterior of the self-consistent twin experiment",
        subtitle=(f"{len(k)} parameter(s); axes in "
                  + ("log-offset from truth" if units == "log" else "physical units")),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
