# soft_sampling.py
"""
Differentiable 'soft-accept' sampling of actions (JR, Jz, Lz) from your
quasi-isothermal total DF, using a sigmoid envelope.

Integrates with:
- potentials.py  (your potentials)
- distributions.py (your DF incl. f_total_disc_from_params)
"""

from typing import Callable, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.nn import sigmoid

# --- import your code ---
from phoenix.potentials import (
    miyamoto_nagai_potential,
    plummer_potential,
    isochrone_potential,
    nfw_potential,
    logarithmic_potential,
    harmonic_potential,
)
from phoenix.distributionfunctions import f_total_disc_from_params

# ---------------------------------------------------------------------
# 1) Potential resolver
# ---------------------------------------------------------------------
def _resolve_phi_and_theta(params: Dict) -> Tuple[Callable, Tuple]:
    """
    Returns (Phi_xyz, theta_tuple).
    Options:
      A) Put a callable directly in params["Phi_xyz"] and a tuple/list in params["theta"].
      B) Or pass params["Phi_name"] in {"MN","Plummer","Isochrone","NFW","Log","Harm"} and
         provide the needed scalar parameters in params (see below).
    """
    # Preferred: directly provide callable + theta
    if "Phi_xyz" in params and "theta" in params:
        Phi_xyz = params["Phi_xyz"]
        theta   = tuple(params["theta"]) if not isinstance(params["theta"], tuple) else params["theta"]
        return Phi_xyz, theta

    # Convenience path via short names
    name = params.get("Phi_name", "MN").lower()
    if name in ("mn", "miyamoto", "miyamoto-nagai", "miyamoto_nagai"):
        # needs (M, a, b)
        Phi_xyz = miyamoto_nagai_potential
        theta = (jnp.asarray(params["M"]),
                 jnp.asarray(params["a"]),
                 jnp.asarray(params["b"]))
        return Phi_xyz, theta

    if name in ("plummer", "plum"):
        # needs (M, a)
        Phi_xyz = plummer_potential
        theta = (jnp.asarray(params["M"]), jnp.asarray(params["a"]))
        return Phi_xyz, theta

    if name in ("isochrone", "iso"):
        # needs (M, a)
        Phi_xyz = isochrone_potential
        theta = (jnp.asarray(params["M"]), jnp.asarray(params["a"]))
        return Phi_xyz, theta

    if name in ("nfw",):
        # needs (M, a)
        Phi_xyz = nfw_potential
        theta = (jnp.asarray(params["M"]), jnp.asarray(params["a"]))
        return Phi_xyz, theta

    if name in ("log", "logarithmic", "logpot"):
        # needs (v0, rcore, p, q)
        Phi_xyz = logarithmic_potential
        theta = (jnp.asarray(params["v0"]),
                 jnp.asarray(params["rcore"]),
                 jnp.asarray(params["p"]),
                 jnp.asarray(params["q"]))
        return Phi_xyz, theta

    if name in ("harm", "harmonic"):
        # needs (Omega, p, q)
        Phi_xyz = harmonic_potential
        theta = (jnp.asarray(params["Omega"]),
                 jnp.asarray(params["p"]),
                 jnp.asarray(params["q"]))
        return Phi_xyz, theta

    raise ValueError(
        "Unrecognized potential. Provide params['Phi_xyz']+params['theta'], "
        "or use params['Phi_name'] in {MN,Plummer,Isochrone,NFW,Log,Harm} with required scalars."
    )


# ---------------------------------------------------------------------
# 2) DF wrapper with the exact signature the 'alien' code expects
# ---------------------------------------------------------------------
def df_total_potential(JR: float, Jz: float, Lz: float, params: Dict) -> jnp.ndarray:
    """
    Scalar -> scalar wrapper that calls your f_total_disc_from_params.
    The 'params' dict should include the DF parameters (thin/thick blocks etc.)
    and EITHER:
      - "Phi_xyz": callable, "theta": tuple/list
      - OR "Phi_name": short name + required scalars (see _resolve_phi_and_theta)
    """
    Phi_xyz, theta = _resolve_phi_and_theta(params)
    return f_total_disc_from_params(JR, Jz, Lz, Phi_xyz, theta, params)


# ---------------------------------------------------------------------
# 3) Soft acceptance + sampler
# ---------------------------------------------------------------------
def soft_acceptance(df_vals: jnp.ndarray,
                    rand_vals: jnp.ndarray,
                    envelope_max: float,
                    tau: float = 0.01) -> jnp.ndarray:
    """
    Sigmoid-soft version of accept/reject. Values in (0,1).
    'envelope_max' should be an upper bound on DF over the proposal box.
    Smaller tau -> sharper step.
    """
    return sigmoid((df_vals / envelope_max - rand_vals) / tau)


def sample_df_potential(key: jax.Array,
                        params: Dict,
                        n_candidates: int,
                        envelope_max: float,
                        tau: float = 0.01,
                        JR_max: float = 200.0,
                        Jz_max: float = 50.0,
                        Lz_max: float = 4000.0):
    """
    Generates candidates uniformly in a box in action-space and returns:
      weighted_candidates: (n, 3) = candidates * soft_weight
      soft_weights:       (n,)

    The function is fully JAX-differentiable w.r.t. 'params' (and via DF).
    """
    # Draw uniform candidates
    key, k1 = random.split(key)
    JR  = random.uniform(k1, (n_candidates,), minval=0.0, maxval=JR_max)
    key, k2 = random.split(key)
    Jz  = random.uniform(k2, (n_candidates,), minval=0.0, maxval=Jz_max)
    key, k3 = random.split(key)
    Lz  = random.uniform(k3, (n_candidates,), minval=0.0, maxval=Lz_max)

    candidates = jnp.stack([JR, Jz, Lz], axis=1)  # (n,3)

    # Resolve Phi,theta ONCE and close over them to keep JIT happy
    Phi_xyz, theta = _resolve_phi_and_theta(params)

    # Vectorized DF evaluator: map (JR,Jz,Lz) -> f_total_disc_from_params(...)
    def _df_one(jr, jz, lz):
        return f_total_disc_from_params(jr, jz, lz, Phi_xyz, theta, params)

    df_vec = jax.jit(vmap(lambda c: _df_one(c[0], c[1], c[2])))

    df_vals = df_vec(candidates)  # (n,)

    # Uniform [0,1]
    key, k4 = random.split(key)
    u = random.uniform(k4, (n_candidates,))

    # Soft weights in (0,1)
    w = soft_acceptance(df_vals, u, envelope_max, tau)  # (n,)
    print("df stats:", float(df_vals.min()), float(df_vals.max()))
    print("ratio max (df/envelope):", float(df_vals.max()/envelope_max))
    print("weights stats:", float(w.min()), float(w.max()), float(w.mean()))
    print("envelope_max:", envelope_max)

    weighted_candidates = candidates * w[:, None]       # (n,3)
    return weighted_candidates, w, candidates, df_vals


# ---------------------------------------------------------------------
# 4) Optional: cheap envelope_max estimator by probing DF on a grid/MC
# ---------------------------------------------------------------------
def estimate_envelope_max(key: jax.Array,
                          params: Dict,
                          n_probe: int = 50_000,
                          JR_max: float = 200.0,
                          Jz_max: float = 50.0,
                          Lz_max: float = 4000.0,
                          safety: float = 1.5) -> float:
    """
    Randomly probe the DF in the proposal box to get a robust upper bound.
    Returns: float envelope_max ≈ safety * max(sampled DF).
    """
    key, k1 = random.split(key)
    JR  = random.uniform(k1, (n_probe,), minval=0.0, maxval=JR_max)
    key, k2 = random.split(key)
    Jz  = random.uniform(k2, (n_probe,), minval=0.0, maxval=Jz_max)
    key, k3 = random.split(key)
    Lz  = random.uniform(k3, (n_probe,), minval=0.0, maxval=Lz_max)

    Phi_xyz, theta = _resolve_phi_and_theta(params)

    def _df_one(jr, jz, lz):
        return f_total_disc_from_params(jr, jz, lz, Phi_xyz, theta, params)

    df_vals = vmap(_df_one)(JR, Jz, Lz)
    m = jnp.max(df_vals)
    # Clip to avoid 0 in degenerate configs
    return float(jnp.clip(safety * m, 1e-20))


# --- plotting helper ---
import numpy as np
import matplotlib.pyplot as plt

def plot_action_histograms(candidates: jnp.ndarray,
                           weights: jnp.ndarray | None = None,
                           bins: int = 80,
                           figsize=(10, 3.2),
                           density=True,
                           titles=("J_R", "J_z", "L_z"),
                           savepath: str | None = None):
    """
    Plot histograms for actions (J_R, J_z, L_z).

    Parameters
    ----------
    candidates : (N,3) JAX array
        Columns are [J_R, J_z, L_z].
    weights : (N,) JAX array or None
        Soft weights. If None, unweighted histograms are shown.
    bins : int
        Number of bins.
    density : bool
        If True, normalize to area=1 (probability density).
    savepath : str or None
        If given, save the figure to this path (e.g. 'actions_hist.png').

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """
    X = np.asarray(candidates)           # convert from JAX to NumPy for matplotlib
    W = None if weights is None else np.asarray(weights)

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.hist(X[:, i], bins=bins, weights=W, density=density)
        ax.set_xlabel(titles[i])
        ax.set_ylabel("density" if density else "count")
        ax.grid(True, linestyle="--", alpha=0.4)
    if savepath:
        fig.savefig(savepath, dpi=150)
    else:
        plt.show()
    return fig, axes
