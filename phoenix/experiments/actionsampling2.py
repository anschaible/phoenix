# soft_sampling.py
"""
Soft-Akzeptanz für die Binney-DF:
- zieht uniforme Kandidaten in einer Box in Actions
- wertet die Gesamt-DF f(J) via f_total_disc_from_params(...) aus
- berechnet weiche Gewichte w in [0,1] mit einer Sigmoid-Glättung
- gibt (candidates, weights, C) zurück; optional kann man danach hart resamplen
"""
from functools import partial
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.nn import sigmoid
from phoenix.distributionfunctions import f_total_disc_from_params

# -------- NO-JIT helpers --------

def _uniform_candidates_nojit(key, n: int, JR_max: float, JZ_max: float, JPHI_max: float):
    key, k1 = random.split(key)
    JR  = random.uniform(k1, (n,), minval=0.0,       maxval=JR_max)
    key, k2 = random.split(key)
    JZ  = random.uniform(k2, (n,), minval=0.0,       maxval=JZ_max)
    key, k3 = random.split(key)
    JPH = random.uniform(k3, (n,), minval=0.0, maxval=JPHI_max)
    cands = jnp.stack([JR, JZ, JPH], axis=1)
    return key, cands

def _estimate_C_nojit(key, Phi_xyz, theta, params, JR_max, JZ_max, JPHI_max,
                      n_probe=16384, safety=2.0) -> float:
    key, probe = _uniform_candidates_nojit(key, n_probe, JR_max, JZ_max, JPHI_max)
    df_vec = vmap(lambda v: f_total_disc_from_params(v[0], v[1], v[2], Phi_xyz, theta, params))
    vals = df_vec(probe)
    # sanitize BEFORE quantile
    vals = jnp.nan_to_num(vals, nan=0.0, posinf=1e300, neginf=0.0)
    # if everything is zero, quantile=0; that’s fine, we’ll bump it below
    q999 = jnp.quantile(vals, 0.999)
    C = safety * q999
    # robust fallback if C<=0 or not finite
    finite_max = jnp.max(jnp.where(jnp.isfinite(vals), vals, 0.0))
    C = jnp.where(jnp.isfinite(C) & (C > 0.0), C, 1.1 * finite_max + 1e-6)
    return float(C)

# -------- JIT inner core (Phi_xyz at arg 1, n_candidates at arg 4 are static) --------

@partial(jax.jit, static_argnums=(1, 4))
def _sample_df_potential_soft_inner(
    key,
    Phi_xyz,                 # static
    theta,
    params: Dict,
    n_candidates: int,       # static
    JR_max: float,
    JZ_max: float,
    JPHI_max: float,
    tau: float,
    envelope_C: float,       # concrete scalar passed in
):
    key, k1 = random.split(key)
    JR  = random.uniform(k1, (n_candidates,), minval=0.0,        maxval=JR_max)
    key, k2 = random.split(key)
    JZ  = random.uniform(k2, (n_candidates,), minval=0.0,        maxval=JZ_max)
    key, k3 = random.split(key)
    JPH = random.uniform(k3, (n_candidates,), minval=-JPHI_max,  maxval=JPHI_max)
    candidates = jnp.stack([JR, JZ, JPH], axis=1)

    df_vec = vmap(lambda v: f_total_disc_from_params(v[0], v[1], v[2], Phi_xyz, theta, params))
    df_vals = df_vec(candidates)

    # sanitize df
    df_vals = jnp.nan_to_num(df_vals, nan=0.0, posinf=1e300, neginf=0.0)
    # make safe C and tau
    Cj = jnp.asarray(envelope_C, dtype=df_vals.dtype)
    Cj = jnp.where(jnp.isfinite(Cj) & (Cj > 0.0), Cj, jnp.maximum(jnp.max(df_vals)*1.1, 1e-6))
    tau = jnp.asarray(tau, dtype=df_vals.dtype)
    tau = jnp.clip(tau, 1e-6)

    key, kU = random.split(key)
    U = random.uniform(kU, (n_candidates,))
    w = jax.nn.sigmoid((df_vals / Cj - U) / tau)
    # optional: zero-out any residual non-finite weights (paranoia)
    w = jnp.where(jnp.isfinite(w), w, 0.0)
    return candidates, w

# -------- public wrapper (DO NOT JIT THIS) --------

def sample_df_potential_soft(
    key,
    Phi_xyz,
    theta,
    params: Dict,
    n_candidates: int,
    JR_max: float = 200.0,
    JZ_max: float = 50.0,
    JPHI_max: float = 3200.0,
    tau: float = 0.02,
    envelope_C: Optional[float] = None,
):
    # compute C OUTSIDE jit
    if envelope_C is None:
        key, kC = random.split(key)
        envelope_C = _estimate_C_nojit(kC, Phi_xyz, theta, params, JR_max, JZ_max, JPHI_max)

    # call jitted inner
    cands, w = _sample_df_potential_soft_inner(
        key, Phi_xyz, theta, params, n_candidates,
        JR_max, JZ_max, JPHI_max, tau, envelope_C
    )
    return cands, w, envelope_C
