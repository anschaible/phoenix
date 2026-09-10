"""
Fit the Phoenix distribution-function + potential model to real GECKOS/MUSE
edge-on galaxies (nGIST-reduced kinematic maps).

Run from the repo root in the `phoenix` conda env:

    conda run -n phoenix python notebooks/geckos/fit_geckos.py            # all galaxies
    conda run -n phoenix python notebooks/geckos/fit_geckos.py NGC3630    # one galaxy

For each galaxy it loads the observed velocity/dispersion (and light) maps
(see geckos_data.py), fits the model with gradient descent (log-space params,
physical bounds, shape-only Poisson self-consistency, bandwidth annealing —
see phoenix.optimization.pipeline), and writes diagnostic plots + fitted
parameters into notebooks/geckos/plots/<NAME>/.

The fit is KINEMATIC-ONLY: only the line-of-sight velocity and dispersion maps
drive the loss (mass weight = 0). The MUSE FLUX is a light (not mass) proxy with
a sharp nucleus the idealized 3-component model cannot reproduce, and including
it dragged the fit and washed out the rotation. Kinematics (km/s) are what
constrain the potential, so we fit those and only *show* the light map for
reference. Two per-galaxy modelling choices matter a lot:
  - L0_bounds: L0 is the qDF angular-momentum scale below which disk orbits
    counter-rotate; a real thin disk rotates coherently, so L0 must be small.
    Left free the fit drifts to large L0 (weak net rotation) and the clean
    rotation dipole is lost, so we bound it small.
  - spheroid_corotation: a central sigma PEAK is the signature of a
    dispersion-supported (non-rotating) bulge, so a galaxy with a hot core
    (NGC 3630) uses ~0.5; a bulge that rotates with the disk uses ~1.0.

Distances and stellar masses are taken from the GECKOS master catalogue
(jvds_geckos_mastercat_all_v20250610): kpc/arcsec = D[Mpc]*1e3/206265,
Mstar_fiducial = 10**lmstar.

Caveats (real, light-based data fed to an idealized 3-component model):
  - FLUX is light used as a stellar-mass proxy (constant M/L); the absolute
    normalization is set to the catalogue Mstar and the free M_disk/M_bulge absorb
    it. The velocities (km/s) carry the physical scale that constrains the potential.
  - For an OFFSET pointing that covers mostly one side of the disk (NGC 3630) the
    field is centred on the nucleus (center_percentile) and the symmetric model is
    compared only where data exist; the stellar-mass normalization is then only
    loosely constrained (one-sided coverage).
"""
import os
import sys
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geckos_data import load_geckos_maps
from phoenix.actions_to_phasespace.actions_to_phasespace_nn import PhoenixMapper
from phoenix.optimization.pipeline import fit, DEFAULT_PARAM_BOUNDS
from phoenix.optimization.observables import generate_edge_on_maps, render_maps_batched

HERE = os.path.dirname(os.path.abspath(__file__))
ARCSEC_PER_RAD = 206265.0

# ---------------------------------------------------------------------------
# Per-galaxy configuration (distances/Mstar from the GECKOS master catalogue)
# ---------------------------------------------------------------------------
GALAXIES = {
    "NGC5010": dict(
        subdir="7_NGC5010", stem="NGC5010_iDR2.0_1_SN100_5_MW_4800-8900",
        d_mpc=42.2, lmstar=10.6833,
        extent_x=7.0, extent_z=4.0, center_percentile=None,
        spheroid_corotation=1.0,   # rotation-dominated, no strong hot core
        L0_bounds=(2.0, 8.0),
        init_pot={'M_halo': 8e11, 'a_halo': 15.0, 'M_disk': 4e10, 'a_disk': 2.0,
                  'b_disk': 0.2, 'M_bulge': 5e9, 'a_bulge': 0.5},
        init_disk={'R0': 4.0, 'Rd': 1.5, 'Sigma0': 1000.0, 'RsigR': 5.0, 'RsigZ': 5.0,
                   'sigmaR0_R0': 60.0, 'sigmaz0_R0': 40.0, 'L0': 3.0, 'Rinit_for_Rc': 4.0},
        init_bulge={'N0_spheroid': 5e9, 'J0_spheroid': 60.0, 'Gamma_spheroid': 1.5,
                    'Beta_spheroid': 4.5, 'eta_spheroid': 1.0},
    ),
    "NGC3630": dict(
        subdir="35_NGC3630", stem="NGC3630_iDR2.0_1_SN100_5_MW_4800-8900",
        d_mpc=30.6, lmstar=10.4904,
        # offset pointing -> centre on the nucleus, one-sided disk out to ~6 kpc
        extent_x=6.0, extent_z=3.0, center_percentile=92,
        spheroid_corotation=0.5,   # hot dispersion-supported core (central sigma peak)
        L0_bounds=(2.0, 5.0),
        init_pot={'M_halo': 4e11, 'a_halo': 10.0, 'M_disk': 4e10, 'a_disk': 2.5,
                  'b_disk': 0.3, 'M_bulge': 3e10, 'a_bulge': 0.3},
        init_disk={'R0': 4.0, 'Rd': 1.5, 'Sigma0': 1000.0, 'RsigR': 5.0, 'RsigZ': 5.0,
                   'sigmaR0_R0': 60.0, 'sigmaz0_R0': 40.0, 'L0': 3.0, 'Rinit_for_Rc': 4.0},
        init_bulge={'N0_spheroid': 3e10, 'J0_spheroid': 25.0, 'Gamma_spheroid': 1.5,
                    'Beta_spheroid': 4.5, 'eta_spheroid': 1.0},
    ),
}

# shared fit hyperparameters
GRID_SIZE = 40
# Many tracers: the model maps are Monte-Carlo estimates, and with too few
# particles each cell's velocity/dispersion is shot-noise-dominated (grainy),
# which inflates the residual against the smooth Voronoi-binned data. More
# tracers -> smoother model maps -> a genuinely lower-noise, better fit.
N_PARTICLES = 35_000          # tracers for the fit (more -> cleaner gradient, better fit)
N_PARTICLES_RENDER = 200_000  # tracers for the final display maps (smooth, low shot noise)
SEED = 0
# PURE KINEMATIC loss: (mass, v_rot, sigma, poisson). Only v_rot and sigma drive
# the fit — mass (light) and the Poisson self-consistency term are both off so the
# optimizer focuses entirely on reproducing the velocity and dispersion fields.
# sigma is up-weighted so the central dispersion peak (a hot bulge signature) is
# reproduced rather than smoothed over.
# (Self-consistency is therefore not enforced; re-enable the Poisson weight if a
# dynamically self-consistent solution is required.)
LOSS_WEIGHTS = (0.0, 2.0, 3.0, 0.0)
LEARNING_RATE = 0.05
N_STEPS = 700
ANNEAL_BANDWIDTH = (4.0, 0.4)
# Tikhonov (ridge) prior regularization toward the physically-motivated initial
# guess (in log-space). The kinematics leave several parameters degenerate (bulge
# DF shape, DF normalizations, disk thickness) so unregularized they drift to
# unphysical values / their bounds while barely changing the fit; a mild anchor
# keeps them sensible at negligible cost to the v_rot/sigma match. Set 0 to disable.
REG_WEIGHT = 0.05


def fit_galaxy(name, cfg, mapper):
    outdir = os.path.join(HERE, "plots", name)
    os.makedirs(outdir, exist_ok=True)
    datadir = os.path.join(HERE, "data", cfg["subdir"], "maps", cfg["stem"])
    kin = os.path.join(datadir, f"{cfg['stem']}_kin_maps.fits")
    sb = os.path.join(datadir, f"{cfg['stem']}_spatial_binning_maps.fits")
    kpc_per_arcsec = cfg["d_mpc"] * 1e3 / ARCSEC_PER_RAD
    Mstar = 10.0 ** cfg["lmstar"]
    EX, EZ = cfg["extent_x"], cfg["extent_z"]

    print(f"\n===== {name} =====")
    print(f"  D = {cfg['d_mpc']} Mpc -> {kpc_per_arcsec:.4f} kpc/arcsec ; "
          f"logM* = {cfg['lmstar']} -> {Mstar:.3g} Msun")
    obs_maps, info = load_geckos_maps(
        kin, sb, kpc_per_arcsec=kpc_per_arcsec, extent_x=EX, extent_z=EZ,
        grid_size=GRID_SIZE, Mstar_fiducial=Mstar, center_percentile=cfg["center_percentile"],
    )
    print(f"  PA={info['position_angle_deg']:.1f} deg  vsys={info['v_systemic_kms']:.1f} km/s  "
          f"filled={info['filled_fraction']*100:.0f}%")
    mass_mask = np.array(obs_maps['mass']) > 0
    vmax_mass = float(np.nanmax(np.array(obs_maps['mass'])))

    # bound L0 small so the disk rotates coherently (see module docstring)
    param_bounds = copy.deepcopy(DEFAULT_PARAM_BOUNDS)
    param_bounds['disk']['L0'] = cfg['L0_bounds']
    # Cap the DISK velocity dispersion to physical values. The default bound (up to
    # 300 km/s) let the fit make an unphysically hot disk to fake the central
    # dispersion, so the bulge's sigma peak never appeared. Capping the disk forces
    # the central sigma peak to come from the (dispersion-supported) bulge instead.
    param_bounds['disk']['sigmaR0_R0'] = (5.0, 120.0)
    param_bounds['disk']['sigmaz0_R0'] = (5.0, 90.0)

    result = fit(
        mapper, obs_maps, cfg["init_pot"], cfg["init_disk"], cfg["init_bulge"],
        N_disk=N_PARTICLES, N_bulge=N_PARTICLES, grid_size=GRID_SIZE,
        extent_x=EX, extent_z=EZ, prng_seed=SEED,
        loss_weights=LOSS_WEIGHTS, poisson_kwargs={},
        learning_rate=LEARNING_RATE, n_steps=N_STEPS, anneal_bandwidth=ANNEAL_BANDWIDTH,
        spheroid_corotation=cfg["spheroid_corotation"], param_bounds=param_bounds,
        reg_weight=REG_WEIGHT,
    )
    h = result['history']
    print(f"  loss {h['loss'][0]:.3f} -> {h['loss'][-1]:.3f}  "
          f"(vrot={h['vrot_loss'][-1]:.3f} sigma={h['sigma_loss'][-1]:.3f} "
          f"reg={h['reg'][-1]:.3f})")

    # Render the fitted model with many tracers and at the bandwidth the fit
    # converged to (final annealing value), batched to bound memory. This removes
    # the Monte-Carlo graininess so the model maps are as smooth as the data while
    # keeping the physical features.
    render_h = ANNEAL_BANDWIDTH[1] if ANNEAL_BANDWIDTH is not None else None
    model_maps = render_maps_batched(
        mapper, result['pot_params'], result['disk_df_params'], result['bulge_df_params'],
        N_disk=N_PARTICLES_RENDER, N_bulge=N_PARTICLES_RENDER, grid_size=GRID_SIZE,
        extent_x=EX, extent_z=EZ, prng_seed=SEED, soft_bin_h=render_h,
        spheroid_corotation=cfg["spheroid_corotation"], chunk=15_000,
    )
    _make_plots(name, cfg, info, obs_maps, model_maps, result, mass_mask, vmax_mass, EX, EZ, outdir)
    print(f"  wrote plots + parameters to {outdir}/")
    return result


def _plot_maps(name, obs_maps, model_maps, mass_mask, vmax_mass, EX, EZ, outdir):
    """Maps comparison (fitted kinematics only) + observed-light reference panel.
    Split out so it can be regenerated without re-running the fit."""
    extent = [-EX, EX, -EZ, EZ]

    def masked(a):
        return np.where(mass_mask, np.array(a), np.nan)

    # 1. observed / model / residual for the FITTED quantities only (v_rot, sigma).
    # The light/mass map is not part of the kinematic-only fit, so it is not shown
    # here; a separate reference panel (below) shows the observed light for context.
    specs = [
        ('v_rot', 'seismic', dict(vmin=-200, vmax=200), 'V$_{LOS}$ (km/s)'),
        ('sigma', 'viridis', dict(vmin=0, vmax=180), '$\\sigma_{LOS}$ (km/s)'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for row, (key, cmap, kw, label) in enumerate(specs):
        im0 = axes[row, 0].imshow(masked(obs_maps[key]), origin='lower', extent=extent, cmap=cmap, aspect='auto', **kw)
        axes[row, 0].set_title(f'Observed: {key}'); fig.colorbar(im0, ax=axes[row, 0], label=label)
        im1 = axes[row, 1].imshow(masked(model_maps[key]), origin='lower', extent=extent, cmap=cmap, aspect='auto', **kw)
        axes[row, 1].set_title(f'Model: {key}'); fig.colorbar(im1, ax=axes[row, 1], label=label)
        resid = np.where(mass_mask, np.array(obs_maps[key]) - np.array(model_maps[key]), np.nan)
        im2 = axes[row, 2].imshow(resid, origin='lower', extent=extent, aspect='auto', cmap='coolwarm', vmin=-60, vmax=60)
        axes[row, 2].set_title(f'Residual: {key}'); fig.colorbar(im2, ax=axes[row, 2], label='obs - model (km/s)')
        for c in range(3):
            axes[row, c].set_xlabel('x [kpc]'); axes[row, c].set_ylabel('z [kpc]')
    fig.suptitle(f'{name}: observed vs. fitted Phoenix model (kinematic-only fit: '
                 f'V$_{{LOS}}$ and $\\sigma_{{LOS}}$)', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{name}_maps_comparison.png'), dpi=110); plt.close(fig)

    # 1b. observed light map, shown separately for reference (NOT fitted)
    fig, axL = plt.subplots(figsize=(6, 4.5))
    imL = axL.imshow(masked(obs_maps['mass']), origin='lower', extent=extent, cmap='magma',
                     norm=LogNorm(vmin=vmax_mass / 1e3, vmax=vmax_mass), aspect='auto')
    axL.set_title(f'{name}: observed light (reference only, not fitted)')
    axL.set_xlabel('x [kpc]'); axL.set_ylabel('z [kpc]')
    fig.colorbar(imL, ax=axL, label='surface brightness (fiducial M$_\\odot$)')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{name}_observed_light.png'), dpi=110); plt.close(fig)


def _make_plots(name, cfg, info, obs_maps, model_maps, result, mass_mask, vmax_mass, EX, EZ, outdir):
    extent = [-EX, EX, -EZ, EZ]
    h = result['history']
    _plot_maps(name, obs_maps, model_maps, mass_mask, vmax_mass, EX, EZ, outdir)

    # 2. convergence — only show the loss components that are actually fitted
    #    (weight > 0), so an unfitted term (e.g. mass) never appears here.
    w_mass, w_vrot, w_sigma, w_pois = LOSS_WEIGHTS
    show_pois = w_pois > 0
    fig, ax = plt.subplots(1, 2 if show_pois else 1, figsize=(13 if show_pois else 7, 4), squeeze=False)
    a0 = ax[0, 0]
    a0.plot(h['loss'], 'k', lw=2, label='total (fitted)')
    if w_mass > 0:
        a0.plot(h['mass_loss'], alpha=0.8, label='mass')
    if w_vrot > 0:
        a0.plot(h['vrot_loss'], alpha=0.8, label='v_rot')
    if w_sigma > 0:
        a0.plot(h['sigma_loss'], alpha=0.8, label='sigma')
    a0.set_yscale('log'); a0.set_xlabel('iteration'); a0.set_ylabel('loss (log)')
    a0.set_title('Kinematic-fit loss'); a0.legend()
    if show_pois:
        ax[0, 1].plot(h['poisson_penalty'], color='tab:red')
        ax[0, 1].set_xlabel('iteration'); ax[0, 1].set_ylabel('Poisson penalty')
        ax[0, 1].set_title('Dynamical self-consistency')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{name}_convergence.png'), dpi=110); plt.close(fig)

    # 3. fitted parameters (text + change-from-init bars)
    def fmt(d):
        return "\n".join(f"    {k:16s} = {v:12.4g}" for k, v in d.items())
    txt = "\n".join([
        f"{name} - fitted Phoenix model parameters", "=" * 46,
        f"distance          : {cfg['d_mpc']} Mpc", f"position angle    : {info['position_angle_deg']:.1f} deg",
        f"systemic velocity : {info['v_systemic_kms']:.1f} km/s", f"final loss        : {h['loss'][-1]:.4f}",
        f"  mass={h['mass_loss'][-1]:.4f} vrot={h['vrot_loss'][-1]:.4f} "
        f"sigma={h['sigma_loss'][-1]:.4f} poisson={h['poisson_penalty'][-1]:.4f}",
        "", "potential parameters:", fmt(result['pot_params']),
        "", "disk DF parameters:", fmt(result['disk_df_params']),
        "", "bulge DF parameters:", fmt(result['bulge_df_params']),
    ])
    with open(os.path.join(outdir, f'{name}_fitted_parameters.txt'), 'w') as fp:
        fp.write(txt + "\n")

    groups = [('potential', cfg['init_pot'], result['pot_params']),
              ('disk DF', cfg['init_disk'], result['disk_df_params']),
              ('bulge DF', cfg['init_bulge'], result['bulge_df_params'])]
    palette = {'potential': 'tab:blue', 'disk DF': 'tab:orange', 'bulge DF': 'tab:green'}
    keys, ratio, colors = [], [], []
    for gname, gi, gf in groups:
        for k in gi:
            keys.append(k); ratio.append(float(gf[k]) / gi[k]); colors.append(palette[gname])
    fig, ax = plt.subplots(figsize=(9, 8))
    yv = np.arange(len(keys))
    ax.barh(yv, ratio, color=colors); ax.axvline(1.0, color='k', ls=':', lw=1)
    ax.set_xscale('log'); ax.set_yticks(yv); ax.set_yticklabels(keys); ax.invert_yaxis()
    ax.set_xlabel('fitted / initial'); ax.set_title(f'{name}: parameter change from initial guess')
    ax.legend(handles=[Patch(facecolor=palette[g], label=g) for g in palette], loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{name}_parameters.png'), dpi=110); plt.close(fig)


def main():
    which = [a for a in sys.argv[1:] if a in GALAXIES] or list(GALAXIES)
    mapper = PhoenixMapper()
    for name in which:
        fit_galaxy(name, GALAXIES[name], mapper)


if __name__ == "__main__":
    main()
