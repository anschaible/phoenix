"""
Load reduced GECKOS/MUSE (nGIST) kinematic maps into the observation format the
Phoenix optimization pipeline expects.

The nGIST output stores, per galaxy, reconstructed 2-D maps on the MUSE sky grid:
  - <..>_kin_maps.fits            : V, SIGMA, H3, H4 (+ errors), km/s
  - <..>_spatial_binning_maps.fits: FLUX (surface brightness), SNR, XBIN, YBIN (arcsec)

`load_geckos_maps` turns those into a dict with keys
  'mass', 'v_rot', 'sigma', 'x_edges', 'z_edges'
on a regular (grid_size x grid_size) grid in kpc, ready to pass to
`phoenix.optimization.pipeline.fit`. The steps:

  1. keep only valid (finite, flux>0) spaxels;
  2. convert the on-sky XBIN/YBIN offsets from arcsec to kpc (needs a distance);
  3. rotate to the galaxy's principal axes (flux-weighted inertia tensor) so the
     major axis lies along +x and the vertical/minor axis along +z — the model is
     built in exactly this edge-on (x = major, z = vertical, y = line of sight) frame;
  4. subtract the flux-weighted systemic velocity;
  5. align the rotation SENSE with the model convention (v_rot increasing toward +x);
  6. bin onto the regular grid (flux-weighted mean for V/SIGMA, flux sum for mass);
  7. rescale the light map to a fiducial stellar mass (see caveats below).

Caveats (this is real, light-based data fed to an idealized model):
  - FLUX is *light*, used as a stellar-*mass* proxy assuming constant M/L. The
    absolute normalization is unknown, so the map is rescaled to `Mstar_fiducial`;
    the fit's free M_disk/M_bulge then absorb that choice. The velocity fields
    (km/s) carry the real physical scale that constrains the potential mass.
  - `kpc_per_arcsec` requires a distance. Default is for NGC 5010 (D ~ 42 Mpc).
"""
import numpy as np
import jax.numpy as jnp
from astropy.io import fits


def load_geckos_maps(
    kin_maps_path: str,
    spatial_binning_path: str,
    kpc_per_arcsec: float = 0.204,   # NGC 5010: D ~ 42 Mpc (H0=70, v_sys~2975 km/s)
    extent_x: float = 7.0,
    extent_z: float = 4.0,
    grid_size: int = 20,
    Mstar_fiducial: float = 3e10,
    center_percentile: float = None,
):
    """Returns (obs_maps, info). `obs_maps` is the pipeline observation dict;
    `info` carries the derived position angle (deg), systemic velocity (km/s) and
    the fraction of grid cells that received data.

    center_percentile : if None (default), the field is centred on the global
        flux-weighted centroid — correct when the galaxy is roughly centred in the
        FOV. When the MUSE pointing is OFFSET so the field covers the nucleus plus
        mostly one side of the disk (e.g. NGC 3630), the centroid is dragged toward
        the covered side and misplaces the centre. Setting e.g. 90 instead centres
        on the flux-weighted centroid of only the brightest (>90th-percentile)
        spaxels — i.e. the nucleus — which is the dynamical centre. The
        principal-axis rotation is then computed about that same centre.
    """
    with fits.open(kin_maps_path) as h:
        V = np.array(h['V'].data)
        S = np.array(h['SIGMA'].data)
    with fits.open(spatial_binning_path) as h:
        F = np.array(h['FLUX'].data)
        X = np.array(h['XBIN'].data)
        Y = np.array(h['YBIN'].data)

    good = np.isfinite(F) & np.isfinite(V) & np.isfinite(S) & (F > 0)
    x = X[good] * kpc_per_arcsec
    y = Y[good] * kpc_per_arcsec
    f = F[good]; v = V[good]; s = S[good]

    # 1. centre — on the nucleus (bright core) if a percentile is given, else the
    # global flux centroid.
    if center_percentile is not None:
        core = f > np.percentile(f, center_percentile)
        wc = f[core]
        x -= np.sum(wc * x[core]) / np.sum(wc)
        y -= np.sum(wc * y[core]) / np.sum(wc)
    else:
        x -= np.sum(f * x) / np.sum(f)
        y -= np.sum(f * y) / np.sum(f)

    # 2. principal-axis rotation (flux-weighted inertia tensor about the centre)
    Ixx = np.sum(f * x * x) / np.sum(f)
    Iyy = np.sum(f * y * y) / np.sum(f)
    Ixy = np.sum(f * x * y) / np.sum(f)
    pa = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    c, sn = np.cos(-pa), np.sin(-pa)
    xr = c * x - sn * y     # major axis  -> pipeline x
    zr = sn * x + c * y     # minor axis  -> pipeline z (vertical)

    # 3. subtract systemic velocity
    vsys = np.sum(f * v) / np.sum(f)
    v = v - vsys

    # 4. bin onto the regular grid
    xe = np.linspace(-extent_x, extent_x, grid_size + 1)
    ze = np.linspace(-extent_z, extent_z, grid_size + 1)
    ix = np.clip(np.digitize(xr, xe) - 1, 0, grid_size - 1)
    iz = np.clip(np.digitize(zr, ze) - 1, 0, grid_size - 1)
    inb = (xr >= -extent_x) & (xr < extent_x) & (zr >= -extent_z) & (zr < extent_z)

    mass = np.zeros((grid_size, grid_size))
    wsum = np.zeros_like(mass); wv = np.zeros_like(mass); ws = np.zeros_like(mass)
    np.add.at(mass, (iz[inb], ix[inb]), f[inb])
    np.add.at(wsum, (iz[inb], ix[inb]), f[inb])
    np.add.at(wv,   (iz[inb], ix[inb]), f[inb] * v[inb])
    np.add.at(ws,   (iz[inb], ix[inb]), f[inb] * s[inb])
    vmap = np.where(wsum > 0, wv / np.maximum(wsum, 1e-30), 0.0)
    smap = np.where(wsum > 0, ws / np.maximum(wsum, 1e-30), 0.0)

    # 5. fiducial stellar-mass normalization (see module docstring caveats)
    mass = mass / mass.sum() * Mstar_fiducial

    # 6. align rotation sense with the model (v_rot increasing toward +x). The
    # observed 'which side approaches' is a sky-orientation convention; the
    # axisymmetric model has a fixed spin sense and rotation direction cannot be
    # flipped by any continuous parameter, so orient the data to match the model.
    xcen = (0.5 * (xe[:-1] + xe[1:]))[None, :] * np.ones_like(vmap)
    if np.sum(mass * vmap * xcen) < 0:
        vmap = -vmap

    obs_maps = {
        'mass':   jnp.array(mass),
        'v_rot':  jnp.array(vmap),
        'sigma':  jnp.array(smap),
        'x_edges': jnp.array(xe),
        'z_edges': jnp.array(ze),
    }
    info = {
        'position_angle_deg': float(np.degrees(pa)),
        'v_systemic_kms': float(vsys),
        'filled_fraction': float(np.mean(wsum > 0)),
        'kpc_per_arcsec': kpc_per_arcsec,
    }
    return obs_maps, info


def dust_lane_mask(obs_maps: dict, flux_ratio_threshold: float = 0.85,
                   z_max: float = 1.5, min_flux_frac: float = 1e-3):
    """
    Flags cells obscured by a dust lane, using the galaxy's own mirror symmetry.

    An edge-on disk is symmetric about its midplane, so at matched |z| the two sides
    should agree. A dust lane breaks that: it attenuates the far side and leaves us
    seeing mostly the near-side foreground, which is kinematically colder, so the
    obscured cells show BOTH low flux and low sigma than their mirror partner.
    Measured on NGC 5010 at |x| < 1.5 kpc the +z side has flux(+z)/flux(-z) = 0.51 and
    sigma(+z)/sigma(-z) = 0.59 at z = +0.3 kpc, recovering to ~1.0 by |z| = 1.5 kpc;
    the outer disk (1.5 < |x| < 4) shows a weaker but still systematic 0.91-0.95. The
    obscuration therefore sits exactly where the central sigma peak should be, and
    fitting it unmasked teaches the model NOT to have a central peak.

    A dust lane is a coherent one-sided band, so the obscured SIDE is determined once
    globally (whichever side of the midplane carries less flux) rather than per cell.
    Thresholding each cell's flux ratio independently is too noisy: at a 0.75
    threshold it scattered flags out to |x| = 5.4 kpc on both sides, removing the
    whole midplane row -- including the unobscured half that has to constrain the fit.

    Detection uses FLUX rather than sigma, because flux attenuation is the direct
    effect of dust while the sigma suppression is a downstream consequence; keying on
    sigma would risk masking away real kinematic structure.

    Only the obscured side is flagged, so the mirror side still constrains the model
    at those heights -- which works because the model is symmetric about z = 0.

    Args:
        obs_maps (dict): Observation dict from `load_geckos_maps` (needs 'mass' and
            'z_edges').
        flux_ratio_threshold (float): Flag a cell when its flux falls below this
            fraction of its mirror partner's.
        z_max (float): Only consider |z| <= z_max (kpc). Beyond the measured recovery
            height the asymmetry is noise, and masking there would just discard data.
        min_flux_frac (float): Skip cells fainter than this fraction of the peak;
            their flux ratios are noise.

    Returns:
        np.ndarray: Boolean array, True where the cell is dust-obscured.
    """
    mass = np.asarray(obs_maps['mass'])
    z_edges = np.asarray(obs_maps['z_edges'])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])

    observed = mass > 0
    bright = mass > mass.max() * min_flux_frac
    near = np.abs(zc)[:, None] <= z_max

    # Which side is obscured? Compare total flux above vs below the midplane within
    # z_max. One global decision, so the mask stays a coherent band.
    upper = observed & near & (zc[:, None] > 0)
    lower = observed & near & (zc[:, None] < 0)
    obscured_is_upper = mass[upper].sum() < mass[lower].sum()
    on_dusty_side = (zc[:, None] > 0) if obscured_is_upper else (zc[:, None] < 0)

    flipped = mass[::-1, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(flipped > 0, mass / flipped, np.inf)

    return observed & bright & near & on_dusty_side & (ratio < flux_ratio_threshold)
