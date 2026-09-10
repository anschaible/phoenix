import jax
import jax.numpy as jnp

from phoenix.distribution_functions.sampling import sample_df_potential, sample_df_importance
from phoenix.distribution_functions.frequencies import kappa, nu, vcirc
from phoenix.distribution_functions.disk import f_disc_from_params
from phoenix.distribution_functions.spheroid import f_double_power_law
from phoenix.potentials.potentials import nfw_potential, plummer_potential, miyamoto_nagai_potential



# HELPER FUNCTIONS
def spheroid_df_wrapper(Jr, Jz, Jphi, Phi_xyz, theta, params):
    return f_double_power_law(Jr, Jz, Jphi, params)


def sample_and_map_particles(
    mapper,
    pot_params: dict,
    disk_df_params: dict,
    bulge_df_params: dict,
    N_disk: int = 100_000,
    N_bulge: int = 100_000,
    prng_seed: int = 42,
    spheroid_corotation: float = 0.5,
    sampler: str = "soft",
):
    """
    Samples the disk and bulge distribution functions (DFs) in the given gravitational 
    potential and maps the resulting actions to 3D phase space via the Phoenix surrogate.

    This function performs differentiable rejection sampling of the actions, queries the 
    neural network to obtain phase-space coordinates, and applies a dynamic net-rotation 
    correction to the bulge population.

    Args:
        mapper: The Phoenix neural network surrogate used to map actions to phase space.
        pot_params (dict): Gravitational potential parameters. Expected keys:
            'M_halo', 'a_halo', 'M_disk', 'a_disk', 'b_disk', 'M_bulge', 'a_bulge'.
        disk_df_params (dict): Parameters governing the disk's distribution function.
        bulge_df_params (dict): Parameters governing the bulge's distribution function.
        N_disk (int, optional): Number of disk candidates to sample. Defaults to 100,000.
        N_bulge (int, optional): Number of bulge candidates to sample. Defaults to 100,000.
        prng_seed (int, optional): Seed for the JAX pseudo-random number generator. Defaults to 42.
        spheroid_corotation (float, optional): Fraction of bulge/spheroid orbits assigned 
            as PROGRADE (co-rotating with the disk). Because the spheroid DF is even in J_phi, 
            a sign must be manually assigned to its azimuthal motion. 
            - 0.5 (default) => Equal prograde/retrograde => Non-rotating, pressure-supported.
            - 1.0 => Fully co-rotating central component.
        sampler (str, optional): How action-space candidates are turned into weights.
            - 'soft' (default): the original soft-acceptance relaxation
              (`sample_df_potential`). Kept as the default so existing results are
              reproducible, but see the warning below.
            - 'importance': importance sampling with a proposal matched to each DF's
              own action scales (`sample_df_importance`). Unbiased and tuning-free.

            WARNING: 'soft' does not faithfully represent the DF for these models.
            Its temperature tau = 0.05 is far larger than the typical f/f_max over
            the sampled action box (median ~8e-6 for the spheroid), so the weight
            degenerates to a function of the uniform random draw alone. Measured
            consequences: the spheroid's DF parameters (J0, Beta) have NO effect on
            the sampled population at all, and the disk is realized at
            sigma_R ~ 80 km/s regardless of the requested sigmaR0_R0 (60 or 12).
            That washes out exactly the fine structure the observables depend on --
            the central velocity gradient and the central-to-disk dispersion
            contrast -- while leaving the gross rotation dipole intact.
            Use 'importance' for anything quantitative.

    Returns:
        tuple: A fully differentiable tracer population containing (x, y, z, vx, vy, vz, weights).
            - x, y, z (Array): 3D spatial coordinates (kpc).
            - vx, vy (Array): In-plane velocities, dynamically reconstructed to account 
              for the applied bulge prograde/retrograde rotation mask (km/s).
            - vz (Array): Vertical velocities (km/s).
            - all_weights (Array): Particle weights, properly scaled by M_disk and M_bulge.
    """
    # 1. Unpack Potential Parameters
    M_halo, a_halo = pot_params['M_halo'], pot_params['a_halo']
    M_disk, a_disk, b_disk = pot_params['M_disk'], pot_params['a_disk'], pot_params['b_disk']
    M_bulge, a_bulge = pot_params['M_bulge'], pot_params['a_bulge']

    # 2. Define the Local Potential
    def total_potential(x, y, z):
        return (nfw_potential(x, y, z, M_halo, a_halo) +
                miyamoto_nagai_potential(x, y, z, M_disk, a_disk, b_disk) +
                plummer_potential(x, y, z, M_bulge, a_bulge))
    # 3. Differentiable Sampling
    key = jax.random.PRNGKey(prng_seed)

    # Disk
    # envelope_max=None -> the sampler auto-calibrates the rejection envelope from
    # the candidates' own max DF value. This is essential for optimization from
    # far-off starts: a fixed envelope computed at a single hard-coded action point
    # underflows to zero when the DF parameters move, dividing by zero and poisoning
    # everything with NaNs. Auto-calibration tracks the DF as the parameters change.
    key, subkey = jax.random.split(key)
    if sampler == "importance":
        # Proposal scales from the DF's own physics: the qDF falls as
        # exp(-kappa*Jr/sigma_R^2) and exp(-nu*Jz/sigma_z^2), so the natural action
        # scales are sigma^2/frequency, evaluated at the DF's reference radius R0.
        # J_phi is set by the guiding-radius distribution, of order Rd * v_circ.
        R0_ref = jnp.maximum(disk_df_params["R0"], 0.1)
        kap0 = jnp.maximum(kappa(total_potential, R0_ref), 1e-3)
        nu0 = jnp.maximum(nu(total_potential, R0_ref), 1e-3)
        vc0 = jnp.maximum(vcirc(total_potential, R0_ref), 1.0)
        cand_disk, w_disk = sample_df_importance(
            df=f_disc_from_params, key=subkey, params=disk_df_params,
            Phi_xyz=total_potential, theta=(), n_candidates=N_disk,
            J_scales=(disk_df_params["sigmaR0_R0"] ** 2 / kap0,
                      disk_df_params["sigmaz0_R0"] ** 2 / nu0,
                      disk_df_params["Rd"] * vc0),
        )
    else:
        cand_disk, w_disk = sample_df_potential(
            df=f_disc_from_params, key=subkey, params=disk_df_params, Phi_xyz=total_potential,
            theta=(), n_candidates=N_disk, envelope_max=None,
            J_bounds=(100.0, 50.0, 3000.0), tau=0.05
        )
    key, subkey = jax.random.split(key)
    angles_disk = jax.random.uniform(subkey, shape=(N_disk, 3), minval=0.0, maxval=2*jnp.pi)

    # Bulge
    key, subkey = jax.random.split(key)
    if sampler == "importance":
        # The double power law turns over at Jtot ~ J0; 0.5*J0 per action was the
        # most efficient of the scales tried (ESS ~ 15% of N, against 0.14% for a
        # uniform draw over the old (500, 500, 500) box).
        J0_b = jnp.maximum(bulge_df_params["J0_spheroid"], 1e-3)
        cand_bulge, w_bulge = sample_df_importance(
            df=spheroid_df_wrapper, key=subkey, params=bulge_df_params,
            Phi_xyz=total_potential, theta=(), n_candidates=N_bulge,
            J_scales=(0.5 * J0_b, 0.5 * J0_b, 0.5 * J0_b),
        )
    else:
        cand_bulge, w_bulge = sample_df_potential(
            df=spheroid_df_wrapper, key=subkey, params=bulge_df_params, Phi_xyz=total_potential,
            theta=(), n_candidates=N_bulge, envelope_max=None,
            J_bounds=(500.0, 500.0, 500.0), tau=0.05
        )
    key, subkey = jax.random.split(key)
    angles_bulge = jax.random.uniform(subkey, shape=(N_bulge, 3), minval=0.0, maxval=2*jnp.pi)

    # 4. Combine and Format Weights
    w_disk_scaled = w_disk / jnp.sum(w_disk) * M_disk
    w_bulge_scaled = w_bulge / jnp.sum(w_bulge) * M_bulge
    all_weights = jnp.concatenate([w_disk_scaled, w_bulge_scaled])

    all_candidates = jnp.vstack([cand_disk, cand_bulge])
    all_angles = jnp.vstack([angles_disk, angles_bulge])

    # Scale potential params for the network
    nn_potentials = jnp.array([M_halo/1e11, a_halo, M_disk/1e11, a_disk, b_disk, M_bulge/1e11, a_bulge])
    potentials_batch = jnp.tile(nn_potentials, (N_disk + N_bulge, 1))

    # 5. Network Mapping
    phase_space = mapper.map_to_phase_space(all_candidates, all_angles, potentials_batch)
    x_raw, y_raw, z_raw = phase_space[:, 0], phase_space[:, 1], phase_space[:, 2]
    vx_raw, vy_raw, vz_raw = phase_space[:, 3], phase_space[:, 4], phase_space[:, 5]

    # 6. Cylindrical Decomposition from Raw Space (Skipping Averages)
    # Safety buffer of 0.05 prevents velocities from exploding near the origin
    R = jnp.maximum(jnp.sqrt(x_raw**2 + y_raw**2), 0.05)
    v_R = (x_raw * vx_raw + y_raw * vy_raw) / R
    v_phi = (x_raw * vy_raw - y_raw * vx_raw) / R

    # 7. Bulge retro-flip (Using JAX random)
    # Assign each bulge orbit prograde (+1) with probability `spheroid_corotation`,
    # retrograde (-1) otherwise.
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(N_bulge,))
    bulge_flip_mask = jnp.where(u < spheroid_corotation, 1.0, -1.0)
    
    # Apply the flip mask ONLY to the bulge indices (from N_disk to the end)
    v_phi = v_phi.at[N_disk:].multiply(bulge_flip_mask)

    # 8. Reconstruct Cartesian Velocities 
    # (Keeping original x, y, z and mapping the flipped v_phi back to vx, vy)
    cos_phi = x_raw / R
    sin_phi = y_raw / R
    
    vx_new = v_R * cos_phi - v_phi * sin_phi
    vy_new = v_R * sin_phi + v_phi * cos_phi

    # Return original coordinates and vz_raw, but updated vx and vy
    return x_raw, y_raw, z_raw, vx_new, vy_new, vz_raw, all_weights



def apply_mass_to_light(all_weights, n_disk: int, ml_ratios=None):
    """
    Converts per-particle MASS weights into per-particle LUMINOSITY weights.

    A real integral-field observation measures surface brightness, not surface
    density, and its kinematics are light-weighted rather than mass-weighted. The
    two differ whenever the stellar populations do: an old, metal-rich bulge has a
    mass-to-light ratio several times that of a younger disk, so it contributes far
    less light per unit mass than the disk does.

    The conversion is per component, `L_i = M_i / Upsilon_component`, applied to the
    particle weights *before* binning. Because every map `bin_maps` produces is a
    weighted sum over particles, dividing the weights turns the density map into a
    surface-brightness map and simultaneously turns the first and second velocity
    moments into light-weighted ones -- which is what an observation actually gives.

    IMPORTANT: only the *observables* should use these weights. The Poisson
    self-consistency penalty is a statement about mass sourcing the potential, so it
    must keep the mass weights; passing luminosity weights there would ask the light
    to generate the gravity.

    Note that a spatially CONSTANT Upsilon is a no-op for a fit: the first map enters
    the loss as a log-space residual and the kinematic maps are ratios of weighted
    sums, so a common factor cancels on both sides. Only a component-dependent
    Upsilon changes the objective. The one exception is cosmetic: `bin_maps` zeroes
    the kinematics below an ABSOLUTE weight of 1e-5, so a pixel just above that
    threshold in mass can drop below it in light. Those pixels lie far below any
    realistic detection floor and do not enter a fit.

    Args:
        all_weights (Array): Particle mass weights as returned by
            `sample_and_map_particles`, disk particles first.
        n_disk (int): Number of leading entries belonging to the disk; the remainder
            are the bulge/spheroid.
        ml_ratios (tuple, optional): `(Upsilon_disk, Upsilon_bulge)` in solar units
            (M_sun / L_sun). `None` (default) returns the weights unchanged, i.e. the
            maps stay mass-weighted and every existing caller is unaffected.

    Returns:
        Array: Luminosity weights, same shape as `all_weights`.
    """
    if ml_ratios is None:
        return all_weights
    ml_disk, ml_bulge = ml_ratios
    n_bulge = all_weights.shape[0] - n_disk
    upsilon = jnp.concatenate([
        jnp.full((n_disk,), ml_disk),
        jnp.full((n_bulge,), ml_bulge),
    ])
    # Floor guards against a zero/negative M/L making the weights blow up or flip sign.
    return all_weights / jnp.maximum(upsilon, 1e-6)

def project_to_sky(x, y, z, vx, vy, vz, inclination_deg: float = 90.0):
    """
    Projects a tracer population onto the sky plane for an arbitrary inclination.

    The model galaxy has its symmetry axis along z and its disk in the x-y plane.
    For an inclination `i` (0 = face-on, 90 = edge-on) the line of sight is
    n_hat = (0, sin i, cos i), which gives

        v_los = v_y sin i + v_z cos i
        x_sky = x                        (kinematic major axis)
        z_sky = -y cos i + z sin i       (kinematic minor axis)

    At i = 90 this reduces exactly to (x, z, v_y), the edge-on convention the rest
    of this module was written against, so the default leaves every existing
    caller's behaviour unchanged.

    Args:
        x, y, z: Particle positions (kpc).
        vx, vy, vz: Particle velocities (km/s). `vx` is accepted for a uniform
            signature but does not enter: the line of sight lies in the y-z plane,
            so v_x is always in the plane of the sky.
        inclination_deg: Inclination in degrees. 90 = edge-on (default).

    Returns:
        tuple: (x_sky, z_sky, v_los) — the two sky coordinates and the
        line-of-sight velocity, ready to hand to `bin_maps`.
    """
    i = jnp.deg2rad(inclination_deg)
    si, ci = jnp.sin(i), jnp.cos(i)
    return x, -y * ci + z * si, vy * si + vz * ci


def bin_maps(
    x, z, vy, all_weights,
    grid_size: int = 30,
    extent_x: float = 15.0,
    extent_z: float = 10.0,
    soft_bin_h: float = None,
):
    """
    Bins a tracer population into differentiable, edge-on observable maps using 
    Gaussian Soft-Binning (Kernel Density Estimation).

    Instead of dropping particles into hard discrete bins (which breaks gradients), 
    each particle is smeared into a 2D Gaussian blob. The grid evaluates the sum 
    of all blobs at each pixel center.

    Args:
        x: Array of particle x-coordinates (shape: [N]).
        z: Array of particle z-coordinates (shape: [N]).
        vy: Array of particle line-of-sight velocities (shape: [N]).
        all_weights: Array of particle masses/weights (shape: [N]).
        grid_size: Number of pixels along one spatial dimension (grid is N x N).
        extent_x: Physical half-width of the grid (from -extent_x to +extent_x).
        extent_z: Physical half-height of the grid (from -extent_z to +extent_z).
        soft_bin_h: Gaussian bandwidth (smoothing scale). If None, defaults to 
            0.25x the pixel width.

    Returns:
        dict: Containing the smoothed 2D maps ('mass', 'v_rot', 'sigma') and 
        the grid edges ('x_edges', 'z_edges') for plotting.
    """
    
    # 1. Grid Definition
    dx = 2.0 * extent_x / grid_size
    dz = 2.0 * extent_z / grid_size

    # Set default smoothing scale (bandwidth)
    if soft_bin_h is None:
        soft_bin_h = jnp.maximum(dx, dz) * 0.25

    # 1D arrays of pixel centers
    X_centers = jnp.linspace(-extent_x + dx/2, extent_x - dx/2, grid_size)
    Z_centers = jnp.linspace(-extent_z + dz/2, extent_z - dz/2, grid_size)

    # 2. Distance Matrix Calculation via Broadcasting
    # We reshape the arrays to create a 3D matrix of shape: (grid_z, grid_x, N_particles)
    # This calculates the distance from EVERY grid pixel to EVERY particle simultaneously.
    
    # Shape: (grid_z, 1, 1) - (1, 1, N_particles) -> (grid_z, 1, N_particles)
    dz_arr = Z_centers[:, None, None] - z[None, None, :]
    
    # Shape: (1, grid_x, 1) - (1, 1, N_particles) -> (1, grid_x, N_particles)
    dx_arr = X_centers[None, :, None] - x[None, None, :]

    # Sum of squared distances. Shape becomes: (grid_z, grid_x, N_particles)
    dist_sq = dx_arr**2 + dz_arr**2

    # 3. Gaussian Kernel Application
    # Calculate the normalized 2D Gaussian density. We multiply by pixel area (dx * dz)
    # so that integrating over the grid recovers the exact original particle mass.
    normalization = (dx * dz) / (2.0 * jnp.pi * soft_bin_h**2)
    kernel = normalization * jnp.exp(-0.5 * dist_sq / soft_bin_h**2)

    # Scale the kernel by the mass/weight of each particle
    w_kernel = all_weights[None, None, :] * kernel

    # 4. Map Construction
    # Sum across the particle axis (-1) to collapse down to a 2D map: (grid_z, grid_x)
    mass_map = jnp.sum(w_kernel, axis=-1)

    # Protect against Division-by-Zero in empty pixels during gradient calculations
    mass_safe = jnp.maximum(mass_map, 1e-12)

    # Calculate Velocity (First Moment: Mean)
    # We ignore pixels with negligible mass (< 1e-5) to prevent numerical instability
    momentum_map = jnp.sum(w_kernel * vy[None, None, :], axis=-1)
    v_rot_map = jnp.where(mass_map > 1e-5, momentum_map / mass_safe, 0.0)

    # Calculate Velocity Dispersion (Second Moment: Variance -> Sigma)
    # Variance = E[v^2] - (E[v])^2
    v2_momentum_map = jnp.sum(w_kernel * (vy**2)[None, None, :], axis=-1)
    v2_map = jnp.where(mass_map > 1e-5, v2_momentum_map / mass_safe, 0.0)
    
    variance_map = v2_map - v_rot_map**2
    sigma_map = jnp.sqrt(jnp.maximum(variance_map, 1e-12))

    # 5. Output Packaging
    x_edges = jnp.linspace(-extent_x, extent_x, grid_size + 1)
    z_edges = jnp.linspace(-extent_z, extent_z, grid_size + 1)

    return {
        'mass': mass_map,
        'v_rot': v_rot_map,
        'sigma': sigma_map,
        'x_edges': x_edges,
        'z_edges': z_edges
    }


def render_maps_batched(
    mapper,
    pot_params: dict,
    disk_df_params: dict,
    bulge_df_params: dict,
    N_disk: int = 200_000,
    N_bulge: int = 200_000,
    grid_size: int = 30,
    extent_x: float = 15.0,
    extent_z: float = 10.0,
    prng_seed: int = 42,
    soft_bin_h: float = None,
    spheroid_corotation: float = 0.5,
    inclination_deg: float = 90.0,
    sampler: str = "soft",
    ml_ratios=None,
    chunk: int = 15_000,
):
    """
    Renders edge-on maps with a large tracer count, binning in chunks to bound memory.

    `bin_maps` builds a dense (grid_z, grid_x, N_particles) kernel array, so a
    high-tracer render (hundreds of thousands of particles, needed to get maps as
    smooth as Voronoi-binned observations) would allocate tens of gigabytes. Here the
    particles are sampled once and then binned in batches of `chunk`, accumulating the
    raw moment sums, which is mathematically identical because every map is a plain
    weighted sum over particles:

        mass      = sum_i w_i K_i
        momentum  = sum_i w_i K_i v_i
        second    = sum_i w_i K_i v_i^2

    The moments are combined only after all batches are accumulated. This is a
    rendering helper for display and diagnostics, not for use inside a gradient step.

    Args:
        mapper: The Phoenix neural network surrogate.
        pot_params, disk_df_params, bulge_df_params (dict): Model parameters.
        N_disk, N_bulge (int): Tracer counts. Large values are the point of this
            function.
        grid_size (int): Output map resolution.
        extent_x, extent_z (float): Physical half-width/half-height of the grid (kpc).
        prng_seed (int): Sampling seed.
        soft_bin_h (float, optional): KDE bandwidth. Pass the bandwidth the fit
            converged to (the final annealing value) so the render is comparable to
            what the loss actually saw; `None` uses `bin_maps`' default.
        spheroid_corotation (float): Bulge prograde fraction.
        inclination_deg (float): Viewing inclination in degrees; 90 = edge-on
            (default). See `project_to_sky`.
        sampler (str): 'soft' or 'importance'; see `sample_and_map_particles`.
        ml_ratios (tuple, optional): `(Upsilon_disk, Upsilon_bulge)`; renders surface
            brightness and light-weighted kinematics instead of mass. See
            `apply_mass_to_light`.
        chunk (int): Particles binned per batch.

    Returns:
        dict: Same keys as `bin_maps` ('mass', 'v_rot', 'sigma', 'x_edges', 'z_edges').
    """
    x, y, z, vx, vy, vz, all_weights = sample_and_map_particles(
        mapper, pot_params, disk_df_params, bulge_df_params,
        N_disk=N_disk, N_bulge=N_bulge, prng_seed=prng_seed,
        spheroid_corotation=spheroid_corotation, sampler=sampler,
    )
    all_weights = apply_mass_to_light(all_weights, N_disk, ml_ratios)
    x, z, vy = project_to_sky(x, y, z, vx, vy, vz, inclination_deg)

    dx = 2.0 * extent_x / grid_size
    dz = 2.0 * extent_z / grid_size
    if soft_bin_h is None:
        soft_bin_h = jnp.maximum(dx, dz) * 0.25

    X_centers = jnp.linspace(-extent_x + dx/2, extent_x - dx/2, grid_size)
    Z_centers = jnp.linspace(-extent_z + dz/2, extent_z - dz/2, grid_size)
    normalization = (dx * dz) / (2.0 * jnp.pi * soft_bin_h**2)

    @jax.jit
    def _chunk_sums(xc, zc, vc, wc):
        dz_arr = Z_centers[:, None, None] - zc[None, None, :]
        dx_arr = X_centers[None, :, None] - xc[None, None, :]
        kernel = normalization * jnp.exp(
            -0.5 * (dx_arr**2 + dz_arr**2) / soft_bin_h**2
        )
        w_kernel = wc[None, None, :] * kernel
        return (jnp.sum(w_kernel, axis=-1),
                jnp.sum(w_kernel * vc[None, None, :], axis=-1),
                jnp.sum(w_kernel * (vc**2)[None, None, :], axis=-1))

    n_total = x.shape[0]
    mass_map = jnp.zeros((grid_size, grid_size))
    momentum_map = jnp.zeros((grid_size, grid_size))
    v2_momentum_map = jnp.zeros((grid_size, grid_size))
    for start in range(0, n_total, chunk):
        stop = min(start + chunk, n_total)
        m, p, q = _chunk_sums(x[start:stop], z[start:stop],
                              vy[start:stop], all_weights[start:stop])
        mass_map = mass_map + m
        momentum_map = momentum_map + p
        v2_momentum_map = v2_momentum_map + q

    # Identical moment combination to bin_maps, applied once to the totals.
    mass_safe = jnp.maximum(mass_map, 1e-12)
    v_rot_map = jnp.where(mass_map > 1e-5, momentum_map / mass_safe, 0.0)
    v2_map = jnp.where(mass_map > 1e-5, v2_momentum_map / mass_safe, 0.0)
    sigma_map = jnp.sqrt(jnp.maximum(v2_map - v_rot_map**2, 1e-12))

    return {
        'mass': mass_map,
        'v_rot': v_rot_map,
        'sigma': sigma_map,
        'x_edges': jnp.linspace(-extent_x, extent_x, grid_size + 1),
        'z_edges': jnp.linspace(-extent_z, extent_z, grid_size + 1),
    }


def blur_maps(
    maps: dict,
    blur_h,
    grid_size: int = 30,
    extent_x: float = 15.0,
    extent_z: float = 10.0,
):
    """
    Blurs a set of observable maps with a Gaussian of width `blur_h` (kpc).

    This is what makes bandwidth annealing a valid coarse-to-fine scheme. The
    model maps are produced by `bin_maps` at the current annealing bandwidth h,
    while an observation is binned once at its own (sharp) bandwidth h_obs. If
    only the model is smoothed, the objective is no longer minimized at the true
    parameters, and the optimizer is actively pulled away from them during the
    wide-bandwidth stages. Blurring the observation to the same effective
    resolution removes that bias.

    The correction is exact for Gaussian soft-binning, because Gaussians compose:
    a KDE at bandwidth h_obs convolved with a Gaussian of width
    `blur_h = sqrt(h**2 - h_obs**2)` is exactly the KDE at bandwidth h. The
    kinematic maps are blurred as mass-weighted moments (the same way a real
    instrument's PSF acts on surface-brightness-weighted kinematics), which is
    likewise exact here since the underlying momentum maps are linear in the
    kernel.

    Args:
        maps (dict): Maps to blur, with keys 'mass', 'v_rot', 'sigma' (extra keys
            such as the grid edges are passed through untouched).
        blur_h: Gaussian width to convolve with, in kpc. Values <= 0 return the
            maps unchanged (the observation cannot be sharpened below its own
            resolution).
        grid_size (int): Resolution of the maps (must match how they were binned).
        extent_x (float): Physical half-width of the grid (kpc).
        extent_z (float): Physical half-height of the grid (kpc).

    Returns:
        dict: The blurred maps, same keys as the input.
    """
    dx = 2.0 * extent_x / grid_size
    dz = 2.0 * extent_z / grid_size
    X_centers = jnp.linspace(-extent_x + dx/2, extent_x - dx/2, grid_size)
    Z_centers = jnp.linspace(-extent_z + dz/2, extent_z - dz/2, grid_size)

    # Floor the width so the kernels stay finite when blur_h -> 0; with a width
    # far below the pixel scale the normalized kernel becomes the identity, which
    # is exactly the desired "no blurring" limit.
    h = jnp.maximum(blur_h, 1e-6)

    # Separable 1D kernels, row-normalized so the blur conserves total mass.
    Kx = jnp.exp(-0.5 * (X_centers[:, None] - X_centers[None, :])**2 / h**2)
    Kx = Kx / jnp.sum(Kx, axis=1, keepdims=True)
    Kz = jnp.exp(-0.5 * (Z_centers[:, None] - Z_centers[None, :])**2 / h**2)
    Kz = Kz / jnp.sum(Kz, axis=1, keepdims=True)

    def _smooth(M):
        return Kz @ M @ Kx.T

    mass = maps['mass']
    # Blur the mass-weighted moments, then divide back out (PSF-like convolution).
    momentum = mass * maps['v_rot']
    second = mass * (maps['v_rot']**2 + maps['sigma']**2)

    mass_b = _smooth(mass)
    mass_safe = jnp.maximum(mass_b, 1e-12)
    v_rot_b = jnp.where(mass_b > 1e-5, _smooth(momentum) / mass_safe, 0.0)
    v2_b = jnp.where(mass_b > 1e-5, _smooth(second) / mass_safe, 0.0)
    sigma_b = jnp.sqrt(jnp.maximum(v2_b - v_rot_b**2, 1e-12))

    out = dict(maps)
    out['mass'] = mass_b
    out['v_rot'] = v_rot_b
    out['sigma'] = sigma_b
    return out


def generate_edge_on_maps(
    mapper,
    pot_params: dict,
    disk_df_params: dict,
    bulge_df_params: dict,
    N_disk: int = 100_000,
    N_bulge: int = 100_000,
    grid_size: int = 30,
    extent_x: float = 15.0,
    extent_z: float = 10.0,
    prng_seed: int = 42,
    soft_bin_h: float = None, # Configurable smoothing bandwidth
    spheroid_corotation: float = 0.5,
    inclination_deg: float = 90.0,
    sampler: str = "soft",
    ml_ratios=None,
):
    """
    End-to-end pipeline for generating fully differentiable edge-on observable kinematic maps 
    of a galaxy model using Gaussian Soft-Binning (KDE).

    This function combines two steps:
    1. Samples tracer particles from the combined disk and bulge distribution 
       functions and maps them to 3D phase space via the Phoenix neural network.
    2. Projects these particles into an edge-on view (x-z plane with line-of-sight 
       velocity vy) and bins them into mass and kinematic maps using Gaussian 
       soft-binning (KDE) to preserve differentiability.

    Args:
        mapper: The Phoenix neural network surrogate used to map actions to phase space.
        pot_params (dict): Gravitational potential parameters (halo, disk, and bulge masses/scales).
        disk_df_params (dict): Parameters governing the disk's distribution function.
        bulge_df_params (dict): Parameters governing the bulge's distribution function.
        N_disk (int, optional): Number of disk particles to sample. Defaults to 100,000.
        N_bulge (int, optional): Number of bulge particles to sample. Defaults to 100,000.
        grid_size (int, optional): The resolution of the output 2D grids (grid_size x grid_size). Defaults to 30.
        extent_x (float, optional): Physical half-width of the spatial grid in the x-direction (kpc). Defaults to 15.0.
        extent_z (float, optional): Physical half-height of the spatial grid in the z-direction (kpc). Defaults to 10.0.
        prng_seed (int, optional): Seed for the JAX pseudo-random number generator. Defaults to 42.
        soft_bin_h (float, optional): Gaussian smoothing bandwidth for KDE. If None, 
            auto-calculates to 0.25x the pixel width.
        spheroid_corotation (float, optional): Fraction of bulge orbits assigned as 
            prograde (co-rotating). 0.5 = non-rotating bulge, 1.0 = fully co-rotating. Defaults to 0.5.
        inclination_deg (float, optional): Viewing inclination in degrees. 90 (default)
            is edge-on and reproduces the original behaviour exactly; lower values tilt
            the model toward face-on. See `project_to_sky`.
        sampler (str, optional): 'soft' or 'importance'; see
            `sample_and_map_particles`. 'importance' is required for the DF
            parameters to actually shape the sampled population.
        ml_ratios (tuple, optional): `(Upsilon_disk, Upsilon_bulge)` mass-to-light
            ratios in solar units. When given, the particle weights are divided by
            their component's Upsilon before binning, so the returned maps are a
            surface-BRIGHTNESS map and LIGHT-weighted kinematics -- what an
            integral-field observation actually measures. `None` (default) keeps the
            maps mass-weighted. See `apply_mass_to_light`.

    Returns:
        dict: A dictionary containing the smoothed edge-on maps and grid edges:
            - 'mass' (Array): 2D map of the projected binning weight -- mass density
              by default, surface luminosity when `ml_ratios` is given. The key name
              is kept so that every downstream consumer (the loss, `blur_maps`, the
              plotting helpers) works unchanged for either weighting.
            - 'v_rot' (Array): 2D map of line-of-sight rotational velocity.
            - 'sigma' (Array): 2D map of line-of-sight velocity dispersion.
            - 'x_edges' (Array): 1D array of spatial bin edges along the x-axis.
            - 'z_edges' (Array): 1D array of spatial bin edges along the z-axis.
    """
    x, y, z, vx, vy, vz, all_weights = sample_and_map_particles(
        mapper, pot_params, disk_df_params, bulge_df_params,
        N_disk=N_disk, N_bulge=N_bulge, prng_seed=prng_seed,
        spheroid_corotation=spheroid_corotation, sampler=sampler,
    )
    bin_weights = apply_mass_to_light(all_weights, N_disk, ml_ratios)
    x_sky, z_sky, v_los = project_to_sky(x, y, z, vx, vy, vz, inclination_deg)
    return bin_maps(x_sky, z_sky, v_los, bin_weights, grid_size=grid_size, extent_x=extent_x,
                     extent_z=extent_z, soft_bin_h=soft_bin_h)

