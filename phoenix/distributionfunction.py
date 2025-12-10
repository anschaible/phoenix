from phoenix.distributionfunctions_disky_Binney import f_total_disc_from_params, f_thin_disc_from_params, f_thick_disc_from_params
from phoenix.distributionfunctions_spheroidal import f_double_power_law

def distributionfunction_combined(Jr, Jz, Lz, Phi_xyz_thin, Phi_xyz_thick, Phi_xyz_spheroid, theta_thin, theta_thick, theta_spheroid, params):
    """
    Combined distribution function for a galaxy with thin disc, thick disc, and spheroidal components.

    Parameters:
      - Jr: Radial action
      - Jz: Vertical action
      - Lz: Angular momentum along z
      - params: Dictionary with DF parameters for each component:
          * 'thin_disc': Parameters for the thin disc DF
          * 'thick_disc': Parameters for the thick disc DF
          * 'spheroid': Parameters for the spheroidal DF
      - Phi_xyz: Callable potential function Phi(x, y, z, *theta)
      - theta: Tuple of parameters for the potential

    Returns:
      - f_total: Value of the combined distribution function at given actions
    """

    factor_thin = params["f_thin"]
    factor_thick = params["f_thick"]
    factor_spheroid = params["f_spheroid"]
    df_thin = f_thin_disc_from_params(Jr, Jz, Lz, Phi_xyz_thin, theta_thin, params)
    df_thick = f_thick_disc_from_params(Jr, Jz, Lz, Phi_xyz_thick, theta_thick, params)
    df_spheroid = f_double_power_law(Jr, Jz, Lz, Phi_xyz_spheroid, theta_spheroid, params)

    f_total = factor_thin * df_thin + factor_thick * df_thick + factor_spheroid * df_spheroid
    return f_total