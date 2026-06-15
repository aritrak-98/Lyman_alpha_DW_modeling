"""
galaxy.py — Computes the intrinsic properties of the galaxies
 
Constants
---------
N_HI_DOT_PER_SFR : float
    Conversion factor from SFR (M_Sun/yr) to ionizing photon rate (s^-1).
    Default: 1e53
 
Functions
---------
compute_sigma_v        : Velocity dispersion from halo mass and redshift
                         (Barkana & Loeb 2001)
intrinsic_lya_profile  : Intrinsic Lyman-alpha line profile
intrinsic_lya_profile_neyer : Neyer 2025 intrinsic profile
                              (arbitrary units, for T_alpha only)
"""

import numpy as np
import astropy.constants as const
import astropy.units as u
 
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
# Lyman-alpha frequency and wavelength
NU_ALPHA   = 2.47e15          # Hz
LAMBDA_ALPHA = 1215.67e-10    # m
 
# Speed of light
C_KM_S = const.c.to(u.km / u.s).value     # km/s
C_M_S  = const.c.value                    # m/s
 
# Lyman-alpha conversion factor: f_alpha * h_planck * nu_alpha (in J)
H_NU_ALPHA = (const.h * NU_ALPHA * u.Hz).to(u.J).value
 
# Ionizing photon rate per unit SFR.
# This model assumes N_HI_dot = N_HI_DOT_PER_SFR * SFR,
# where SFR is in M_Sun/yr and N_HI_dot is in photons/s.
# Modify this constant if your simulation uses a different conversion.
N_HI_DOT_PER_SFR = 1e53      # photons s^-1 / (M_Sun yr^-1)


# UV continuum conversion factor (Sipple & Lidz 2023)
K_UV = 1.15e-28   # M_Sun/yr / (erg/s/Hz)

 
# Lyman-alpha escape fraction and recombination fraction (default values)
DEFAULT_F_ESC   = 0.02        # escape fraction of ionizing photons
DEFAULT_F_ALPHA = 0.68        # fraction of recombinations producing Lya photons


# Neyer 2025 profile parameters (default values)
NEYER_A_PEAK = 0.011
NEYER_B_PEAK = 10**(-0.440)
NEYER_A_WIND = -0.088
NEYER_B_WIND = 1.399
NEYER_F_P    = 0.93099
 


# ---------------------------------------------------------------------------
# Velocity dispersion
# ---------------------------------------------------------------------------

def compute_sigma_v(M_h, z, h):
    """
    Compute the velocity dispersion of a galaxy/halo.
 
    Uses the analytic scaling relation from Barkana & Loeb (2001):
        sigma_v = 20 * (M_h / (1e8 * h^-1))^(1/3) * ((1+z) / 10)^(1/2)
 
    Parameters
    ----------
    M_h : float or array-like
        Halo mass in M_Sun.
    z : float
        Redshift.
    h : float
        Dimensionless Hubble parameter (H0 / 100 km/s/Mpc).
 
    Returns
    -------
    float or np.ndarray
        Velocity dispersion in km/s.
 
    References
    ----------
    Barkana & Loeb (2001), Physics Reports, 349, 125.
 
    Examples
    --------
    >>> sigma = compute_sigma_v(M_h=1e10, z=7.0, h=0.6774)
    >>> sigma = compute_sigma_v(M_h=np.array([1e9, 1e10, 1e11]), z=7.0, h=0.6774)
    """
    M_h = np.asarray(M_h, dtype=np.float64)
    sigma_v = 20.0 * (M_h / (1e8 * h**(-1)))**(1.0/3.0) * ((1.0 + z) / 10.0)**(0.5)
    return sigma_v


# ---------------------------------------------------------------------------
# Intrinsic Lyman-alpha profile
# ---------------------------------------------------------------------------

def intrinsic_lya_profile(delta, sigma_v, SFR, vel_out, 
                          f_esc=DEFAULT_F_ESC, f_alpha=DEFAULT_F_ALPHA):

    """
    Compute the intrinsic (pre-IGM) Lyman-alpha line profile.
 
    The profile is a Gaussian in velocity space, centred at 2 * vel_out
    (the factor of 2 accounts for the double scattering in the outflowing spherical shell),
    normalised so that its integral gives the total intrinsic luminosity.
 
    The ionizing photon rate is computed internally as:
        N_HI_dot = N_HI_DOT_PER_SFR * SFR

    Parameters
    ----------
    delta : np.ndarray, shape (N_delta,)
        Fractional frequency offset from Lyman-alpha line center,
    sigma_v : np.ndarray, shape (N_gal,)
        Velocity dispersion of each galaxy in km/s.
    SFR : np.ndarray, shape (N_gal,)
        Star formation rate of each galaxy in M_Sun/yr.
    vel_out : np.ndarray, shape (N_gal,)
        Outflow velocity of each galaxy in km/s.
    f_esc : float, optional
        Escape fraction of ionizing photons. Default: 0.02.
    f_alpha : float, optional
        Fraction of recombinations that produce a Lyman-alpha photon.
        Default: 0.68.

    Returns
    -------
    np.ndarray, shape (N_gal, N_delta)
        Intrinsic Lyman-alpha luminosity profile in W / (km/s).
 
    Notes
    -----
    The peak of the Gaussian is at del_v = 2 * vel_out.
 
    Examples
    --------
    >>> profile = intrinsic_lya_profile(delta, sigma_v, SFR, vel_out)
    >>> profile.shape   # (N_gal, N_delta)
    """
    # Reshape for broadcasting: (N_gal, 1) against (1, N_delta)
    delta    = np.asarray(delta, dtype=np.float64)[None, :]          # (1, N_delta)
    sigma_v  = np.asarray(sigma_v, dtype=np.float64)[:, None]        # (N_gal, 1)
    SFR      = np.asarray(SFR, dtype=np.float64)[:, None]            # (N_gal, 1)
    vel_out  = np.asarray(vel_out, dtype=np.float64)[:, None]        # (N_gal, 1)
 
    # Velocity offset from line centre
    del_v = C_KM_S * delta                         # km/s, shape (1, N_delta)
 
    # Ionizing photon rate and intrinsic luminosity
    N_HI_dot = N_HI_DOT_PER_SFR * SFR             # photons/s, shape (N_gal, 1)
    L_alpha  = f_alpha * (1.0 - f_esc) * H_NU_ALPHA * N_HI_dot   # W
 
    # Gaussian profile centred at 2 * vel_out
    profile = (
        (L_alpha / np.sqrt(2.0 * np.pi * sigma_v**2))
        * np.exp(-((del_v - 2.0 * vel_out)**2) / (2.0 * sigma_v**2))
    )
 
    return profile    # W / (km/s), shape (N_gal, N_delta)


# ---------------------------------------------------------------------------
# Neyer 2025 intrinsic Lyman-alpha profile
# ---------------------------------------------------------------------------
 
def _neyer_delta_v_tilde(delta, sigma_v):
    """
    Dimensionless velocity offset for the Neyer 2025 profile.
 
    Parameters
    ----------
    delta : np.ndarray, shape (1, N_delta) or (N_delta,)
        Fractional frequency offset from line centre.
    sigma_v : np.ndarray, shape (N_gal, 1) or (N_gal,)
        Velocity dispersion in km/s.
 
    Returns
    -------
    np.ndarray
        Dimensionless velocity offset.
    """
    return (C_KM_S * delta) / (NEYER_A_PEAK * sigma_v + 300.0 * NEYER_B_PEAK)
 
 
def _neyer_beta(sigma_v):
    """
    Wind parameter beta for the Neyer 2025 profile.
 
    Parameters
    ----------
    sigma_v : np.ndarray
        Velocity dispersion in km/s.
 
    Returns
    -------
    np.ndarray
        Beta parameter.
    """
    return NEYER_A_WIND * (sigma_v / 300.0) + NEYER_B_WIND
 
 
def intrinsic_lya_profile_neyer(delta, sigma_v, symmetric=False):
    """
    Compute the Neyer 2025 intrinsic Lyman-alpha line profile.
 
    The Neyer 2025 profile is a double-peaked profile.
 
    Returns the profile shape in **arbitrary units** (no normalization).
    This is sufficient for T_alpha computation since the normalization
    cancels in the ratio.
 
    WARNING: This profile should NOT be passed to compute_luminosities()
    since the absolute scale is not physical — use intrinsic_lya_profile()
    instead for absolute luminosity calculations.
 
    Parameters
    ----------
    delta : np.ndarray, shape (N_delta,)
        Fractional frequency offset from Lyman-alpha line centre.
    sigma_v : np.ndarray, shape (N_gal,)
        Velocity dispersion of each galaxy in km/s.
    symmetric : bool, optional
        If True, set the wind parameter beta=0 for a symmetric profile.
        If False, use the wind term from Neyer 2025. Default: False.
 
    Returns
    -------
    np.ndarray, shape (N_gal, N_delta)
        Intrinsic Lyman-alpha profile shape in arbitrary units.
 
    References
    ----------
    Neyer et al. (2025)
 
    Examples
    --------
    >>> profile = intrinsic_lya_profile_neyer(delta, sigma_v)
    >>> profile.shape   # (N_gal, N_delta)
 
    For T_alpha computation:
 
    >>> J_v = intrinsic_lya_profile_neyer(delta, sigma_v)
    >>> num = trapezoid(J_v * np.exp(-tau_DW), nu_rest, axis=1)
    >>> denom = trapezoid(J_v, nu_rest, axis=1)
    >>> T_alpha = num / denom    # normalisation cancels
    """
    # Reshape for broadcasting
    delta   = np.asarray(delta,   dtype=np.float64)[None, :]   # (1, N_delta)
    sigma_v = np.asarray(sigma_v, dtype=np.float64)[:, None]   # (N_gal, 1)
 
    # Dimensionless velocity offset
    delvt = _neyer_delta_v_tilde(delta, sigma_v)
 
    # Wind parameter (0 if symmetric)
    if symmetric:
        beta = 0.0
    else:
        beta = _neyer_beta(sigma_v)
 
    # Numerically stable computation of profile = delvt^2 * sech^2(z) * exp(w * sqrt(pi)/3 * beta)
    # where z = sqrt(pi^3/54) * f_p^3 * delvt^3 and w = f_p^3 * delvt^3
    #
    # sech^2(z) = (2 / (e^z + e^-z))^2
    # log(sech^2(z)) = log(4) - 2 * log(e^z + e^-z)
    #                = log(4) - 2 * (|z| + log(1 + e^-2|z|))    [stable for large |z|]
    #                = -2 * (|z| + log(1 + e^-2|z|) - log(2))
 
    w         = NEYER_F_P**3 * delvt**3
    z         = np.sqrt(np.pi**3 / 54.0) * w
    az        = np.abs(z)
    log_sech2 = -2.0 * (az + np.log1p(np.exp(-2.0 * az)) - np.log(2.0))
    log_wind  = (np.sqrt(np.pi) / 3.0) * beta * w
 
    # log_J = log(delvt^2) + log(sech^2) + log(wind term)
    # At delvt=0 the log(delvt^2) is -inf, but the profile itself is 0 there
    with np.errstate(divide='ignore'):
        log_J = np.log(delvt**2) + log_sech2 + log_wind
 
    profile = np.exp(log_J)
 
    # exp(-inf) = 0, so delvt=0 gives profile=0 correctly
    # Any remaining NaN (e.g. from 0 * inf) is replaced with 0
    profile = np.where(np.isnan(profile), 0.0, profile)
 
    return profile    # arbitrary units, shape (N_gal, N_delta)