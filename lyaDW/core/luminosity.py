"""
luminosity.py — Luminosities and rest-frame Lyman-alpha equivalent width computation.
 
Computes rest-frame equivalent widths (REW) and luminosities for each galaxy,
given the transmission coefficients from transmission.compute_T_alpha().

This is a not so accurate computation of the REWs. So this code should be treated as optional for now.

"""

import numpy as np
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import h5py
import os
 
from lyaDW.utils.galaxy import (
    intrinsic_lya_profile,
    DEFAULT_F_ESC,
    DEFAULT_F_ALPHA,
    NU_ALPHA,
    LAMBDA_ALPHA,
    C_KM_S,
    K_UV,
)



# ---------------------------------------------------------------------------
# Physical constants and defaults
# ---------------------------------------------------------------------------
 
DEFAULT_LAMBDA_UV = 1500e-10    # UV continuum reference wavelength (m)
DEFAULT_B         = -1.7        # UV spectral slope (Dijkstra & Wyithe 2010)

# The UV spectral slope is fixed for now.



# ---------------------------------------------------------------------------
# The main function the user needs to call
# ---------------------------------------------------------------------------
 
def compute_luminosities(T_alpha, delta, sigma_v, SFR, sim, z_s,
                         f_vel_out=0.0, f_esc=DEFAULT_F_ESC, f_alpha=DEFAULT_F_ALPHA,
                         b=DEFAULT_B, lambda_UV=DEFAULT_LAMBDA_UV,
                         save=False, savepath='./', filename=None):
    """
    Compute the luminosities and rest-frame Lyman-alpha equivalent widths.
 
    Parameters
    ----------
    T_alpha : array-like, shape (N,)
        Transmission coefficients, as returned by transmission.compute_T_alpha().
    delta : np.ndarray, shape (N_delta,)
        Fractional frequency offset grid, as returned by optical_depth.compute_tau().
    sigma_v : array-like, shape (N,)
        Velocity dispersion of each galaxy in km/s.
    SFR : array-like, shape (N,)
        Star formation rate of each galaxy in M_Sun/yr.
    sim : lyaDW.Simulation
        Simulation context.
    z_s : float
        Source redshift. Resolved to the closest available redshift in sim.
    f_vel_out : float, optional
        Outflow velocity as a multiple of sigma_v. Default: 0.0.
    f_esc : float, optional
        Escape fraction of ionizing photons. Default: 0.02.
    f_alpha : float, optional
        Fraction of recombinations producing a Lyman-alpha photon. Default: 0.68.
    b : float, optional
        UV spectral slope. Default: -1.7 (Dijkstra & Wyithe 2010).
    lambda_UV : float, optional
        UV continuum reference wavelength in metres. Default: 1500e-10 m.
    save : bool, optional
        If True, save results to an HDF5 file. Default: False.
    savepath : str, optional
        Directory to save output files. Default: './'.
    filename :  str, optional
        Name of the output file. Default: None
 
    Returns
    -------
    results : dict with keys:
        'REW'           : np.ndarray, shape (N,) — rest-frame EW in Angstroms
        'L_alpha_int'   : np.ndarray, shape (N,) — intrinsic Lya luminosity (erg/s)
        'L_alpha_trans' : np.ndarray, shape (N,) — transmitted Lya luminosity (erg/s)
        'L_UV_nu'       : np.ndarray, shape (N,) — UV luminosity density (erg/s/Hz)
 
    Examples
    --------
    >>> results = lyaDW.core.luminosity.compute_luminosities(
    ...     T_alpha, delta, sigma_v, SFR, sim, z_s=6.94,
    ...     f_vel_out=0.0, b=-1.7,
    ...     save=True, savepath='./output/'
    ... )
    """
    # ---- Resolve redshift ---- #
    z_s_resolved = sim.resolve_redshift(z_s)

    # ---- Validate inputs ---- #
    T_alpha = np.asarray(T_alpha, dtype=np.float64)
    delta = np.asarray(delta,   dtype=np.float64)
    sigma_v = np.asarray(sigma_v, dtype=np.float64)
    SFR = np.asarray(SFR,     dtype=np.float64)
 
    N_gal = len(T_alpha)

    if len(sigma_v) != N_gal:
        raise ValueError(
            f"sigma_v length ({len(sigma_v)}) must match T_alpha length ({N_gal})."
        )
    if len(SFR) != N_gal:
        raise ValueError(
            f"SFR length ({len(SFR)}) must match T_alpha length ({N_gal})."
        )
    if np.any((T_alpha < 0) | (T_alpha > 1)):
        raise ValueError("T_alpha values must be in [0, 1].")
 
    # ---- Frequency and wavelength grids ---- #
    nu_rest = NU_ALPHA / (1.0 + delta)      # Hz, decreasing
    dnu_dv = -nu_rest / C_KM_S            # Hz / (km/s)
 
    # ---- Outflow velocities ---- #
    vel_out = f_vel_out * sigma_v
 
    # ---- Intrinsic Lyman-alpha profile ---- #
    J_v = intrinsic_lya_profile(
        delta, sigma_v, SFR, vel_out,
        f_esc=f_esc, f_alpha=f_alpha
    )                                       # W/(km/s), shape (N_gal, N_delta)
 
    # Convert to W/Hz
    J_nu = J_v / np.abs(dnu_dv[None, :])   # shape (N_gal, N_delta)
 
    # ---- Intrinsic Lyman-alpha luminosity (W) ---- #
    nu_rest_rev = jnp.array(nu_rest[::-1],  dtype=jnp.float64)
    J_nu_rev = jnp.array(J_nu[:, ::-1],  dtype=jnp.float64)
 
    L_alpha_int_jax = trapezoid(J_nu_rev, nu_rest_rev, axis=1)
    L_alpha_int = np.asarray(L_alpha_int_jax, dtype=np.float64)
 
    # Convert W -> erg/s
    L_alpha_int_ergs = L_alpha_int * 1e7
 
    # ---- Transmitted Lyman-alpha luminosity (erg/s) ---- #
    L_alpha_trans = L_alpha_int_ergs * T_alpha
 
    # ---- UV continuum luminosity density (erg/s/Hz) ---- #
    L_UV_nu = SFR / K_UV
 
    # ---- Rest-frame equivalent width (Angstroms) ---- #
    REW = (
        (L_alpha_trans / L_UV_nu)
        * (LAMBDA_ALPHA / NU_ALPHA)
        * (lambda_UV / LAMBDA_ALPHA) ** (b + 2.0)
    ) * 1e10    # convert m to Angstroms
 
    # ---- Result dictionary ---- #
    results = {
        'REW':           REW,
        'L_alpha_int':   L_alpha_int_ergs,
        'L_alpha_trans': L_alpha_trans,
        'L_UV_nu':       L_UV_nu,
    }
 
    # ---- Save (optional) ---- #
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_simname = f"{sim.simname}" if sim.simname else ""
        if filename is None:
            filename = f"lum_rew_{sim_simname}_z{z_s_resolved:.2f}_fvelout{f_vel_out}_b{abs(b)}.h5"
        filepath = os.path.join(savepath, filename)
 
        with h5py.File(filepath, 'w') as f:
            lum = f.create_group('luminosities')
            lum.create_dataset('L_alpha_int', data=L_alpha_int_ergs)
            lum.create_dataset('L_alpha_trans', data=L_alpha_trans)
            lum.create_dataset('L_UV_nu', data=L_UV_nu)
 
            rew = f.create_dataset('REW', data=REW)
            rew.attrs['units'] = 'Angstroms'
            rew.attrs['b'] = b
            rew.attrs['lambda_UV'] = lambda_UV
            rew.attrs['f_vel_out'] = f_vel_out
            rew.attrs['z_s'] = z_s_resolved
            if sim.simname:
                rew.attrs['simulation'] = sim.simname
 
        print(f"Saved equivalent widths and luminosities to: {filepath}")
 
    return results
