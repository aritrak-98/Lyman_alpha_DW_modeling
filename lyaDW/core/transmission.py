"""
transmission.py — Lyman-alpha transmission coefficient computation.
 
For each galaxy, this code computes the IGM Lyman-alpha transmission coefficient T_alpha by
integrating the attenuated Lyman-alpha profile over frequency and dividing
by the intrinsic (unattenuated) profile integral.
 
    T_alpha = integral(J_nu * exp(-tau_DW) dnu) / integral(J_nu dnu)

"""


import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import h5py
import os
 
from lyaDW.utils.galaxy import (
    intrinsic_lya_profile,
    DEFAULT_F_ESC,
    DEFAULT_F_ALPHA,
    NU_ALPHA,
    C_KM_S,
)



# ---------------------------------------------------------------------------
# Internal JAX functions
# ---------------------------------------------------------------------------

@jax.jit
def _compute_T_alpha_batched(J_nu_int, tau_dw, nu_rest):
    """
    Compute T_alpha for a batch of galaxies.
 
    T_alpha = integral(J_nu * exp(-tau_DW) dnu) / integral(J_nu dnu)
 
    Parameters
    ----------
    J_nu_int : jnp.ndarray, shape (N_gal, N_delta)
        Intrinsic Lyman-alpha profile in W/Hz.
    tau_dw : jnp.ndarray, shape (N_gal, N_delta)
        Damping wing optical depth.
    nu_rest : jnp.ndarray, shape (N_delta,)
        Rest-frame frequency array in Hz. Expected to be decreasing.
 
    Returns
    -------
    jnp.ndarray, shape (N_gal,)
        Transmission coefficient T_alpha in [0, 1].
    """
    # Attenuated profile — exp(-inf) = 0 for delta < 0
    J_nu_att = J_nu_int * jnp.exp(-tau_dw)
 
    # Reverse along frequency axis for trapezoid (needs increasing x)
    nu_rest_rev  = nu_rest[::-1]
    J_nu_int_rev = J_nu_int[:, ::-1]
    J_nu_att_rev = J_nu_att[:, ::-1]
 
    # Integrate over frequency for each galaxy (axis=1)
    num   = trapezoid(J_nu_att_rev, nu_rest_rev, axis=1)
    denom = trapezoid(J_nu_int_rev, nu_rest_rev, axis=1)
 
    # Guard against division by zero (galaxies with zero SFR)
    T_alpha = jnp.where(denom > 0.0, num / denom, 0.0)
 
    return T_alpha




# ---------------------------------------------------------------------------
# The main function the user needs to call
# ---------------------------------------------------------------------------
 
def compute_T_alpha(tau_dw, delta, z_s, sigma_v, SFR, sim,
                    f_vel_out=0.0, f_esc=DEFAULT_F_ESC, f_alpha=DEFAULT_F_ALPHA,
                    batch_size=1000, save=False, savepath='./', filename=None):
    """
    Compute the Lyman-alpha IGM transmission coefficient T_alpha for each galaxy.
 
    T_alpha is the fraction of intrinsic Lyman-alpha luminosity that survives
    IGM absorption:
 
        T_alpha = integral(J_nu * exp(-tau_DW) dnu) / integral(J_nu dnu)
 
    Parameters
    ----------
    tau_dw : np.ndarray, shape (N, N_delta)
        Damping wing optical depth, as returned by optical_depth.compute_tau().
    delta : np.ndarray, shape (N_delta,)
        Fractional frequency offset grid, as returned by optical_depth.compute_tau().
    z_s : float
        Source redshift. Resolved to the closest available redshift in sim.
    sigma_v : array-like, shape (N,)
        Velocity dispersion of each galaxy in km/s.
    SFR : array-like, shape (N,)
        Star formation rate of each galaxy in M_Sun/yr.
    sim : lyaDW.Simulation
        Simulation context.
    f_vel_out : float, optional
        Outflow velocity as a multiple of sigma_v. Default: 0.0.
    f_esc : float, optional
        Escape fraction of ionizing photons. Default: 0.02.
    f_alpha : float, optional
        Fraction of recombinations producing a Lyman-alpha photon. Default: 0.68.
    batch_size : int, optional
        Number of galaxies per processing batch. Default: 1000.
    save : bool, optional
        If True, save T_alpha to an HDF5 file. Default: False.
    savepath : str, optional
        Directory to save the output file. Default: './'.
    filename :  str, optional
        Name of the output file. Default: None
 
    Returns
    -------
    T_alpha : np.ndarray, shape (N,)
        Transmission coefficient for each galaxy, in [0, 1].
 
    Examples
    --------
    >>> T_alpha = lyaDW.core.transmission.compute_T_alpha(
    ...     tau_dw, delta, z_s=6.94,
    ...     sigma_v=sigma_v, SFR=SFR, sim=sim,
    ...     f_vel_out=0.0,
    ...     save=True, savepath='./output/'
    ... )
    """
    # ---- Resolve redshift ---- #
    z_s_resolved = sim.resolve_redshift(z_s)
 
    # ---- Validate and convert inputs to float64 ---- #
    tau_dw = np.asarray(tau_dw,  dtype=np.float64)
    delta = np.asarray(delta,   dtype=np.float64)
    sigma_v = np.asarray(sigma_v, dtype=np.float64)
    SFR = np.asarray(SFR,     dtype=np.float64)
 
    N_gal = tau_dw.shape[0]
 
    if tau_dw.shape[1] != len(delta):
        raise ValueError(
            f"tau_dw shape {tau_dw.shape} is inconsistent with "
            f"delta length {len(delta)}. "
            f"Make sure you pass the delta returned by optical_depth.compute()."
        )
    if len(sigma_v) != N_gal:
        raise ValueError(
            f"sigma_v length ({len(sigma_v)}) must match "
            f"number of galaxies in tau_dw ({N_gal})."
        )
    if len(SFR) != N_gal:
        raise ValueError(
            f"SFR length ({len(SFR)}) must match "
            f"number of galaxies in tau_dw ({N_gal})."
        )
 
    # ---- Frequency grids ---- #
    nu_rest = (NU_ALPHA / (1.0 + delta)).astype(np.float64)   # Hz, decreasing
    dnu_dv = -nu_rest / C_KM_S                               # Hz / (km/s)
 
    # ---- Outflow velocities ---- #
    vel_out = f_vel_out * sigma_v    # km/s
 
    # ---- Process in batches ---- #
    num_batches = (N_gal + batch_size - 1) // batch_size
    T_alpha = np.zeros(N_gal, dtype=np.float64)
 
    nu_rest_jax = jnp.array(nu_rest, dtype=jnp.float64)
 
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N_gal)
 
        # Intrinsic profile in W/(km/s), shape (batch, N_delta)
        J_v_batch = intrinsic_lya_profile(
            delta,
            sigma_v[start:end],
            SFR[start:end],
            vel_out[start:end],
            f_esc=f_esc,
            f_alpha=f_alpha
        )
 
        # Convert to W/Hz
        J_nu_batch = J_v_batch / np.abs(dnu_dv[None, :])
 
        # Compute T_alpha for this batch
        J_nu_jax = jnp.array(J_nu_batch, dtype=jnp.float64)
        tau_jax  = jnp.array(tau_dw[start:end], dtype=jnp.float64)
 
        batch_T = _compute_T_alpha_batched(J_nu_jax, tau_jax, nu_rest_jax)
        T_alpha[start:end] = np.asarray(batch_T, dtype=np.float64)
 
    # ---- Save (optional) ---- #
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_simname = f"{sim.simname}" if sim.simname else ""
        if filename is None:
            filename = f"T_alpha_{sim_simname}_z{z_s_resolved:.2f}_fvelout{f_vel_out}.h5"
        filepath = os.path.join(savepath, filename)
 
        with h5py.File(filepath, 'w') as f:
            ds = f.create_dataset('T_alpha', data=T_alpha)
            ds.attrs['z_s'] = z_s_resolved
            ds.attrs['f_vel_out'] = f_vel_out
            ds.attrs['f_esc'] = f_esc
            ds.attrs['f_alpha'] = f_alpha
            if sim.simname:
                ds.attrs['simulation'] = sim.simname
 
        print(f"Saved T_alpha to: {filepath}")
 
    return T_alpha