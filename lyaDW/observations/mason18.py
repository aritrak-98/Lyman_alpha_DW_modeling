"""
mason18.py — Lyman-alpha fraction computation using the Mason et al. 2018 EW model.

The observed EW distribution at z=6 is modelled as a mixture of
an exponential and a delta function at EW=0 (Mason et al. 2018):

    P_6(EW | M_UV) = A(M_UV) / EW_c(M_UV) * exp(-EW / EW_c(M_UV)) * H(EW)
                   + [1 - A(M_UV)] * delta(EW)

where:
    A(M_UV)    = 0.65 + 0.1 * tanh[3 * (M_UV + 20.75)]
    EW_c(M_UV) = 31 + 12 * tanh[4 * (M_UV + 20.25)]
    H(EW)      = Heaviside step function

P_6 is the observed EW distribution at z=6. To get the intrinsic EW,
we deconvolve by T_alpha_z6:
    EW_int = EW_obs_z6 / T_alpha_z6

At redshift z, the observed EW is then:
    EW_obs(z) = EW_int * T_alpha(z) = EW_obs_z6 * T_alpha(z) / T_alpha_z6

Reference
---------
Mason et al. (2018), ApJ, 856, 2
"""

import numpy as np
import os
import pickle
from tqdm import tqdm
import sys


# ---------------------------------------------------------------------------
# EW thresholds for Lyman-alpha fraction
# ---------------------------------------------------------------------------

EW_THRESHOLD_25 = 25.0    # Angstrom
EW_THRESHOLD_10 = 10.0    # Angstrom


# ---------------------------------------------------------------------------
# Mason+18 model parameters
# ---------------------------------------------------------------------------

def _A_MUV(M_UV):
    """
    Fraction of galaxies with Lyman-alpha emission (Mason+18 Eq.).

    Parameters
    ----------
    M_UV : float or np.ndarray
        UV absolute magnitude.

    Returns
    -------
    float or np.ndarray
        Emission fraction A(M_UV) in [0, 1].
    """
    return 0.65 + 0.1 * np.tanh(3.0 * (M_UV + 20.75))


def _EW_c_MUV(M_UV):
    """
    Characteristic EW scale (Mason+18 Eq.).

    Parameters
    ----------
    M_UV : float or np.ndarray
        UV absolute magnitude.

    Returns
    -------
    float or np.ndarray
        Characteristic EW in Angstrom.
    """
    return 31.0 + 12.0 * np.tanh(4.0 * (M_UV + 20.25))


def _sample_EW_mason18_z6(M_UV_arr, N_samples, rng):
    """
    Draw EW samples from the Mason+18 P_6(EW | M_UV) distribution.

    For each galaxy, samples EW from:
        P_6(EW | M_UV) = A/EW_c * exp(-EW/EW_c) * H(EW) + (1-A) * delta(EW)

    This is a mixture: with probability A, draw from Exp(EW_c);
    with probability (1-A), EW=0.

    Parameters
    ----------
    M_UV_arr : np.ndarray, shape (N_gal,)
        UV absolute magnitudes of galaxies.
    N_samples : int
        Number of Monte Carlo samples.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    EW_samples : np.ndarray, shape (N_samples, N_gal)
        EW samples in Angstrom.
    """
    N_gal  = len(M_UV_arr)
    A      = _A_MUV(M_UV_arr)     # (N_gal,)
    EW_c   = _EW_c_MUV(M_UV_arr)  # (N_gal,)

    # Draw from exponential distribution: Exp(EW_c)
    EW_exp = rng.exponential(scale=EW_c[None, :],
                             size=(N_samples, N_gal))   # (N_samples, N_gal)

    # With probability (1-A), set EW=0
    u      = rng.uniform(size=(N_samples, N_gal))
    emits  = u < A[None, :]                             # True if galaxy emits

    return np.where(emits, EW_exp, 0.0)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def calc_lya_fraction(T_alpha_z6, T_alpha_z, M_UV_z6, M_UV_z, z, sim,
                      M_UV_1=-20.25, M_UV_2=-18.75,
                      N_samples=1000, N_T_samples=100,
                      seed=1216, save=False, savepath='./',
                      filename=None):
    """
    Compute the Lyman-alpha fraction using the Mason et al. 2018 EW model.

    P_6(EW | M_UV) is the observed EW distribution at z=6. To get the
    intrinsic EW, we deconvolve by T_alpha_z6:
        EW_int = EW_obs_z6 / T_alpha_z6

    At redshift z, the observed EW is then:
        EW_obs(z) = EW_int * T_alpha(z) = EW_obs_z6 * T_alpha(z) / T_alpha_z6

    The Lyman-alpha fraction is:
        X_25 = fraction of galaxies with EW_obs > 25 Angstrom
        X_10 = fraction of galaxies with EW_obs > 10 Angstrom

    Parameters
    ----------
    T_alpha_z6 : np.ndarray, shape (N_gal_z6,)
        IGM transmission at z=6.
    T_alpha_z : np.ndarray, shape (N_gal_z,)
        IGM transmission at target redshift z.
    M_UV_z6 : np.ndarray, shape (N_gal_z6,)
        UV absolute magnitudes at z=6.
    M_UV_z : np.ndarray, shape (N_gal_z,)
        UV absolute magnitudes at target redshift z.
    z : float
        Target redshift.
    sim : lyaDW.Simulation
        Simulation context.
    M_UV_1 : float, optional
        Faint end of M_UV selection. Default: -20.25.
    M_UV_2 : float, optional
        Bright end of M_UV selection. Default: -18.75.
    N_samples : int, optional
        Number of Monte Carlo samples for uncertainty estimation. Default: 1000.
    N_T_samples : int, optional
        Number of T_alpha samples per Monte Carlo sample. Default: 100.
    seed : int, optional
        Random seed. Default: 1216.
    save : bool, optional
        If True, save results to a pickle file. Default: False.
    savepath : str, optional
        Directory to save output. Default: './'.
    filename : str, optional
        Output filename. Default: None.

    Returns
    -------
    dict with keys:
        'X_25'    : np.ndarray, shape (N_samples,) — Lya fraction EW > 25 A at z
        'X_10'    : np.ndarray, shape (N_samples,) — Lya fraction EW > 10 A at z
        'X_25_z6' : np.ndarray, shape (N_samples,) — Lya fraction EW > 25 A at z=6
        'X_10_z6' : np.ndarray, shape (N_samples,) — Lya fraction EW > 10 A at z=6
        'z'       : float — resolved redshift

    Notes
    -----
    - M_UV selection is applied to both z=6 and target z samples.
    - If no galaxies pass the M_UV cut, returns zeros with a warning.
    - X_25_z6 and X_10_z6 are computed directly from P_6 (no additional
      T_alpha modulation) since P_6 is already the observed distribution.
    - T_ratio_matrix[i, j] = T_z_samples[i] / T_z6_samples[j] is precomputed
      outside the loop for efficiency.

    Examples
    --------
    >>> result = lyaDW.observations.mason18.calc_lya_fraction(
    ...     T_alpha_z6, T_alpha_z,
    ...     M_UV_z6, M_UV_z,
    ...     z=7.33, sim=sim,
    ...     N_samples=1000, N_T_samples=100,
    ...     save=True, savepath='./output/'
    ... )
    >>> print(np.percentile(result['X_25'], [16, 50, 84]))
    """
    import warnings

    # --- Setup ---
    z_resolved = sim.resolve_redshift(z)
    rng        = np.random.default_rng(seed)

    # --- Apply M_UV cuts ---
    T_z6   = np.asarray(T_alpha_z6, dtype=np.float64)
    T_z    = np.asarray(T_alpha_z,  dtype=np.float64)
    M_UV_z6 = np.asarray(M_UV_z6,   dtype=np.float64)
    M_UV_z  = np.asarray(M_UV_z,    dtype=np.float64)

    uv_mask_z6 = (M_UV_z6 >= M_UV_1) & (M_UV_z6 < M_UV_2)
    uv_mask_z  = (M_UV_z  >= M_UV_1) & (M_UV_z  < M_UV_2)

    T_z6 = T_z6[uv_mask_z6]
    T_z  = T_z[uv_mask_z]

    M_UV_z6_sel = M_UV_z6[uv_mask_z6]
    M_UV_z_sel  = M_UV_z[uv_mask_z]

    # --- Check for empty samples ---
    if len(T_z6) == 0 or len(T_z) == 0:
        which = 'z' if len(T_z) == 0 else 'z=6'
        warnings.warn(
            f"No galaxies found in M_UV bin [{M_UV_1}, {M_UV_2}] at "
            f"{which}={z_resolved:.2f}. Returning X_25=0, X_10=0.",
            UserWarning, stacklevel=2
        )
        # Still compute z=6 anchor from P_6 directly
        EW_z6 = _sample_EW_mason18_z6(M_UV_z6_sel, N_samples, rng)
        X_25_z6 = np.mean(EW_z6 > EW_THRESHOLD_25, axis=-1)
        X_10_z6 = np.mean(EW_z6 > EW_THRESHOLD_10, axis=-1)

        results = {
            'X_25':    np.zeros(N_samples, dtype=np.float64),
            'X_10':    np.zeros(N_samples, dtype=np.float64),
            'X_25_z6': X_25_z6,
            'X_10_z6': X_10_z6,
            'z':       z_resolved,
        }
        if save:
            os.makedirs(savepath, exist_ok=True)
            sim_label = f"_{sim.simname}" if sim.simname else ""
            fname     = filename or f"lya_fraction_mason18{sim_label}_z{z_resolved:.2f}.pkl"
            with open(os.path.join(savepath, fname), 'wb') as f_out:
                pickle.dump(results, f_out)
        return results

    # --- z=6 anchor: sample directly from P_6 (already observed) ---
    EW_z6   = _sample_EW_mason18_z6(M_UV_z6_sel, N_samples, rng)
    X_25_z6 = np.mean(EW_z6 > EW_THRESHOLD_25, axis=-1)   # (N_samples,)
    X_10_z6 = np.mean(EW_z6 > EW_THRESHOLD_10, axis=-1)   # (N_samples,)

    # --- Build T_ratio_matrix = T_z / T_z6 via histogram CDF sampling ---
    T_bins_z6 = np.linspace(0.0, np.nanmax(T_z6) + 1e-6, 50)
    T_bins_z  = np.linspace(0.0, np.nanmax(T_z)  + 1e-6, 50)

    hist_z6, edges_z6 = np.histogram(T_z6, bins=T_bins_z6, density=True)
    hist_z,  edges_z  = np.histogram(T_z,  bins=T_bins_z,  density=True)

    cdf_z6 = np.cumsum(hist_z6) / np.cumsum(hist_z6)[-1]
    cdf_z  = np.cumsum(hist_z)  / np.cumsum(hist_z)[-1]

    centers_z6 = 0.5 * (edges_z6[1:] + edges_z6[:-1])
    centers_z  = 0.5 * (edges_z[1:]  + edges_z[:-1])

    u_z6   = rng.random(N_T_samples)
    u_z    = rng.random(N_T_samples)
    idx_z6 = np.clip(np.searchsorted(cdf_z6, u_z6), 0, len(centers_z6) - 1)
    idx_z  = np.clip(np.searchsorted(cdf_z,  u_z),  0, len(centers_z)  - 1)

    T_z6_samples = centers_z6[idx_z6]    # (N_T_samples,)
    T_z_samples  = centers_z[idx_z]      # (N_T_samples,)

    # T_ratio_matrix[i, j] = T_z_samples[i] / T_z6_samples[j]
    T_z6_samples_safe = np.where(T_z6_samples > 0, T_z6_samples, 1e-12)
    T_ratio_matrix    = T_z_samples[:, None] / T_z6_samples_safe[None, :]
    # shape: (N_T_samples, N_T_samples)

    # --- Monte Carlo loop ---
    X_25 = np.zeros(N_samples, dtype=np.float64)
    X_10 = np.zeros(N_samples, dtype=np.float64)

    for j in tqdm(range(N_samples),
                  desc=f"Computing Lya fraction (Mason+18) at z={z_resolved:.2f}",
                  disable=not sys.stdout.isatty()):

        # Sample N_T_samples galaxies from M_UV-selected sample at target z
        gal_idx   = rng.choice(len(M_UV_z6_sel), size=N_T_samples, replace=True)
        EW_obs_z6 = _sample_EW_mason18_z6(
            M_UV_z6_sel[gal_idx], 1, rng
        )[0]   # (N_T_samples,)

        # EW_obs(z) = EW_obs_z6 * T_z / T_z6
        # shape: (N_T_samples, N_T_samples)
        EW_obs_z = EW_obs_z6[None, :] * T_ratio_matrix

        X_25[j] = np.mean(EW_obs_z > EW_THRESHOLD_25)
        X_10[j] = np.mean(EW_obs_z > EW_THRESHOLD_10)

    results = {
        'X_25':    X_25,
        'X_10':    X_10,
        'X_25_z6': X_25_z6,
        'X_10_z6': X_10_z6,
        'z':       z_resolved,
    }

    # --- Save (optional) ---
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_label = f"_{sim.simname}" if sim.simname else ""
        fname     = filename or f"lya_fraction_mason18{sim_label}_z{z_resolved:.2f}.pkl"
        with open(os.path.join(savepath, fname), 'wb') as f_out:
            pickle.dump(results, f_out)
        print(f"Saved Mason+18 Lya fraction to: {os.path.join(savepath, fname)}")

    return results