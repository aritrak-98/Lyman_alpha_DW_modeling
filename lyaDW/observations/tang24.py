"""
tang24.py — Lyman-alpha emitter fraction using Tang et al. (2024) as z=6 anchor.
 
Uses the observed log-normal EW distribution from Tang et al. (2024) at z=6
to predict the Lyman-alpha emitter fraction at higher redshifts, by convolving
the z=6 distribution with the IGM transmission ratio P(T_z6 / T_z).
 
The Lyman-alpha emitter fraction is defined as the fraction of galaxies with
rest-frame EW above a threshold (25 Å or 10 Å).
 
----------
get_EW_distribution_z6(EW_obs, N_samples, bright=True)
    Sample the Tang+24 log-normal EW distribution at z=6.
 
calc_lya_fraction(T_alpha_z6, T_alpha_z, z, sim, ...)
    Compute the Lyman-alpha emitter fraction at redshift z.
"""

import numpy as np
import pickle
import os
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from scipy.integrate import simpson
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tang+24 EW distribution parameters
# ---------------------------------------------------------------------------
 
# Bright galaxies: -20.25 < M_UV < -18.75
_TANG24_BRIGHT = {
    'exp_mu_med':   8.0,
    'exp_mu_plus':  4.0,
    'exp_mu_minus': 3.0,
    'sigma_med':    1.85,
    'sigma_plus':   0.42,
    'sigma_minus':  0.33,
}
 
# Faint galaxies: -18.75 < M_UV < -17.25
# (estimated by fitting Tang+24 parameters — use with caution)
_TANG24_FAINT = {
    'exp_mu_med':   22.40,
    'exp_mu_plus':  1.89,
    'exp_mu_minus': 1.97,
    'sigma_med':    1.16,
    'sigma_plus':   0.08,
    'sigma_minus':  0.08,
}
 
# EW thresholds (Angstroms)
EW_THRESHOLD_25 = 25.0
EW_THRESHOLD_10 = 10.0


# ---------------------------------------------------------------------------
# EW probability distribution functions
# ---------------------------------------------------------------------------

def _p_EW_lognormal(EW, mu, sigma):
    """
    Log-normal EW probability distribution.
 
    Parameters
    ----------
    EW : np.ndarray, shape (N_EW,)
        EW values in Angstroms.
    mu : np.ndarray, shape (N_samples,)
        Log-normal mean parameter samples.
    sigma : np.ndarray, shape (N_samples,)
        Log-normal width parameter samples.
 
    Returns
    -------
    np.ndarray, shape (N_samples, N_EW)
        Probability density values.
    """
    mu = mu[:, None]
    sigma = sigma[:, None]
    return (
        np.exp(-(((np.log(EW) - mu)**2) / (2.0 * sigma**2)))
        / (np.sqrt(2.0 * np.pi) * sigma * EW)
    )


@jax.jit
def _p_EW_lognormal_jax(EW, mu, sigma):
    """
    JAX-compiled log-normal EW probability distribution (scalar mu, sigma).
 
    Parameters
    ----------
    EW : jnp.ndarray, shape (N_EW,)
        EW values.
    mu : float
        Log-normal mean parameter.
    sigma : float
        Log-normal width parameter.
 
    Returns
    -------
    jnp.ndarray, shape (N_EW,)
        Probability density values.
    """
    return (
        jnp.exp(-(((jnp.log(EW) - mu)**2) / (2.0 * sigma**2)))
        / (jnp.sqrt(2.0 * jnp.pi) * sigma * EW)
    )


# ---------------------------------------------------------------------------
# Tang+24 parameter sampling
# ---------------------------------------------------------------------------
def get_EW_distribution_z6(EW_obs, N_samples=1000, bright=True, seed=None):
    """
    Sample the Tang et al. (2024) log-normal EW distribution at z=6.
 
    Draws N_samples realisations of (mu, sigma) from the asymmetric
    uncertainties reported in Tang+24, and evaluates the log-normal
    EW PDF for each realisation.
 
    Parameters
    ----------
    EW_obs : np.ndarray, shape (N_EW,)
        EW grid in Angstroms at which to evaluate the distribution.
    N_samples : int, optional
        Number of (mu, sigma) parameter samples. Default: 1000.
    bright : bool, optional
        If True, use bright galaxy parameters (-20.25 < M_UV < -18.75).
        If False, use faint galaxy parameters (-18.75 < M_UV < -17.25).
        Default: True.
    seed : int or None, optional
        Random seed for reproducibility. Default: None.
 
    Returns
    -------
    P_EW_z6 : np.ndarray, shape (N_samples, N_EW)
        EW probability density samples at z=6.
    mu_samples : np.ndarray, shape (N_samples,)
        Sampled log-normal mean parameters.
    sigma_samples : np.ndarray, shape (N_samples,)
        Sampled log-normal width parameters.

    References
    ----------
    Tang et al. (2024).
 
    Examples
    --------
    >>> EW_obs = np.logspace(-5, 5, 200)
    >>> P_z6, mu_s, sigma_s = get_EW_distribution_z6(EW_obs, N_samples=1000)
    """
    rng = np.random.default_rng(seed)
    params = _TANG24_BRIGHT if bright else _TANG24_FAINT
 
    k_mu = rng.standard_normal(N_samples)
    k_sigma = rng.standard_normal(N_samples)
 
    # Sample mu (in log space, accounting for asymmetric uncertainties)
    exp_mu_med = params['exp_mu_med']
    exp_mu_plus = params['exp_mu_plus']
    exp_mu_minus = params['exp_mu_minus']
 
    mu_med = np.log(exp_mu_med)
    mu_plus = np.log(exp_mu_med + exp_mu_plus)  - np.log(exp_mu_med)
    mu_minus = np.log(exp_mu_med) - np.log(exp_mu_med - exp_mu_minus)
 
    mu_samples = np.where(
        k_mu < 0,
        mu_med + k_mu * mu_minus,
        mu_med + k_mu * mu_plus
    )

    # Sample sigma (symmetric uncertainties, clipped to positive)
    sigma_med = params['sigma_med']
    sigma_plus = params['sigma_plus']
    sigma_minus = params['sigma_minus']
 
    sigma_samples = np.where(
        k_sigma < 0,
        sigma_med + k_sigma * sigma_minus,
        sigma_med + k_sigma * sigma_plus
    )
    sigma_samples = np.clip(sigma_samples, 1e-6, None)
 
    # Evaluate PDF for each sample
    P_EW_z6 = _p_EW_lognormal(EW_obs, mu_samples, sigma_samples)
 
    return P_EW_z6, mu_samples, sigma_samples



# ---------------------------------------------------------------------------
# Predicted EW distribution at redshift z
# ---------------------------------------------------------------------------
 
@jax.jit
def _compute_P_EW_z(EW_obs, mu, sigma, T_ratio_matrix):
    """
    Compute the predicted EW distribution at redshift z for a single
    (mu, sigma) realisation.
 
    Convolves the z=6 log-normal distribution with the transmission
    ratio P(T_z6 / T_z):
 
        P(EW | z) = (1/N^2) * sum_{T_z6, T_z} (T_z6/T_z) * p_z6(EW * T_z6/T_z)
 
    Parameters
    ----------
    EW_obs : jnp.ndarray, shape (N_EW,)
        EW grid in Angstroms.
    mu : float
        Log-normal mean parameter at z=6.
    sigma : float
        Log-normal width parameter at z=6.
    T_ratio_matrix : jnp.ndarray, shape (N_T6, N_Tz)
        Matrix of T_z6 / T_z ratios.
 
    Returns
    -------
    jnp.ndarray, shape (N_EW,)
        Predicted EW probability density at redshift z.
    """
    def body(EW):
        EW_scaled = EW * T_ratio_matrix
        P_i = _p_EW_lognormal_jax(EW_scaled, mu, sigma)
        return jnp.sum(T_ratio_matrix * P_i)
 
    P = jax.vmap(body)(EW_obs)
    P = P / (T_ratio_matrix.shape[0] * T_ratio_matrix.shape[1])
 
    return P


# ---------------------------------------------------------------------------
# Lyman-alpha emitter fractions
# ---------------------------------------------------------------------------
 
def calc_lya_fraction(T_alpha_z6, T_alpha_z, z, sim,
                      M_UV_z6=None, M_UV_z=None,
                      M_UV_1=-20.25, M_UV_2=-18.75,
                      f_vel_out=0.0, EW_obs=None, N_samples=1000,
                      N_T_samples=1000, bright=True,
                      seed=1216, save=False, savepath='./',
                      filename=None):
    """
    Compute the Lyman-alpha emitter fraction at redshift z.
 
    Uses the Tang+24 EW distribution at z=6 as an anchor, and predicts
    the EW distribution at redshift z by convolving with the IGM
    transmission ratio P(T_z6 / T_z).
 
    The Lyman-alpha fraction is defined as the fraction of galaxies with
    rest-frame EW >= 25 Å (X_25) or >= 10 Å (X_10).
 
    Parameters
    ----------
    T_alpha_z6 : np.ndarray, shape (N_z6,)
        Transmission coefficients at z~6, from transmission.compute().
    T_alpha_z : np.ndarray, shape (N_z,)
        Transmission coefficients at target redshift z.
    z : float
        Target redshift. Resolved via sim.resolve_redshift().
    sim : lyaDW.Simulation
        Simulation context.
    M_UV_z6 : np.ndarray, shape (N_z6,) or None, optional
        UV magnitudes at z~6. Used to apply M_UV_1/M_UV_2 cut.
        If None, no magnitude cut is applied.
    M_UV_z : np.ndarray, shape (N_z,) or None, optional
        UV magnitudes at target redshift z. If None, no cut applied.
    M_UV_1 : float, optional
        Bright end of UV magnitude bin. Default: -20.25.
    M_UV_2 : float, optional
        Faint end of UV magnitude bin. Default: -18.75.
    f_vel_out : float, optional
        Outflow velocity as a multiple of sigma_v. Default: 0.0.
    EW_obs : np.ndarray or None, optional
        EW grid in Angstroms. If None, uses np.logspace(-5, 5, 200).
    N_samples : int, optional
        Number of Tang+24 (mu, sigma) parameter samples. Default: 1000.
    N_T_samples : int, optional
        Number of transmission samples drawn from P(T). Default: 1000.
    bright : bool, optional
        If True, use Tang+24 bright galaxy parameters. Default: True.
    seed : int, optional
        Random seed for reproducibility. Default: 1216.
    save : bool, optional
        If True, save results to a pickle file. Default: False.
    savepath : str, optional
        Directory to save the output file. Default: './'.
    filename :  str, optional
        Name of the output file. Default: None
 
    Returns
    -------
    results : dict with keys:
        'X_25' : np.ndarray, shape (N_samples,)
            Lyman-alpha fraction (EW >= 25 Å) for each parameter sample.
        'X_10' : np.ndarray, shape (N_samples,)
            Lyman-alpha fraction (EW >= 10 Å) for each parameter sample.
        'z'    : float
            Resolved target redshift.
        'X_25_z6' : np.ndarray, shape (N_samples,)
            Lyman-alpha fraction at z=6 (EW >= 25 Å).
        'X_10_z6' : np.ndarray, shape (N_samples,)
            Lyman-alpha fraction at z=6 (EW >= 10 Å).
 
    Notes
    -----
    - The z=6 fractions are also returned so you can directly compare
      with the Tang+24 anchor values.
    - X_25 and X_10 are arrays of length N_samples, reflecting the
      uncertainty in the Tang+24 parameters. Take the median and
      16th/84th percentiles for plotting.
 
    Examples
    --------
    >>> results = lyaDW.observations.tang24.calc_lya_fraction(
    ...     T_alpha_z6, T_alpha_z, z=7.0, sim=sim,
    ...     N_samples=1000, save=True, savepath='./output/'
    ... )
    >>> X_25_median = np.median(results['X_25'])
    >>> X_25_lo, X_25_hi = np.percentile(results['X_25'], [16, 84])
    """
    # ---- Setup ---- #
    z_resolved = sim.resolve_redshift(z)
    rng = np.random.default_rng(seed)
 
    if EW_obs is None:
        EW_obs = np.logspace(-5, 5, 200)
    EW_obs = np.asarray(EW_obs, dtype=np.float64)
 
    mask_25 = EW_obs >= EW_THRESHOLD_25
    mask_10 = EW_obs >= EW_THRESHOLD_10
 
    # ---- Apply M_UV cuts if provided ---- #
    T_z6 = np.asarray(T_alpha_z6, dtype=np.float64)
    T_z = np.asarray(T_alpha_z, dtype=np.float64)
 
    if M_UV_z6 is not None:
        M_UV_z6 = np.asarray(M_UV_z6, dtype=np.float64)
        uv_mask_z6 = (M_UV_z6 >= M_UV_1) & (M_UV_z6 < M_UV_2)
        T_z6 = T_z6[uv_mask_z6]
 
    if M_UV_z is not None:
        M_UV_z = np.asarray(M_UV_z, dtype=np.float64)
        uv_mask_z = (M_UV_z >= M_UV_1) & (M_UV_z < M_UV_2)
        T_z = T_z[uv_mask_z]

    # --- Check for empty samples after M_UV masking ---
    if len(T_z) == 0 or len(T_z6) == 0:
        import warnings
        which = 'z' if len(T_z) == 0 else 'z=6'
        warnings.warn(
            f"No galaxies found in M_UV bin [{M_UV_1}, {M_UV_2}] at {which}={z_resolved:.2f}. "
            f"Returning X_25=0, X_10=0.",
            UserWarning, stacklevel=2
        )
        # Compute z=6 anchor fractions even if target z is empty
        P_obs_z6, _, _ = get_EW_distribution_z6(
            EW_obs, N_samples=N_samples, bright=bright, seed=seed
        )
        X_25_z6 = simpson(P_obs_z6[:, EW_obs >= EW_THRESHOLD_25],
                          x=EW_obs[EW_obs >= EW_THRESHOLD_25], axis=-1)
        X_10_z6 = simpson(P_obs_z6[:, EW_obs >= EW_THRESHOLD_10],
                          x=EW_obs[EW_obs >= EW_THRESHOLD_10], axis=-1)
        results = {
            'X_25':    np.zeros(N_samples, dtype=np.float64),
            'X_10':    np.zeros(N_samples, dtype=np.float64),
            'X_25_z6': X_25_z6,
            'X_10_z6': X_10_z6,
            'z':       z_resolved,
        }
 
        if save:
            os.makedirs(savepath, exist_ok=True)
            sim_label  = f"_{sim.label}" if sim.label else ""
            bright_str = 'bright' if bright else 'faint'
            filename   = f"lya_fraction{sim_label}_z{z_resolved:.2f}_{bright_str}.pkl"
            filepath   = os.path.join(savepath, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved Lya fraction (empty M_UV bin) to: {filepath}")
 
        return results

 
    # ---- Sample Tang+24 parameters at z=6 ---- #
    P_obs_z6, mu_samples, sigma_samples = get_EW_distribution_z6(
        EW_obs, N_samples=N_samples, bright=bright, seed=seed
    )
 
    # ---- z=6 Lyman-alpha fractions (anchor) ---- #
    X_25_z6 = simpson(P_obs_z6[:, mask_25], x=EW_obs[mask_25], axis=-1)
    X_10_z6 = simpson(P_obs_z6[:, mask_10], x=EW_obs[mask_10], axis=-1)
 
    # ---- Build transmission PDFs via histogram sampling ---- #
    # Sample N_T_samples transmission values from the empirical distributions
    T_bins_z6 = np.linspace(0.0, np.nanmax(T_z6) + 1e-6, 50)
    T_bins_z  = np.linspace(0.0, np.nanmax(T_z)  + 1e-6, 50)
 
    hist_z6, edges_z6 = np.histogram(T_z6, bins=T_bins_z6, density=True)
    hist_z, edges_z = np.histogram(T_z, bins=T_bins_z, density=True)
 
    cdf_z6 = np.cumsum(hist_z6) / np.cumsum(hist_z6)[-1]
    cdf_z = np.cumsum(hist_z) / np.cumsum(hist_z)[-1]
 
    centers_z6 = 0.5 * (edges_z6[1:] + edges_z6[:-1])
    centers_z = 0.5 * (edges_z[1:] + edges_z[:-1])
 
    # Draw samples via inverse CDF
    u_z6 = rng.random(N_T_samples)
    u_z = rng.random(N_T_samples)
    idx_z6 = np.searchsorted(cdf_z6, u_z6)
    idx_z = np.searchsorted(cdf_z,  u_z)
 
    # Clip to valid range
    idx_z6 = np.clip(idx_z6, 0, len(centers_z6) - 1)
    idx_z = np.clip(idx_z, 0, len(centers_z) - 1)
 
    T_z6_samples = centers_z6[idx_z6]
    T_z_samples  = centers_z[idx_z]
 
    # T_ratio_matrix[i, j] = T_z6_samples[i] / T_z_samples[j]
    # Guard against division by zero
    T_z_samples_safe = np.where(T_z_samples > 0, T_z_samples, 1e-12)
    T_ratio_matrix = jnp.array(
        T_z6_samples[:, None] / T_z_samples_safe[None, :]
    )
 
    # ---- Compute Lyman-alpha fraction at redshift z ---- #
    EW_obs_jax = jnp.array(EW_obs)
 
    X_25 = np.zeros(N_samples, dtype=np.float64)
    X_10 = np.zeros(N_samples, dtype=np.float64)
 
    for j in tqdm(range(N_samples), desc=f"Computing Lya fraction at z={z_resolved:.2f}"):
        mu = float(mu_samples[j])
        sigma = float(sigma_samples[j])
 
        P_EW_z = np.asarray(
            _compute_P_EW_z(EW_obs_jax, mu, sigma, T_ratio_matrix)
        )
 
        X_25[j] = simpson(P_EW_z[mask_25], x=EW_obs[mask_25])
        X_10[j] = simpson(P_EW_z[mask_10], x=EW_obs[mask_10])
 
    results = {
        'X_25':     X_25,
        'X_10':     X_10,
        'X_25_z6':  X_25_z6,
        'X_10_z6':  X_10_z6,
        'z':        z_resolved,
    }
 
    # --- Save (optional) ---
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_simname  = f"{sim.simname}" if sim.simname else ""
        bright_str = 'bright' if bright else 'faint'
        if filename is None:
            filename   = f"lya_fraction_{sim_simname}_z{z_resolved:.2f}_fvelout{f_vel_out}_{bright_str}.pkl"
        filepath   = os.path.join(savepath, filename)
 
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
 
        print(f"Saved Lya fraction to: {filepath}")
 
    return results