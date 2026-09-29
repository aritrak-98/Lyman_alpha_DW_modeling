"""
optical_depth.py - computes the Lyman-alpha damping wing optical depth of the IGM 

For each galaxy, it computes the damping wing optical depth tau_DW as a function
of the frequency offset delta. Photons blueward of the Lyman-alpha line centre
(delta < 0) are assumed to be completely absorbed by the IGM (tau = inf),
consistent with the Gunn-Peterson trough.
 
The computation uses Gauss-Legendre quadrature and is fully JAX-accelerated
and vmapped over all galaxies.
 

"""

import numpy as np
import jax
import jax.numpy as jnp
import h5py
import os
from numpy.polynomial.legendre import leggauss
from tqdm import tqdm 


# ---------------------------------------------------------------------------
# Lyman-alpha physical constants
# ---------------------------------------------------------------------------
 
# Quantum damping wing parameter R_alpha (dimensionless)
R_ALPHA = 2.02e-8
 
# Lyman-alpha frequency (Hz)
NU_ALPHA = 2.47e15



# ---------------------------------------------------------------------------
# Internal JAX functions
# ---------------------------------------------------------------------------
 
@jax.jit
def _del_I_integrand(x):
    """
    Integrand for the computing damping wing tau.
 
    I(x) = integral of x^4.5 / (1 - x)^2 dx
 
    Parameters
    ----------
    x : jnp.ndarray
        Dimensionless variable x = (1 + z) / ((1 + z_s)(1 + delta)).
        z is any redshift between z_beg and z_end. z_s is the source redshift.
 
    Returns
    -------
    jnp.ndarray
        Integrand values.
    """
    return (x**4.5) / (1.0 - x)**2



def _build_gauss_legendre(n_quad=64):
    """
    Precompute Gauss-Legendre nodes and weights.
 
    Parameters
    ----------
    n_quad : int
        Number of quadrature points. Default = 64
 
    Returns
    -------
    nodes : jnp.ndarray, shape (n_quad,)
    weights : jnp.ndarray, shape (n_quad,)
    """
    nodes_np, weights_np = leggauss(n_quad)
    return jnp.array(nodes_np, dtype=np.float64), jnp.array(weights_np, dtype=np.float64)


@jax.jit
def _tau_DW_batched(delta, z_s, x_HI_mean, z_beg, z_end, Omega_b_h2, Omega_m_h2, nodes, weights):
    """
    Compute tau_DW for all galaxies and delta values.
 
    For delta < 0 (blueward of line centre), tau_DW = inf, consistent
    with complete IGM absorption in the Gunn-Peterson trough.
 
    Parameters
    ----------
    delta : jnp.ndarray, shape (N_delta,)
        Fractional frequency offsets.
    z_s : float
        Source redshift.
    x_HI_mean : float
        Volume-averaged mean neutral fraction of the IGM.
    z_beg : jnp.ndarray, shape (N_gal,)
        IGM start redshift per galaxy.
    z_end : float
        IGM end redshift.
    Omega_b_h2 : float
        Omega_b * h^2.
    Omega_m_h2 : float
        Omega_m * h^2.
    nodes : jnp.ndarray
        Gauss-Legendre nodes
    weights : jnp.ndarray
        Gauss-Legendre weights
 
    Returns
    -------
    jnp.ndarray, shape (N_gal, N_delta)
        Damping wing optical depth.
    """

    def _integrate_gauss_single(x0, x1):
        mid = 0.5 * (x0 + x1)
        half = 0.5 * (x1 - x0)
        x = half * nodes + mid
        y = _del_I_integrand(x)
        return half * jnp.sum(weights * y)

    # Gunn-Peterson optical depth   # Miralda-Escude (1998), Dijkstra (2014).
    Y = 0.24
    tau_gp = (
        4e5
        * ((1.0 + z_s) / 7.0) ** 1.5
        * (Omega_b_h2 / 0.0225)
        * (Omega_m_h2 / 0.132) ** (-0.5)
        * ((1.0 - Y) / 0.76)
    )

    z_end = jnp.ones(z_beg.size)*z_end
    
    # x limits: shape (N_gal, N_delta)
    x_beg = (1.0 + z_beg[:, None]) / ((1.0 + z_s) * (1.0 + delta[None, :]))
    x_end = (1.0 + z_end[:, None]) / ((1.0 + z_s) * (1.0 + delta[None, :]))

    # vmap over galaxies (axis 0) and delta values (axis 1)
    v_integrate = jax.vmap(
        jax.vmap(_integrate_gauss_single, in_axes=(0, 0)),
        in_axes=(0, 0)
    )
    del_I = v_integrate(x_end, x_beg)

    # tau_DW: shape (N_gal, N_delta)
    tau_dw = (
        (tau_gp / jnp.pi)
        * R_ALPHA
        * x_HI_mean
        * (1.0 + delta[None, :]) ** 1.5
        * del_I
    )

    # Blueward photons (delta < 0) are completely absorbed by the IGM
    tau_dw = jnp.where(delta[None, :] >= 0.0, tau_dw, jnp.inf)
 
    return tau_dw



# ---------------------------------------------------------------------------
# The function the user needs to call
# ---------------------------------------------------------------------------
 
# Default delta grid — fractional frequency offset from Lyman-alpha line centre
DEFAULT_DELTA = np.arange(-0.05, 0.05 + 5e-5, 5e-5)


def compute_tau(z_beg, z_s, sim, z_end=5.5, delta=None, n_quad=64, 
                batch_size=10000, save=False, savepath='./', filename=None):
    """
    Compute the Lyman-alpha damping wing optical depth tau_DW for each galaxy.
 
    tau_DW is computed as a function of the fractional frequency offset delta,
    for each galaxy characterised by its IGM start redshift z_beg.
 
    Photons blueward of the Lyman-alpha line centre (delta < 0) are assigned
    tau_DW = inf, consistent with complete absorption in the Gunn-Peterson trough.
 
    Parameters
    ----------
    z_beg : array-like, shape (N,)
        IGM start redshift for each galaxy, as returned by igm_redshift.compute_z_beg.
    z_s : float
        Source redshift. Resolved to the closest available redshift in sim.
    sim : lyaDW.Simulation
        Simulation context. Must have mean_xHI set for z_s, either via
        set_HII_cube() or set_mean_xHI().
    z_end : float, optional
        Redshift at which the neutral IGM ends. Default: 5.5.
    delta : np.ndarray, shape (N_delta,), optional
        Fractional frequency offset grid. If None, uses the default grid:
        np.arange(-0.05, 0.05 + 5e-5, 5e-5). Default: None.
    n_quad : int, optional
        Number of Gauss-Legendre quadrature points for the delta_I integral.
        Higher values give more accurate results at the cost of speed.
        Default: 64.
    batch_size : int, optional
        Number of galaxies to process per JAX batch. Reduce if running out
        of memory. Default: 5000.
    save : bool, optional
        If True, save tau_DW to an HDF5 file. Default: False.
    savepath : str, optional
        Directory to save the output file. Used only if save=True.
        Default: './'.
    filename :  str, optional
        Name of the output file. Default: None
 
    Returns
    -------
    tau_dw : np.ndarray, shape (N, N_delta)
        Damping wing optical depth for each galaxy and delta value.
        Values are inf for delta < 0.
    delta : np.ndarray, shape (N_delta,)
        The delta grid used (returned for convenience if default was used).
 
    Notes
    -----
    - The mean neutral fraction x_HI_mean is retrieved automatically from
      sim.get_mean_xHI(z_s). Make sure to call sim.set_HII_cube() or
      sim.set_mean_xHI() before calling this function.
    - Processing is done in batches of batch_size galaxies to manage memory.
 
    Examples
    --------
    >>> tau_dw, delta = lyaDW.core.optical_depth.compute_tau(
    ...     z_beg, z_s=6.94, sim=sim,
    ...     z_end=5.5, n_quad=64,
    ...     save=True, savepath='./output/'
    ... )
    """
    # ---- Resolve redshift and get mean x_HI ---- #
    z_s_resolved = sim.resolve_redshift(z_s)
    x_HI_mean = sim.get_mean_xHI(z_s_resolved)

    # ---- Set up delta grid ---- #
    if delta is None:
        delta = DEFAULT_DELTA.copy()
    delta = np.asarray(delta, dtype=np.float64)

    # ---- Validate inputs ---- #
    z_beg = np.asarray(z_beg, dtype=np.float64)
    if z_beg.ndim != 1:
        raise ValueError(f"z_beg must be a 1D array, got shape {z_beg.shape}.")
    if z_end >= z_s_resolved:
        raise ValueError(f"z_end ({z_end}) must be less than z_s ({z_s_resolved}).")

    # ---- Cosmological parameters ---- #
    cosmo      = sim.cosmo
    Omega_b_h2 = cosmo['Omega_b_h2']
    Omega_m_h2 = cosmo['Omega_m_h2']
 
    # ---- Precompute Gauss-Legendre nodes and weights ---- #
    nodes, weights = _build_gauss_legendre(n_quad)


    # ---- JIT compile once before the batch loop ---- #
    jitted_tau_DW = jax.jit(_tau_DW_batched)

    # ---- Process in batches ---- #
    N_gal = len(z_beg)
    num_batches = (N_gal + batch_size - 1) // batch_size
    tau_dw = np.empty((N_gal, len(delta)), dtype=np.float64)
 
    delta_jax = jnp.array(delta, dtype=jnp.float64)

    for batch_idx in tqdm(range(num_batches)):
        start = batch_idx * batch_size
        end   = min(start + batch_size, N_gal)
 
        batch_z_beg = jnp.array(z_beg[start:end])

        batch_tau = jitted_tau_DW(
            delta_jax, z_s_resolved, x_HI_mean,
            batch_z_beg, z_end,
            Omega_b_h2, Omega_m_h2,
            nodes, weights
        )
        tau_dw[start:end, :] = np.asarray(batch_tau, dtype=np.float64)

    # ---- Save (optional) ---- #
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_simname = f"{sim.simname}" if sim.simname else ""
        if filename is None:
            filename  = f"tau_dw_{sim_simname}_z{z_s_resolved:.2f}.h5"
        filepath  = os.path.join(savepath, filename)
 
        with h5py.File(filepath, 'w') as f:
            ds = f.create_dataset('tau_dw', data=tau_dw)
            ds.attrs['z_s'] = z_s_resolved
            ds.attrs['z_end'] = z_end
            ds.attrs['x_HI_mean'] = x_HI_mean
            ds.attrs['n_quad'] = n_quad
            if sim.simname:
                ds.attrs['simulation'] = sim.simname
 
            f.create_dataset('delta', data=delta)
            f.create_dataset('z_beg', data=z_beg)
 
        print(f"Saved tau_DW to: {filepath}")
 
    return tau_dw, delta