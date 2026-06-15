"""
igm_redshift.py - computes the starting redshift from the bubble sizes

For each galaxy, it finds the redshift z_beg such that the distance
from z_beg to z_s equals the ionized bubble radius R_ion. This gives the
redshift at which the neutral IGM begins along the line of sight.
 
The bisection is fully JAX-accelerated and vmapped over all galaxies.

"""

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import os
import sys


# ---------------------------------------------------------------------------
# Internal JAX functions
# ---------------------------------------------------------------------------


@jax.jit
def _distance_integrand(z, d_H0, Omega_m, Omega_l):
    """
    Integrand for comoving distance: d_H0 / E(z).
 
    JAX-compatible version for use inside JIT-compiled bisection.
 
    Parameters
    ----------
    z : float or jnp.ndarray
        Redshift.
    d_H0 : float
        Hubble distance in Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_l : float
        Dark energy density parameter.
 
    Returns
    -------
    float or jnp.ndarray
        Integrand value in Mpc.
    """

    return d_H0 * 1.0/(jnp.sqrt((1 + z)**3*Omega_m + Omega_l))

@jax.jit
def _distance_between_z(z_lo, z_hi, d_H0, Omega_m, Omega_l, num_points=100):
    """
    Compute distance between z_lo and z_hi using JAX trapezoid rule.
 
    Used inside the bisection loop where scipy cannot be called.
 
    Parameters
    ----------
    z_lo : float
        Lower redshift bound.
    z_hi : float
        Upper redshift bound.
    d_H0 : float
        Hubble distance in Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_l : float
        Dark energy density parameter.
    num_points : int, optional
        Number of quadrature points. Default: 100.
 
    Returns
    -------
    float
        Distance in Mpc.
    """

    z_points = jnp.linspace(z_lo, z_hi, num_points)

    integrand = _distance_integrand(z_points, d_H0, Omega_m, Omega_l)
    return trapezoid(integrand, z_points)    # in Mpc



def _bisection_batched(R_ion_arr, z_s, z_end, d_H0, Omega_m, Omega_l, tol=1e-6):
    """
    Compute z_beg for all galaxies via JAX vmap over the bisection.
 
    Parameters
    ----------
    R_ion_arr : jnp.ndarray, shape (N,)
        Bubble radii in Mpc.
    z_s : float
        Source redshift.
    z_end : float
        Lower bound redshift for bisection.
    d_H0 : float
        Hubble distance in Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_l : float
        Dark energy density parameter.
 
    Returns
    -------
    jnp.ndarray, shape (N,)
        z_beg values for each galaxy.
    """

    def _bisection_single(R_ion):
        """
        Find z_beg for a single galaxy via JAX bisection.
 
        Solves: distance_between_z(z_beg, z_s) - R_ion = 0
        using a bisection search over [z_end, z_s].
     
        Parameters
        ----------
        R_ion : float
            Ionized bubble radius in Mpc.
            
        Returns
        -------
        float
            z_beg — the redshift at which the neutral IGM begins.
        """

        def f(z_beg):
            return _distance_between_z(z_beg, z_s, d_H0, Omega_m, Omega_l) - R_ion
    
        def cond_fun(state):
            z_hi, z_lo = state
            return (z_hi - z_lo) > tol
    
        def body_fun(state):
            z_hi, z_lo = state
            z_mid = 0.5 * (z_hi + z_lo)
    
            sign = f(z_mid)*f(z_lo) < 0
    
            def true_fn():
                return z_mid, z_lo
    
            def false_fn():
                return z_hi, z_mid
    
            z_hi, z_lo = jax.lax.cond(sign, true_fn, false_fn)
            return z_hi, z_lo

        def run_bisection():
            init_state = (z_s, z_end)
            z_hi_new, z_lo_new = jax.lax.while_loop(cond_fun, body_fun, init_state)
            return 0.5*(z_hi_new + z_lo_new)

        # If R_ion >= d_total, bubble extends beyond z_end — clamp to z_end
        d_total = _distance_between_z(z_end, z_s, d_H0, Omega_m, Omega_l)
        return jax.lax.cond(
            R_ion >= d_total,
            lambda: jnp.float64(z_end),
            lambda: run_bisection()
        )
    
        #init_state = (z_s, z_end)
        #z_hi_new, z_lo_new = jax.lax.while_loop(cond_fun, body_fun, init_state)
        #
        #return 0.5*(z_hi_new + z_lo_new)

    batched_z_beg = jax.vmap(_bisection_single, in_axes=(0))

    return batched_z_beg(R_ion_arr)


#------------------------------------------------------------------------------------
# The function that the user needs to call
#------------------------------------------------------------------------------------

def compute_z_beg(R_ion, z_s, sim, z_end=5.5, 
                  R_ion_units='Mpc', save=False, savepath='./',
                  filename=None):
    """
    Compute the IGM starting redshift z_beg for each galaxy.
 
    For each galaxy, finds z_beg such that the distance from
    z_beg to z_s equals the ionized bubble radius R_ion. This is the
    redshift at which the damping wing optical depth integration begins.
 
    Parameters
    ----------
    R_ion : array-like, shape (N,)
        Ionized bubble radii. Must be in Mpc (see R_ion_units).
    z_s : float
        Source redshift.
    sim : lyaDW.Simulation
        Simulation context carrying cosmological parameters.
    z_end : float, optional
        Lower redshift bound — the redshift at which the neutral IGM ends
        (we assume it to be where reionization is effectively complete). Default: 5.5.
    R_ion_units : str, optional
        Units of R_ion. Currently only 'Mpc' is supported. If your R_ion
        is in other units, convert before calling this function (e.g. using
        bubble_sizes.compute with boxsize_units set). Default: 'Mpc'.
    save : bool, optional
        If True, save z_beg to an HDF5 file. Default: False.
    savepath : str, optional
        Directory to save the output file. Used only if save=True.
        Default: './'.
    filename :  str, optional
        Name of the output file. Default: None
 
    Returns
    -------
    z_beg : np.ndarray, shape (N,)
        Redshift at which the neutral IGM begins for each galaxy.
 
    Raises
    ------
    ValueError
        If R_ion_units is not 'Mpc'.
        If any R_ion value is negative.
        If z_end >= z_s.
 
    Notes
    -----
    - R_ion is treated as the full bubble radius with no internal scaling.
    - The bisection searches over [z_end, z_s] for each galaxy.
    - Galaxies with R_ion = 0 will return z_beg = z_s.
 
    Examples
    --------
    >>> z_beg = lyaDW.core.igm_redshift.compute_z_beg(
    ...     R_ion, z_s=6.94, sim=sim,
    ...     z_end=5.5,
    ...     save=True, savepath='./output/'
    ... )
    """
    # ---- Validate units ---- #
    if R_ion_units != 'Mpc':
        raise ValueError(
            f"R_ion_units='{R_ion_units}' is not supported. "
            f"Please convert R_ion to Mpc before calling this function. "
            f"You can do this by setting boxsize_units in bubble_sizes.compute()."
        )
 
    # ---- Resolve redshift ---- #
    z_s_resolved = sim.resolve_redshift(z_s)
    print('Closest redshift = ', z_s_resolved)
 
    # ---- Validate z_end < z_s ---- #
    if z_end >= z_s_resolved:
        raise ValueError(
            f"z_end ({z_end}) must be less than z_s ({z_s_resolved})."
        )


    # ---- Cosmological parameters ---- #
    cosmo = sim.cosmo
    d_H   = 1.0 / (sim.h * 100.0) * 2.998e5   # c / H0 in Mpc  (c in km/s)
    Omega_m = cosmo['Omega_m']
    Omega_l = cosmo['Omega_l']
 
    # ---- Validate R_ion values ---- #
    R_ion = np.asarray(R_ion, dtype=np.float64)
 
    if R_ion.ndim != 1:
        raise ValueError(
            f"R_ion must be a 1D array, got shape {R_ion.shape}."
        )
 
    # Validate R_ion values
    if np.any(R_ion < 0):
        raise ValueError("R_ion values must be non-negative.")

    # ---- Run batched bisection ---- #
    R_ion_jax = jnp.array(R_ion)
    z_beg_jax = _bisection_batched(R_ion_jax, z_s_resolved, z_end, d_H, Omega_m, Omega_l)
    z_beg = np.asarray(z_beg_jax, dtype=np.float64)


    # ---- Save (optional) ---- #
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_name = f"{sim.simname}" if sim.simname else ""
        if filename is None:
            filename  = f"z_beg_{sim_name}_z{z_s_resolved:.2f}.h5"
        filepath  = os.path.join(savepath, filename)
 
        with h5py.File(filepath, 'w') as f:
            ds = f.create_dataset('z_beg', data=z_beg)
            ds.attrs['z_s']   = z_s_resolved
            ds.attrs['z_end'] = z_end
            if sim.simname:
                ds.attrs['simulation'] = sim.simname
 
            # Also save R_ion for reference
            f.create_dataset('R_ion', data=R_ion)
 
        print(f"Saved z_beg to: {filepath}")
 
    return z_beg