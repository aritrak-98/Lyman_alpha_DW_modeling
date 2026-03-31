"""
cosmology.py — Some cosmological utility functions for lyaDW.
 
 
Functions
---------
hubble_distance       : Hubble distance d_H = c/H0 in Mpc
comoving_distance     : Comoving distance to redshift z in Mpc
tau_GP                : Gunn-Peterson optical depth at redshift z_s
"""
 
import numpy as np
import scipy.integrate as integrate
import astropy.constants as const
import astropy.units as u
 
 
# Speed of light in km/s
C_KM_S = const.c.to(u.km / u.s).value
 
 
def hubble_distance(sim):
    """
    Compute the Hubble distance d_H = c / H0 in Mpc.
 
    Parameters
    ----------
    sim : lyaDW.Simulation
        Simulation context carrying cosmological parameters.
 
    Returns
    -------
    float
        Hubble distance in Mpc.
 
    Examples
    --------
    >>> d_H = hubble_distance(sim)
    """
    H0 = sim.h * 100.0                        # km/s/Mpc
    return C_KM_S / H0
 
 
def _distance_integrand(z, d_H, Omega_m, Omega_l):
    """
    Integrand for comoving distance: d_H / E(z), where E(z) = H(z)/H0.
 
    Parameters
    ----------
    z : float or array
        Redshift.
    d_H : float
        Hubble distance in Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_l : float
        Dark energy density parameter.
 
    Returns
    -------
    float or array
        Integrand value in Mpc.
    """
    return d_H / np.sqrt((1.0 + z)**3 * Omega_m + Omega_l)
 
 
def comoving_distance(z, sim):
    """
    Compute the comoving distance from z=0 to redshift z in Mpc.
 
    Uses scipy.integrate.quad for accurate numerical integration.
 
    Parameters
    ----------
    z : float
        Redshift.
    sim : lyaDW.Simulation
        Simulation context carrying cosmological parameters.
 
    Returns
    -------
    float
        Comoving distance in Mpc.
 
    Examples
    --------
    >>> d_c = comoving_distance(6.94, sim)
    """
    d_H = hubble_distance(sim)
    d_c, _ = integrate.quad(
        _distance_integrand, 0.0, z,
        args=(d_H, sim.Omega_m, sim.Omega_l)
    )
    return d_c
 
 
def luminosity_distance(z, sim):
    """
    Compute the luminosity distance to redshift z in Mpc.
 
    Parameters
    ----------
    z : float
        Redshift.
    sim : lyaDW.Simulation
        Simulation context carrying cosmological parameters.
 
    Returns
    -------
    float
        Luminosity distance in Mpc.
 
    Examples
    --------
    >>> d_L = luminosity_distance(6.94, sim)
    """
    d_c = comoving_distance(z, sim)
    return (1.0 + z) * d_c
 
 
def tau_GP(z_s, sim):
    """
    Compute the Gunn-Peterson (GP) optical depth at source redshift z_s.
 
    This is the optical depth of a fully neutral IGM at the Lyman-alpha
    line centre, used as the prefactor in the damping wing calculation.
 
    The formula used is:
        tau_GP = 4e5 * ((1+z_s)/7)^(3/2)
                     * (Omega_b_h2 / 0.0225)
                     * (Omega_m_h2 / 0.132)^(-1/2)
                     * ((1 - Y) / 0.76)
    where Y = 0.24 is the helium mass fraction.
 
    Parameters
    ----------
    z_s : float
        Source redshift.
    sim : lyaDW.Simulation
        Simulation context carrying cosmological parameters
        (uses Omega_b_h2 and Omega_m_h2 from sim.cosmo).
 
    Returns
    -------
    float
        Gunn-Peterson optical depth (dimensionless).
 
    References
    ----------
    Miralda-Escude (1998), Dijkstra (2014).
 
    Examples
    --------
    >>> tau = tau_GP(6.94, sim)
    """
    Y = 0.24                                  # helium mass fraction
    cosmo = sim.cosmo
    Omega_b_h2 = cosmo['Omega_b_h2']
    Omega_m_h2 = cosmo['Omega_m_h2']
 
    tau = (
        4e5
        * ((1.0 + z_s) / 7.0) ** (3.0 / 2.0)
        * (Omega_b_h2 / 0.0225)
        * (Omega_m_h2 / 0.132) ** (-0.5)
        * ((1.0 - Y) / 0.76)
    )
    return tau
