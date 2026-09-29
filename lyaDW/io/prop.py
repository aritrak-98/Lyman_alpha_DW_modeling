import numpy as np
import h5py

from lyaDW.utils.galaxy import compute_sigma_v
import lyaDW




# ---------------------------------------------------------------------------
# Load bubble sizes
# ---------------------------------------------------------------------------

def load_bubble_sizes(filepath):
    """
    Read bubble sizes that was computed.

    Parameters
    ----------
    filepath : str
        Path to the bubble size files.

    Returns
    -------
    R_ion : np.ndarray, shape (N,)
        Bubble sizes in Mpc.

    Raises
    ------
    ValueError
        If sim.snap_halo_arr is not set.

    Examples
    --------
    >>> R_ion = thesan.load_bubble_sizes(
    ...   filepath='./output/bubble_sizes_Thesan_z9.01.h5'
    ... )
    """
    
    with h5py.File(filepath, 'r') as f:
        data = f['R_ion']
        R_ion = data[:]
        output_units = data.attrs['units']
        output_z = data.attrs['z']
        output_xHII_threshold = data.attrs['xHII_threshold']
        output_boxsize_units = data.attrs['boxsize_units']
        simname = data.attrs['simulation']

    print(f"  Simulation     = {simname}")
    print(f"  R_ion units    = {output_units}")
    print(f"  z_s            = {output_z}")
    print(f"  xHII_threshold = {output_xHII_threshold}")
    print(f"  Box units      = {output_boxsize_units}")
    
    return R_ion


# ---------------------------------------------------------------------------
# Load IGM begining redshift
# ---------------------------------------------------------------------------

def load_z_beg_uniform(filepath):
    """
    Read redshifts where the neutral regions begin.

    Parameters
    ----------
    filepath : str
        Path to the z_beg files.

    Returns
    -------
    z_beg : np.ndarray, shape (N,)
        z_beg

    Raises
    ------
    ValueError
        If sim.snap_halo_arr is not set.

    Examples
    --------
    >>> z_beg = thesan.load_z_beg_uniform(
    ...   filepath='./output/z_beg_Thesan_z9.01.h5'
    ... )
    """
    
    with h5py.File(filepath, 'r') as f:
        data = f['z_beg']
        z_beg = data[:]
        z_s = data.attrs['z_s']
        z_end = data.attrs['z_end']
        simname = data.attrs['simulation']

    print(f"  Simulation     = {simname}")
    print(f"  z_s            = {z_s}")
    print(f"  z_end          = {z_end}")

    return z_beg



# ---------------------------------------------------------------------------
# Load sightlines data
# ---------------------------------------------------------------------------

def load_sightlines(filepath):
    """
    Load sightline data saved by traverse_sightlines().

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file saved by traverse_sightlines().

    Returns
    -------
    z_arr : np.ndarray, shape (N_gal, max_cells+1)
        Redshift bin edges per sightline. Padded with -1 sentinel.
    x_HI_arr : np.ndarray, shape (N_gal, max_cells)
        Neutral fraction per cell. Padded with -1 sentinel.
    z_beg : np.ndarray, shape (N_gal,)
        Redshift of first neutral cell per galaxy.
    n_cells : np.ndarray, shape (N_gal,), dtype=int
        Number of valid cells per sightline (before padding).

    Notes
    -----
    The returned arrays are identical to what traverse_sightlines() returns,
    so they can be passed directly to compute_tau_dw() without modification.

    Examples
    --------
    >>> z_arr, x_HI_arr, z_beg, n_cells = lyaDW.io.prop.load_sightlines(
    ...     './output/sightlines_Thesan_z9.01.h5'
    ... )
    """

    with h5py.File(filepath, 'r') as f:
        z_arr    = f['z_arr'][:]
        x_HI_arr = f['x_HI_arr'][:]
        z_beg    = f['z_beg'][:]
        n_cells  = f['n_cells'][:]

        z_s   = f.attrs['z_s']
        z_end = f.attrs['z_end']
        simname = f.attrs['simulation']

    print(f"Loaded sightlines from: {filepath}")
    print(f"Simulation  = {simname}")
    print(f"  z_s       = {z_s:.4f}")
    print(f"  z_end     = {z_end:.4f}")
    print(f"  N_gal     = {z_arr.shape[0]}")
    print(f"  max_cells = {x_HI_arr.shape[1]}")

    return z_arr, x_HI_arr, z_beg, n_cells



# ---------------------------------------------------------------------------
# Load optical depth data
# ---------------------------------------------------------------------------

def load_tau_dw_patchy(filepath):

    """
    Load optical depth data in the patchy scenario.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 files.

    Returns
    -------
    tau_dw1 : np.ndarray, shape (N_gal, N_delta)
        Damping wing optical depth. Values are inf for delta < 0. This is for z > z_beg
    tau_dw2 : np.ndarray, shape (N_gal, N_delta)
        Damping wing optical depth. Values are inf for delta < 0. This is for z > z_s
    delta  : np.ndarray, shape (N_delta,)
        The delta grid used.


    Examples
    --------
    >>> tau_dw1, tau_dw2, delta = lyaDW.io.prop.load_tau_dw(
    ...     './output/tau_dw_patchy_Thesan_z9.01.h5'
    ... )
    """

    with h5py.File(filepath, 'r') as f:
        tau_dw1 = f['tau_dw1'][:]
        tau_dw2 = f['tau_dw2'][:]
        delta  = f['delta'][:]

        z_s   = f['tau_dw1'].attrs['z_s']
        z_end = f['tau_dw1'].attrs['z_end']
        simname = f['tau_dw1'].attrs['simulation']

    print(f"Loaded tau_dw from: {filepath}")
    print(f"Simulation  = {simname}")
    print(f"  z_s       = {z_s:.4f}")
    print(f"  z_end     = {z_end:.4f}")
    print(f"  N_gal     = {tau_dw1.shape[0]}")
    
    return tau_dw1, tau_dw2, delta



def load_tau_dw_uniform(filepath):

    """
    Load optical depth data in the uniform scenario.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 files.

    Returns
    -------
    tau_dw : np.ndarray, shape (N_gal, N_delta)
        Damping wing optical depth. Values are inf for delta < 0.
    delta  : np.ndarray, shape (N_delta,)
        The delta grid used.


    Examples
    --------
    >>> tau_dw, delta = lyaDW.io.prop.load_tau_dw(
    ...     './output/tau_dw_uniform_Thesan_z9.01.h5'
    ... )
    """

    with h5py.File(filepath, 'r') as f:
        tau_dw = f['tau_dw'][:]
        delta  = f['delta'][:]

        z_s   = f['tau_dw'].attrs['z_s']
        z_end = f['tau_dw'].attrs['z_end']
        simname = f['tau_dw'].attrs['simulation']

    print(f"Loaded tau_dw from: {filepath}")
    print(f"Simulation  = {simname}")
    print(f"  z_s       = {z_s:.4f}")
    print(f"  z_end     = {z_end:.4f}")
    print(f"  N_gal     = {tau_dw.shape[0]}")
    
    return tau_dw, delta


# ---------------------------------------------------------------------------
# Load transmission data
# ---------------------------------------------------------------------------

def load_transmission_alpha(filepath, neyer=False):

    """
    Load transmission data.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 files.
    neyer : boolean
        If True, loads the transmission data for the double peaked profile in Neyer+25

    Returns
    -------
    T_alpha : np.ndarray, shape (N,)
        Transmission coefficient for each galaxy, in [0, 1].

    Examples
    --------
    >>> T_alpha = lyaDW.io.prop.load_transmission_data(
    ...     './output/T_alpha_Thesan_z9.01_fvelout0.0.h5'
    ... )
    """
    if(neyer == False):
        with h5py.File(filepath, 'r') as f:
            data    = f['T_alpha']
            T_alpha = data[:]
    
            z_s   = data.attrs['z_s']
            f_vel_out = data.attrs['f_vel_out']
            f_esc = data.attrs['f_esc']
            f_alpha = data.attrs['f_alpha']
            simname = data.attrs['simulation']
    
        print(f"Loaded T_alpha from: {filepath}")
        print(f"Simulation      = {simname}")
        print(f"  z_s           = {z_s:.4f}")
        print(f"  f_vel_out     = {f_vel_out:.2f}")
        print(f"  N_gal         = {T_alpha.shape[0]}")
        print(f"  f_esc         = {f_esc:.4f}")
        print(f"  f_alpha       = {f_alpha:.4f}")
    
        return T_alpha

    else:
        with h5py.File(filepath, 'r') as f:
            data    = f['T_alpha']
            T_alpha = data[:]

            z_s   = data.attrs['z_s']
            simname = data.attrs['simulation']

        print(f"Loaded T_alpha from: {filepath}")
        print(f"Simulation      = {simname}")
        print(f"  z_s           = {z_s:.4f}")
        print(f"  N_gal         = {T_alpha.shape[0]}")

        return T_alpha


# ---------------------------------------------------------------------------
# Load luminosity data
# ---------------------------------------------------------------------------


def load_luminosities(filepath):

    """
    Load luminosity data.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 files.

    Returns
    -------
    L_alpha_int : np.ndarray, shape (N_gal,)
        intrinsic Lya luminosity (erg/s)
    L_alpha_trans :  np.ndarray, shape (N_gal,)
        transmitted Lya luminosity (erg/s)
    L_UV_nu : np.ndarray, shape (N_gal,)
        UV luminosity density (erg/s/Hz)
    REW : np.ndarray, shape (N_gal,)
        rest-frame EW in Angstroms

    Examples
    --------
    >>> lum = lyaDW.io.prop.load_luminosities(
    ...     './output/lum_rew_Thesan_z9.01_fvelout0.0_b1.7.h5'
    ... )
    """
    
    with h5py.File(filepath, 'r') as f:
        lum = f['luminosities']

        L_alpha_int   = lum['L_alpha_int'][:]
        L_alpha_trans = lum['L_alpha_trans'][:]
        L_UV_nu       = lum['L_UV_nu'][:]

        REW_data = f['REW']
        REW      = REW_data[:]
        
        units     = REW_data.attrs['units']
        b         = REW_data.attrs['b']
        f_vel_out = REW_data.attrs['f_vel_out']
        z_s       = REW_data.attrs['z_s']
        simname   = REW_data.attrs['simulation']

    print(f"Loaded results from: {filepath}")
    print(f"Simulation      = {simname}")
    print(f"  z_s           = {z_s:.4f}")
    print(f"  f_vel_out     = {f_vel_out:.2f}")
    print(f"  N_gal         = {L_alpha_int.shape[0]}")

    return L_alpha_int, L_alpha_trans, L_UV_nu, REW