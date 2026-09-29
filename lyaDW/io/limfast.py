"""
!!! IN PROGRESS !!! USE WITH CAUTION !!!

io/limfast.py — Convenience reader functions for the LIMFAST simulation.

These functions read LIMFAST simulation outputs and return plain numpy
arrays ready to be passed into the lyaDW core pipeline. They do not
perform any physics — they are purely I/O helpers.

LIMFAST uses binary cube files for the HII fraction field and text files
for halo catalogs (derived from MiniUchuu).

Functions
---------
load_HII_cube       : Read the 3D HII fraction field from binary xHI boxes
load_halos          : Read halo positions and masses from MiniUchuu halo lists
load_halos_limfast  : Read LIMFAST-specific halo list (includes SFR column). 
                      Similar to the MiniUchuu halo lists, but the positions are stored
                      as fractions of the boxsize.

"""

import numpy as np
import glob
import os
import sys

from lyaDW.utils.galaxy import compute_sigma_v


# ---------------------------------------------------------------------------
# Binary cube readers (from Jason (Guochao) Sun / LIMFAST)
# ---------------------------------------------------------------------------

def _load_binary_data(filepath, dtype=np.float32):
    """
    Read a binary data file written in little-endian float32 format.

    Parameters
    ----------
    filepath : str
        Path to the binary file.
    dtype : np.dtype, optional
        Data type. Default: np.float32.

    Returns
    -------
    np.ndarray
        Flat array of data values.
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    _data = np.frombuffer(data, dtype=np.float32)
    if sys.byteorder == 'big':
        _data = _data.byteswap()
    return _data


def _parse_dim_from_filename(filename):
    """
    Parse the cube dimension N from a LIMFAST filename.

    Parameters
    ----------
    filename : str
        Base filename (not full path).

    Returns
    -------
    int
        Cube dimension N (number of cells per side).
    """
    base = os.path.basename(filename)
    parts = base.split('_')

    if base.endswith('lighttravel'):
        dim = int(parts[-3])
    else:
        dim = int(parts[-2])

    return dim


def _read_cube(filepath):
    """
    Read a LIMFAST binary cube file and reshape to 3D.

    Parameters
    ----------
    filepath : str
        Path to the binary file.

    Returns
    -------
    np.ndarray, shape (N, N, N)
        3D data cube, Fortran-ordered.
    """
    data = _load_binary_data(filepath)
    dim  = _parse_dim_from_filename(filepath)
    data = data.reshape((dim, dim, dim))              # step 1: C-order first
    data = data.reshape((dim, dim, dim), order='F')   # step 2: Fortran-order
    return data


# ---------------------------------------------------------------------------
# HII fraction cube
# ---------------------------------------------------------------------------

def load_HII_cube(sim, z, filepath=None):
    """
    Read the 3D HII fraction field from LIMFAST binary xHI box files.

    Reads the xHI (neutral fraction) cube and converts to xHII = 1 - xHI.
    Also registers the cube with the simulation context via sim.set_HII_cube(),
    so that sim.get_mean_xHI(z) becomes available automatically.

    Parameters
    ----------
    sim : lyaDW.Simulation
        Simulation context. Used for redshift resolution.
    z : float
        Redshift. Resolved to the closest available redshift in sim.
    filepath : str, optional
        Path to the directory containing LIMFAST box files.
        Files are expected to match the pattern: xH_z0{z:05.2f}_*
        If None, raises a ValueError.

    Returns
    -------
    HII_cube : np.ndarray, shape (N, N, N)
        3D HII fraction field.
    BoxSize : float
        Spatial extent of the box in Mpc. Parsed from the filename.

    Raises
    ------
    ValueError
        If filepath is None or no matching file is found.

    Examples
    --------
    >>> HII_cube, BoxSize = lyaDW.io.limfast.load_HII_cube(
    ...     sim, z=7.03, filepath='/path/to/LIMFAST/Boxes/'
    ... )
    """
    if filepath is None:
        raise ValueError(
            "filepath must be provided. "
            "Pass the path to the directory containing LIMFAST box files."
        )

    z_resolved = sim.resolve_redshift(z)

    # Find xHI file
    pattern = os.path.join(filepath, f'xH_z0{z_resolved:05.2f}_*')
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No LIMFAST xHI file found matching pattern: {pattern}"
        )
    if len(matches) > 1:
        import warnings
        warnings.warn(
            f"Multiple files match pattern {pattern}. Using: {matches[0]}",
            UserWarning, stacklevel=2
        )

    xHI_cube = _read_cube(matches[0])
    HII_cube = (1.0 - xHI_cube).astype(np.float64)

    # Parse BoxSize from filename: ..._NNN_LLL_... where LLL is box size in Mpc
    base  = os.path.basename(matches[0])
    parts = base.split('_')
    try:
        BoxSize = float(parts[-1][:3])
    except (ValueError, IndexError):
        import warnings
        warnings.warn(
            f"Could not parse BoxSize from filename {base}. Setting BoxSize=None.",
            UserWarning, stacklevel=2
        )
        BoxSize = None

    # Register with sim
    sim.set_HII_cube(z_resolved, HII_cube)

    return HII_cube, BoxSize


def load_delta_cube(sim, z, filepath=None):
    """
    Read the 3D density field from LIMFAST binary delta_x box files.    

    Parameters
    ----------
    sim : lyaDW.Simulation
        Simulation context. Used for redshift resolution.
    z : float
        Redshift. Resolved to the closest available redshift in sim.
    filepath : str, optional
        Path to the directory containing LIMFAST box files.
        Files are expected to match the pattern: updated_smoothed_deltax*
        If None, raises a ValueError.

    Returns
    -------
    delta_cube : np.ndarray, shape (N, N, N)
        3D density field.

    Raises
    ------
    ValueError
        If filepath is None or no matching file is found.

    Examples
    --------
    >>> delta_cube, BoxSize = lyaDW.io.limfast.load_delta_cube(
    ...     sim, z=7.03, filepath='/path/to/LIMFAST/Boxes/'
    ... )
    """
    if filepath is None:
        raise ValueError(
            "filepath must be provided. "
            "Pass the path to the directory containing LIMFAST box files."
        )

    z_resolved = sim.resolve_redshift(z)

    # Find density file
    pattern = os.path.join(filepath, f'updated_smoothed_deltax*')
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No LIMFAST xHI file found matching pattern: {pattern}"
        )
    if len(matches) > 1:
        import warnings
        warnings.warn(
            f"Multiple files match pattern {pattern}. Using: {matches[0]}",
            UserWarning, stacklevel=2
        )

    deltax_cube = _read_cube(matches[0])
    deltax_cube = (deltax_cube).astype(np.float64)

    return deltax_cube


def load_delta_T_cube(sim, z, filepath=None):
    """
    Read the 3D differential brightness temperature from LIMFAST.    

    Parameters
    ----------
    sim : lyaDW.Simulation
        Simulation context. Used for redshift resolution.
    z : float
        Redshift. Resolved to the closest available redshift in sim.
    filepath : str, optional
        Path to the directory containing LIMFAST box files.
        Files are expected to match the pattern: delta_T_*
        If None, raises a ValueError.

    Returns
    -------
    delta_T_cube : np.ndarray, shape (N, N, N)
        3D differential brightness temperature field.

    Raises
    ------
    ValueError
        If filepath is None or no matching file is found.

    Examples
    --------
    >>> delta_T_cube, BoxSize = lyaDW.io.limfast.load_delta_T_cube(
    ...     sim, z=7.03, filepath='/path/to/LIMFAST/Boxes/'
    ... )
    """
    if filepath is None:
        raise ValueError(
            "filepath must be provided. "
            "Pass the path to the directory containing LIMFAST box files."
        )

    z_resolved = sim.resolve_redshift(z)

    # Find density file
    pattern = os.path.join(filepath, f'delta_T_*')
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No LIMFAST xHI file found matching pattern: {pattern}"
        )
    if len(matches) > 1:
        import warnings
        warnings.warn(
            f"Multiple files match pattern {pattern}. Using: {matches[0]}",
            UserWarning, stacklevel=2
        )

    delta_T_cube = _read_cube(matches[0])
    delta_T_cube = (delta_T_cube).astype(np.float64)

    return delta_T_cube



def load_sfrd_cube(sim, z, filepath=None):
    """
    Read the 3D star-formation rate density box from LIMFAST.    

    Parameters
    ----------
    sim : lyaDW.Simulation
        Simulation context. Used for redshift resolution.
    z : float
        Redshift. Resolved to the closest available redshift in sim.
    filepath : str, optional
        Path to the directory containing LIMFAST box files.
        Files are expected to match the pattern: SFRD_*
        If None, raises a ValueError.

    Returns
    -------
    sfrd_cube : np.ndarray, shape (N, N, N)
        3D sfrd field.

    Raises
    ------
    ValueError
        If filepath is None or no matching file is found.

    Examples
    --------
    >>> sfrd_cube, BoxSize = lyaDW.io.limfast.load_sfrd_cube(
    ...     sim, z=7.03, filepath='/path/to/LIMFAST/Boxes/'
    ... )
    """
    if filepath is None:
        raise ValueError(
            "filepath must be provided. "
            "Pass the path to the directory containing LIMFAST box files."
        )

    z_resolved = sim.resolve_redshift(z)

    # Find density file
    pattern = os.path.join(filepath, f'SFRD_*')
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No LIMFAST xHI file found matching pattern: {pattern}"
        )
    if len(matches) > 1:
        import warnings
        warnings.warn(
            f"Multiple files match pattern {pattern}. Using: {matches[0]}",
            UserWarning, stacklevel=2
        )

    sfrd_cube = _read_cube(matches[0])
    sfrd_cube = (sfrd_cube).astype(np.float64)

    return sfrd_cube


# ---------------------------------------------------------------------------
# Halo catalogs
# ---------------------------------------------------------------------------

def load_halos(sim, z, filepath=None, M_h_min=0.0, M_h_max=np.inf):
    """
    Read halo positions and masses from MiniUchuu halo lists.

    MiniUchuu halo list format (columns):
        0: halo mass (M_Sun)
        1: X position (Mpc)
        2: Y position (Mpc)
        3: Z position (Mpc)

    Parameters
    ----------
    sim : lyaDW.Simulation
        Simulation context. Used for redshift resolution.
    z : float
        Redshift. Resolved to the closest available redshift in sim.
    filepath : str, optional
        Path to the directory containing MiniUchuu halo list files.
        Files are expected to match: MiniUchuu_halolist_z{zp}_*
        If None, raises a ValueError.
    M_h_min : float, optional
        Minimum halo mass in M_Sun. Default: 0.
    M_h_max : float, optional
        Maximum halo mass in M_Sun. Default: inf.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Halo positions in Mpc (X, Y, Z).
    halo_masses : np.ndarray, shape (N,)
        Halo masses in M_Sun.

    Raises
    ------
    ValueError
        If filepath is None or no matching file is found.

    Examples
    --------
    >>> pos, M_halo = lyaDW.io.limfast.load_galaxies(
    ...     sim, z=7.03, filepath='/path/to/MiniUchuu_halolists/', M_h_min=1e8
    ... )
    """
    if filepath is None:
        raise ValueError(
            "filepath must be provided. "
            "Pass the path to the directory containing MiniUchuu halo list files."
        )

    z_resolved = sim.resolve_redshift(z)
    zp = f"{z_resolved:.2f}".replace('.', 'p')
    pattern = os.path.join(filepath, f'MiniUchuu_halolist_z{zp}_*')
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No MiniUchuu halo list found matching pattern: {pattern}"
        )

    halos = np.loadtxt(matches[0])

    halo_masses = halos[:, 0]
    positions = halos[:, 1:4]

    # Apply mass filter
    mask = (halo_masses >= M_h_min) & (halo_masses < M_h_max)

    return positions[mask], halo_masses[mask]


def load_halos_limfast(sim, z, filepath=None, M_h_min=0.0, M_h_max=np.inf):
    """
    Read LIMFAST-specific halo list (derived from MiniUchuu, includes SFR).

    LIMFAST halo list format (columns):
        0: halo mass (M_Sun)
        1: X position (fractional boxsize units — X Mpc / BoxSize Mpc)
        2: Y position (fractional boxsize units)
        3: Z position (fractional boxsize units)
        4: scaled halo mass
        5: star formation rate (M_Sun/yr)

    Parameters
    ----------
    sim : lyaDW.Simulation
        Simulation context. Used for redshift resolution.
    z : float
        Redshift. Resolved to the closest available redshift in sim.
    filepath : str, optional
        Path to the directory containing LIMFAST halo list files.
        Files are expected to match: updated_halos_z{z:05.2f}_*
        If None, raises a ValueError.
    M_h_min : float, optional
        Minimum halo mass in M_Sun. Default: 0.
    M_h_max : float, optional
        Maximum halo mass in M_Sun. Default: inf.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Halo positions in fractional boxsize units.
        Multiply by BoxSize to get physical positions in Mpc.
    halo_masses : np.ndarray, shape (N,)
        Halo masses in M_Sun.
    SFR : np.ndarray, shape (N,)
        Star formation rates in M_Sun/yr.

    Raises
    ------
    ValueError
        If filepath is None or no matching file is found.

    Notes
    -----
    Positions are in fractional boxsize units (col / BoxSize). Multiply by
    BoxSize before passing to core pipeline functions.

    Examples
    --------
    >>> pos, M_halo, SFR = lyaDW.io.limfast.load_halos_limfast(
    ...     sim, z=7.03, filepath='/path/to/LIMFAST/Halo_lists/', M_h_min=1e8
    ... )
    >>> pos_mpc = pos * BoxSize    # convert to Mpc
    """
    if filepath is None:
        raise ValueError(
            "filepath must be provided. "
            "Pass the path to the directory containing LIMFAST halo list files."
        )

    z_resolved = sim.resolve_redshift(z)
    pattern = os.path.join(filepath, f'updated_halos_z{z_resolved:06.2f}_*')
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No LIMFAST halo list found matching pattern: {pattern}"
        )

    halos = np.loadtxt(matches[0])

    halo_masses = halos[:, 0]
    positions = halos[:, 1:4]
    SFR = halos[:, 5]

    # Apply mass filter
    mask = (halo_masses >= M_h_min) & (halo_masses < M_h_max)

    return positions[mask], halo_masses[mask], SFR[mask]


# ---------------------------------------------------------------------------
# Velocity dispersion
# ---------------------------------------------------------------------------

def load_sigma_v(halo_masses, z, sim):
    """
    Compute velocity dispersions analytically from halo masses.

    Uses the Barkana & Loeb (2001) scaling relation via
    lyaDW.utils.galaxy.compute_sigma_v(). No file reading is involved.

    Parameters
    ----------
    halo_masses : array-like, shape (N,)
        Halo masses in M_Sun, as returned by load_galaxies() or
        load_halos_limfast().
    z : float
        Redshift.
    sim : lyaDW.Simulation
        Simulation context. Uses sim.h for the Hubble parameter.

    Returns
    -------
    sigma_v : np.ndarray, shape (N,)
        Velocity dispersions in km/s.

    References
    ----------
    Barkana & Loeb (2001), Physics Reports, 349, 125.

    Examples
    --------
    >>> sigma_v = lyaDW.io.limfast.load_sigma_v(M_halo, z=7.03, sim=sim)
    """
    return compute_sigma_v(halo_masses, z, sim.h)