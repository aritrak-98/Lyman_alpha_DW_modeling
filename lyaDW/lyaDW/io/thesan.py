"""
io/thesan.py — Convenience reader functions for the Thesan simulation.

These functions read Thesan simulation outputs and return plain numpy/jax
arrays ready to be passed into the lyaDW core pipeline. They do not perform
any physics — they are purely I/O helpers.

All functions require a Simulation instance for redshift resolution and
snapshot index lookup (via sim.get_snaps()).

Functions
---------
load_HII_cube   : Read the 3D HII fraction field from cartesian outputs
load_galaxies   : Read galaxy positions, halo masses, and stellar masses

"""

import numpy as np
import h5py

from lyaDW.utils.galaxy import compute_sigma_v
import lyaDW


# ---------------------------------------------------------------------------
# HII fraction cube
# ---------------------------------------------------------------------------

def load_HII_cube(thesan_dir, sim, z):
    """
    Read the 3D HII fraction field from Thesan cartesian outputs.

    Also registers the cube with the simulation context via sim.set_HII_cube(),
    so that sim.get_mean_xHI(z) becomes available automatically.

    Parameters
    ----------
    thesan_dir : str
        Path to the Thesan simulation directory.
    sim : lyaDW.Simulation
        Simulation context. Must have snap_cartesian_arr set.
    z : float
        Redshift. Resolved to the closest available redshift in sim.

    Returns
    -------
    HII_cube : np.ndarray, shape (N, N, N)
        3D HII fraction field.
    BoxSize : float
        Spatial extent of the box in ckpc/h.

    Raises
    ------
    ValueError
        If sim.snap_cartesian_arr is not set.

    Examples
    --------
    >>> HII_cube, BoxSize = lyaDW.io.thesan.load_HII_cube(thesan_dir, sim, z=6.94)
    """
    if sim.snap_cartesian_arr is None:
        raise ValueError(
            "sim.snap_cartesian_arr is not set. "
            "Provide snap_cartesian_arr when creating the Simulation object."
        )

    z_resolved = sim.resolve_redshift(z)
    _, snap_cartesian = sim.get_snaps(z_resolved)

    # Read header
    header_file = (
        f'{thesan_dir}/cartesian512_{snap_cartesian:03d}/'
        f'cartesian512_{snap_cartesian:03d}.0.hdf5'
    )
    with h5py.File(header_file, 'r') as f:
        g = f['Header']
        NumFiles = g.attrs['NumFiles']
        NumPixels = g.attrs['NumPixels']
        BoxSize = g.attrs['BoxSize']

    # Read HII fraction across all subfiles
    HII_fraction = np.zeros(NumPixels**3, dtype=np.float64)
    offset = 0

    for sub in range(NumFiles):
        filename = (
            f'{thesan_dir}/cartesian512_{snap_cartesian:03d}/'
            f'cartesian512_{snap_cartesian:03d}.{sub}.hdf5'
        )
        with h5py.File(filename, 'r') as f:
            n_buf = len(f['HII_Fraction'])
            HII_fraction[offset:offset + n_buf] = f['HII_Fraction'][:]
            offset += n_buf

    HII_cube = HII_fraction.reshape(NumPixels, NumPixels, NumPixels)

    # Register with sim so mean_xHI is available downstream
    sim.set_HII_cube(z_resolved, HII_cube)

    return HII_cube, float(BoxSize)


# ---------------------------------------------------------------------------
# Galaxy / subhalo properties
# ---------------------------------------------------------------------------

def load_galaxies(thesan_dir, sim, z, M_h_min=0.0, M_h_max=np.inf):
    """
    Read subhalo positions, halo masses, stellar masses, 
    and star formation rates from Thesan group catalogs.

    Only subhalos with non-zero stellar mass and halo mass within
    [M_h_min, M_h_max] are returned.

    Parameters
    ----------
    thesan_dir : str
        Path to the Thesan simulation directory.
    sim : lyaDW.Simulation
        Simulation context. Must have snap_halo_arr set.
    z : float
        Redshift. Resolved to the closest available redshift in sim.
    M_h_min : float, optional
        Minimum halo mass in M_Sun. Default: 0.
    M_h_max : float, optional
        Maximum halo mass in M_Sun. Default: inf.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Subhalo positions in ckpc/h.
    halo_masses : np.ndarray, shape (N,)
        Subhalo halo masses in M_Sun.
    stellar_masses : np.ndarray, shape (N,)
        Subhalo stellar masses in M_Sun.
    sfr: np.ndarray, shape (N,)
        Star-formation rate in M_Sun/yr.

    Raises
    ------
    ValueError
        If sim.snap_halo_arr is not set.

    Examples
    --------
    >>> pos, M_halo, M_star = lyaDW.io.thesan.load_galaxies(
    ...     thesan_dir, sim, z=6.94, M_h_min=1e8
    ... )
    """
    if sim.snap_halo_arr is None:
        raise ValueError(
            "sim.snap_halo_arr is not set. "
            "Provide snap_halo_arr when creating the Simulation object."
        )

    z_resolved = sim.resolve_redshift(z)
    snap_halo, _ = sim.get_snaps(z_resolved)

    # Read header for h and NumFiles
    header_file = (
        f'{thesan_dir}/groups_{snap_halo:03d}/'
        f'fof_subhalo_tab_{snap_halo:03d}.0.hdf5'
    )
    with h5py.File(header_file, 'r') as f:
        g = f['Header']
        NumFiles = g.attrs['NumFiles']
        h = g.attrs['HubbleParam']

    # Read across all subfiles
    pos_list = []
    mass_list = []
    mass_type_list = []
    sfr_list = []

    for sub in range(NumFiles):
        filename = (
            f'{thesan_dir}/groups_{snap_halo:03d}/'
            f'fof_subhalo_tab_{snap_halo:03d}.{sub}.hdf5'
        )
        with h5py.File(filename, 'r') as f:
            sh = f['Subhalo']
            pos_list.append(sh['SubhaloPos'][:])
            mass_list.append(sh['SubhaloMass'][:])
            mass_type_list.append(sh['SubhaloMassType'][:])
            sfr_list.append(sh['SubhaloSFR'][:])

    SFR = np.concatenate(sfr_list)                                    # M_sun/yr
    positions = np.concatenate(pos_list, axis=0)                      # ckpc/h
    halo_masses = np.concatenate(mass_list) * 1e10 / h                # M_Sun
    mass_type = np.concatenate(mass_type_list, axis=0) * 1e10 / h
    stellar_masses = mass_type[:, 4]                                  # M_Sun

    # Filter: non-zero stellar mass and within halo mass range
    mask = (
        (stellar_masses > 0.0) &
        (halo_masses >= M_h_min) &
        (halo_masses < M_h_max)
    )

    return np.float64(positions[mask]), np.float64(halo_masses[mask]), np.float64(stellar_masses[mask]), np.float64(SFR[mask])
