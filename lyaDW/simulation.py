"""
simulation.py - provides the information about the simulation

Information includes:-
    1. The available redshifts (or snapshots) for a given simulation
    2. Cosmological parameters
    3. HII fraction cubes and derived mean neutral fraction (one per redshift)
"""

import warnings
import numpy as np

class Simulation:
    """
    Parameters
    ----------
    redshifts : array-like
        The redshifts available in this simulation (e.g. snapshot redshifts).
    h : float
        Dimensionless Hubble parameter (H0 / 100 km/s/Mpc).
    Omega_m : float
        Matter density parameter.
    Omega_l : float
        Dark energy (cosmological constant) density parameter.
    Omega_b : float
        Baryon density parameter.
    simname : str, optional
        A human-readable name for the simulation (e.g. 'Thesan', 'LIMFAST').
        Used in warnings and optionally in output filenames.
    snap_halo_arr : array-like, optional
        Snapshot indices corresponding to halo catalogs, one per redshift.
        Only needed for simulations with Thesan-style snapshot numbering.
    snap_cartesian_arr : array-like, optional
        Snapshot indices corresponding to cartesian (grid) outputs, one per redshift.
        Only needed for simulations with Thesan-style snapshot numbering.



    Examples
    --------
    >>> import numpy as np
    >>> import lyaDW
    >>> sim = lyaDW.Simulation(
    ...     redshifts=np.array([6.10, 6.94, 7.33]),
    ...     h=0.6774, Omega_m=0.309, Omega_l=0.691, Omega_b=0.0486,
    ...     simname='Thesan'
    ... )
    >>> sim.set_HII_cube(z=6.94, HII_cube=my_cube)
    >>> x_HI = sim.get_mean_xHI(6.94)
    """

    def __init__(self, redshifts, h, Omega_m, Omega_l, Omega_b, simname=None, snap_halo_arr=None, snap_cartesian_arr=None):

        self.redshifts = np.asarray(redshifts, dtype=float)
        self.h = float(h)
        self.Omega_m = float(Omega_m)
        self.Omega_l = float(Omega_l)
        self.Omega_b = float(Omega_b)
        self.simname = simname
        self.snap_halo_arr = (np.asarray(snap_halo_arr) if snap_halo_arr is not None else None)
        self.snap_cartesian_arr = (np.asarray(snap_cartesian_arr) if snap_cartesian_arr is not None else None)


        # Validate that snapshot arrays match the redshift arrays
        if snap_halo_arr is not None:
            if len(snap_halo_arr) != len(redshifts):
                raise ValueError(f"snap_halo_arr length ({len(snap_halo_arr)}) must match redshifts length ({len(redshifts)}).")

        
        if snap_cartesian_arr is not None:
            if len(snap_cartesian_arr) != len(redshifts):
                raise ValueError(f"snap_halo_arr length ({len(snap_cartesian_arr)}) must match redshifts length ({len(redshifts)}).")

        self.HII_cubes = {}        # 3D numpy array
        self.mean_xHI = {}         # float


    # ------------------------------------------------------------------
    # Redshift resolution
    # ------------------------------------------------------------------
    
    def resolve_redshift(self, z, strict=False):
        """
        Return the closest available redshift in the simulation grid.
 
        Parameters
        ----------
        z : float
            Requested redshift.
        strict : bool, optional
            If True, raise a ValueError when z is not exactly in the grid.
            Default is False (emit a warning instead).
 
        Returns
        -------
        float
            The closest redshift in self.redshifts.
 
        Raises
        ------
        ValueError
            If strict=True and z is not exactly in the redshift grid.
        """
        idx = np.argmin(np.abs(self.redshifts - z))
        z_closest = self.redshifts[idx]
        delta_z = abs(z_closest - z)

        if delta_z > 1e-3:      # not an exact match
            sim_name = f"({self.simname})" if self.simname else ""
            msg = (
                f"Requested z={z} not in redshift grid{sim_name}. "
                f"Using closest available: z={z_closest:.4f} (delta_z={delta_z:.4f}). "
                f"Available redshifts: {self.redshifts}"
            )
            if strict:
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning, stacklevel=2)

        return float(z_closest)



    # ------------------------------------------------------------------
    # HII cube and mean x_HI
    # ------------------------------------------------------------------
 
    def set_HII_cube(self, z, HII_cube):
        """
        Register a 3D HII fraction cube at a given redshift.
 
        The volume-averaged mean neutral fraction is computed automatically
        as mean_xHI = 1 - mean(HII_cube) and cached internally.
 
        Parameters
        ----------
        z : float
            Redshift of the cube. Will be snapped to the closest available
            redshift in the grid (with a warning if not exact).
        HII_cube : array-like
            3D array of HII fraction values. Shape should be (N, N, N).
 
        Notes
        -----
        This is the primary way to provide neutral fraction information.
        If you only have the mean value (not the full cube), use
        set_mean_xHI() instead.
        """
        z_resolved = self.resolve_redshift(z)
        cube = np.asarray(HII_cube, dtype=np.float64)
 
        if cube.ndim != 3:
            raise ValueError(
                f"HII_cube must be a 3D array, got shape {cube.shape}."
            )
 
        self.HII_cubes[z_resolved] = cube
        self.mean_xHI[z_resolved] = float(1.0 - np.mean(cube))
        
 
    def set_mean_xHI(self, z, mean_xHI):
        """
        Manually set the volume-averaged mean neutral fraction at a redshift.
 
        Use this only if you do not have the full 3D HII fraction cube
        available. If you have the cube, use set_HII_cube() instead —
        it will compute the mean automatically.
 
        Parameters
        ----------
        z : float
            Redshift. Will be snapped to the closest available redshift
            in the grid (with a warning if not exact).
        mean_xHI : float
            Volume-averaged mean neutral hydrogen fraction of the IGM.
            This is a single scalar, NOT the spatially resolved HII cube.
            Must be in [0, 1].
        """
        if not (0.0 <= mean_xHI <= 1.0):
            raise ValueError(
                f"mean_xHI must be between 0 and 1, got {mean_xHI}."
            )
 
        z_resolved = self.resolve_redshift(z)
        self.mean_xHI[z_resolved] = float(mean_xHI)
        

    def get_HII_cube(self, z):
        """
        Return the 3D HII fraction cube at redshift z, if available.
 
        Parameters
        ----------
        z : float
            Redshift. Will be snapped to the closest available redshift.
 
        Returns
        -------
        np.ndarray
            3D HII fraction array of shape (N, N, N).
 
        Raises
        ------
        KeyError
            If no HII cube has been set for this redshift. Note that
            set_mean_xHI() does not store a cube — only set_HII_cube() does.
        """
        z_resolved = self.resolve_redshift(z)
 
        if z_resolved not in self.HII_cubes:
            raise KeyError(
                f"No HII cube available for z={z_resolved}. "
                f"Call set_HII_cube(z, cube) first."
            )
 
        return self.HII_cubes[z_resolved]
        
 
    def get_mean_xHI(self, z):
        """
        Return the volume-averaged mean neutral fraction at redshift z.
 
        Parameters
        ----------
        z : float
            Redshift. Will be snapped to the closest available redshift.
 
        Returns
        -------
        float
            Volume-averaged mean neutral hydrogen fraction of the IGM.
 
        Raises
        ------
        KeyError
            If no HII cube or mean_xHI has been set for this redshift.
        """
        z_resolved = self.resolve_redshift(z)
 
        if z_resolved not in self.mean_xHI:
            raise KeyError(
                f"No mean_xHI available for z={z_resolved}. "
                f"Call set_HII_cube(z, cube) or set_mean_xHI(z, value) first."
            )
 
        return self.mean_xHI[z_resolved]
 
 
    # ------------------------------------------------------------------
    # Snapshot index helpers
    # ------------------------------------------------------------------
 
    def get_snaps(self, z):
        """
        Return the halo and cartesian snapshot indices for a given redshift.
 
        Parameters
        ----------
        z : float
            Redshift. Will be snapped to the closest available redshift.
 
        Returns
        -------
        snap_halo : int or None
            Halo catalog snapshot index, or None if not provided.
        snap_cartesian : int or None
            Cartesian output snapshot index, or None if not provided.
        """
        z_resolved = self.resolve_redshift(z)
        idx = np.argmin(np.abs(self.redshifts - z_resolved))
 
        snap_halo = (int(self.snap_halo_arr[idx]) 
                     if self.snap_halo_arr is not None else None)
        snap_cartesian = (int(self.snap_cartesian_arr[idx]) 
                          if self.snap_cartesian_arr is not None else None)

        #print('(Snap halo, Snap cartesian)')
        return snap_halo, snap_cartesian
 
    # ------------------------------------------------------------------
    # Cosmological parameters
    # ------------------------------------------------------------------
 
    @property
    def cosmo(self):
        """
        Return a dict of cosmological parameters, including derived quantities.
 
        Returns
        -------
        dict with keys: h, Omega_m, Omega_l, Omega_b, Omega_m_h2, Omega_b_h2, z
        """
        return {
            'h':          self.h,
            'Omega_m':    self.Omega_m,
            'Omega_l':    self.Omega_l,
            'Omega_b':    self.Omega_b,
            'Omega_m_h2': self.Omega_m * self.h**2,
            'Omega_b_h2': self.Omega_b * self.h**2,
        }
 
    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
 
    def __repr__(self):
        name = f"'{self.simname}'" if self.simname else "unlabeled"
        return (
            f"Simulation({name}, "
            f"z=[{self.redshifts.min():.2f}..{self.redshifts.max():.2f}], "
            f"h={self.h}, Omega_m={self.Omega_m})"
        )
 
        
        