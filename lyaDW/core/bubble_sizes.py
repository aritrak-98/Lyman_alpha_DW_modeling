"""
bubble_sizes.py - computes the bubble size around a galaxy using a mean-free-path approach

For each galaxy, it shoots a ray in a random direction through the HII fraction
field and measures the distance until the field drops below a threshold.
This is the mean-free-path (MFP) approach to estimating ionized bubble sizes.
 
The computation is fully JAX-accelerated and vmapped over all galaxies.

"""

import numpy as np
import h5py
import os
import sys
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
import warnings


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
 
# Supported boxsize unit conversions -> Mpc
# Each entry is a callable that takes (value, h) and returns value in Mpc
_UNIT_CONVERSIONS = {
    'ckpc/h': lambda x, h: x / (h * 1000.0),
    'kpc/h':  lambda x, h: x / (h * 1000.0),
    'kpc':    lambda x, h: x / 1000.0,
    'Mpc/h':  lambda x, h: x / h,
    'Mpc':    lambda x, h: x,
}


#------------------------------------------------------------------
# Convert any known boxsize units to Mpc
#------------------------------------------------------------------

def convert_to_Mpc(value, boxsize_units, h):
    """
    Convert distance value to Mpc given its units.

    Parameters
    ----------
    value :  float or np.ndarray
        Distance value(s) to convert.
    boxsize_units : str
        Units of the input value. One of: 'ckpc/h', 'kpc/h', 'kpc', 'Mpc/h', 'Mpc'. 
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    float or np.ndarray
        Distance in Mpc.
    """
    if boxsize_units not in _UNIT_CONVERSIONS:
        raise ValueError(
            f"Unknown boxsize units '{boxsize_units}'."
            f"Supported: {list(_UNIT_CONVERSIONS.keys())}"
        )

    return _UNIT_CONVERSIONS[boxsize_units](value, h)


#------------------------------------------------------------------
# Internal JAX functions
#------------------------------------------------------------------

def _build_interpolator(HII_cube, BoxSize):
    """
    Build a linear interpolator over the HII fraction cube.
 
    Parameters
    ----------
    HII_cube : jnp.ndarray, shape (N, N, N)
        HII fraction field.
    BoxSize : float
        Spatial extent of the box (in any consistent units).
 
    Returns
    -------
    RegularGridInterpolator
        JAX interpolator with linear interpolation and zero fill outside bounds.
    """

    N = HII_cube.shape[0]
    grid_pts = jnp.linspace(0.0, BoxSize, N, endpoint=False)
    return RegularGridInterpolator(
        (grid_pts, grid_pts, grid_pts),
        HII_cube, 
        method='linear', 
        bounds_error=False, 
        fill_value=0.0
    )



def _compute_mfp_batched(galaxy_pos, HII_interp, xHII_threshold, step_size, BoxSize, seed, direction):
    """
    Compute MFP bubble sizes for all galaxies via JAX vmap.
 
    Parameters
    ----------
    galaxy_pos : jnp.ndarray, shape (N, 3)
        Galaxy positions in box units.
    HII_interp : RegularGridInterpolator
        Interpolator over the HII fraction cube.
    xHII_threshold : float
        HII fraction threshold to define the ionized bubble boundary.
    step_size : float
        Step size for the random walk (typically BoxSize / NumPixels).
    BoxSize : float
        Spatial extent of the periodic box.
    seed : int
        Random seed for reproducibility.
    direction : ndarray(3, )
        User defined unit vector direction for all galaxies.
        If None, then random directions for all galaxies are generated using
        the seed value. Default: None.
 
    Returns
    -------
    jnp.ndarray, shape (N,)
        MFP bubble sizes in the same units as BoxSize.
    """

    def _compute_mfp_single(pos, key):
        """
        Compute the MFP bubble size for a single galaxy.
    
        Shoots a ray from 'pos' in a random direction, with each iteration being traversed by 'step_size',
        until encountering a neutral region set by 'xHII_threshold'. Periodicity of the box is also taken 
        into account.
    
        Parameters
        ----------
        pos :  jnp.ndarray, shape (3,)
            Galaxy position in box units
        key :  jax.random.PRNGKey
            JAX random key for direction sampling.
    
        Returns
        -------
        mfp_dist : float
            MFP bubble size in the same units as BoxSize.
        """
        if direction is None:
            # Sample a random isotropic direction
            # Split key into two subkeys: one for phi, one for cos_theta
            key_phi, key_costheta = jax.random.split(key)
        
            phi = jax.random.uniform(key_phi, (), minval=0, maxval=2*jnp.pi)
            cos_theta = jax.random.uniform(key_costheta, (), minval=-1, maxval=1)
            sin_theta = jnp.sqrt(1 - cos_theta**2)
            cos_phi = jnp.cos(phi)
            sin_phi = jnp.sin(phi)
        
            direction_vec = jnp.array([
                sin_theta*cos_phi, 
                sin_theta*sin_phi, 
                cos_theta
            ])
        else:
            direction_vec = direction/jnp.linalg.norm(direction)
    
        def cond_fun(state):
            current_pos, dist = state
            wrapped_pos = current_pos % BoxSize
            x_HII_val = HII_interp(wrapped_pos[None, :])[0]
            #return x_HII_val > xHII_threshold
            return (x_HII_val > xHII_threshold) & (dist < 5*BoxSize)
        
        def body_fun(state):
            current_pos, dist = state
            new_dist = dist + step_size
            new_pos = current_pos + direction_vec * step_size         
            return new_pos, new_dist
    
        init_state = (pos, 0.0)    # initial postion and MFP distance for the galaxy
        _, mfp_dist = jax.lax.while_loop(cond_fun, body_fun, init_state)
    
        return mfp_dist

    keys = jax.random.split(jax.random.PRNGKey(seed), galaxy_pos.shape[0])
    batched_mfp = jax.vmap(_compute_mfp_single, in_axes=(0, 0))
    
    return batched_mfp(galaxy_pos, keys)
    

#------------------------------------------------------------------------------------
# The function that the user needs to call
#------------------------------------------------------------------------------------

def compute_bubble_sizes(galaxy_pos, HII_cube, BoxSize, sim, z, 
                         xHII_threshold=0.5, seed=1216, 
                         direction=None,
                         boxsize_units=None, save=False, savepath='/.',
                         filename=None):

    """
    Compute ionized bubble sizes around galaxies using the mean-free-path method.

    For each galaxy, we shoot a ray in a random isotropic direction through the HII fraction field.
    The bubble size is the distance from the center of the galaxy until the HII fraction drops below 'xHII_threshold'.

    Parameters
    ----------
    galaxy_pos : array-like, shape (N, 3)
        Galaxy positions in the same units as BoxSize.
    HII_cube : array-like, shape (P, P, P)
        3D HII fraction field. Must be a cube (all dimensions equal).
        NumPixels (number of pixels) is inferred as P = HII_cube.shape[0].
    BoxSize : float
        Spatial extent of the periodic box. Units must match galaxy_pos.
    sim : lyaDW.Simulation
        Simulation context. Used for redshift resolution, cosmological
        parameters (needed if boxsize_units is set), and file naming.
    z : float
        Redshift of the snapshot. Resolved to closest available redshift
        in sim.redshifts.
    xHII_threshold : float, optional
        HII fraction threshold that defines the ionized bubble boundary.
        Default: 0.5.
    seed : int, optional
        JAX random seed for reproducible direction sampling. Default: 1216 (must be this number right?).
    direction : ndarray(3, )
        User defined unit vector direction for all galaxies.
        If None, then random directions for all galaxies are generated using
        the seed value. Default: None.
    boxsize_units : str or None, optional
        Units of BoxSize (and therefore of the output R_ion (bubble size)). If provided,
        R_ion is converted to Mpc. If None, R_ion is returned in the same
        units as BoxSize. Supported: 'ckpc/h', 'kpc/h', 'kpc', 'Mpc/h', 'Mpc'.
        Default: None.
    save : bool, optional
        If True, save R_ion to an HDF5 file. Default: False.
    savepath :  str, optional
        Directory to save the output file. Used only if save=True.
        Default: './'.
    filename :  str, optional
        Name of the output file. Default: None

    Returns
    -------
    R_ion : np.ndarray, shape (N,)
        Bubble sizes around each galaxy. Units mathch BoxSize if boxsize_units is None, otherwise in Mpc.

    Notes
    -----
    - The step size for the random walk is step_size = BoxSize /  NumPixels.
    - The box is treated as periodic.
    - Galaxies starting outside an ionized region (HII fraction < threshold
      at their position) will have R_ion = 0.


    Examples
    --------
    >>> R_ion = lyaDW.core.bubble_sizes.compute_bubble_sizes(
    ...     galaxy_pos, HII_cube, BoxSize=646.92,
    ...     sim=sim, z=6.94,
    ...     xHII_threshold=0.5,
    ...     boxsize_units='cMpc/h',
    ...     save=True, savepath='./bubble_size_outputs/'
    ... )
    """
    # ----- Resolve redshift ----- #
    z_resolved = sim.resolve_redshift(z)


    # ----- Validate inputs ----- #
    HII_cube = jnp.array(HII_cube, dtype=jnp.float64)
    galaxy_pos = jnp.array(galaxy_pos, dtype=jnp.float64)

    if HII_cube.ndim != 3:
        raise ValueError(
            f"HII_cube must be a 3D array, got shape {HII_cube.shape}."
        )

    if HII_cube.shape[0] != HII_cube.shape[1] or HII_cube.shape[1] != HII_cube.shape[2]:
        raise ValueError(
            f"HII_cube must be cubic (N x N x N), got shape {HII_cube.shape}."
        )

    if galaxy_pos.ndim != 2 or galaxy_pos.shape[1] != 3:
        raise ValueError(
            f"galaxy_pos must have shape (N, 3), got {galaxy_pos.shape}."
        )

    # ---- Build interpolator and compute step size ---- #
    NumPixels = HII_cube.shape[0]
    step_size = BoxSize / NumPixels

    HII_interp = _build_interpolator(HII_cube, BoxSize)


    # ---- Compute MFP bubble sizes ---- #
    R_ion = _compute_mfp_batched(galaxy_pos, HII_interp, xHII_threshold, step_size, BoxSize, seed, direction)   # JAX array
    R_ion = np.asarray(R_ion, dtype=np.float64)  # Numpy array in the final version


    # ---- Unit conversion (optional) ---- #
    if boxsize_units is not None:
        R_ion = convert_to_Mpc(R_ion, boxsize_units, sim.h).astype(np.float64)
        output_units = 'Mpc'
        print('R_ion is in ', output_units)
    else:
        output_units = 'boxsize_units'
        print('R_ion is in ', output_units)


    # ---- Save (optional) ---- #
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_name = f"{sim.simname}" if sim.simname else ""
        if filename is None:
            filename = f"bubble_sizes_{sim_name}_z{z_resolved:.2f}.h5"
        filepath = os.path.join(savepath, filename)

        with h5py.File(filepath, 'w') as f:
            ds = f.create_dataset('R_ion', data=R_ion)
            ds.attrs['units'] = output_units
            ds.attrs['z'] = z_resolved
            ds.attrs['xHII_threshold'] = xHII_threshold
            ds.attrs['boxsize_units'] = str(boxsize_units)
            ds.attrs['seed'] = seed
            if sim.simname:
                ds.attrs['simulation'] = sim.simname

        print(f"Saved bubble sizes to: {filepath}")

    return R_ion
    