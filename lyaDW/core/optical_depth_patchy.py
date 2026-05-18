"""
optical_depth_patchy.py — Lyman-alpha damping wing optical depth using patchy reionization along the line of sight.

An alternative to optical_depth.py that computes tau_DW by walking each
galaxy's line of sight cell-by-cell through the HII fraction field, using
the local neutral fraction x_HI at each cell rather than a uniform mean.

The sightline direction for each galaxy should be drawn using the same random seed
as bubble_sizes.compute(), so that both modules trace the same ray. Default: 1216

Three functions are provided:
  - traverse_sightlines : walks sightlines and stores (z_i, x_HI_i) per cell
  - load_sightlines     : loads saved sightline data from HDF5
  - compute_tau_dw      : computes patchy tau_DW from stored sightline data

----------
traverse_sightlines(galaxy_pos, HII_cube_interp_all, BoxSize, sim, z_s, z_end, ...)
    Walk sightlines and return per-cell redshifts, neutral fractions, and z_beg.

load_sightlines(filepath)
    Load sightline data previously saved by traverse_sightlines().

compute_tau_dw(z_arr, x_HI_arr, z_beg, n_cells, z_s, sim, z_end, ...)
    Compute patchy tau_DW from sightline data.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator as ScipyRGI
from joblib import Parallel, delayed
import h5py
import os
import tempfile
import shutil
from numpy.polynomial.legendre import leggauss
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


# ---------------------------------------------------------------------------
# Lyman-alpha physical constants
# ---------------------------------------------------------------------------

R_ALPHA  = 2.02e-8    # Quantum damping wing parameter (dimensionless)
NU_ALPHA = 2.47e15    # Lyman-alpha frequency (Hz)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_random_directions(N_gal, seed):
    """
    Generate random isotropic unit direction vectors using the same
    JAX PRNGKey logic as bubble_sizes.compute().

    Parameters
    ----------
    N_gal : int
        Number of galaxies.
    seed : int
        Random seed — must match the seed used in bubble_sizes.compute()
        to ensure the same sightline directions are used.

    Returns
    -------
    directions : np.ndarray, shape (N_gal, 3)
        Unit direction vectors.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), N_gal)

    def sample_direction(key):
        key_phi, key_theta = jax.random.split(key)
        phi       = jax.random.uniform(key_phi,   (), minval=0.0,  maxval=2.0 * jnp.pi)
        cos_theta = jax.random.uniform(key_theta, (), minval=-1.0, maxval=1.0)
        sin_theta = jnp.sqrt(1.0 - cos_theta**2)
        return jnp.array([sin_theta * jnp.cos(phi),
                          sin_theta * jnp.sin(phi),
                          cos_theta])

    directions = jax.vmap(sample_direction)(keys)
    return np.asarray(directions, dtype=np.float64)


def _build_gauss_legendre(n_quad):
    """
    Precompute Gauss-Legendre nodes and weights in float64.

    Parameters
    ----------
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    nodes : np.ndarray, shape (n_quad,)
    weights : np.ndarray, shape (n_quad,)
    """
    nodes, weights = leggauss(n_quad)
    return nodes.astype(np.float64), weights.astype(np.float64)


def _del_I_integrand(x):
    """
    Integrand for the damping wing spectral profile: x^4.5 / (1 - x)^2.

    Parameters
    ----------
    x : np.ndarray
        Dimensionless frequency variable.

    Returns
    -------
    np.ndarray
        Integrand values.
    """
    return (x**4.5) / (1.0 - x)**2


def _integrate_gauss_batched(z_beg_cells, z_end_cells, x_HI_cells,
                              delta_pos, z_s, tau_gp, nodes, weights):
    """
    Compute tau_DW contributions for all valid cells and all delta values
    in a single batched numpy operation.

    Parameters
    ----------
    z_beg_cells : np.ndarray, shape (N_cells,)
        Near-edge redshifts of valid cells.
    z_end_cells : np.ndarray, shape (N_cells,)
        Far-edge redshifts of valid cells.
    x_HI_cells : np.ndarray, shape (N_cells,)
        Neutral fractions of valid cells.
    delta_pos : np.ndarray, shape (N_delta,)
        Positive delta values only (delta >= 0).
    z_s : float
        Source redshift.
    tau_gp : float
        Gunn-Peterson optical depth prefactor.
    nodes : np.ndarray, shape (n_quad,)
        Gauss-Legendre nodes.
    weights : np.ndarray, shape (n_quad,)
        Gauss-Legendre weights.

    Returns
    -------
    np.ndarray, shape (N_delta,)
        Total tau_DW contribution summed over all valid cells.
    """
    # x limits: shape (N_cells, N_delta)
    x_beg = (1.0 + z_beg_cells[:, None]) / ((1.0 + z_s) * (1.0 + delta_pos[None, :]))
    x_end = (1.0 + z_end_cells[:, None]) / ((1.0 + z_s) * (1.0 + delta_pos[None, :]))

    # Gauss-Legendre quadrature — shape (N_cells, N_delta, n_quad)
    mid    = 0.5 * (x_beg + x_end)
    half   = 0.5 * (x_beg - x_end)
    x_quad = half[:, :, None] * nodes[None, None, :] + mid[:, :, None]
    y_quad = _del_I_integrand(x_quad)

    # Integrate: shape (N_cells, N_delta)
    # I(x) integral
    I_vals = half * np.sum(weights[None, None, :] * y_quad, axis=2)

    # tau_DW per cell per delta: shape (N_cells, N_delta)
    tau_cells = (
        (tau_gp / np.pi)
        * R_ALPHA
        * x_HI_cells[:, None]
        * (1.0 + delta_pos[None, :]) ** 1.5
        * I_vals
    )

    # Sum over all cells: shape (N_delta,)
    # tau_dw contribution from all cells along line-of-sight
    return np.sum(tau_cells, axis=0)


def _tau_GP(z_s, Omega_b_h2, Omega_m_h2):
    """
    Gunn-Peterson optical depth prefactor.

    Parameters
    ----------
    z_s : float
        Source redshift.
    Omega_b_h2 : float
        Omega_b * h^2.
    Omega_m_h2 : float
        Omega_m * h^2.

    Returns
    -------
    float
        Gunn-Peterson optical depth.
    """
    Y = 0.24
    return (4e5
            * ((1.0 + z_s) / 7.0) ** 1.5
            * (Omega_b_h2 / 0.0225)
            * (Omega_m_h2 / 0.132) ** (-0.5)
            * ((1.0 - Y) / 0.76))


def _build_distance_lookup(z_s, z_end, d_H, Omega_m, Omega_l, n_lookup=100000):
    """
    Build a lookup table mapping comoving distance from source to redshift.

    Uses cumulative_trapezoid for vectorized computation — much faster
    than calling the bisection solver for each step.

    This is the main computational difference from optical_depth.py.
    Here, instead of solving for the redshift of ith cell z_i from the source z_s
    using a bisection solver, we are using a pre-computed lookup table that provides 
    z_i for a given comoving distance of the ith cell d_i from the source

    Parameters
    ----------
    z_s : float
        Source redshift.
    z_end : float
        End redshift (lower bound).
    d_H : float
        Hubble distance in Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_l : float
        Dark energy density parameter.
    n_lookup : int, optional
        Number of lookup table points. Default: 100000.

    Returns
    -------
    d_fine : np.ndarray, shape (n_lookup,)
        Comoving distance from z_end to each point, increasing.
        d_fine[0] = 0 at z_end, d_fine[-1] = total distance at z_s.
    z_fine_rev : np.ndarray, shape (n_lookup,)
        Redshifts paired with d_fine, increasing from z_end to z_s.
    d_total : float
        Total comoving distance from z_end to z_s in Mpc.

    Notes
    -----
    For np.interp lookup given distance from source (mfp_dist_mpc):
        z_i = np.interp(d_total - mfp_dist_mpc, d_fine, z_fine_rev)
    """
    z_fine    = np.linspace(z_s, z_end, n_lookup)
    integrand = d_H / np.sqrt((1.0 + z_fine)**3 * Omega_m + Omega_l)
    d_fine    = cumulative_trapezoid(integrand[::-1], z_fine[::-1], initial=0.0)
    d_total   = d_fine[-1]
    return d_fine, z_fine[::-1], d_total


def _save_cubes_as_memmap(HII_cube_interp_all, tmpdir):
    """
    Save each HII cube as a memory-mapped .npy file.

    Extracts the raw cube arrays from the interpolator objects and saves
    them to disk. Workers then load these via np.load(..., mmap_mode='r'),
    which allows the OS to share physical memory pages across processes
    without copying.

    Since we all now using all the HII cubes from z_s to z_end, this approach
    makes things significantly fast.

    Parameters
    ----------
    HII_cube_interp_all : dict
        Dictionary mapping redshift -> RegularGridInterpolator.
    tmpdir : str
        Directory to save the .npy files.

    Returns
    -------
    memmap_paths : dict
        Dictionary mapping redshift -> path to .npy file.
    BoxSize : float
        Spatial extent of the box (inferred from grid points).
    NumPixels : int
        Number of pixels per dimension (inferred from grid points).
    """
    memmap_paths = {}
    BoxSize      = None
    NumPixels    = None

    for z_key, interp in HII_cube_interp_all.items():
        fpath = os.path.join(tmpdir, f'HII_z{z_key:.4f}.npy')
        np.save(fpath, np.asarray(interp.values, dtype=np.float64))
        memmap_paths[z_key] = fpath

        if BoxSize is None:
            grid    = interp.grid[0]
            N       = len(grid)
            BoxSize  = float(grid[-1] + (grid[1] - grid[0]))  # endpoint not included
            NumPixels = N

    return memmap_paths, BoxSize, NumPixels


def _build_interp_from_memmap(fpath, BoxSize, NumPixels):
    """
    Build a scipy RegularGridInterpolator from a memory-mapped .npy file.

    Uses scipy instead of JAX to avoid JAX re-initialization issues in
    forked worker processes, enabling true process-based parallelism.
    In optical_depth.py, we are using the JAX interpolator than SciPy one.

    Parameters
    ----------
    fpath : str
        Path to the .npy file containing the HII cube.
    BoxSize : float
        Spatial extent of the box.
    NumPixels : int
        Number of pixels per dimension.

    Returns
    -------
    ScipyRGI
        Scipy RegularGridInterpolator built from the memory-mapped cube.
    """
    cube     = np.load(fpath, mmap_mode='r')
    grid_pts = np.linspace(0.0, BoxSize, NumPixels, endpoint=False)
    return ScipyRGI(
        (grid_pts, grid_pts, grid_pts),
        cube,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )


def _traverse_single_sightline(gal_idx, pos, direction, z_s, z_end,
                               memmap_paths, avail_z,
                               BoxSize, NumPixels, step_size,
                               d_fine, z_fine_rev, d_total, h):
    """
    Walk a single galaxy's sightline and record per-cell data.

    This is the worker function called in parallel by traverse_sightlines.
    Defined at module level so joblib can pickle it.

    Loads HII cubes from memory-mapped files — avoids pickling large
    interpolator objects. Each worker builds its own interpolators from
    the shared memory-mapped arrays.

    Parameters
    ----------
    gal_idx : int
        Galaxy index.
    pos : np.ndarray, shape (3,)
        Galaxy position in box units.
    direction : np.ndarray, shape (3,)
        Unit direction vector for the sightline.
    z_s : float
        Source redshift.
    z_end : float
        Redshift at which the walk stops.
    memmap_paths : dict
        Dictionary mapping redshift -> path to memory-mapped .npy file.
    avail_z : np.ndarray
        Available redshifts sorted ascending.
    BoxSize : float
        Spatial extent of the periodic box.
    NumPixels : int
        Number of pixels per dimension.
    step_size : float
        Step size in box units.
    d_fine : np.ndarray
        Lookup table: comoving distance from z_end (increasing).
    z_fine_rev : np.ndarray
        Lookup table: redshifts paired with d_fine (increasing from z_end).
    d_total : float
        Total comoving distance from z_end to z_s in Mpc.
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    z_arr_gal : list of float
        Redshift bin edges for this sightline from z_s to z_end. 
        So it is same for all the galaxies at z_s. 
    x_HI_arr_gal : list of float
        Neutral fraction per cell. 
        Different for all the galaxies because of different directions.
    z_beg_gal : float
        Redshift of first neutral cell, or z_end if fully ionized.
    n_cells_gal : int
        Number of cells walked.
    """
    # Build interpolators from memory-mapped files — shared across processes
    # via OS page cache, no data copying
    interp_cache = {}
    for z_key, fpath in memmap_paths.items():
        interp_cache[z_key] = _build_interp_from_memmap(fpath, BoxSize, NumPixels)

    pos           = pos.copy()
    z_i           = z_s
    mfp_dist      = 0.0
    found_neutral = False
    z_beg_gal     = z_end

    z_arr_gal    = [z_i]
    x_HI_arr_gal = []

    # Find closest available snapshot >= z_i using searchsorted
    idx    = np.searchsorted(avail_z, z_i)
    snap_z = avail_z[min(idx, len(avail_z) - 1)]

    while np.abs(z_i - z_end) >= 1e-2:

        # Get local HII fraction at current position
        wrapped_pos = pos % BoxSize
        x_HII_val   = float(interp_cache[snap_z]([[wrapped_pos[0], wrapped_pos[1], wrapped_pos[2]]])[0])
        x_HI        = 1.0 - x_HII_val

        x_HI_arr_gal.append(x_HI)

        # Record first neutral cell
        if not found_neutral and x_HI > 0.5:
            z_beg_gal     = z_i
            found_neutral = True

        # Step forward
        mfp_dist += step_size
        pos      += direction * step_size

        # Convert distance to redshift via lookup table
        mfp_dist_mpc = mfp_dist / (h * 1000.0)    # ckpc/h → Mpc
        z_i = float(np.interp(d_total - mfp_dist_mpc, d_fine, z_fine_rev))

        z_arr_gal.append(z_i)

        # Update snapshot using searchsorted
        idx    = np.searchsorted(avail_z, z_i)
        snap_z = avail_z[min(idx, len(avail_z) - 1)]

    return z_arr_gal, x_HI_arr_gal, z_beg_gal, len(x_HI_arr_gal)


def _compute_tau_dw_single(gal_idx, z_arr_gal, x_HI_arr_gal, z_beg_gal,
                           n_c, z_s, tau_gp, delta, mask_pos, nodes, weights):
    """
    Compute patchy tau_DW for a single galaxy.

    Worker function for joblib parallelization in compute_tau_dw.
    Defined at module level so joblib can pickle it.

    Parameters
    ----------
    gal_idx : int
        Galaxy index.
    z_arr_gal : np.ndarray, shape (n_c+1,)
        Redshift bin edges for this galaxy (unpadded).
    x_HI_arr_gal : np.ndarray, shape (n_c,)
        Neutral fraction per cell (unpadded).
    z_beg_gal : float
        Redshift of first neutral cell.
    n_c : int
        Number of valid cells.
    z_s : float
        Source redshift.
    tau_gp : float
        Gunn-Peterson optical depth prefactor.
    delta : np.ndarray, shape (N_delta,)
        Full delta grid.
    mask_pos : np.ndarray, shape (N_delta,), dtype=bool
        Mask for delta >= 0.
    nodes : np.ndarray
        Gauss-Legendre nodes.
    weights : np.ndarray
        Gauss-Legendre weights.

    Returns
    -------
    np.ndarray, shape (N_delta,)
        tau_DW for this galaxy.
    """
    delta_pos = delta[mask_pos]
    tau       = np.zeros(len(delta), dtype=np.float64)

    # Extract valid cells
    z_beg_all = z_arr_gal[:n_c]           # All the redshifts of the near-edge of each cell
    z_end_all = z_arr_gal[1:n_c + 1]      # All the redshifts of the far-edge of each cell
    x_HI_all  = x_HI_arr_gal[:n_c]        # All the x_HI of each cell

    # This checks for valid cells.
    # Only the cells with redshift below z_beg_gal (the redshift of the first neutral cell outside the ionized region)
    # contributes to tau_DW
    valid_mask = (
        (z_beg_all >= 0.0) &
        (z_end_all >= 0.0) &
        (x_HI_all  >  0.0) &
        (z_beg_all <= z_beg_gal)
    )

    z_beg_cells = z_beg_all[valid_mask]
    z_end_cells = z_end_all[valid_mask]
    x_HI_cells  = x_HI_all[valid_mask]

    if len(z_beg_cells) == 0:
        tau[~mask_pos] = np.inf
        return tau

    # Single batched call for all valid cells x all delta values
    tau[mask_pos] = _integrate_gauss_batched(
        z_beg_cells, z_end_cells, x_HI_cells,
        delta_pos, z_s, tau_gp, nodes, weights
    )
    tau[~mask_pos] = np.inf

    return tau


# ---------------------------------------------------------------------------
# Functions to be called by the user
# ---------------------------------------------------------------------------

def traverse_sightlines(galaxy_pos, HII_cube_interp_all, BoxSize, sim,
                        z_s, z_end, seed=1216, n_jobs=-1,
                        save=False, savepath='./', tmpdir=None):
    """
    Walk each galaxy's line of sight through the HII fraction field and
    record the redshift and neutral fraction at each cell.

    Should use the same random seed as bubble_sizes.compute() so that sightline
    directions are consistent between the two modules. Sightlines are
    processed in parallel using joblib with loky backend.

    HII cubes are saved as memory-mapped files so that worker processes
    share the same physical memory pages without copying — avoiding the
    pickling overhead of the interpolator objects.

    Parameters
    ----------
    galaxy_pos : array-like, shape (N, 3) or (3,)
        Galaxy positions in the same units as BoxSize. A single position
        of shape (3,) is automatically promoted to (1, 3).
    HII_cube_interp_all : dict
        Dictionary mapping redshift -> RegularGridInterpolator for the
        HII fraction field. Keys must be available redshifts in sim.
    BoxSize : float
        Spatial extent of the periodic box, in the same units as galaxy_pos.
    sim : lyaDW.Simulation
        Simulation context carrying cosmological parameters.
    z_s : float
        Source redshift. Resolved to closest available redshift in sim.
    z_end : float
        Redshift at which the sightline walk stops.
    seed : int, optional
        Random seed. Must match the seed used in bubble_sizes.compute()
        to ensure the same sightline directions. Default: 1216.
    n_jobs : int, optional
        Number of parallel jobs for joblib. -1 uses all available cores.
        1 disables parallelism (useful for debugging). Default: -1.
    save : bool, optional
        If True, save sightline data to HDF5. Default: False.
    savepath : str, optional
        Directory to save output. Default: './'.
    tmpdir : str, optional
        Directory to temporarily store the HII cube files in a memory map. 
        This is deleted after completion.

    Returns
    -------
    z_arr : np.ndarray, shape (N_gal, max_cells+1)
        Redshift bin edges per sightline. Padded with -1 beyond actual length.
    x_HI_arr : np.ndarray, shape (N_gal, max_cells)
        Neutral fraction per cell per sightline. Padded with -1.
    z_beg : np.ndarray, shape (N_gal,)
        Redshift of first neutral cell (x_HI > 0.5) per galaxy.
        Set to z_end if sightline is fully ionized.
    n_cells : np.ndarray, shape (N_gal,), dtype=int
        Number of cells per sightline (before padding).

    Notes
    -----
    - Padding sentinel value is -1. Use n_cells[i] to recover the actual
      data for galaxy i: z_arr[i, :n_cells[i]+1], x_HI_arr[i, :n_cells[i]].
    - The sightline walk stops when |z_i - z_end| < 1e-2.
    - It is recommended to pass only galaxies with R_ion > 0.
    - HII cubes are saved as temporary memory-mapped .npy files and cleaned
      up automatically after the parallel computation completes.
    - A distance-to-redshift lookup table is precomputed once and shared
      across all workers, replacing the slow per-step bisection solver.

    Examples
    --------
    >>> z_arr, x_HI_arr, z_beg, n_cells = lyaDW.core.optical_depth_patchy.traverse_sightlines(
    ...     galaxy_pos, HII_cube_interp_all, BoxSize,
    ...     sim=sim, z_s=7.33, z_end=5.5, seed=1216, n_jobs=-1
    ... )
    """
    z_s_resolved = sim.resolve_redshift(z_s)

    # Ensure 2D — handles single galaxy input (3,) → (1, 3)
    galaxy_pos = np.atleast_2d(np.asarray(galaxy_pos, dtype=np.float64))
    N_gal      = galaxy_pos.shape[0]

    # Cosmological parameters
    cosmo   = sim.cosmo
    d_H     = 1.0 / (sim.h * 100.0) * 2.998e5    # c / H0 in Mpc
    Omega_m = cosmo['Omega_m']
    Omega_l = cosmo['Omega_l']

    # Generate random directions in main process
    directions = _get_random_directions(N_gal, seed)

    # Available redshifts for snap lookup (sorted ascending)
    avail_z = np.sort(sim.redshifts)

    # Precompute distance-to-redshift lookup table once for all workers
    print("Building distance-to-redshift lookup table...")
    d_fine, z_fine_rev, d_total = _build_distance_lookup(
        z_s_resolved, z_end, d_H, Omega_m, Omega_l
    )

    # Save HII cubes as memory-mapped files — workers load these directly
    # avoiding pickling of large interpolator objects
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix='lyaDW_sightlines_')
    else:
        tmpdir = tempfile.mkdtemp(prefix='lyaDW_sightlines_', dir=tmpdir)
    print(f"Saving HII cubes to temporary memory-mapped files in {tmpdir}...")

    try:
        memmap_paths, _, NumPixels = _save_cubes_as_memmap(HII_cube_interp_all, tmpdir)
        step_size = BoxSize / NumPixels

        print(f"Traversing {N_gal} sightlines using {n_jobs} jobs...")

        with tqdm_joblib(desc='Traversing sightlines', total=N_gal):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_traverse_single_sightline)(
                    gal_idx,
                    galaxy_pos[gal_idx],
                    directions[gal_idx],
                    z_s_resolved, z_end,
                    memmap_paths, avail_z,
                    BoxSize, NumPixels, step_size,
                    d_fine, z_fine_rev, d_total,
                    sim.h
                )
                for gal_idx in range(N_gal)
            )

    finally:
        # Always clean up temporary files even if an error occurs
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("Cleaned up temporary memory-mapped files.")

    # Unpack results
    z_arr_list    = [r[0] for r in results]
    x_HI_arr_list = [r[1] for r in results]
    z_beg_all     = np.array([r[2] for r in results], dtype=np.float64)
    n_cells_all   = np.array([r[3] for r in results], dtype=np.int32)

    # Convert lists to padded numpy arrays
    max_cells = int(n_cells_all.max())

    z_arr_all = np.full((N_gal, max_cells + 1), -1.0, dtype=np.float64)

    for i in range(N_gal):
        n = n_cells_all[i]
        z_arr_all[i, :n + 1] = z_arr_list[i]
    del z_arr_list

    x_HI_arr_all = np.full((N_gal, max_cells), -1.0, dtype=np.float64)

    for i in range(N_gal):
        n = n_cells_all[i]
        x_HI_arr_all[i, :n] = x_HI_arr_list[i]
    del x_HI_arr_list

    print(f"Done. Max cells per sightline: {max_cells}")

    # --- Save (optional) ---
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_name = f"_{sim.simname}" if sim.simname else ""
        filename  = f"sightlines{sim_name}_z{z_s_resolved:.2f}.h5"
        filepath  = os.path.join(savepath, filename)

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('z_arr',    data=z_arr_all)
            f.create_dataset('x_HI_arr', data=x_HI_arr_all)
            f.create_dataset('z_beg',    data=z_beg_all)
            f.create_dataset('n_cells',  data=n_cells_all)

            f.attrs['z_s']      = z_s_resolved
            f.attrs['z_end']    = z_end
            f.attrs['seed']     = seed
            f.attrs['sentinel'] = -1.0
            if sim.simname:
                f.attrs['simulation'] = sim.simname

        print(f"Saved sightline data to: {filepath}")

    return z_arr_all, x_HI_arr_all, z_beg_all, n_cells_all


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
    >>> z_arr, x_HI_arr, z_beg, n_cells = lyaDW.core.optical_depth_patchy.load_sightlines(
    ...     './output/sightlines_Thesan_z9.01.h5'
    ... )
    >>> tau_dw, delta = lyaDW.core.optical_depth_patchy.compute_tau_dw(
    ...     z_arr, x_HI_arr, z_beg, n_cells, z_s=9.01, sim=sim
    ... )
    """
    with h5py.File(filepath, 'r') as f:
        z_arr    = f['z_arr'][:]
        x_HI_arr = f['x_HI_arr'][:]
        z_beg    = f['z_beg'][:]
        n_cells  = f['n_cells'][:]

        print(f"Loaded sightlines from: {filepath}")
        print(f"  z_s       = {f.attrs['z_s']:.4f}")
        print(f"  z_end     = {f.attrs['z_end']:.4f}")
        print(f"  seed      = {f.attrs['seed']}")
        print(f"  N_gal     = {z_arr.shape[0]}")
        print(f"  max_cells = {x_HI_arr.shape[1]}")

    return z_arr, x_HI_arr, z_beg, n_cells


def compute_neutral_blob_sizes(z_s, z_arr, x_HI_arr, z_beg, z_end, n_cells, step_size_mpc,
                               xHI_threshold=0.5):
    """
    Compute the sizes of neutral blobs along each galaxy's sightline.
 
    For each galaxy, finds consecutive neutral cells (x_HI > xHI_threshold)
    starting from z_beg onward and measures the comoving path length through
    each neutral region.
 
 
    Parameters
    ----------
    z_s : float
        The redshift at which the galaxies reside
    z_arr : np.ndarray, shape (N_gal, max_cells+1)
        Redshift bin edges per sightline, as returned by traverse_sightlines().
    x_HI_arr : np.ndarray, shape (N_gal, max_cells)
        Neutral fraction per cell, as returned by traverse_sightlines().
    z_beg : np.ndarray, shape (N_gal,)
        Redshift of first neutral cell per galaxy.
    z_end : float
        Final redshift to calculate the neutral blob sizes.
    n_cells : np.ndarray, shape (N_gal,), dtype=int
        Number of valid cells per sightline (before padding).
    step_size_mpc : float
        Step size in Mpc (= step_size_box_units / (h * 1000)).
    xHI_threshold : float, optional
        Neutral fraction threshold. Default: 0.5.
 
    Returns
    -------
    blob_sizes : list of np.ndarray
        List of length N_gal. Each element is a 1D array of neutral blob
        sizes in Mpc for that galaxy. Empty array if no neutral blobs found.
 
    Examples
    --------
    >>> step_size_mpc = BoxSize / NumPixels / (sim.h * 1000)
    >>> blob_sizes = lyaDW.core.optical_depth_patchy.compute_neutral_blob_sizes(
    ...     z_arr, x_HI_arr, z_beg, n_cells, step_size_mpc
    ... )
    >>> # Flatten all blob sizes across all galaxies
    >>> all_blobs = np.concatenate(blob_sizes)
    >>> print(f"Mean neutral blob size: {np.mean(all_blobs):.2f} Mpc")
    """
    from itertools import groupby
 
    n_cells  = np.asarray(n_cells,  dtype=np.int32)
    z_beg    = np.asarray(z_beg,    dtype=np.float64)
    N_gal    = z_arr.shape[0]
 
    all_blob_sizes = []
 
    for gal_idx in tqdm(range(N_gal)):
        n_c      = n_cells[gal_idx]
        z_beg_g  = z_beg[gal_idx]
 
        # Get valid cells from z_beg onward
        z_cells   = z_arr[gal_idx, :n_c]
        xHI_cells = x_HI_arr[gal_idx, :n_c]
 
        # Mask: valid (not sentinel) and at or beyond the neutral boundary
        mask      = (z_cells >= z_end) & (z_cells <= z_beg_g)
        xHI_valid = xHI_cells[mask]
 
        if len(xHI_valid) == 0:
            all_blob_sizes.append(np.array([], dtype=np.float64))
            continue
 
        # Find consecutive runs of neutral cells using groupby
        blob_sizes = []
        for is_neutral, group in groupby(xHI_valid > xHI_threshold):
            run_length = sum(1 for _ in group)
            if is_neutral:
                blob_sizes.append(run_length * step_size_mpc)
 
        all_blob_sizes.append(np.array(blob_sizes, dtype=np.float64))
 
    return all_blob_sizes


def compute_tau_dw(z_arr, x_HI_arr, z_beg, n_cells, z_s, sim,
                   z_end=5.5, delta=None, n_quad=8, n_jobs=-1,
                   save=False, savepath='./'):
    """
    Compute the patchy Lyman-alpha damping wing optical depth tau_DW.

    Sums tau_DW contributions from all neutral cells along each sightline
    in a single batched numpy operation per galaxy. Galaxies are processed
    in parallel using joblib.

    Parameters
    ----------
    z_arr : np.ndarray, shape (N_gal, max_cells+1)
        Redshift bin edges per sightline, as returned by traverse_sightlines().
        Padded with -1 sentinel beyond actual length.
    x_HI_arr : np.ndarray, shape (N_gal, max_cells)
        Neutral fraction per cell, as returned by traverse_sightlines().
        Padded with -1 sentinel.
    z_beg : np.ndarray, shape (N_gal,)
        Redshift of first neutral cell per galaxy.
    n_cells : np.ndarray, shape (N_gal,), dtype=int
        Number of valid cells per sightline (before padding).
    z_s : float
        Source redshift. Resolved to closest available redshift in sim.
    sim : lyaDW.Simulation
        Simulation context carrying cosmological parameters.
    z_end : float, optional
        Redshift at which the neutral IGM ends. Default: 5.5.
    delta : np.ndarray, shape (N_delta,), optional
        Fractional frequency offset grid. If None, uses default grid:
        np.arange(-0.05, 0.05 + 5e-5, 5e-5). Default: None.
    n_quad : int, optional
        Number of Gauss-Legendre quadrature points. Default: 8.
        Accuracy tests show n_quad=8 gives machine-precision results
        for this integrand, while being 8x faster than n_quad=64.
    n_jobs : int, optional
        Number of parallel jobs for joblib. -1 uses all available cores.
        1 disables parallelism (useful for debugging). Default: -1.
    save : bool, optional
        If True, save tau_DW to HDF5. Default: False.
    savepath : str, optional
        Directory to save output. Default: './'.

    Returns
    -------
    tau_dw : np.ndarray, shape (N_gal, N_delta)
        Patchy damping wing optical depth. Values are inf for delta < 0.
    delta : np.ndarray, shape (N_delta,)
        The delta grid used.

    Notes
    -----
    - Only cells with z_beg_cell <= z_beg[gal] are included (neutral region).
    - Cells with x_HI = 0 (fully ionized) contribute zero optical depth.
    - All valid cells per galaxy are batched into a single numpy operation.
    - Memory usage per worker: O(N_cells * N_delta * n_quad) floats.
      With n_quad=8, N_cells~6500, N_delta~1000: ~400 MB per worker.

    Examples
    --------
    >>> tau_dw, delta = lyaDW.core.optical_depth_patchy.compute_tau_dw(
    ...     z_arr, x_HI_arr, z_beg, n_cells,
    ...     z_s=7.33, sim=sim, z_end=5.5,
    ...     save=True, savepath='./output/'
    ... )
    """
    from lyaDW.core.optical_depth import DEFAULT_DELTA

    # --- Setup ---
    z_s_resolved = sim.resolve_redshift(z_s)

    if delta is None:
        delta = DEFAULT_DELTA.copy()
    delta   = np.asarray(delta,   dtype=np.float64)
    n_cells = np.asarray(n_cells, dtype=np.int32)
    z_beg   = np.asarray(z_beg,   dtype=np.float64)

    N_gal    = z_arr.shape[0]
    mask_pos = delta >= 0

    # Cosmological parameters
    cosmo      = sim.cosmo
    Omega_b_h2 = cosmo['Omega_b_h2']
    Omega_m_h2 = cosmo['Omega_m_h2']
    tau_gp     = _tau_GP(z_s_resolved, Omega_b_h2, Omega_m_h2)

    # Gauss-Legendre nodes and weights
    nodes, weights = _build_gauss_legendre(n_quad)

    print(f"Computing patchy tau_DW for {N_gal} galaxies using {n_jobs} jobs...")

    with tqdm_joblib(desc='Computing patchy tau_DW', total=N_gal):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_tau_dw_single)(
                gal_idx,
                z_arr[gal_idx, :n_cells[gal_idx] + 1],
                x_HI_arr[gal_idx, :n_cells[gal_idx]],
                z_beg[gal_idx],
                n_cells[gal_idx],
                z_s_resolved, tau_gp,
                delta, mask_pos,
                nodes, weights
            )
            for gal_idx in range(N_gal)
        )

    tau_dw = np.array(results, dtype=np.float64)

    # --- Save (optional) ---
    if save:
        os.makedirs(savepath, exist_ok=True)
        sim_name = f"_{sim.simname}" if sim.simname else ""
        filename  = f"tau_dw_patchy{sim_name}_z{z_s_resolved:.2f}.h5"
        filepath  = os.path.join(savepath, filename)

        with h5py.File(filepath, 'w') as f:
            ds = f.create_dataset('tau_dw', data=tau_dw)
            ds.attrs['z_s']    = z_s_resolved
            ds.attrs['z_end']  = z_end
            ds.attrs['n_quad'] = n_quad
            if sim.simname:
                ds.attrs['simulation'] = sim.simname

            f.create_dataset('delta', data=delta)
            f.create_dataset('z_beg', data=z_beg)

        print(f"Saved patchy tau_DW to: {filepath}")

    return tau_dw, delta