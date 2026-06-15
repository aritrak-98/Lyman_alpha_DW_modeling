"""
lyaDW — Lyman-alpha Damping Wing modeling package.
 
A simulation-agnostic toolkit for computing Lyman-alpha damping wing
optical depths, transmission coefficients, and equivalent width distributions.
 
Typical usage
-------------
    import lyaDW
    import numpy as np
 
    sim = lyaDW.Simulation(
        redshifts=np.array([6.10, 6.94, 7.33, ...]),
        h=0.6774, Omega_m=0.309, Omega_l=0.691, Omega_b=0.0486,
        simname='Thesan'
    )
 
    sim.set_HII_cube(z=6.94, HII_cube=my_cube)
 
    # Full pipeline — all accessible via lyaDW.*
    R_ion   = lyaDW.core.bubble_sizes.compute(...)
    z_beg   = lyaDW.core.igm_redshift.compute(...)
    tau_dw  = lyaDW.core.optical_depth.compute(...)
    T_alpha = lyaDW.core.transmission.compute(...)
    ew      = lyaDW.core.equivalent_width.compute(...)
 
    # IO helpers
    HII_cube, BoxSize = lyaDW.io.thesan.load_HII_cube(...)
 
    # Observational anchors
    results = lyaDW.observations.tang24.calc_lya_fraction(...)
"""

import jax
# Enable float64 support in JAX — required to avoid overflow of large
# physical constants (e.g. N_HI_dot = 1e53 * SFR overflows float32)
jax.config.update("jax_enable_x64", True)


from .simulation import Simulation
from . import core
from . import utils
from . import io
from . import observations

__version__ = "0.2.0"
__all__ = ["Simulation", "core", "utils", "io", "observations"]