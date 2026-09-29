"""
lyaDW.observations — EW calibration using observed EW distributions from Tang et al. (2024)
 
Modules
-------
tang24 : Tang et al. (2024) log-normal EW distribution at z=6,
         used to predict the Lyman-alpha emitter fraction at higher redshifts.
"""
 
from . import tang24
from . import mason18