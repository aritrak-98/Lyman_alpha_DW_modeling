# Lyman-alpha Damping Wing transmission modeling

This package models the Lyman-alpha damping wing transmission around each star-forming galaxy. The code needs the ionized hydrogen field and the galaxy properties like position, halo mass, stellar mass, star-formate rate to work. It incorporates a Gaussian distribution for the intrinsic Lyman-alpha profile and accounts for galactic outflows by parameterizing it as a redward shift in the center of the profile.

Details of each function is well-documented as doc-strings. Here's a brief description of the modules. The main package directory in called ```lyaDW```. The core physics modules are in the ```/core``` directory. The package was tested using Thesan and Limfast outputs, so it has some useful loading modules for these simulations in the ```/io``` directory. It has some usefule utility packages in the ```/utils``` directory. The ```simulation.py``` code is used to define the simulation the user is using (more details in the doc-string). The optional use of calculating the equivalent width fractions can be done using the ```tang24.py``` code in ```/observations```.

```
├── core                            # core physics packages
│   ├── bubble_sizes.py             # calculates the bubble sizes around each galaxy using the mean-free path approach
│   ├── igm_redshift.py             # calculates the starting redshift of the IGM outside each bubble
│   ├── __init__.py
│   ├── luminosity.py               # calculates the Lyman-alpha intrinsic and transmitted luminosities as well as the UV luminosity for each galaxy
│   ├── optical_depth_patchy.py     # calculates the Lyman-alpha damping wing optical depth in a patchy IGM scenario
│   ├── optical_depth.py            # calculates the Lyman-alpha damping wing optical depth
│   └── transmission.py             # calculates the transmission coefficients
├── io                              # useful I/O modules
│   ├── __init__.py
│   ├── limfast.py                  # I/O for LIMFAST
│   ├── prop.py                     # I/O for general functions
│   └── thesan.py                   # I/O for Thesan
├── observations
│   ├── __init__.py
│   ├── mason18.py                  # optional P(EW) calculations (Mason et al. 2018)
│   └── tang24.py                   # optional P(EW) calculations (Tang et al. 2024)
├── utils                           # useful utilities
│   ├── cosmology.py
│   ├── galaxy.py
│   └── __init__.py
├── __init__.py
└── simulation.py                   # for defining the simulation
```

The whole package can be imported as 

```import lyaDW```

or each of the module can be imported as

```
import lyaDW.core.bubble_sizes
import lyaDW.io.thesan
import lyaDW.utils.galaxy
```

## Steps to run the code:

1. Load in your x_HII field and galaxy properties.
2. Calculate the bubble sizes around each galaxy.
3. Compute the redshift at which the neutral region starts outside the bubble.
4. Compute the Lyman-alpha damping wing optical depth around the galaxy.
5. Calculate the transmission coefficients defined as a ratio between transmitted and intrinsic Lyman-alpha luminosities.

### Optional steps:

This code was designed to calculate the Lyman-alpha equivalent width (EW) fractions by calibrating to the observed EW distribution at z ~ 6 ([Tang et al. 2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.2701T/abstract)). So given the transmission coefficients and the observed EW distribution at z = 6, it can calculate the observed EW distributions at higher redshifts by convolving the tranmission distribution with the EW distrbution at z = 6.


The ```Example_notebook.ipynb``` file shows an example of how to run the pipeline. However, it is advised to use a pipeline in a ```.py``` code instead of a Jupyter notebook as some of module can take some time to run depending on the number of galaxies. In the example notebook, only a subset of galaxies were selected for faster calculations.


### Updates to version 0.2.0

The package now includes 
- more I/O modules for easily reading data files
- damping wing optical depth calculation in patchy IGM scenario

### Updates to version 0.2.0

New intrinsic Lyman-alpha profile from Neyer+25
Exponential EW distribution from Mason+18

## Dependencies:

1. numpy
2. scipy
3. astropy
4. pickle
5. jax
6. h5py
7. tqdm


Aritra Kundu acknowledges the use of the AI model Claude Sonnet 4.6 in helping to turn the modeling pipeline into a Python package.
