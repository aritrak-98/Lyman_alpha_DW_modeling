# Lyman-alpha Damping Wing transmission modeling

This package models the Lyman-alpha damping wing transmission around each star-forming galaxy. The code needs the ionized hydrogen field and the galaxy properties like position, halo mass, stellar mass, star-formate rate to work. It incorporates a Gaussian distribution for the intrinsic Lyman-alpha profile and accounts for galactic outflows by parameterizing it as a redward shift in the center of the profile.

Details of each function is well-documented as doc-strings. Here's a brief description of the modules. The main package directory in called ```lyaDW```. The core physics modules are in the ```/core``` directory. The package was tested using Thesan and Limfast outputs, so it has some useful loading modules for these simulations in the ```/io``` directory. It has some usefule utility packages in the ```/utils``` directory. The ```simulation.py``` code is used to define the simulation the user is using (more details in the doc-string). The optional use of calculating the equivalent width fractions can be done using the ```tang24.py``` code in ```/observations```.

```
.
├── core                    # core physics packages
│   ├── bubble_sizes.py     # calculates the bubbles sizes around each galaxy using the mean-free path approach
│   ├── igm_redshift.py     # calculates the starting redshift of the IGM outside each bubble
│   ├── luminosity.py       # calculates the Lyman-alpha intrinsic and transmitted luminosities as well as the UV luminosity for each galaxy
│   ├── optical_depth.py    # calculates the Lyman-alpha damping wing optical depth
│   └── transmission.py     # calculates the transmission coefficients
├── io                      # useful I/O module
│   ├── limfast.py          # useful I/O module for Limfast
│   └── thesan.py           # useful I/O module for Thesan
├── observations
│   └── tang24.py           # optional P(EW) calculations
├── simulation.py           # for defining the simulation
└── utils                   # useful utilities
    ├── cosmology.py      
    └── galaxy.py
```

The whole package can be imported as 

```import lyaDW```

or each of the module can be imported as

```import lyaDW.core.bubble_sizes```

```
import lyaDW.io.thesan
import lyaDW.utils.galaxy
```

## Steps to run the code:

1. Load in your x_HII field and galaxy properties.
2. Calculate the bubble sizes around each galaxy.
3. Compute the redshift at which the neutral region stars outside the bubble.
4. Compute the Lyman-alpha damping wing optical depth around galaxy.
5. Calculate the transmission coefficients defined as a ratio between emitted and intrinsic Lyman-alpha luminosities.

### Optional steps:

This code was designed to calculate the Lyman-alpha equivalent width (EW) fractions by calibrating to the observed EW distribution at z ~ 6 ([Tang et al. 2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.2701T/abstract)). So given the transmission coefficients and the observed EW distribution at z = 6, it can calculate the observed EW distributions at higher redshifts by convolving the tranmission distribution with the EW distrbution at z = 6.


The ```Example_notebook.ipynb``` file shows an example of how to run the pipeline. However, it is advised to use a pipeline in a ```.py``` code instead of a Jupyter notebook as some of module can take some time to run depending on the number of galaxies. In the example notebook, only a subset of galaxies were selected for faster calculations.
