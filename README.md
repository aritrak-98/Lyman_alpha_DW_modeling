# Lyman-alpha Damping Wing transmission modeling

This package models the Lyman-alpha damping wing transmission around each star-forming galaxy. The code needs the ionized hydrogen field and the galaxy properties like position, halo mass, stellar mass, star-formate rate to work. It incorporates a Gaussian distribution for the intrinsic Lyman-alpha profile and accounts for galactic outflows by parameterizing it as a redward shift in the center of the profile.

Details of each function is well-documented as doc-strings.

## Steps to run the code:

1. Load in your x_HII field and galaxy properties.
2. Calculate the bubble sizes around each galaxy.
3. Compute the redshift at which the neutral region stars outside the bubble.
4. Compute the Lyman-alpha damping wing optical depth around galaxy.
5. Calculate the transmission coefficients defined as a ratio between emitted and intrinsic Lyman-alpha luminosities.

### Optional steps:

This code was designed to calculate the Lyman-alpha equivalent width (EW) fractions by calibrating to the observed EW distribution at z ~ 6 (Tang et al. 2024). So given the transmission coefficients and the observed EW distribution at z = 6, it can calculate the observed EW distributions at higher redshifts by convolving the tranmission distribution with the EW distrbution at z = 6.
