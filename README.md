# StarspotMapping: Mapping starspots from photometric variability 

### The code to calculate the spotted flux is available in "macula.py" (version 0.0.0 on 23/10/27)
### The code to visualize the hemisphere is also available in "visualize.py" (version 0.0.0 on 23/11/17)
### The inference code with a parallel tempering or No U-turn sampler in JAX/NumPyro is also available in "Example_NumPyro.py" (version 0.0.0 on 24/02/14)


## Installation

```bash
pip install git+https://github.com/KaiIkuta/StarspotMapping.git
```

## Import (To be uploaded)

```python
import jax.numpy as jnp
import spotmap
from spotmap import spotflux

#Time series in 20 days
t_array = jnp.linspace(0, 20, 1000)

#Parameters
params = {
    "radius": jnp.array([0.1]),        # Spot radius (rad)
    "phi": jnp.array([1.0]),           # Spot latitude (rad)
    "lam": jnp.array([0.0]),           # Spot longitude (rad)
    "period_rot": 11.05,               # Stellar equatorial period (day)
    "incl": jnp.deg2rad(90.0),         # Stellar inclination (rad)
    "ld_star": jnp.array([3.0, -4.54, 4.01, -1.35]), # Stellar limb-darkening coefficients
    "f_spot": 0.48,                    # Spot intensity
    
    # If spot size is variable (if skipped, stable spot size is adopted)
    "t_ref": jnp.array([0.0]),         # Time of maximum radius (day) from start time
    "ing": jnp.array([0.0]),           # Emergence rate (day)
    "eg": jnp.array([0.0]),            # Decay rate (day)
    "life": jnp.array([1e6]),          # Stable duration (day)
    "kappa": 0.0,                      # Degree of differential rotation
    "ld_spot": jnp.array([3.0, -4.54, 4.01, -1.35]) # Spot limb-darkening coefficients
}

flux = spotflux().relative_flux(params, t_array)

```



## Paper I
Implementation of spotted flux with variable spot toward Kepler and TESS light curve ([Ikuta et al. 2020, ApJ, 902, 73](https://ui.adsabs.harvard.edu/abs/2020ApJ...902...73I/abstract)):

Supplementary figures for joint posteriors of all parameters 

## Paper II
Delving into the relation with flares on M-dwarf flare stars AU Mic, YZ CMi, and EV Lac ([Ikuta et al. 2023, ApJ, 648, 64](https://ui.adsabs.harvard.edu/abs/2023ApJ...948...64I/abstract)):

Flare tables and Supplementary figures for joint posteriors of models in Appendix A 

## Applications

Quantifying the Transit Light Source Effect of a young M-dwarf K2-25 ([Mori, Ikuta et al. 2024, MNRAS, 530, 167](https://ui.adsabs.harvard.edu/abs/2024MNRAS.tmp..863M/abstract))

Delving into the relation with a prominence eruption on a young solar-type star EK Dra ([Namekata, Ikuta et al. 2024, ApJ, 976, 255](https://ui.adsabs.harvard.edu/abs/2024ApJ...976..255N/abstract))

Exploring the relation with Zeeman Doppler Imaging and multiwavelength variability for a young solar-type star EK Dra ([Ikuta et al. 2026, ApJ, 1001, 18](https://ui.adsabs.harvard.edu/abs/2026ApJ..1001...18I/abstract)) 


## Related papers

Blue/Red asymmetries from superflares on an M-dwarf flare star YZ CMi ([Kajikiya, Namekata, Notsu, Ikuta et al. 2025, ApJ, 985, 136](https://ui.adsabs.harvard.edu/abs/2025ApJ...985..136K/abstract))

Probable postflare loop from a superflare on an M-dwarf flare star EV Lac ([Ichihara, ..., Ikuta et al. 2025, PASJ, 77, 1025](https://ui.adsabs.harvard.edu/abs/2025PASJ...77.1025I/abstract))

Joint Doppler imaging with spot mapping on a K-dwarf flare star PW And ([Lee, ..., Ikuta et al. 2026, A&A, 707, A24](https://ui.adsabs.harvard.edu/abs/2026A%26A...707A..24L/abstract)) and a young solar-type star EK Dra (Lee et al., in prep.)

Spot map and activity on high-resolution spectrum for a RS CVn star (Nakasone et al., in prep.)

Dynamical Doppler imaging with spot mapping ([jaxsmap](https://github.com/KaiIkuta/jaxsmap); Ikuta et al., in prep.)

## Erratum and Remarks (as of 25/02/07)

### Paper I

Page 3: In Equation 11, $\csc_k$ -> $\csc \beta_k$

Page 3: the tempering parameter (inverse temperature) $\beta_l = \exp (-7(l-1)/2 )$

Page 4: In Equation 15, multiply $\Lambda_k$ (deg) by $\pi/180$ to the unit of (rad)

Page 5, 6: In the footnotes of Table 1 and 2, the log uniform (Jeffreys) prior is described by $1/(\theta \log(b/a))$ (correct representation in Paper II)


### Paper II

Page 4: Above Equation 1, According not to Shibayama et al (2013) but to Maehara et al. (2021)

Page 9: The values of the reduced chi-square and p-value are incorrect (The conclusion does not change; see [Kajikiya et al. 2025b](https://ui.adsabs.harvard.edu/abs/2024ApJ...976..255N/abstract)).
