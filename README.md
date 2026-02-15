# StarspotMapping: Mapping starspots from photometric variability 

### The code to calculate the spotted flux is available in "macula.py" (version 0.0.0 on 23/10/27)
### The code to visualize the hemisphere is also available in "visualize.py" (version 0.0.0 on 23/11/17)
### The inference code with a parallel tempering or No U-turn sampler in JAX/NumPyro is also available in "Example_NumPyro.py" (version 0.0.0 on 24/02/14)




## Paper I

Supplementary figures for joint posteriors of all parameters ([Ikuta et al. 2020, ApJ, 902, 73](https://ui.adsabs.harvard.edu/abs/2020ApJ...902...73I/abstract))

## Paper II

Flare tables and Supplementary figures for joint posteriors of models in Appendix A ([Ikuta et al. 2023, ApJ, 648, 64](https://ui.adsabs.harvard.edu/abs/2023ApJ...948...64I/abstract))

## Applications

Quantifying the Transit Light Source Effect of a young M-dwarf ([Mori, Ikuta et al. 2024, MNRAS, 530, 167](https://ui.adsabs.harvard.edu/abs/2024MNRAS.tmp..863M/abstract))

Delving into the relation with a prominence eruption on a young solar-type star ([Namekata, Ikuta et al. 2024, ApJ, 976, 255](https://ui.adsabs.harvard.edu/abs/2024ApJ...976..255N/abstract))

Exploring the relation with Zeeman Doppler Imaging and multiwavelength variability for a young solar-type star ([Ikuta et al., submitted to ApJ](https://ui.adsabs.harvard.edu/abs/2024tsc3.confE..12I/abstract)) 

Spot map and activity on high-resolution spectrum for a RS CVn star (Nakasone et al., in prep.)

## Related papers

Blue/Red asymmetries from superflares on an M-dwarf flare star YZ CMi ([Kajikiya, Namekata, Notsu, Ikuta et al. 2025, ApJ, 985, 136](https://ui.adsabs.harvard.edu/abs/2025ApJ...985..136K/abstract))

Probable postflare loop from a superflare on an M-dwarf flare star EV Lac ([Ichihara, ..., Ikuta et al. 2025, PASJ, 77, 1025](https://ui.adsabs.harvard.edu/abs/2025PASJ...77.1025I/abstract))

Joint Doppler imaging with spot mapping on a K-dwarf flare star PW And ([Lee, ..., Ikuta et al. 2026, A&A, in press](https://ui.adsabs.harvard.edu/abs/2025arXiv251112190L/abstract))


## Erratum and Remarks (as of 25/02/07)

### Paper I

Page 3: In Equation 11, $\csc_k$ -> $\csc \beta_k$

Page 3: the tempering parameter (inverse temperature) $\beta_l = \exp (-7(l-1)/2 )$

Page 4: In Equation 15, multiply $\Lambda_k$ (deg) by $\pi/180$ to the unit of (rad)

Page 5, 6: In the footnotes of Table 1 and 2, the log uniform (Jeffreys) prior is described by $1/(\theta \log(b/a))$ (correct representation in Paper II)


### Paper II

Page 4: Above Equation 1, According not to Shibayama et al (2013) but to Maehara et al. (2021)

Page 9: The values of the reduced chi-square and p-value are incorrect (The conclusion does not change; see [Kajikiya et al. 2025b](https://ui.adsabs.harvard.edu/abs/2024ApJ...976..255N/abstract)).
