# StarspotMapping

### The code to calculate the spotted flux is available in "macula.py" (version 0.0.0 on 23/10/27)
### The code to visualize the hemisphere is also available in "visualize.py" (version 0.0.0 on 23/11/17)

I have been implementing the code of starspot mapping with JAX/NumPyro (Ikuta et al., in preparation).

## Paper I

Supplementary figures for joint posteriors of all parameters ([Ikuta et al. 2020, ApJ, 902, 73](https://ui.adsabs.harvard.edu/abs/2020ApJ...902...73I/abstract))

## Paper II

Flare tables and Supplementary figures for joint posteriors of models in Appendix A ([Ikuta et al., 2023, 648, 64](https://ui.adsabs.harvard.edu/abs/2023ApJ...948...64I/abstract))





## Erratum and Remarks (as of 24/02/03)

### Paper I

Page 3: In Equation 11, $\csc_k$ -> $\csc \beta_k$

Page 3: the tempering parameter (inverse temperature) $\beta_l = \exp (-7(l-1)/2 )$

Page 4: In Equation 15, multiply $\Lambda_k$ (deg) by $\pi/180$ to the unit of (rad)

Page 5, 6: In the footnotes of Table 1 and 2, the log uniform (Jeffreys) prior is described by $1/(\theta \log(b/a))$ (correct representation in Paper II)


### Paper II

Page 4: Above Equation 1, According not to Shibayama et al (2013) but to Maehara et al. (2021)
