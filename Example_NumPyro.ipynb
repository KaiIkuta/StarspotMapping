#Inference of parameters in macula.py with JAX/NumPyro
#macula.py may be updated to module for user-friendly code
#Released version 0.0.0 on 2024/02/14

import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

import numpyro

numpyro.set_platform('cpu')
numpyro.set_host_device_count(jax.device_count())

import numpyro.distributions as dist
from numpyro.distributions import Normal
from numpyro.infer import MCMC, NUTS


#Parallel tempering in NumPyro from Tensorflow
from tensorflow_probability.python.internal.backend import jax as tf
import tensorflow_probability as tfp; tfp = tfp.substrates.jax
tfd = tfp.distributions
from numpyro.contrib.tfp.mcmc import TFPKernel

##############################################
### Spotted flux (macula.py optimized for fast computation with JAX) 


#Spot relative intensity
f_spot = 0.48
#Stellar inclination
incl = 60.
incl = jnp.deg2rad(incl)

#Nonlinear limb-darkening case (Table 6 in Ikuta 23)
ld_star = jnp.array([3.00,-4.54,4.01,-1.35])
ld_spot = jnp.copy(ld_star)

#Flux by the stellar limb-darkening law (the first and second terms in Equation 10 in Ikuta et al. 2020)
flux_lim = 1. - ld_star[0]/5. - ld_star[1]*2./6. - ld_star[2]*3./7. -ld_star[3]*4./8.

flux_star = (1.-jnp.sum(ld_star))-(1.-jnp.sum(ld_spot))*f_spot

flux_spot = ld_star-ld_spot*f_spot



#The coordinate to the line of sight (Equation 15 in Ikuta 20) with the shape of (data point, spot_number)
@jax.jit
def cos_beta(phi,lam,period,kappa,t):
    z =jnp.cos(incl)*jnp.sin(phi).reshape(1,-1) #(1,num_phi)
    z =jnp.dot(jnp.ones([t.size]).reshape(-1,1),z) #(num_t,1)*(1,num_phi) = (num_t,num_phi)
    phase =jnp.dot((2.*jnp.pi*t/period).reshape(-1,1),(1-kappa*jnp.sin(phi)*jnp.sin(phi)).reshape(1,-1)) #(num_t,1)*(1,num_phi) = (num_t,num_phi)
    phase = phase + jnp.dot(jnp.ones([t.size]).reshape(-1,1),lam.reshape(1,-1))
    z = z + jnp.dot(jnp.ones([t.size]).reshape(-1,1),jnp.sin(incl)*jnp.cos(phi).reshape(1,-1))*jnp.cos(phase)
    z =jnp.where(z<-1.,-1.,z)
    z =jnp.where(z>1.,1.,z)
    return z


#Projected area (Equation 11 in Ikuta 20) with the shape of (data point, spot_number)
#The example case (Figure 6b in Ikuta 23)
@jax.jit
def projected_area(alpha,beta):
    # the middle term of the second case
    area = -jnp.cos(alpha)*jnp.cos(beta)/(jnp.sin(alpha)*jnp.sin(beta)+1e-15)
    area =jnp.where(area<-1.,-1.,area)
    area =jnp.where(area>1.,1.,area)
    area =jnp.arccos(area)
    area *=jnp.cos(beta)*jnp.sin(alpha)*jnp.sin(alpha)

    beta_alpha =jnp.cos(alpha)/(jnp.sin(beta)+1e-15)
    beta_alpha =jnp.where(beta_alpha>1.,1.,beta_alpha)
    # former and latter term of the second case (in case of the first/third case, return 0)
    area +=jnp.arccos(beta_alpha) -jnp.cos(alpha)*jnp.sin(beta)*jnp.sqrt(1-beta_alpha*beta_alpha+1e-15)
    return area/jnp.pi


#Angular radius of spot (Equation 14 in Ikuta 20) with the shape of (data point, spot_number)
@jax.jit
def alpha_t(radius,t_ref,ing,eg,life,t):
  t_before =jnp.dot(t.reshape(-1,1),jnp.ones([radius.size]).reshape(1,-1))
  t_before = ((t_before-t_ref)+life/2.)/ing
  t_after =jnp.dot(t.reshape(-1,1),jnp.ones([radius.size]).reshape(1,-1))
  t_after = ((t_ref-t_after)+life/2.)/eg
  s =jnp.where(t_before<t_after,t_before,t_after) +1.
  s =jnp.where(s<1.,s,1.)
  s *=radius
  s =jnp.where(s>0.,s,0.)
  return s

#The third term of the flux from spots (Equation 10 in Ikuta 20) with the shape of (data point, spot_number)
@jax.jit
def spotted_flux(phi, lam, period,kappa,radius,t_ref,ing,eg,life,t):
    alpha = alpha_t(radius,t_ref,ing,eg,life,t)
    beta =jnp.arccos(cos_beta(phi,lam,period,kappa,t))

    zeta_pos =jnp.cos(beta+alpha)
    zeta_pos =jnp.where(zeta_pos<0.,0.,zeta_pos)

    beta_alpha = beta-alpha
    beta_alpha =jnp.where(beta_alpha<0.,0.,beta_alpha)
    zeta_neg =jnp.cos(beta_alpha)
    zeta_neg =jnp.where(zeta_neg<0.,0.,zeta_neg)

    flux_area = flux_star
    flux_area += 2./3.*flux_spot[1]*(zeta_pos*zeta_pos+zeta_pos*zeta_neg+zeta_neg*zeta_neg)/(zeta_pos+zeta_neg+1e-15)
    flux_area += 1./2.*flux_spot[3]*(zeta_pos*zeta_pos+zeta_neg*zeta_neg)
    #If nonlinear limb-darkening law
    flux_area += 4./5.*flux_spot[0]*(zeta_pos*zeta_pos*jnp.sqrt(zeta_pos)-zeta_neg*zeta_neg*jnp.sqrt(zeta_neg))/(zeta_pos*zeta_pos-zeta_neg*zeta_neg+1e-15)
    flux_area += 4./7.*flux_spot[2]*(zeta_pos*zeta_pos*zeta_pos*jnp.sqrt(zeta_pos)-zeta_neg*zeta_neg*zeta_neg*jnp.sqrt(zeta_neg))/(zeta_pos*zeta_pos-zeta_neg*zeta_neg+1e-15)
    return projected_area(alpha,beta)*flux_area

#Relative flux (Equation 19 in Ikuta 20)
@jax.jit
def relative_flux(phi, lam, period,kappa,radius,t_ref,ing,eg,life,t):
    f =  flux_lim -jnp.sum(spotted_flux(phi, lam, period,kappa,radius, t_ref,ing,eg,life,t),axis=1) #Summation of contribution from all spots
    f_ave =jnp.mean(f)
    f = f/f_ave-1.
    return f



##############################################
### Fixed spot parameters and initialization of inference parameters 

em = jnp.array([1e12,1e12])
dec = jnp.array([1e12,1e12])
life = jnp.array([1000., 1000.]) #Stable size in the observed window
tref = jnp.array([27/2., 27/2.])

init = {
    "log_p" : jnp.array(jnp.log(4.3573)),
    "kappa" : jnp.array(0.002),
    "lat": jnp.deg2rad(jnp.array([24.89,55.89])),
    "lon": jnp.deg2rad(jnp.array([60.35,-84.11])),
    "rad": jnp.deg2rad(jnp.array([13.44,14.36]))
}
num_spot = 2


##############################################
### Synthetic data 

#Two-min cadence
cad = 0.00138888888
#Observed time, flux, its error
t = jnp.linspace(0,23.048,int(23.048/cad))
y =  relative_flux(init["lat"],init["lon"],jnp.exp(init["log_p"]),init["kappa"],init["rad"],tref,em,dec,life,t)
yerr = jnp.ones(t.size)*1e-4



##############################################
### NumPyro model


def numpyro_model(time,flux_obs,flux_err):
    log_p = numpyro.sample("log_p",dist.Uniform(0.,3.))
    period = numpyro.deterministic("period",jnp.exp(log_p))
    kappa = numpyro.sample("kappa",dist.Uniform(-0.3,0.3))
    lat = numpyro.sample("lat",dist.Uniform(-jnp.pi/2.,jnp.pi/2.).expand([num_spot])) #Spots are not discerned 
    lon = numpyro.sample("lon",dist.Uniform(-jnp.pi,jnp.pi).expand([num_spot]))
    rad = numpyro.sample("rad",dist.Uniform(1e-3,jnp.pi/2.).expand([num_spot]))

    flux_mod = relative_flux(lat,lon,period,kappa,rad,tref,em,dec,life,time)

    numpyro.sample("obs", dist.Normal(flux_mod, flux_err), obs=flux_obs)


##############################################
### If parallel tempering as in Ikuta et al. 2023

inverse_temperatures = (jnp.exp(-3.5))**tf.range(10)
pt_step = tf.tensordot(1./jnp.sqrt(inverse_temperatures),jnp.array([0.00025,0.005,0.005,0.0001,0.0001,0.0001,0.0001,0.00025,0.005,0.005,0.005,0.005, 0.001,0.001]),axes=0)
print(pt_step)

def make_kernel_fn(target_log_prob_fn):
    return tfp.mcmc.MetropolisHastings(tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn, step_size=pt_step,num_leapfrog_steps=10))

nuts_kernel = TFPKernel[tfp.mcmc.ReplicaExchangeMC](numpyro_model, inverse_temperatures=inverse_temperatures, swap_proposal_fn =  tfp.mcmc.default_swap_proposal_fn(prob_swap=0.5), make_kernel_fn=make_kernel_fn)

### If No U-tern sampler 

nuts_kernel = NUTS(numpyro_model, max_tree_depth = 5, forward_mode_differentiation=True, find_heuristic_step_size = True,step_size=1e-4,target_accept_prob=0.4)

### MCMC inference 

mcmc = MCMC(
    nuts_kernel,
    num_warmup=50000,
    num_samples=100000,
    num_chains=1,
    #chain_method="vectorized",
    progress_bar=True,
)
rng_key = jax.random.PRNGKey(154101678)
rng_key_, rng_key = jax.random.split(rng_key)


### Table for derived parameters

import arviz as az

data = az.from_numpyro(mcmc)

az.summary(data, var_names=[v for v in data.posterior.data_vars])

az.plot_trace(data)

