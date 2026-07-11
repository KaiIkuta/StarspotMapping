import jax
import jax.numpy as jnp
from spotmap.spot_flux import spotflux
from functools import partial


eps = 1e-5


@partial(jax.jit, static_argnums=(2,))
def single_starspot_simulator_split(theta_single, time,n_spots):

    log_p = theta_single[0]
    period = jnp.exp(log_p)

    kappa = theta_single[1]

    i_lat = 2
    i_lon = i_v + n_spots
    i_rad = i_lon + n_spots
    i_tref = i_rad + n_spots
    i_ing = i_tref + n_spots
    i_eg = i_ing + n_spots
    i_life = i_eg + n_spots
    i_end = i_life + n_spots

    lat = jnp.clip(theta_single[i_v:i_lat], -jnp.pi/2. + eps, jnp.pi/2. - eps)
    lon = jnp.clip(theta_single[i_lon:i_rad], -jnp.pi + eps, jnp.pi - eps)
    rad = jnp.clip(theta_single[i_rad:i_tref], 1e-4, jnp.pi / 2.0)
    tref = jnp.clip(theta_single[i_tref:i_ing], time1[0] + eps, time2[-1] - eps)
    ing = jnp.exp(jnp.clip(theta_single[i_ing:i_eg], -5.0, 10.0))
    eg = jnp.exp(jnp.clip(theta_single[i_eg:i_life], -5.0, 10.0))
    life = jnp.exp(jnp.clip(theta_single[i_eg:i_life], -5.0, 10.0))

    flux_abs = spotflux().relative_flux(params, time)

    flux_mod1_norm = (flux_abs1 / jnp.mean(flux_abs1)) - 1.0

    return flux_mod_norm

@jax.jit
def physics_simulator_batch_split(theta_batch, time):
    return jax.vmap(single_starspot_simulator_split, in_axes=(0, None, None))(theta_batch, time,n_spot)

def sample_prior_n_spots(rng_key, n_samples, time_array, n_spots):
    keys = jax.random.split(rng_key, 7)
    
    log_p = jax.random.uniform(keys[0], (n_samples, 1), minval=jnp.log(2.6), maxval=jnp.log(3.1))
  
    lat_unsorted = jax.random.uniform(keys[1], (n_samples, n_spots), minval=-1.0, maxval=1.0)
    lat = jnp.sort(lat_unsorted, axis=1) 
    
    lon = jax.random.uniform(keys[2], (n_samples, n_spots), minval=-jnp.pi, maxval=jnp.pi)
    rad = jax.random.uniform(keys[3], (n_samples, n_spots), minval=1e-3, maxval=jnp.pi / 2.0)
    tref = jax.random.uniform(keys[4], (n_samples, n_spots), minval=time_array[0], maxval=time_array[-1])
    log_em = jax.random.uniform(keys[5], (n_samples, n_spots), minval=-3.0, maxval=7.0)
    log_dec = jax.random.uniform(keys[6], (n_samples, n_spots), minval=-3.0, maxval=7.0)

    return jnp.concatenate([log_p, lat, lon, rad, tref, log_em, log_dec], axis=1)
