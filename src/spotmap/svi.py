from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.infer.reparam import NeuTraReparam
import numpyro.handlers as handlers

guide = autoguide.AutoBNAFNormal(numpyro_model,init_loc_fn=numpyro.infer.init_to_value(values=init_params),num_flows=2) 
optimizer = numpyro.optim.Adam(step_size=1e-4)
svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO())

rng_key, svi_key, sample_key = jax.random.split(jax.random.PRNGKey(159613900), 3)

svi_result = svi.run(svi_key, 10000, time1=t1, time2=t2, flux_obs=f, flux_err=ferr)

neutra = NeuTraReparam(guide, svi_result.params)
latent_param_names = ["log_p", "kappa", "lat", "lon", "rad", "tref", "log_em", "log_dec"]
model_neutra = handlers.reparam(numpyro_model, config=lambda site: neutra if site["name"] in latent_param_names else None)



#sampling from approximated posterior 
predictive = Predictive(guide, params=svi_result.params, num_samples=5000)

samples = predictive(sample_key, time1=t1, time2=t2, flux_obs=f, flux_err=ferr)
