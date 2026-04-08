from numpyro.contrib.nested_sampling import NestedSampler

ns = NestedSampler(numpyro_model, constructor_kwargs={"num_live_points": 1000, "max_samples": 10000},termination_kwargs={"dlogZ": 1e-4})

ns.run(jax.random.PRNGKey(42), t, f, ferr)
samples = ns.get_samples(jax.random.PRNGKey(0), num_samples=10000)
