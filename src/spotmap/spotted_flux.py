import jax
import jax.numpy as jnp
from functools import partial
from spotmap.geometry import spotgeometry

class spotted_flux(spotgeometry):
    def __init__(self, eps=1e-15):
        super().__init__(eps)

    def _spotted_flux_single(self, params, t):
        alpha = self.alpha_t(params["radius"], params.get("t_ref", 0.0), params.get("ing", 0.0), params.get("eg", 0.0), params.get("life", 1e6), t)
        beta = jnp.arccos(self.cos_beta(params["phi"], params["lam"], params["period_rot"], params.get("kappa", 0.0), t, params["incl"]))

        zeta_pos = jnp.cos(beta + alpha)
        zeta_pos = jnp.where(zeta_pos < 0., 0., zeta_pos)

        beta_alpha = beta - alpha
        beta_alpha = jnp.where(beta_alpha < 0., 0., beta_alpha)
        zeta_neg = jnp.cos(beta_alpha)
        zeta_neg = jnp.where(zeta_neg < 0., 0., zeta_neg)

        ld_star = params["ld_star"]
        ld_spot = params.get("ld_spot", ld_star)
        f_spot = params["f_spot"]

        flux_star = (1. - jnp.sum(ld_star)) - (1. - jnp.sum(ld_spot)) * f_spot
        flux_spot_arr = ld_star - ld_spot * f_spot

        flux_area = flux_star
        flux_area += 2./3. * flux_spot_arr[1] * (zeta_pos*zeta_pos + zeta_pos*zeta_neg + zeta_neg*zeta_neg) / (zeta_pos + zeta_neg + self.eps)
        flux_area += 1./2. * flux_spot_arr[3] * (zeta_pos*zeta_pos + zeta_neg*zeta_neg)
        flux_area += 4./5. * flux_spot_arr[0] * (zeta_pos*zeta_pos*jnp.sqrt(zeta_pos + self.eps) - zeta_neg*zeta_neg*jnp.sqrt(zeta_neg + self.eps)) / (zeta_pos*zeta_pos - zeta_neg*zeta_neg + self.eps)
        flux_area += 4./7. * flux_spot_arr[2] * (zeta_pos*zeta_pos*zeta_pos*jnp.sqrt(zeta_pos + self.eps) - zeta_neg*zeta_neg*zeta_neg*jnp.sqrt(zeta_neg + self.eps)) / (zeta_pos*zeta_pos - zeta_neg*zeta_neg + self.eps)
        
        return self.projected_area(alpha, beta) * flux_area

    def _relative_flux_single(self, params, t):
        ld_star = params["ld_star"]
        flux_lim = 1. - ld_star[0]/5. - ld_star[1]*2./6. - ld_star[2]*3./7. - ld_star[3]*4./8.
        spot_contrib = jnp.sum(self._spotted_flux_single(params, t), axis=1)
        f = flux_lim - spot_contrib
        f_ave = jnp.mean(f)
        return f / f_ave - 1.

    @partial(jax.jit, static_argnums=(0,))
    def relative_flux(self, params, t):
        ld_star = params.get("ld_star")
        ld_spot = params.get("ld_spot")
        f_spot = params.get("f_spot")

        ld_star_axis = 0 if (ld_star is not None and jnp.ndim(ld_star) == 2) else None
        ld_spot_axis = 0 if (ld_spot is not None and jnp.ndim(ld_spot) == 2) else None
        f_spot_axis  = 0 if (f_spot is not None and jnp.ndim(f_spot) == 1) else None

        if ld_star_axis is not None or ld_spot_axis is not None or f_spot_axis is not None:
            axes_dict = {k: None for k in params.keys()}
            if "ld_star" in axes_dict: axes_dict["ld_star"] = ld_star_axis
            if "ld_spot" in axes_dict: axes_dict["ld_spot"] = ld_spot_axis
            if "f_spot"  in axes_dict: axes_dict["f_spot"]  = f_spot_axis

            mapped_func = jax.vmap(spotted_flux._relative_flux_single, in_axes=(None, axes_dict, None))
            return mapped_func(self, params, t)
        else:
            return self._relative_flux_single(params, t)
