import jax
import jax.numpy as jnp
from functools import partial
from jaxoplanet.orbits.keplerian import Central, Body, System
from spotmap.geometry import spotgeometry

class spotcrossed_flux(spotgeometry):
    def __init__(self, eps=1e-15, n_annuli=300):
        super().__init__(eps) 
        self.n_annuli = n_annuli
        self.r_annuli = jnp.linspace(1e-5, 0.99999, self.n_annuli)
        self.mu_r = jnp.sqrt(1.0 - self.r_annuli**2)

    @partial(jax.jit, static_argnums=(0,))
    def circleangle(self, r, p, z): #Calculation of delta(r) from p (=Rp/Rs) and z (=sqrt(xp^2+xp^2)) 
        safe_denom = jnp.where(2.0 * z * r == 0.0, self.eps, 2.0 * z * r)
        cos_val = jnp.clip((r*r + z*z - p*p) / safe_denom, -1.0, 1.0)
        mid_val = jnp.arccos(cos_val)
        ans_p_gt_z = jnp.where(r < p - z, jnp.pi, jnp.where(r < p + z, mid_val, 0.0))
        ans_p_le_z = jnp.where(r < z - p, 0.0, jnp.where(r < z + p, mid_val, 0.0)) 
        return jnp.where(p > z, ans_p_gt_z, ans_p_le_z)

    @partial(jax.jit, static_argnums=(0,))
    def ellipse_angle(self, r, alpha, beta): #Calculation of gamma(r) from alpha and beta (I note beta -> pi/2 - beta in Beky et al. 2014) 
        a = jnp.sin(alpha)
        b = jnp.sin(alpha) * jnp.cos(beta)
        z = jnp.cos(alpha) * jnp.sin(beta)
        
        concentric_mask = (z <= 0.0)
        z = jnp.clip(z, 0.0, 1.0)
        
        z_sq, a_sq = z * z, a * a
        b_0_mask = (a_sq + z_sq >= 1.0)
        
        denom_A = 1.0 - a_sq - z_sq
        safe_A = jnp.where(denom_A == 0.0, 1.0, denom_A)
        A = z_sq / safe_A
        
        bound1, bound2, bound3 = b - z, z - b, b + z
        val_inside_sqrt = jnp.maximum(0.0, z_sq - A * (r*r - z_sq - a_sq))
        safe_r = jnp.where(r == 0.0, self.eps, r)
        
        val_arccos = (z - (-z + jnp.sqrt(val_inside_sqrt)) / safe_A) / safe_r
        ans_intersect = jnp.arccos(jnp.clip(val_arccos, -1.0, 1.0))
        
        ans_main = jnp.where(r < bound1, jnp.pi,
                     jnp.where(r < bound2, 0.0,
                     jnp.where(r < bound3, ans_intersect, 0.0)))
        
        ans_b0 = jnp.where(r >= z, jnp.arccos(jnp.clip(z / safe_r, -1.0, 1.0)), 0.0)
        ans = jnp.where(b_0_mask, ans_b0, ans_main)
        ans = jnp.where(alpha <= 0.0, 0.0, ans)
        return jnp.where(concentric_mask, jnp.where(r < a, jnp.pi, 0.0), ans)

    @partial(jax.jit, static_argnums=(0,))
    def spot_sky_coords(self, phi, lam, period, kappa, t, incl, spin_orbit):
        t_2d = t[:, None]
        phi_2d = phi[None, :]
        lam_2d = lam[None, :]
        
        phase = (2. * jnp.pi * t_2d / period) * (1. - kappa * jnp.sin(phi_2d)**2) + lam_2d
        
        z_los = jnp.cos(incl) * jnp.sin(phi_2d) + jnp.sin(incl) * jnp.cos(phi_2d) * jnp.cos(phase)
        x_sky = jnp.cos(phi_2d) * jnp.sin(phase)
        y_sky = jnp.sin(incl) * jnp.sin(phi_2d) - jnp.cos(incl) * jnp.cos(phi_2d) * jnp.cos(phase)
        
        x_orb = x_sky * jnp.cos(spin_orbit) + y_sky * jnp.sin(spin_orbit)
        y_orb = -x_sky * jnp.sin(spin_orbit) + y_sky * jnp.cos(spin_orbit)
        
        return x_orb, y_orb, jnp.clip(z_los, -1.0 + self.eps, 1.0 - self.eps)

    @partial(jax.jit, static_argnums=(0,))
    def planet_sky_coords(self, period_orb, t0, b, r_p, ecc, omega, t):
        body = Body(
            period=period_orb, 
            time_transit=t0, 
            impact_param=b, 
            radius=r_p, 
            eccentricity=ecc, 
            omega_peri=omega
        )
        system = System(Central()).add_body(body)
        
        x_sys, y_sys, z_sys_los = system.relative_position(t)
        x_p, y_p = x_sys[0], y_sys[0]
        z_p_los = z_sys_los[0]
        
        z_p = jnp.where(z_p_los > 0, jnp.sqrt(x_p**2 + y_p**2), 1000.0)
        return x_p, y_p, z_p

    def _spotted_transit_flux_single(self, params, t, px, py, pz, f_r):

        alpha_arr = self.alpha_t(params["radius"], params.get("t_ref", 0.0), params.get("ing", 0.0), params.get("eg", 0.0), params.get("life", 1e6), t)
        sx, sy, sz_los = self.spot_sky_coords(params["phi"], params["lam"], params["period_rot"], params.get("kappa", 0.0), t, params["incl"], params.get("spin_orbit", 0.0))
        beta_arr = jnp.arccos(sz_los)
        
        r_p = params["r_p"]
        f_spot = params["f_spot"]
        
        def calc_single_spot(alpha, beta, ex, ey, sz_val, px_val, py_val, pz_val):
            visible = (sz_val > 0) & (alpha > 0)
            
            sa_t = jnp.where(visible, self.ellipse_angle(self.r_annuli, alpha, beta), 0.0)
            pa_t = self.circleangle(self.r_annuli, r_p, pz_val)
            
            z_ell = jnp.sqrt(ex**2 + ey**2)
            d_sq = (px_val - ex)**2 + (py_val - ey)**2
            
            denom = jnp.where(2.0 * pz_val * z_ell == 0.0, 1.0, 2.0 * pz_val * z_ell)
            ps_angle = jnp.where((z_ell == 0.0) | (pz_val == 0.0), 0.0, 
                                 jnp.arccos(jnp.clip((pz_val**2 + z_ell**2 - d_sq) / denom, -1.0, 1.0)))
            
            s_c = f_spot - 1.0
            oot_contrib = jnp.sum(s_c * sa_t * self.r_annuli * f_r)
            
            mask1 = ps_angle > pa_t + sa_t
            mask2 = (~mask1) & (sa_t > ps_angle + pa_t)
            mask3 = (~mask1) & (~mask2) & (pa_t <= ps_angle + sa_t)
            mask3a = mask3 & ((2.0 * jnp.pi - ps_angle) >= (pa_t + sa_t))
            mask3b = mask3 & (~mask3a)
            
            val1 = s_c * sa_t
            val2 = s_c * (sa_t - pa_t)
            val3a = 0.5 * s_c * (sa_t + ps_angle - pa_t)
            val3b = s_c * (jnp.pi - pa_t)
            
            updates = (mask1 * val1) + (mask2 * val2) + (mask3a * val3a) + (mask3b * val3b)
            transit_contrib = jnp.sum(updates * self.r_annuli * f_r)
            
            return oot_contrib, transit_contrib

        v_calc = jax.vmap(jax.vmap(calc_single_spot, in_axes=(0,0,0,0,0,None,None,None)), 
                                                     in_axes=(0,0,0,0,0,0,0,0))
        
        ex_arr = sx * jnp.cos(alpha_arr)
        ey_arr = sy * jnp.cos(alpha_arr)
        
        oot_c, transit_c = v_calc(alpha_arr, beta_arr, ex_arr, ey_arr, sz_los, px, py, pz)
        return oot_c, transit_c

    def _relative_transit_flux_single(self, params, t):
        r_p = params["r_p"]
        px, py, pz = self.planet_sky_coords(params["period_orb"], params["t0"], params["b"], r_p, params.get("ecc", 0.0), params.get("omega", 0.0), t)
        
        ld_star = params["ld_star"]
        f_r = 1.0 - ld_star[0]*(1.0 - self.mu_r**0.5) - ld_star[1]*(1.0 - self.mu_r) - ld_star[2]*(1.0 - self.mu_r**1.5) - ld_star[3]*(1.0 - self.mu_r**2)
        oot_base_flux = jnp.sum(jnp.pi * self.r_annuli * f_r)

        pa_t = jax.vmap(self.circleangle, in_axes=(None, None, 0))(self.r_annuli, r_p, pz)
        planet_transit_flux = jnp.sum((jnp.pi - pa_t) * self.r_annuli * f_r, axis=1)
        
        oot_c, transit_c = self._spotted_transit_flux_single(params, t, px, py, pz, f_r)
        
        oot_base = oot_base_flux + jnp.sum(oot_c, axis=1)
        transit_flux = planet_transit_flux + jnp.sum(transit_c, axis=1)
        
        transit_flux = jnp.where(pz >= 1.0 + r_p, oot_base, transit_flux)
        
        return (transit_flux / oot_base) - 1.0

    @partial(jax.jit, static_argnums=(0,))
    def relative_transit_flux(self, params, t):
        ld_star = params.get("ld_star")
        f_spot = params.get("f_spot")

        ld_star_axis = 0 if (ld_star is not None and jnp.ndim(ld_star) == 2) else None
        f_spot_axis  = 0 if (f_spot is not None and jnp.ndim(f_spot) == 1) else None

        if ld_star_axis is not None or f_spot_axis is not None:
            axes_dict = {k: None for k in params.keys()}
            if "ld_star" in axes_dict: axes_dict["ld_star"] = ld_star_axis
            if "f_spot"  in axes_dict: axes_dict["f_spot"]  = f_spot_axis

            mapped_func = jax.vmap(SpotrodModel._relative_transit_flux_single, in_axes=(None, axes_dict, None))
            return mapped_func(self, params, t)
        else:
            return self._relative_transit_flux_single(params, t)
