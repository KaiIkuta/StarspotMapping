#Spot-crossing code in Beky et al. (2014) for spot map in Kipping (2012)
#To be debugged and connected with the macula, jaxoplanet, and tinygp

import jax
import jax.numpy as jnp
from jaxoplanet.orbits import KeplerianOrbit
from spotmap import macula_jax


eps = 1e-15
f_spot = 0.48
incl = jnp.deg2rad(60.)
spin_orbit = jnp.deg2rad(0.) 


ld_star = jnp.array([3.00, -4.54, 4.01, -1.35])

n_annuli = 300
r_annuli = jnp.linspace(1e-5, 0.99999, n_annuli)
mu_r = jnp.sqrt(1.0 - r_annuli**2)
f_r = 1.0 - ld_star[0]*(1.0 - mu_r**0.5) - ld_star[1]*(1.0 - mu_r) - ld_star[2]*(1.0 - mu_r**1.5) - ld_star[3]*(1.0 - mu_r**2)
oot_base_flux = jnp.sum(jnp.pi * r_annuli * f_r) # スポット無しの基準フラックス


@jax.jit
def circleangle(r, p, z):

    safe_denom = jnp.where(2.0 * z * r == 0.0, eps, 2.0 * z * r)
    cos_val = jnp.clip((r*r + z*z - p*p) / safe_denom, -1.0, 1.0)
    mid_val = jnp.arccos(cos_val)
    ans_p_gt_z = jnp.where(r < p - z, jnp.pi, jnp.where(r < p + z, mid_val, 0.0))
    ans_p_le_z = jnp.where(r < z - p, 0.0, jnp.where(r < z + p, mid_val, 0.0))
    return jnp.where(p > z, ans_p_gt_z, ans_p_le_z)

@jax.jit
def ellipse_angle_macula(r, alpha, beta):
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
    safe_r = jnp.where(r == 0.0, eps, r)
    
    val_arccos = (z - (-z + jnp.sqrt(val_inside_sqrt)) / safe_A) / safe_r
    ans_intersect = jnp.arccos(jnp.clip(val_arccos, -1.0, 1.0))
    
    ans_main = jnp.where(r < bound1, jnp.pi,
                 jnp.where(r < bound2, 0.0,
                 jnp.where(r < bound3, ans_intersect, 0.0)))
    
    ans_b0 = jnp.where(r >= z, jnp.arccos(jnp.clip(z / safe_r, -1.0, 1.0)), 0.0)
    ans = jnp.where(b_0_mask, ans_b0, ans_main)
    ans = jnp.where(alpha <= 0.0, 0.0, ans)
    return jnp.where(concentric_mask, jnp.where(r < a, jnp.pi, 0.0), ans)


@jax.jit
def spot_sky_coords(phi, lam, period, kappa, t):
    t_2d = t.reshape(-1, 1)
    phi_2d = phi.reshape(1, -1)
    lam_2d = lam.reshape(1, -1)
    
    phase = (2. * jnp.pi * t_2d / period) * (1. - kappa * jnp.sin(phi_2d)**2) + lam_2d
    
    z_los = jnp.cos(incl) * jnp.sin(phi_2d) + jnp.sin(incl) * jnp.cos(phi_2d) * jnp.cos(phase)
    x_sky = jnp.cos(phi_2d) * jnp.sin(phase)
    y_sky = jnp.sin(incl) * jnp.sin(phi_2d) - jnp.cos(incl) * jnp.cos(phi_2d) * jnp.cos(phase)
    

    x_orb = x_sky * jnp.cos(spin_orbit) + y_sky * jnp.sin(spin_orbit)
    y_orb = -x_sky * jnp.sin(spin_orbit) + y_sky * jnp.cos(spin_orbit)
    
    return x_orb, y_orb, jnp.clip(z_los, -1.0 + eps, 1.0 - eps)

@jax.jit
def planet_sky_coords(period_orb, t0, b, r_p, ecc, omega, t):

    orbit = KeplerianOrbit(period=period_orb, time_transit=t0, impact_param=b, radius=r_p, eccentricity=ecc, omega_peri=omega)
    x_p, y_p, z_p_los = orbit.relative_position(t)

    z_p = jnp.where(z_p_los > 0, jnp.sqrt(x_p**2 + y_p**2), 1000.0)
    return x_p, y_p, z_p


@jax.jit
def spotted_transit_flux(phi, lam, period, kappa, radius, t_ref, ing, eg, life, t, px, py, pz, r_p):

    alpha_arr = alpha_t(radius, t_ref, ing, eg, life, t)
    sx, sy, sz_los = spot_sky_coords(phi, lam, period, kappa, t)
    beta_arr = jnp.arccos(sz_los)
    

    def calc_single_spot(alpha, beta, ex, ey, sz_val, px_val, py_val, pz_val):
        visible = (sz_val > 0) & (alpha > 0)
        

        sa_t = jnp.where(visible, ellipse_angle_macula(r_annuli, alpha, beta), 0.0)
        pa_t = circleangle(r_annuli, r_p, pz_val)
        

        z_ell = jnp.sqrt(ex**2 + ey**2)
        d_sq = (px_val - ex)**2 + (py_val - ey)**2
        
        denom = jnp.where(2.0 * pz_val * z_ell == 0.0, 1.0, 2.0 * pz_val * z_ell)
        ps_angle = jnp.where((z_ell == 0.0) | (pz_val == 0.0), 0.0, 
                             jnp.arccos(jnp.clip((pz_val**2 + z_ell**2 - d_sq) / denom, -1.0, 1.0)))
        
        s_c = f_spot - 1.0
        

        oot_contrib = jnp.sum(s_c * sa_t * r_annuli * f_r)
        

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
        transit_contrib = jnp.sum(updates * r_annuli * f_r)
        
        return oot_contrib, transit_contrib


    v_calc = jax.vmap(jax.vmap(calc_single_spot, in_axes=(0,0,0,0,0,None,None,None)), 
                                                 in_axes=(0,0,0,0,0,0,0,0))
    

    ex_arr = sx * jnp.cos(alpha_arr)
    ey_arr = sy * jnp.cos(alpha_arr)
    
    oot_c, transit_c = v_calc(alpha_arr, beta_arr, ex_arr, ey_arr, sz_los, px, py, pz)
    return oot_c, transit_c


@jax.jit
def relative_transit_flux(phi, lam, period_rot, kappa, radius, t_ref, ing, eg, life, t, 
                          period_orb, t0, b, r_p, ecc, omega):


    px, py, pz = planet_sky_coords(period_orb, t0, b, r_p, ecc, omega, t)
    

    pa_t = jax.vmap(circleangle, in_axes=(None, None, 0))(r_annuli, r_p, pz)
    planet_transit_flux = jnp.sum((jnp.pi - pa_t) * r_annuli * f_r, axis=1)
    

    oot_c, transit_c = spotted_transit_flux(phi, lam, period_rot, kappa, radius, t_ref, ing, eg, life, t, px, py, pz, r_p)
    

    oot_base = oot_base_flux + jnp.sum(oot_c, axis=1)
    transit_flux = planet_transit_flux + jnp.sum(transit_c, axis=1)
    

    transit_flux = jnp.where(pz >= 1.0 + r_p, oot_base, transit_flux)
    
    return (transit_flux / oot_base) - 1.0
