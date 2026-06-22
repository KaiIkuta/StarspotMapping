#Spot-crossing code in Beky et al. (2014) for spot map in Kipping (2012)
#To be debugged and connected with the macula, jaxoplanet, and tinygp

import jax
import jax.numpy as jnp
from spotmap import macula_jax

eps = 1e-15

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
    
    z_sq = z * z
    a_sq = a * a
    b_0_mask = (a_sq + z_sq >= 1.0)
    
    denom_A = 1.0 - a_sq - z_sq
    safe_denom_A = jnp.where(denom_A == 0.0, 1.0, denom_A)
    A = z_sq / safe_denom_A
    
    bound1, bound2, bound3 = b - z, z - b, b + z
    
    val_inside_sqrt = jnp.maximum(0.0, z_sq - A * (r*r - z_sq - a_sq))
    safe_A = jnp.where(A == 0.0, 1.0, A)
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
def transit_chord_overlap(r, pz, r_p, ex, ey, z_ell, px, py, sa_t, f_spot):

    pa_t = circleangle(r, r_p, pz)
    values = jnp.pi - pa_t 
    
    d_sq = (px - ex)**2 + (py - ey)**2 # (N_spots,)
    denom = 2.0 * pz * z_ell
    safe_denom = jnp.where(denom == 0.0, 1.0, denom)
    
    ps_angle = jnp.where((z_ell == 0.0) | (pz == 0.0), 0.0, 
                         jnp.arccos(jnp.clip((pz**2 + z_ell**2 - d_sq) / safe_denom, -1.0, 1.0)))
                         
    ps_a = jnp.expand_dims(ps_angle, 1) # (N_spots, 1)
    pa_a = jnp.expand_dims(pa_t, 0)     # (1, N_r)
    s_c  = jnp.expand_dims(f_spot - 1.0, 1) 
    
    mask1 = ps_a > pa_a + sa_t
    val1 = s_c * sa_t
    
    mask2 = (~mask1) & (sa_t > ps_a + pa_a)
    val2 = s_c * (sa_t - pa_a)
    
    mask3 = (~mask1) & (~mask2) & (pa_a <= ps_a + sa_t)
    mask3a = mask3 & ((2.0 * jnp.pi - ps_a) >= (pa_a + sa_t))
    val3a = 0.5 * s_c * (sa_t + ps_a - pa_a)
    mask3b = mask3 & (~mask3a)
    val3b = s_c * (jnp.pi - pa_a)
    
    # 算術マスク演算による超高速化
    updates = (mask1 * val1) + (mask2 * val2) + (mask3a * val3a) + (mask3b * val3b)
    return values + jnp.sum(updates, axis=0)


@jax.jit
def relative_transit_flux(phi, lam, period, kappa, radius, t_ref, ing, eg, life, t, px, py, pz, r_p, r_annuli, f_r, f_spot_arr):

    alpha = alpha_t(radius, t_ref, ing, eg, life, t) # (N_t, N_spots)
    

    phase = (2.*jnp.pi*t/period).reshape(-1,1) * (1-kappa*jnp.sin(phi)*jnp.sin(phi)).reshape(1,-1) + lam.reshape(1,-1)
    
    z_los = jnp.cos(incl)*jnp.sin(phi).reshape(1,-1) + jnp.sin(incl)*jnp.cos(phi).reshape(1,-1)*jnp.cos(phase)
    x_sky = jnp.cos(phi).reshape(1,-1)*jnp.sin(phase)
    y_sky = jnp.sin(incl)*jnp.sin(phi).reshape(1,-1) - jnp.cos(incl)*jnp.cos(phi).reshape(1,-1)*jnp.cos(phase)
    
    z_los = jnp.clip(z_los, -1.0 + eps, 1.0 - eps)
    beta = jnp.arccos(z_los) # (N_t, N_spots)
    

    def scan_body(carry, step_inputs):
        px_t, py_t, pz_t, alpha_t_arr, beta_t_arr, sx_t, sy_t, sz_t = step_inputs

        visible = (sz_t > 0) & (alpha_t_arr > 0)
        

        ex_t = jnp.where(visible, sx_t * jnp.cos(alpha_t_arr), 0.0)
        ey_t = jnp.where(visible, sy_t * jnp.cos(alpha_t_arr), 0.0)
        z_ell = jnp.where(visible, jnp.sqrt(ex_t**2 + ey_t**2), 0.0)
        

        v_ellipse = jax.vmap(ellipse_angle_macula, in_axes=(None, 0, 0))
        sa_t = v_ellipse(r_annuli, alpha_t_arr, beta_t_arr) # (N_spots, N_r)
        sa_t = jnp.where(visible[:, None], sa_t, 0.0)
        

        s_c = jnp.where(visible, f_spot_arr - 1.0, 0.0)[:, None]
        darkening = s_c * sa_t
        ootflux_t = jnp.sum((jnp.pi + jnp.sum(darkening, axis=0)) * r_annuli * f_r)
        

        values_t = transit_chord_overlap(
            r_annuli, pz_t, r_p, ex_t, ey_t, z_ell, px_t, py_t, sa_t, jnp.where(visible, f_spot_arr, 1.0)
        )

        flux_t = jnp.sum(r_annuli * f_r * values_t) / ootflux_t
        

        ans = jnp.where(pz_t >= 1.0 + r_p, 1.0, flux_t)
        return None, ans

    scan_inputs = (px, py, pz, alpha, beta, x_sky, y_sky, z_los)
    _, final_fluxes = jax.lax.scan(scan_body, None, scan_inputs)
    
    return final_fluxes
