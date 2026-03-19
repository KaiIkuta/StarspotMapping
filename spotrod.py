#Spot-crossing code in Beky et al. (2014) for spot map in Kipping (2012)
#To be debugged and connected with the macula, jaxoplanet, and tinygp

import jax
import jax.numpy as jnp
from typing import Tuple

def integratetransit(
    planetx: jax.Array,
    planety: jax.Array,
    z: jax.Array,
    p: float,
    r: jax.Array,
    f: jax.Array,
    spotx: jax.Array,
    spoty: jax.Array,
    spotradius: jax.Array,
    spotcontrast: jax.Array,
    planetangle: jax.Array,
) -> jax.Array:
    """Calculate integrated flux of a spotty star transited by a planet."""
    
    k = spotx.shape[0]

    if k == 0:
        ootflux = jnp.pi * jnp.sum(r * f)
        flux_drop = jnp.sum((jnp.pi - planetangle) * r * f, axis=1) / ootflux
        return jnp.where(z < 1.0 + p, flux_drop, 1.0)

    spotcenterdistance = jnp.sqrt((spotx**2 + spoty**2) * (1.0 - spotradius**2))


    v_ellipseangle = jax.vmap(ellipseangle, in_axes=(None, 0, 0))
    spotangle = v_ellipseangle(r, spotradius, spotcenterdistance) 
  
    darkening = jnp.expand_dims(spotcontrast - 1.0, axis=1) * spotangle
    ootflux = jnp.sum((jnp.pi + jnp.sum(darkening, axis=0)) * r * f)


    def scan_body(carry, step_inputs):
        """1タイムステップ (t) 分のフラックス計算"""
        px, py, z_t, pa_t = step_inputs
        
        values = jnp.pi - pa_t

        dsquared = (px - spotx * jnp.sqrt(1.0 - spotradius**2))**2 + \
                   (py - spoty * jnp.sqrt(1.0 - spotradius**2))**2

        denom = 2.0 * z_t * spotcenterdistance
        safe_denom = jnp.where(denom == 0.0, 1.0, denom)
        arccos_val = (z_t**2 + spotcenterdistance**2 - dsquared) / safe_denom
        arccos_val = jnp.clip(arccos_val, -1.0, 1.0)

        ps_angle = jnp.where((spotcenterdistance == 0.0) | (z_t == 0.0),
                             0.0, jnp.arccos(arccos_val))


        ps_a = jnp.expand_dims(ps_angle, 1)          
        pa_a = jnp.expand_dims(pa_t, 0)              
        s_a = spotangle                              
        s_c = jnp.expand_dims(spotcontrast - 1.0, 1) 


        mask1 = ps_a > pa_a + s_a
        val1 = s_c * s_a

        mask2 = (~mask1) & (s_a > ps_a + pa_a)
        val2 = s_c * (s_a - pa_a)

        mask3 = (~mask1) & (~mask2) & (pa_a <= ps_a + s_a)
        mask3a = mask3 & ((2.0 * jnp.pi - ps_a) >= (pa_a + s_a))
        val3a = 0.5 * s_c * (s_a + ps_a - pa_a)

        mask3b = mask3 & (~mask3a)
        val3b = s_c * (jnp.pi - pa_a)

        updates = (mask1 * val1) + (mask2 * val2) + (mask3a * val3a) + (mask3b * val3b)

        values = values + jnp.sum(updates, axis=0)
        flux = jnp.sum(r * f * values) / ootflux
        

        flux_out = jnp.where(z_t >= 1.0 + p, 1.0, flux)

        return None, flux_out

    scan_inputs = (planetx, planety, z, planetangle)
    _, final_fluxes = jax.lax.scan(scan_body, None, scan_inputs)

    return final_fluxes


def elements(
    deltaT: jax.Array,
    period: float,
    a: float,
    k: float,
    h: float,
) -> Tuple[jax.Array, jax.Array]:
    """Calculate orbital elements eta and xi."""
    e = jnp.sqrt(k * k + h * h)
    l = 1.0 - jnp.sqrt(1.0 - k * k - h * h)

    lam_circ = 0.5 * jnp.pi + 2 * jnp.pi * deltaT / period
    eta_circ = a * jnp.sin(lam_circ)
    xi_circ = a * jnp.cos(lam_circ)

    Mdot = 2.0 * jnp.pi / period
    omega = jnp.arctan2(h, k)
    
    Emid = jnp.pi - jnp.arcsin(k * e / (jnp.sqrt(k * k + h * h * (1.0 - e * e)) + 1e-10)) \
           - jnp.arctan2(k, -h * jnp.sqrt(1.0 - k * k - h * h) + 1e-10)
    Mmid = Emid - e * jnp.sin(Emid)

    M = Mdot * deltaT + Mmid
    lam = M + omega

    tol = 10.0 * jnp.pi / (43200.0 * period)

    def cond_func(val):
        E, E_old = val
        return jnp.max(jnp.abs(E - E_old)) > tol

    def body_func(val):
        E, _ = val
        E_new = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E) + 1e-10)
        return E_new, E

    E_initial = M
    E_old_initial = M + 2.0 * tol
    E_final, _ = jax.lax.while_loop(cond_func, body_func, (E_initial, E_old_initial))

    p_val = e * jnp.sin(E_final)
    eta_ecc = a * (jnp.sin(lam + p_val) - k * p_val / (2.0 - l) - h)
    xi_ecc = a * (jnp.cos(lam + p_val) + h * p_val / (2.0 - l) - k)

    eta = jnp.where(e == 0.0, eta_circ, eta_ecc)
    xi = jnp.where(e == 0.0, xi_circ, xi_ecc)
    return eta, xi


def circleangle(r: jax.Array, p: float, z: jax.Array) -> jax.Array:
    """Calculate half central angle of arc of circle covered by planet."""
    safe_denom = jnp.where(2.0 * z * r == 0.0, 1e-10, 2.0 * z * r)
    cos_val = jnp.clip((r*r + z*z - p*p) / safe_denom, -1.0, 1.0)
    mid_val = jnp.arccos(cos_val)

    ans_p_gt_z = jnp.where(r < p - z, jnp.pi,
                 jnp.where(r < p + z, mid_val, 0.0))

    ans_p_le_z = jnp.where(r < z - p, 0.0,
                 jnp.where(r < z + p, mid_val, 0.0))

    return jnp.where(p > z, ans_p_gt_z, ans_p_le_z)


def ellipseangle(r: jax.Array, a: float, z: float) -> jax.Array:
    """Calculate half central angle of arc of circle covered by ellipse."""
    concentric_mask = (z <= 0.0)
    z = jnp.clip(z, 0.0, 1.0)
    zsquared = z * z
    asquared = a * a

    b_0_mask = (asquared + zsquared >= 1.0)
    
    denom_b = 1.0 - asquared
    safe_denom_b = jnp.where(denom_b == 0.0, 1.0, denom_b)
    inner_sqrt = jnp.maximum(0.0, 1.0 - zsquared / safe_denom_b)
    b = a * jnp.sqrt(inner_sqrt)

    denom_A = 1.0 - asquared - zsquared
    safe_denom_A = jnp.where(denom_A == 0.0, 1.0, denom_A)
    A = zsquared / safe_denom_A

    bound1 = b - z
    bound2 = z - b
    bound3 = b + z

    val_inside_sqrt = jnp.maximum(0.0, zsquared - A * (r*r - zsquared - asquared))
    safe_A = jnp.where(A == 0.0, 1.0, A)
    safe_r = jnp.where(r == 0.0, 1e-10, r)

    val_arccos = (z - (-z + jnp.sqrt(val_inside_sqrt)) / safe_A) / safe_r
    ans_intersect = jnp.arccos(jnp.clip(val_arccos, -1.0, 1.0))

    ans_main = jnp.where(r < bound1, jnp.pi,
                 jnp.where(r < bound2, 0.0,
                 jnp.where(r < bound3, ans_intersect, 0.0)))

    ans_b0 = jnp.where(r >= z, jnp.arccos(jnp.clip(z / safe_r, -1.0, 1.0)), 0.0)

    ans = jnp.where(b_0_mask, ans_b0, ans_main)
    ans = jnp.where(a <= 0.0, 0.0, ans)
    ans = jnp.where(concentric_mask, jnp.where(r < a, jnp.pi, 0.0), ans)

    return ans
