import jax
import jax.numpy as jnp
from functools import partial

class SpotGeometry:
    def __init__(self, eps=1e-15):
        self.eps = eps

    @partial(jax.jit, static_argnums=(0,))
    def cos_beta(self, phi, lam, period, kappa, t, incl):
        t_2d = t[:, None]
        phi_2d = phi[None, :]
        lam_2d = lam[None, :]
        
        phase = (2. * jnp.pi * t_2d / period) * (1. - kappa * jnp.sin(phi_2d)**2) + lam_2d
        z = jnp.cos(incl) * jnp.sin(phi_2d) + jnp.sin(incl) * jnp.cos(phi_2d) * jnp.cos(phase)
        return jnp.clip(z, -1.0 + self.eps, 1.0 - self.eps)

    @partial(jax.jit, static_argnums=(0,))
    def alpha_t(self, radius, t_ref, ing, eg, life, t):
        t_2d = t[:, None]
        radius_2d = radius[None, :]
        
        t_before = ((t_2d - t_ref) + life / 2.) / ing
        t_after = ((t_ref - t_2d) + life / 2.) / eg
        
        s = jnp.where(t_before < t_after, t_before, t_after) + 1.
        s = jnp.clip(s, 0.0, 1.0) * radius_2d
        return s

    @partial(jax.jit, static_argnums=(0,))
    def projected_area(self, alpha, beta):
        area = -jnp.cos(alpha) * jnp.cos(beta) / (jnp.sin(alpha) * jnp.sin(beta) + self.eps)
        area = jnp.clip(area, -1.0 + self.eps, 1.0 - self.eps)
        area = jnp.arccos(area)
        area *= jnp.cos(beta) * jnp.sin(alpha) * jnp.sin(alpha)

        beta_alpha = jnp.cos(alpha) / (jnp.sin(beta) + self.eps)
        beta_alpha = jnp.clip(beta_alpha, -1. + self.eps, 1. - self.eps)
        
        area += jnp.arccos(beta_alpha) - jnp.cos(alpha) * jnp.sin(beta) * jnp.sqrt(1. - beta_alpha * beta_alpha + self.eps)
        return area / jnp.pi
