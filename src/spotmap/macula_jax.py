#Analytical spotted model of macula.py with JAX 
#Implemented in Example_NumPyro.py

import jax
import jax.numpy as jnp


eps = 1e-15
###

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
    return jnp.clip(z, -1.0 + eps, 1.0 - eps) ###Fixed


#Projected area (Equation 11 in Ikuta 20) with the shape of (data point, spot_number)
#The example case (Figure 6b in Ikuta 23)
@jax.jit
def projected_area(alpha,beta):
    # the middle term of the second case
    area = -jnp.cos(alpha)*jnp.cos(beta)/(jnp.sin(alpha)*jnp.sin(beta)+eps)
    area = jnp.clip(area, -1.0 + eps, 1.0 - eps) ###Fixed
    area =jnp.arccos(area)
    area *=jnp.cos(beta)*jnp.sin(alpha)*jnp.sin(alpha)

    beta_alpha =jnp.cos(alpha)/(jnp.sin(beta)+eps)
    beta_alpha = jnp.clip(beta_alpha, -1.+eps, 1.-eps) ###Fixed
    # former and latter term of the second case (in case of the first/third case, return 0)
    area +=jnp.arccos(beta_alpha) -jnp.cos(alpha)*jnp.sin(beta)*jnp.sqrt(1.-beta_alpha*beta_alpha+eps)
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
    flux_area += 2./3.*flux_spot[1]*(zeta_pos*zeta_pos+zeta_pos*zeta_neg+zeta_neg*zeta_neg)/(zeta_pos+zeta_neg+eps)
    flux_area += 1./2.*flux_spot[3]*(zeta_pos*zeta_pos+zeta_neg*zeta_neg)
    #If nonlinear limb-darkening law
    flux_area += 4./5.*flux_spot[0]*(zeta_pos*zeta_pos*jnp.sqrt(zeta_pos+eps)-zeta_neg*zeta_neg*jnp.sqrt(zeta_neg+eps))/(zeta_pos*zeta_pos-zeta_neg*zeta_neg+eps) ###Fixed
    flux_area += 4./7.*flux_spot[2]*(zeta_pos*zeta_pos*zeta_pos*jnp.sqrt(zeta_pos+eps)-zeta_neg*zeta_neg*zeta_neg*jnp.sqrt(zeta_neg+eps))/(zeta_pos*zeta_pos-zeta_neg*zeta_neg+eps) ###Fixed
    return projected_area(alpha,beta)*flux_area

#Relative flux (Equation 19 in Ikuta 20)
@jax.jit
def relative_flux(phi, lam, period,kappa,radius,t_ref,ing,eg,life,t):
    f =  flux_lim -jnp.sum(spotted_flux(phi, lam, period,kappa,radius, t_ref,ing,eg,life,t),axis=1) #Summation of contribution from all spots
    f_ave =jnp.mean(f)
    f = f/f_ave-1.
    return f

