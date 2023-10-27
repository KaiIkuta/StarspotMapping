#Calculation of spotted flux with any number of variable circular spots as in Ikuta et al. (2020)
#This code is based on macula code described by analytic formulae in Kipping (2012)
#Released version 0.0.0 on 2023/10/27



import numpy as np

#The example parameters are set in the case of the two-spot model for EV Lac Cycle 2 (Table 6 in Ikuta et al. 2023)

#Spot relative intensity
f_spot = 0.48
#Stellar inclination
incl = 60
incl = np.deg2rad(incl)


#Stellar equatorial rotation period
period = 4.3573
#The degree of differential rotation
kappa = 0.002

#Latitude, longitude, radius for each spot
lat = np.array([24.89,55.89])
lon = np.array([60.35,-84.11])
rad = np.array([13.44,14.36])
##You can calculate the relative flux as  "relative_flux(lat, lon, rad)" from the functions below (Figure 1e and 9a in Ikuta et al. 2023)



#The parameters are set for stable spots in Ikuta et al. (2023)
#If variable spot is needed as Equation 14 in Ikuta et al. (2020), you can change the parameters
#Emergence, decay, and stable duration
ing = np.array([1e12,1e12])
eg = np.array([1e12,1e12])
life = np.array([0, 0])
#Reference time
t_ref = np.array([27/2., 27/2.])

#time
t = np.linspace(0,27,100)

#Quadratic limb-darkening case (Appendix A in Ikuta 23)
#u1, u2 = 0.16, 0.44
#ld_star = np.array([0,u1+u2*2.,0,-u2])

#Nonlinear limb-darkening case (Table 6 in Ikuta 23)
ld_star = np.array([3.00,-4.54,4.01,-1.35])


ld_spot = np.copy(ld_star)



#Flux by the stellar limb-darkening law (the first and second terms in Equation 10 in Ikuta et al. 2020)
flux_lim = 1 - ld_star[0]/5. - ld_star[1]*2./6. - ld_star[2]*3./7. -ld_star[3]*4./8.




#The coordinate to the line of sight (Equation 15 in Ikuta 20) with the shape of (data point, spot_number)
def cos_beta(phi,lam):
    z = np.cos(incl)*np.sin(phi).reshape(1,-1)
    z = np.dot(np.ones([t.size]).reshape(-1,1),z)
    phase = np.dot((2.*np.pi*t/period).reshape(-1,1),(1-kappa*np.sin(phi)*np.sin(phi)).reshape(1,-1))
    z = z+np.sin(incl)*np.cos(phi)*np.cos(lam)*np.cos(phase)
    z = z-np.sin(incl)*np.cos(phi)*np.sin(lam)*np.sin(phase)
    z = np.where(z<-1.,-1.,z)
    z = np.where(z>1.,1.,z)
    return z


#Projected area (Equation 11 in Ikuta 20) with the shape of (data point, spot_number)
#The example case (Figure 6b in Ikuta 23)
def projected_area(alpha,beta):
    # the middle term of the second case
    area = -np.cos(alpha)*np.cos(beta)/(np.sin(alpha)*(np.sin(beta)+1e-9)) 
    area = np.where(area<-1.,-1.,area)
    area = np.where(area>1.,1.,area)
    area = np.arccos(area)
    area *= np.cos(beta)*np.sin(alpha)*np.sin(alpha)

    beta_alpha = np.cos(alpha)/(np.sin(beta)+1e-9)
    beta_alpha = np.where(beta_alpha>1.,1.,beta_alpha)
    # former and latter term of the second case (in case of the first/third case, return 0)
    area += np.arccos(beta_alpha) - np.cos(alpha)*np.sin(beta)*np.sqrt(1-beta_alpha*beta_alpha+1e-9)
    return area/np.pi


#Angular radius of spot (Equation 14 in Ikuta 20) with the shape of (data point, spot_number)
def alpha_t(radius):
  t_before = np.dot(t.reshape(-1,1),np.ones([radius.size]).reshape(1,-1))
  t_before = ((t_before-t_ref)+life/2.)/ing
  t_after = np.dot(t.reshape(-1,1),np.ones([radius.size]).reshape(1,-1))
  t_after = ((t_ref-t_after)+life/2.)/eg
  s = np.where(t_before<t_after,t_before,t_after) +1.
  s = np.where(s<1.,s,1.)
  s *= np.deg2rad(radius)
  s = np.where(s>0.,s,0.)
  return s

#The third term of the flux from spots (Equation 10 in Ikuta 20) with the shape of (data point, spot_number)
def spotted_flux(phi, lam, radius):
    alpha = alpha_t(radius)
    beta =np.arccos(cos_beta(np.deg2rad(phi),np.deg2rad(lam)))

    zeta_pos = np.cos(beta+alpha)
    zeta_pos = np.where(zeta_pos<0.,0.,zeta_pos)

    beta_alpha = beta-alpha
    beta_alpha = np.where(beta_alpha<0.,0.,beta_alpha)
    zeta_neg = np.cos(beta_alpha)
    zeta_neg = np.where(zeta_neg<0.,0.,zeta_neg)

    flux_spot = ((1.-np.sum(ld_star))-(1.-np.sum(ld_spot))*f_spot)
    flux_spot += 2./3.*(ld_star[1]-ld_spot[1]*f_spot)*(zeta_pos*zeta_pos+zeta_pos*zeta_neg+zeta_neg*zeta_neg)/(zeta_pos+zeta_neg+1e-9)
    flux_spot += 1./2.*(ld_star[3]-ld_spot[3]*f_spot)*(zeta_pos*zeta_pos+zeta_neg*zeta_neg)
    #If nonlinear limb-darkening law
    flux_spot += 4./5.*(ld_star[0]-ld_spot[0]*f_spot)*(zeta_pos*zeta_pos*np.sqrt(zeta_pos)-zeta_neg*zeta_neg*np.sqrt(zeta_neg))/(zeta_pos*zeta_pos-zeta_neg*zeta_neg+1e-9)
    flux_spot += 4./7.*(ld_star[2]-ld_spot[2]*f_spot)*(zeta_pos*zeta_pos*zeta_pos*np.sqrt(zeta_pos)-zeta_neg*zeta_neg*zeta_neg*np.sqrt(zeta_neg))/(zeta_pos*zeta_pos-(zeta_neg*zeta_neg+1e-9))
    return projected_area(alpha,beta)*flux_spot

#Relative flux (Equation 19 in Ikuta 20)
def relative_flux(phi, lam, radius):
    f = flux_lim - np.sum(spotted_flux(phi, lam, radius),axis=1) #Summation of contribution from all spots
    f_ave = np.mean(f)
    f = f/f_ave-1.
    return f
