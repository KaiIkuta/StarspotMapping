#Visualized stellar hemisphere with any number of variable circular spots as Figures 13-14 (Ikuta et al. 2020) and Figures 5-9 (Ikuta et al. 2023) 
#This code works after the code in "macula.py"
#Released version 0.0.0 on 2023/11/17

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#Visualized time
t_vis = 0

fig = plt.figure(figsize=(6,6),tight_layout=True)
plt.rcParams['legend.fontsize']=32
plt.rcParams['legend.frameon'] = False

ax = fig.add_subplot(1,1,1)
ax.xaxis.set_label_position('top')
ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
ax.tick_params(bottom=False,left=False,right=False,top=False)

#The number of mesh point
mesh_bin = 300

#Stellar disk with the limb-darkening
r = np.linspace(0, 1, mesh_bin)
r = np.sqrt(r)
nu = np.linspace(0, 2*np.pi, mesh_bin)
r, nu = np.meshgrid(r, nu)
x, y = r*np.cos(nu), r*np.sin(nu)
mu = 1.0-x**2-y**2+1e-9
z = 1.-ld_star[0]*(1.0-np.sqrt(mu))-ld_star[1]*(1.0-mu)-ld_star[2]*(1.0-mu*np.sqrt(mu))-ld_star[3]*(1.0-mu*mu)
ax.pcolormesh(x, y, z,cmap='gist_gray',norm=Normalize(vmin=0.1, vmax=1.0),rasterized=True,zorder=0)
ax.set_xlim(-1.3,1.3)
ax.set_ylim(-1.3,1.3)

#Stellar rotation axis
ax.axvline(x=0.0,ymin=0.5+np.sin(incl)*0.5/1.3,ymax=0.5+0.5*1.2/1.3,color="gray",linewidth=2.0,rasterized=True,zorder=5)
ax.axvline(x=0.0,ymin=0.5-0.5*1.2/1.3,ymax=0.5-0.5*1.0/1.3,color="gray",linewidth=2.0,rasterized=True,zorder=-5)

#Spot on the surface as the solid angle
for i in range(lat.size):
    alpha = np.linspace(0,1.0,mesh_bin)
    rad_t = alpha_t(rad)[np.where(np.abs(t-t_vis)<2*cad)[0][0],i]
    alpha = np.arccos(1-(1-np.cos(rad_t))*alpha)
    nu = np.linspace(0.,2.*np.pi,mesh_bin)
    alpha, nu = np.meshgrid(alpha, nu)
    phi = np.deg2rad(lat[i])
    lam = np.deg2rad(lon[i])+(2.*np.pi*t_vis/period)*(1.-kappa*np.sin(phi)*np.sin(phi))
    #The coordinate to the line of sight (Equations A17-A19 in Kipping 12)
    dx = np.cos(lam)*np.sin(alpha)*np.sin(nu)+np.sin(lam)*(np.cos(phi)*np.cos(alpha)-np.sin(phi)*np.sin(alpha)*np.cos(nu))
    dy = np.cos(phi)*np.sin(alpha)*np.cos(nu)+np.sin(phi)*np.cos(alpha)
    dz = np.cos(lam)*np.cos(phi)*np.cos(alpha)-np.sin(alpha)*(np.sin(lam)*np.sin(nu)+np.cos(lam)*np.sin(phi)*np.cos(nu))
    dy, dz = dy*np.sin(incl)-dz*np.cos(incl), dy*np.cos(incl)+dz*np.sin(incl)
    if np.all(dz<0):
        continue
    mu = 1.0-dx**2-dy**2+1e-9
    z = 1.0-ld_spot[0]*(1.0-np.sqrt(mu))-ld_spot[1]*(1.0-mu)-ld_spot[2]*(1.0-mu*np.sqrt(mu))-ld_spot[3]*(1.0-mu*mu)
    z = np.where(dz>0.,z*f_spot,z)
    ax.scatter(dx[dz>=0],dy[dz>=0],c=z[dz>=0],cmap='gist_gray',norm=Normalize(vmin=0.1, vmax=1.0),rasterized=True,s=2,zorder=1)
    ax.set_rasterized(True)
