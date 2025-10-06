# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 21:05:06 2025

@author: dkoch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import networkx as nx
import os
import sys
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jacfwd
from matplotlib.colors import LogNorm

#paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join( os.path.dirname( __file__ ), '..' ))

import functions_v08 as fun
 
import _utils_251004 as utils
 

options = {"node_color": "lightblue", "node_size": 220, "linewidths": 0.1, "width": 0.3}
inCm = 1/2.54 # convert inch to cm for plotting


# Testmodels
def saddleNodeBif_NF(t,z,para): # normal form of saddle-node bifurcation in 2D

    a=para

    dx= a + z[0]*z[0]
    dy= a + z[1]*z[1]
    # dy= - z[1]
         
    return jnp.array([dx, dy])

#%%##############################################################################################
#
# PART 1: 
# Show that the algorithm is specific for ghosts and oscillatory transients and
# does not pick up other origins of transient states such as saddles or slow/critical manifolds
# from slow-fast systems.
#
#################################################################################################

##########
# SADDLE #
##########

# We first show a phase space analysis for the relevant ghost criteria and then show that the algorithm does
# not pick up the saddle
print("Perform manual analyses for saddle model...")

alpha=0.01 # Saddle & Stable FP
F = lambda x: saddleNodeBif_NF(0,x,alpha) # for calculation of eigenvalues

# Define grid
xmin=-1;xmax=1
ymin=-1;ymax=1

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# Q-values on grid
Q = fun.qOnGrid(saddleNodeBif_NF,alpha,Xg,Yg)

# Eigenvalues on grid
ev1, ev2 = fun.eigValsOnGrid(F, Xg, Yg)

# simulate trajectory approaching saddle
dt = 0.01; t_end = 20; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
IC = [0.44721,-1]
sol = solve_ivp(saddleNodeBif_NF, (0,t_end), IC, rtol=1.e-6, atol=1.e-6,t_eval=time, args=([alpha]), method='RK45') 

  
#%% ######## PLOT 1: flow, Q-values and trajectory
print("... flow, Q-values and trajectory ...")

plt.figure(figsize=(13*inCm,11*inCm))
ax = plt.gca()

# Flow
def flow_model(t,z):
    return saddleNodeBif_NF(t,z,alpha)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

# Define log scale range
vmin = 1e-3   # Avoid zero or negative values
vmax = 1

im = plt.imshow(Q, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')

# trajectory
plt.plot(sol.y[0,:],sol.y[1,:],lw=3,color='blue')

plt.xlabel('$x$'); plt.ylabel('$y$')
plt.xlim(xmin,xmax);plt.ylim(ymin,ymax)

#%%######## PLOT 2: Eigenvalues


point = np.array([0.0,0.0])


J_fun = utils.make_jacfun(saddleNodeBif_NF, alpha)

eigValues, eigVectors = jnp.linalg.eig(J_fun(point))            # eigenvalues

eigVecs = np.asarray(eigVectors)



print("... eigenvalue distribution in slow areas of the phase space ...")

# Threshold for Q
Q_threshold = 0.0  # adjust as needed

# Mask grid where Q is above the threshold
ev1_masked = np.ma.masked_where(Q >= Q_threshold, ev1)
ev2_masked = np.ma.masked_where(Q >= Q_threshold, ev2)

plt.figure(figsize=(30*inCm,13*inCm))

# scale range
vmin = -1
vmax = 1

plt.subplot(1,2,1)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D') 
   
plt.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,0.5],arrowsize=1.2,linewidth=1.1)

im = plt.imshow(ev1, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=vmin, vmax=vmax)
# plt.plot(sol.y[0,:],sol.y[1,:],lw=3,color='blue')
plt.colorbar(im, label='$\\lambda_1$')
plt.ylabel('$x_2$')


plt.scatter(*point, color='k')  # base point
for i in range(2):
    plt.quiver(*point, *eigVecs[:, i], scale=5, angles='xy', scale_units='xy', color=f'C{i+2}', label=f'λ={eigValues[i]:.7f}')

plt.legend(loc='upper left')

plt.subplot(1,2,2)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
plt.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,0.5],arrowsize=1.2,linewidth=1.1)

im = plt.imshow(ev2, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=vmin, vmax=vmax)
# plt.plot(sol.y[0,:],sol.y[1,:],lw=3,color='blue')
plt.colorbar(im, label='$\\lambda_2$')

##################################################
plt.scatter(*point, color='k')  # base point
for i in range(2):
    plt.quiver(*point, *eigVecs[:, i], scale=5, angles='xy', scale_units='xy', color=f'C{i+2}', label=f'λ={eigValues[i]:.7f}')


plt.legend(loc='upper left')

plt.tight_layout()
