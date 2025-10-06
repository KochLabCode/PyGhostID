# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 23:30:27 2025

@author: Daniel Koch
"""
#%% Import packages
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

options = {"node_color": "lightblue", "node_size": 220, "linewidths": 0.1, "width": 0.3}
inCm = 1/2.54 # convert inch to cm for plotting


# Testmodels
def saddleNodeBif_NF(t,z,para): # normal form of saddle-node bifurcation in 2D

    a=para

    dx= a + z[0]*z[0]
    dy= - z[1]
         
    return jnp.array([dx, dy])

def vanDerPol_2g(t,z,para):
    eps,alpha=para
    dx=(1/eps)*(z[1]-(1/3)*z[0]**3+z[0])
    dy = -z[0] + alpha*(z[1] - (1/3)*z[1]**3)
    return jnp.array([dx, dy])

def model_GRU_ghostCycle(t,h,para):
 
    alpha,beta = para
   
    dh1=(1/2)*(jnp.tanh((beta/2)*(jnp.cos(alpha)*h[0]-jnp.sin(alpha)*h[1]))-h[0])
    dh2=(1/2)*(jnp.tanh((beta/2)*(jnp.sin(alpha)*h[0]+jnp.cos(alpha)*h[1]))-h[1])
    return jnp.array([dh1, dh2])
 
alpha=0.15 # ghost
beta = 3

paraGC = [alpha,beta]


def ghostNetwork(t, x, para):
    x_off, a, b, c, d, tau, A = para  # unpack parameters

    dx = (a * (x + x_off)**3 + b * (x + x_off)**2 + c * (x + x_off) + d) / tau \
     + 6 * sigmoid(100 * (jnp.matmul(A, x) - 0.95))

    return dx
    
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

alpha=-0.2 # Saddle & Stable FP
F = lambda x: saddleNodeBif_NF(0,x,alpha) # for calculation of eigenvalues

# Define grid
xmin=-1;xmax=1
ymin=-1;ymax=1

Ng=150
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

  
######### PLOT 1: flow, Q-values and trajectory
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

######### PLOT 2: Eigenvalues
print("... eigenvalue distribution in slow areas of the phase space ...")

# Threshold for Q
Q_threshold = 0.018  # adjust as needed

# Mask grid where Q is above the threshold
ev1_masked = np.ma.masked_where(Q >= Q_threshold, ev1)
ev2_masked = np.ma.masked_where(Q >= Q_threshold, ev2)

plt.figure(figsize=(30*inCm,13*inCm))

# scale range
vmin = -1
vmax = 1

plt.subplot(1,2,1)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
plt.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

im = plt.imshow(ev1_masked, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=vmin, vmax=vmax)
plt.plot(sol.y[0,:],sol.y[1,:],lw=3,color='blue')
plt.colorbar(im, label='$\\lambda_1$')
plt.ylabel('$x_2$')

plt.subplot(1,2,2)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
plt.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

im = plt.imshow(ev2_masked, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=vmin, vmax=vmax)
plt.plot(sol.y[0,:],sol.y[1,:],lw=3,color='blue')
plt.colorbar(im, label='$\\lambda_2$')

plt.tight_layout()


######### PLOT 3: Q-values along trajectory
print("... Q-values along trajectory ...")

J_fun = jacfwd(F)

plt.figure(figsize=(6,6))

idcsQmin = []

Q_ts = []
for pt in sol.y.T:
    Q_ts.append(fun.qAtPt(saddleNodeBif_NF,alpha,pt))

Q_ts = np.asarray(Q_ts)
pQ = -np.log(Q_ts)

dur_ghost_min = 1 # set small to make saddle candidate of being identified as ghost

idx_minima, widths = find_peaks(pQ,width=dur_ghost_min/dt)

plt.plot(sol.t,Q_ts,'-k',lw=2)
plt.plot(sol.t[idx_minima],Q_ts[idx_minima],'xr',lw=2,ms=10)

plt.ylabel('$Q$',fontsize=14)
plt.xlabel('$time$',fontsize=14)
plt.yscale("log")


######### PLOT >3: Eigenvalues along trajectory segment
print("... eigenvalues along trajectory segment.")

ev_steps = 20

for i in range(len(idx_minima)):

    eigVals = []        
    for pt in sol.y.T[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps]:
        
        jac = J_fun(jnp.asarray(pt))
        
        eigVals.append(np.linalg.eigvals(jac))
        
    eigVals = np.asarray(eigVals)
    
    plt.figure()

    for ii in range(2):
        if abs(np.mean(eigVals[:,ii]))<0.1:
            plt.subplot(1,2,1)
            plt.title("$\\lambda_{i}$ close to 0")
        else:
            plt.subplot(1,2,2)
            plt.title("$\\lambda_{i}$ far from 0")
        plt.plot(sol.t[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps],eigVals[:,ii],'-',lw=2,label=f'$\\lambda_{ii+1}$')
        plt.ylabel('$\\lambda_{i}$',fontsize=14)
        plt.xlabel('$time$',fontsize=14)  
        plt.legend(fontsize=12)
    plt.tight_layout()
    

#%% ######## PLOT xxxx: Eigenvalues with Qminima

# Threshold for Q
Q_threshold = 0.018  # adjust as needed

# Mask grid where Q is above the threshold
ev1_masked = np.ma.masked_where(Q >= Q_threshold, ev1)
ev2_masked = np.ma.masked_where(Q >= Q_threshold, ev2)

plt.figure(figsize=(30*inCm,13*inCm))

plt.subplot(1,2,1)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
plt.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

im = plt.imshow(ev1_masked, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=ev1_masked.min(), vmax=ev1_masked.max())
plt.plot(sol.y[0,:],sol.y[1,:],lw=3,color='blue')
plt.plot(sol.y[0,idx_minima],sol.y[1,idx_minima],'xr',lw=2,ms=10)
plt.colorbar(im, label='$\\lambda_1$')
plt.ylabel('$x_2$')

plt.subplot(1,2,2)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
plt.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

im = plt.imshow(ev2_masked, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=-1, vmax=1)
plt.plot(sol.y[0,:],sol.y[1,:],lw=3,color='blue')
plt.plot(sol.y[0,idx_minima],sol.y[1,idx_minima],'xr',lw=2,ms=10)
plt.colorbar(im, label='$\\lambda_2$')

plt.tight_layout()
    
#%% ghost ID
# import functions as fun

Trj = sol.y.T
ghostSeq = fun.ghostID_v08(saddleNodeBif_NF,alpha,dt,Trj,0.05)
print("Analysis ghostID algorithm. Number of identified ghost states in saddle model: ", len(ghostSeq))
    
#%%################################
# Van der Pol without/with ghosts #
###################################

# We first show a phase space analysis for the relevant ghost criteria and then show that the algorithm does
# not pick up the slow manifold, but does pick up the two ghosts for high enough alpha

print("Perform manual analyses for modified Van der Pol model...")

F = lambda x: vanDerPol_2g(0,x,[eps,alpha]) # for calculation of eigenvalues

eps=0.1 # time scale separation

induceGhosts = False

if induceGhosts == False:
    alpha=0
    t_end = 4
else:
    alpha=3.145-0.01 #2 ghosts
    t_end = 50

# define grid
xmin=-2.5;xmax=2.5
ymin=-2.1;ymax=2.1

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# # Q-values on grid
Q = fun.qOnGrid(vanDerPol_2g,[eps,alpha],Xg,Yg)

# # Eigenvalues on grid
# ev1, ev2 = fun.eigValsOnGrid(F, Xg, Yg)
  
# simulate trajectory
dt = 0.01; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
sol = solve_ivp(vanDerPol_2g, (0,t_end), [-1,-1], rtol=1.e-6, atol=1.e-6,t_eval=time, args=([[eps,alpha]]), method='RK45') 

#%%####### PLOT 1: flow, Q-values and trajectory
print("... flow, Q-values and trajectory ...")

plt.figure(figsize=(13*inCm,11*inCm))
ax = plt.gca()


def flow_model(t,z):
    return vanDerPol_2g(t,z,[eps,alpha])

# Flow
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
# ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

# Define log scale range
vmin = 1e-2   # Avoid zero or negative values
vmax = 1000

im = plt.imshow(Q, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')

# plt.plot(sol.y[0,:],sol.y[1,:],'-',lw=0.5,color='m',alpha=1)

plt.xlabel('$x$'); plt.ylabel('$y$')
plt.xlim(xmin,xmax);plt.ylim(ymin,ymax)


#%%######## PLOT 2: Eigenvalues
print("... eigenvalue distribution in slow areas of the phase space ...")

# Threshold for Q
Q_threshold = 20  # adjust as needed

# Mask grid where Q is above the threshold
ev1_masked = np.ma.masked_where(Q >= Q_threshold, ev1)
ev2_masked = np.ma.masked_where(Q >= Q_threshold, ev2)

plt.figure(figsize=(30*inCm,13*inCm))

# scale range
vmin = -100
vmax = 100

plt.subplot(1,2,1)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
plt.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

im = plt.imshow(ev1_masked, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=vmin, vmax=vmax)
plt.plot(sol.y[0,:],sol.y[1,:],'-',lw=0.5,color='m',alpha=1)
plt.colorbar(im, label='$\\lambda_1$')
plt.ylabel('$x_2$')

plt.subplot(1,2,2)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
plt.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

im = plt.imshow(ev2_masked, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=vmin, vmax=vmax)
plt.plot(sol.y[0,:],sol.y[1,:],'-',lw=0.5,color='m',alpha=1)
plt.colorbar(im, label='$\\lambda_2$')

plt.tight_layout()



#%%######## PLOT 3: Q-values along trajectory
print("... Q-values along trajectory ...")

plt.figure(figsize=(6,6))

idcsQmin = []

Q_ts = []
for pt in sol.y.T:
    Q_ts.append(fun.qAtPt(vanDerPol_2g,[eps,alpha],pt))

Q_ts = np.asarray(Q_ts)
pQ = -np.log(Q_ts)

dur_ghost_min = 0.1
idx_minima, widths = find_peaks(pQ,width=dur_ghost_min/dt)

plt.plot(sol.t,Q_ts,'-k',lw=2)
plt.plot(sol.t[idx_minima],Q_ts[idx_minima],'xr',lw=2,ms=10)

plt.ylabel('$Q$',fontsize=14)
plt.xlabel('$time$',fontsize=14)
plt.yscale("log")

#%%######## PLOT >3: Eigenvalues along trajectory segment
print("... eigenvalues along trajectory segment.")

ev_steps = 40

F = lambda x: vanDerPol_2g(0, x, [eps,alpha])
J_fun = jacfwd(F)

def xNC(x):
    return x**3/3-x

xNC_pts = np.array([x_range,xNC(x_range)])


for i in range(3):

    eigVals = []        
    distXNC = []
    
    for pt in sol.y.T[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps]:
        
        jac = J_fun(jnp.asarray(pt))
        
        eigVals.append(np.linalg.eigvals(jac))
        
        distances = np.asarray([np.linalg.norm(xNC_pts[:,ii]-pt) for ii in range(xNC_pts.shape[1])])
        distXNC.append(np.min(distances))
        
    eigVals = np.asarray(eigVals)
    distXNC = np.asarray(distXNC)
    
    plt.figure()

    for ii in range(2):
        if abs(np.mean(eigVals[:,ii]))<0.1:
            plt.subplot(2,2,3)
            plt.plot(sol.t[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps],distXNC,'-k',label='distance to slow manifold')
            if not induceGhosts:
                plt.vlines(sol.t[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps][np.min(np.where(distXNC>0.05))],ymin=np.min(distXNC),ymax=np.max(distXNC),color='r')
            plt.xlabel('$time$',fontsize=14)
            plt.ylabel('distance to x-NC',fontsize=14)
            plt.subplot(2,2,1)
            plt.title("$\\lambda_{i}$ close to 0")
        else:
            plt.subplot(2,2,4)
            plt.plot(sol.t[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps],distXNC,'-k',label='distance to slow manifold')
            if not induceGhosts:
                plt.vlines(sol.t[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps][np.min(np.where(distXNC>0.05))],ymin=np.min(distXNC),ymax=np.max(distXNC),color='r')
            plt.xlabel('$time$',fontsize=14)
            plt.ylabel('distance to x-NC',fontsize=14)
            plt.subplot(2,2,2)
            plt.title("$\\lambda_{i}$ far from 0")
            
        plt.plot(sol.t[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps],eigVals[:,ii],'-',lw=2,label=f'$\\lambda_{ii+1}$')
        if not induceGhosts:
            plt.vlines(sol.t[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps][np.min(np.where(distXNC>0.05))],ymin=np.min(eigVals[:,ii]),ymax=np.max(eigVals[:,ii]),color='r')
        
        plt.ylabel('$\\lambda_{i}$',fontsize=14)
        plt.xlabel('$time$',fontsize=14)  
        plt.legend(fontsize=12)
    plt.tight_layout()

#%% ######## PLOT xxxx: Eigenvalues with Qminima
import matplotlib.patches as patches

# Threshold for Q
Q_threshold = 20  # adjust as needed

# Mask grid where Q is above the threshold
ev1_masked = np.ma.masked_where(Q >= Q_threshold, ev1)
ev2_masked = np.ma.masked_where(Q >= Q_threshold, ev2)

plt.figure(figsize=(30*inCm,13*inCm))

# # scale range
# vmin = -100
# vmax = 100

plt.subplot(1,2,1)
ax = plt.gca()
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)


vmax = max(abs(ev1_masked.max()),abs(ev1_masked.min()))

im = ax.imshow(ev1_masked, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=-vmax, vmax=vmax)
ax.plot(sol.y[0,:],sol.y[1,:],'-',lw=0.5,color='m',alpha=1)

# create circle patch
for i in idx_minima:
    circle = patches.Circle((sol.y[0,i], sol.y[1,i]), 0.05, fill=False, edgecolor='red', linewidth=1)
    # add to axes
    ax.add_patch(circle)

plt.colorbar(im, label='$\\lambda_1$')
ax.set_ylabel('$x_2$')

plt.subplot(1,2,2)
ax = plt.gca()

U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)

vmax = max(abs(ev2_masked.max()),abs(ev2_masked.min()))

im = ax.imshow(ev2_masked, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='coolwarm',vmin=-vmax, vmax=vmax)
# create circle patch
for i in idx_minima:
    circle = patches.Circle((sol.y[0,i], sol.y[1,i]), 0.05, fill=False, edgecolor='red', linewidth=1)
    # add to axes
    ax.add_patch(circle)

ax.plot(sol.y[0,:],sol.y[1,:],'-',lw=0.5,color='m',alpha=1)
plt.colorbar(im, label='$\\lambda_2$')

plt.tight_layout()


#%% ghost ID

Trj = sol.y.T
ghostSeq = fun.ghostID_v08(vanDerPol_2g,[eps,alpha],dt,Trj,0.05,peak_kwargs={"prominence":5},ctrlOutputs={"ctrl_qplot":True,"ctrl_evplot":False})

print("Analysis ghostID algorithm. Number of identified ghost states: ", len(ghostSeq))
#%%
if induceGhosts:
    
    plt.figure(figsize=(13*inCm,11*inCm))
    ax = plt.gca()
    
    # Flow
    U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
    ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)
    
    for g in ghostSeq:
        plt.plot(g["position"][0],g["position"][1],'o',color='red',ms=10)
    
    plt.plot(sol.y[0,:],sol.y[1,:],'-',lw=0.5,color='blue',alpha=0.5)
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.xlim(xmin,xmax);plt.ylim(ymin,ymax)
    
    # Turn sequences into adjacency matrix:
    
    ghostSeq_ = [ghostSeq]

    labels = []

    for s in ghostSeq_:
        for i in s:
            if i["id"][:1]=="G" and not i["id"] in labels:
                labels.append(i["id"])

    ng = len(labels)
    adjM_g = np.zeros((ng,ng))

    seqIDs = [[g["id"] for g in s] for s in ghostSeq_]

    for s in seqIDs:
        for i in range(len(s)-1):
            e_out = labels.index(s[i])
            e_in = labels.index(s[i+1])
            if adjM_g[e_out,e_in]==0:
                adjM_g[e_out,e_in]=1

    # plot
    A = adjM_g
    N = A.shape[0]
    nodeColors = [(0.8, 0.8, 1, 1.0)] * N # all nodes have same color

    fig = plt.figure(figsize=(7, 7))
    fun.drawNetwork(A, nodeColors, labels)
    plt.show()
    
#%%##################################################################################
#
# PART 2: Identification and visualization of ghost structures (channels, cycles) in
# example models
#
#####################################################################################

#########################################################################
# GRU ghost cycle, model from https://doi.org/10.3389/fncom.2021.678158 #
#########################################################################

dt = 0.05; t_end = 700; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  

ICs = [0.5,0.1]

sol = solve_ivp(model_GRU_ghostCycle, (0,t_end), ICs, rtol=1.e-6, atol=1.e-6,t_eval=time, args=([paraGC]), method='RK45') 

################################### Q along trajectory
# plt.figure(figsize=(6,6))

# idcsQmin = []

# Q_ts = []
# for pt in sol.y.T:
#     Q_ts.append(fun.qAtPt(model_GRU_ghostCycle,paraGC,pt))

# Q_ts = np.asarray(Q_ts)
# pQ = -np.log(Q_ts)

# dur_ghost_min = 10
# idx_minima, widths = find_peaks(pQ,width=dur_ghost_min/dt)

# plt.plot(sol.t,Q_ts,'-k',lw=2)
# plt.plot(sol.t[idx_minima],Q_ts[idx_minima],'xr',lw=2,ms=10)

# plt.ylabel('$Q$',fontsize=14)
# plt.xlabel('$time$',fontsize=14)
# plt.yscale("log")

################### Eigenvalues along trajectory segments (first 4 only)
# ev_steps = 20

# F = lambda x: model_GRU_ghostCycle(0, x, paraGC)
# J_fun = jacfwd(F)

# for i in range(4):

#     eigVals = []        
#     for pt in sol.y.T[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps]:
        
#         jac = J_fun(jnp.asarray(pt))
        
#         eigVals.append(np.linalg.eigvals(jac))
        
#     eigVals = np.asarray(eigVals)
    
#     plt.figure()

#     for ii in range(2):
#         if abs(np.mean(eigVals[:,ii]))<0.1:
#             plt.subplot(1,2,1)
#             plt.title("$\\lambda_{i}$ close to 0")
#         else:
#             plt.subplot(1,2,2)
#             plt.title("$\\lambda_{i}$ far from 0")
#         plt.plot(sol.t[idx_minima[i]-ev_steps:idx_minima[i]+ev_steps],eigVals[:,ii],'-',lw=2,label=f'$\\lambda_{ii+1}$')
#         plt.ylabel('$\\lambda_{i}$',fontsize=14)
#         plt.xlabel('$time$',fontsize=14)  
#         plt.legend(fontsize=12)
#     plt.tight_layout()
    
#%%############## ghostID #####################

Trj = sol.y.T
# ghostSeq = [fun.ghostID_v06(model_GRU_ghostCycle,paraGC,dt,Trj,plot_ctrl=True)]
ghostSeq = [fun.ghostID_v08(model_GRU_ghostCycle,paraGC,dt,Trj,0.05,peak_kwargs={"prominence":2},ctrlOutputs={"ctrl_qplot":True,"ctrl_evplot":False})
]
print("Analysis ghostID algorithm. Number of identified ghost states: ", len(ghostSeq[0]))

# Turn sequences into adjacency matrix:

labels = []

for s in ghostSeq:
    for i in s:
        if i["id"][:1]=="G" and not i["id"] in labels:
            labels.append(i["id"])

ng = len(labels)
adjM_g = np.zeros((ng,ng))

seqIDs = [[g["id"] for g in s] for s in ghostSeq]

for s in seqIDs:
    for i in range(len(s)-1):
        e_out = labels.index(s[i])
        e_in = labels.index(s[i+1])
        if adjM_g[e_out,e_in]==0:
            adjM_g[e_out,e_in]=1

# plot
A = adjM_g
N = A.shape[0]
nodeColors = [(0.8, 0.8, 1, 1.0)] * N # all nodes have same color

fig = plt.figure(figsize=(9, 9))
fun.drawNetwork(A, nodeColors, labels)
plt.show()

#%%######################
# Ghost networks        #
#########################

# define network topology

rnd_network = True # set to 'False' to analyse a simple linear chain leading to a ghost channel
rndNet = 2 # 0 no seed, 1: simple ghost structure, 2: complex ghost structure with oscillatory transients

if rndNet == 0:
    np.random.seed()
    n = np.random.randint(6,15)
    p_inhibitory = 0.4
elif rndNet == 1:
    rseed=5
    np.random.seed(rseed)
    n = 10
    p_inhibitory = 0.4
elif rndNet == 2:
    rseed=29
    np.random.seed(rseed)
    n = 15
    p_inhibitory = 0.6

if rnd_network:
    # define network topology
    if rndNet==0:
        G = nx.erdos_renyi_graph(n, 1/np.sqrt(n), directed=True)
    else:
        G = nx.erdos_renyi_graph(n, 1/np.sqrt(n), seed=rseed, directed=True)
        
    A = nx.to_numpy_array(G)

    # define colors for plotting
    colormap = plt.get_cmap('rainbow')
    colors = colormap(np.linspace(0, 1, n))

    # introduce inhibitory links
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] == 1:
                if np.random.rand() < p_inhibitory:
                    A[i,j] = -1
else:
    A = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0]
        ])

n = A.shape[0]

# set parameters
tau_ = np.ones(n) # timescales
paraGNet = [-2,-0.5,2,0,-4.743,tau_,A]

# timecourse simulation

dt = 0.02; t_end = 350; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  

if rnd_network:
    ICs1 = np.random.rand(n)*8
    ICs2 = np.random.rand(n)*8
    ICs3 = np.random.rand(n)*8
else:
    ICs1 = np.array([1,0,0])*8
    ICs2 = np.array([0,1,0])*8
    ICs3 = np.array([0,0,1])*8

sol1 = solve_ivp(ghostNetwork, (0,t_end), ICs1, rtol=1.e-6, atol=1.e-6,t_eval=time, args=([paraGNet]), method='RK45') 
sol2 = solve_ivp(ghostNetwork, (0,t_end), ICs2, rtol=1.e-6, atol=1.e-6,t_eval=time, args=([paraGNet]), method='RK45') 
sol3 = solve_ivp(ghostNetwork, (0,t_end), ICs3, rtol=1.e-6, atol=1.e-6,t_eval=time, args=([paraGNet]), method='RK45') 

#%% ghost ID
Trj1 = sol1.y.T
Trj2 = sol2.y.T
Trj3 = sol3.y.T

ghostSeq1 = fun.ghostID_v08(ghostNetwork,paraGNet,dt,Trj1,0.05,peak_kwargs={"prominence":6},ctrlOutputs={"ctrl_qplot":True,"ctrl_evplot":False})
ghostSeq2 = fun.ghostID_v08(ghostNetwork,paraGNet,dt,Trj2,0.05,peak_kwargs={"prominence":6})
ghostSeq3 = fun.ghostID_v08(ghostNetwork,paraGNet,dt,Trj3,0.05,peak_kwargs={"prominence":6})

# rename ghosts for consistency accross trajectories
   
allSeqs = [ghostSeq1,ghostSeq2,ghostSeq3]

allSeqs_relabeled = fun.unifyIDs(allSeqs.copy())

# plot timecourses with labels
ghostSeq1_,ghostSeq2_,ghostSeq3_ = allSeqs_relabeled
plt.figure(figsize=(9, 14.5))
plt.subplot(3,1,1)
for i in range(n):
    plt.plot(time,sol1.y[i,:])
for i in range(len(ghostSeq1_)):
    plt.text(ghostSeq1_[i]["time"]-10, 7, ghostSeq1_[i]["id"])
plt.ylim(-0.1,8)
plt.xlabel('time'); plt.ylabel('value');
plt.subplot(3,1,2)
for i in range(n):
    plt.plot(time,sol2.y[i,:])
for i in range(len(ghostSeq2_)):
    plt.text(ghostSeq2_[i]["time"]-10, 7, ghostSeq2_[i]["id"])
plt.ylim(-0.1,8)
plt.xlabel('time'); plt.ylabel('value');
plt.subplot(3,1,3)
for i in range(n):
    plt.plot(time,sol3.y[i,:])
for i in range(len(ghostSeq3_)):
    plt.text(ghostSeq3_[i]["time"]-10, 7, ghostSeq3_[i]["id"])
plt.ylim(-0.1,8)
plt.xlabel('time'); plt.ylabel('value');


# Turn sequences into adjacency matrix:

labels = []

for s in allSeqs_relabeled:
    for i in s:
        if i["id"][:1]=="G" and not i["id"] in labels:
            labels.append(i["id"])
for s in allSeqs_relabeled:
    for i in s:
        if i["id"][:1]=="O" and not i["id"] in labels:
            labels.append(i["id"])

ng = len(labels)

adjM_g = np.zeros((ng,ng))


seqIDs = [[g["id"] for g in s] for s in allSeqs_relabeled]

for s in seqIDs:
    for i in range(len(s)-1):
        e_out = labels.index(s[i])
        e_in = labels.index(s[i+1])
        if adjM_g[e_out,e_in]==0:
            adjM_g[e_out,e_in]=1
            
# plot ghost structure graph

A = adjM_g
N = A.shape[0]
nodeColors = [(0.8, 0.8, 1, 1.0)] * N # all nodes have same color

fig = plt.figure(figsize=(9, 9))
fun.drawNetwork(A, nodeColors, labels)
plt.show()
# %%
