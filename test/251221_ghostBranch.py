# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 20:49:54 2025

@author: dkoch
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 23:30:27 2025

@author: Daniel Koch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import networkx as nx
import os
import sys
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jacfwd
from matplotlib.colors import LogNorm

#paths
# Path to project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add src folder (parent of PyGhostID) to Python path
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)

from PyGhostID import core as fun
from PyGhostID import _utils as utils

import functions_v08 as funOld

print(fun.__file__)
print(utils.__file__)
 

options = {"node_color": "lightblue", "node_size": 220, "linewidths": 0.1, "width": 0.3}
inCm = 1/2.54 # convert inch to cm for plotting

def wunderling_model(t, Z, para_model):
    Z = np.asarray(Z, dtype=np.float64) 
    # print(np.round(t,3), end = '\r')
    
    # Unpack parameters
    d = para_model['d']
    GMT  = para_model['GMT']
    Tcrits = para_model['Tcrits'] 
    Taus = para_model['Taus'] 
    mat_inter = para_model['mat_inter'] 

    # intrinsic = a * x * (1 - x**2)
    intrinsic = -Z**3 + Z + np.sqrt(4/27)*GMT/Tcrits
    # Coupling effects: sum over j of C_ij * x_j
    coupling = d/10* mat_inter @ (Z + 1)
    # Total derivative
    dZdt = (intrinsic + coupling)/Taus

    return np.asarray(dZdt)  


def wunderling_modeljx(t, Z, para_model):
    # print(np.round(t,3), end = '\r')
    
    # Unpack parameters
    d = para_model['d']
    GMT  = para_model['GMT']
    Tcrits = para_model['Tcrits'] 
    Taus = para_model['Taus'] 
    mat_inter = para_model['mat_inter'] 

    # intrinsic = a * x * (1 - x**2)
    intrinsic = -Z**3 + Z + np.sqrt(4/27)*GMT/Tcrits
    # Coupling effects: sum over j of C_ij * x_j
    coupling = d/10* mat_inter @ (Z + 1)
    # Total derivative
    dZdt = (intrinsic+coupling)/Taus

    return jnp.array(dZdt)


def ghostNetwork(t, x, para):
    x_off, a, b, c, d, tau, A = para  # unpack parameters

    dx = (a * (x + x_off)**3 + b * (x + x_off)**2 + c * (x + x_off) + d) / tau \
     + 6 * sigmoid(100 * (jnp.matmul(A, x) - 0.95))

    return dx

def vanDerPol_2g(t,z,para):
    eps,alpha=para
    dx=(1/eps)*(z[1]-(1/3)*z[0]**3+z[0])
    dy = -z[0] + alpha*(z[1] - (1/3)*z[1]**3)
    return jnp.array([dx, dy])


#%% ######################
# Wunderling model #######
##########################


para_model = {
    "d": 0.2,
    "GMT": 1.51,
    "mat_inter": np.array([[1,0],[1,0]]),
    "Tcrits": np.array([1.5,1.5]),
    "Taus": np.array([50,50])
    }

def flow_model(t,z):
    return wunderling_model(t,z,para_model)

# Define grid
xmin=-1.5;xmax=1.5
ymin=-1.5;ymax=1.5

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# Q-values on grid
Q, coords = fun.qOnGrid(wunderling_modeljx,para_model,coords=[x_range,y_range], jit=True)

x0s = [jnp.array([-0.7,-0.7]), jnp.array([1,-0.6]), jnp.array([-0.6,1])]

qminima = [fun.find_local_Qminimum(wunderling_modeljx, x0, para_model,tol_glob=1e-6,method="BFGS",verbose=False)[0] for x0 in x0s]

#%% phase space: Qmin, eigenvectors, ICs and trjs
plt.figure(figsize=(16*inCm,14*inCm))
plt.title(f"$\\tau_1 = {para_model['Taus'][0]},\\tau_2 = {para_model['Taus'][1]}$")
ax = plt.gca()

U,V=funOld.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,0.5],arrowsize=1.2,linewidth=1.1)


ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)

vmin = np.min(Q)   # Avoid zero or negative values
vmax = np.max(Q)
im = plt.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')
 
for i in range(3):
    plt.scatter(*qminima[i], color='w')  # qminima



for i in range(len(qminima)):
    
    ic, eigVals, eigVecs = utils.icAtQmin(qminima[i], -0.1,2, wunderling_modeljx, para_model)
    
    for ii in range(2):
        plt.quiver(*qminima[i], *eigVecs[:,ii], scale=5, angles='xy', scale_units='xy', color=f'C{2*i+ii}', label=f'λ={eigVals[ii]:.7f}')
        

    plt.scatter(*ic, marker='x',color='white')
    
    dt = 10
    t_eval = np.asarray(np.arange(0, 1e3, dt), dtype=np.float64)
    tF = t_eval[-1]

    sol = solve_ivp(wunderling_modeljx, [0,tF], np.real(ic),  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)
    ax.plot(sol.y[0,:],sol.y[1,:],'--w',lw=1)
    
    
plt.legend(loc='upper right')


#%% ghost ID from ICs

ghostSeqs=[]

for i in range(len(qminima)):
    
    ic, eigVals, eigVecs = utils.icAtQmin(qminima[i], -0.1,2,  wunderling_modeljx, para_model)
    
    dt = 10
    t_eval = np.asarray(np.arange(0, 1e3, dt), dtype=np.float64)
    tF = t_eval[-1]

    sol = solve_ivp(wunderling_modeljx, [0,tF], np.real(ic),  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)
    Trj=sol.y.T
    ghostSeq = fun.ghostID(wunderling_modeljx,para_model,dt,Trj,0.05,peak_kwargs={"prominence":0.5,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False}) #

    ghostSeqs.append(ghostSeq)
    

#%% 
gS_unified = fun.unifyIDs(ghostSeqs)

# gS_unified = fun.unifyIDs(ghostSeqs)


uniqueGhosts = fun.uniqueGhosts(gS_unified)

plt.figure(figsize=(16*inCm,14*inCm))
plt.title(f"$\\tau_1 = {para_model['Taus'][0]},\\tau_2 = {para_model['Taus'][1]}$")
ax = plt.gca()

U,V=funOld.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,0.5],arrowsize=1.2,linewidth=1.1)


ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)

vmin = np.min(Q)   # Avoid zero or negative values
vmax = np.max(Q)
im = plt.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')
 
for i in range(len(uniqueGhosts)):
    plt.scatter(*uniqueGhosts[i]["position"], color='w')  # qminima

uniqueGhosts.pop(1)

#%% wrapper

def wunderling_modeljx(t, Z, para_model):
    # print(np.round(t,3), end = '\r')
    
    # Unpack parameters
    d = para_model['d']
    GMT  = para_model['GMT']
    Tcrits = para_model['Tcrits'] 
    Taus = para_model['Taus'] 
    mat_inter = para_model['mat_inter'] 

    # intrinsic = a * x * (1 - x**2)
    intrinsic = -Z**3 + Z + np.sqrt(4/27)*GMT/Tcrits
    # Coupling effects: sum over j of C_ij * x_j
    coupling = d/10* mat_inter @ (Z + 1)
    # Total derivative
    dZdt = (intrinsic+coupling)/Taus

    return jnp.array(dZdt)

def wunderling_modeljx_listpara(t, Z, para_model):
   
    
    # Unpack parameters
    d, GMT, Tcrits, Taus, mat_inter = para_model
    
    para_model_ = {
        "d": d,
        "GMT": GMT,
        "mat_inter": mat_inter,
        "Tcrits": Tcrits,
        "Taus": Taus
        }   

    return wunderling_modeljx(t, Z, para_model_)


dt = 10
t_eval = np.asarray(np.arange(0, 1e3, dt), dtype=np.float64)
tF = t_eval[-1]

pars = [para_model[k] for k in ["d", "GMT", "Tcrits", "Taus", "mat_inter"]]
pars[1] = 1.61
sol = solve_ivp(wunderling_modeljx_listpara, [0,tF], np.real(ic),  method='LSODA', 
                                    args=(pars,), t_eval=t_eval)

plt.figure()
plt.plot(sol.t,sol.y[1,:],'-b',lw=1)

#%% ghostContinuation

pars = [para_model[k] for k in ["d", "GMT", "Tcrits", "Taus", "mat_inter"]]


gpos0, pars0, gSeq0 =  fun.ghostBranch1D(uniqueGhosts[0], wunderling_modeljx_listpara, pars, 1, 20, 0.1, 1e3, 5, delta=0.2, icStep=0.4, mode="closest", 
                             epsilon_gid=0.1,solve_ivp_method='LSODA', qmin_method="BFGS",qmin_tol=1e-6,peak_kwargs={"prominence":0.1,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False})

gpos1, pars1, gSeq1 =  fun.ghostBranch1D(uniqueGhosts[1], wunderling_modeljx_listpara, pars, 1, 20, 0.1, 1e3, 5, delta=0.2, icStep=0.4, mode="closest", 
                             epsilon_gid=0.1,solve_ivp_method='LSODA', qmin_method="BFGS",qmin_tol=1e-6,peak_kwargs={"prominence":0.1,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False})

gpos2, pars2, gSeq2 =  fun.ghostBranch1D(uniqueGhosts[2], wunderling_modeljx_listpara, pars, 1, 20, 0.1, 1e3, 5, delta=0.2, icStep=0.4, mode="closest", 
                             epsilon_gid=0.1,solve_ivp_method='LSODA', qmin_method="BFGS",qmin_tol=1e-6,peak_kwargs={"prominence":0.1,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False})



#%% '

plt.figure()
plt.plot(pars0,gpos0[:,1],'--o',color='gray')
plt.plot(pars1,gpos1[:,1],'-d',color='gray')
plt.plot(pars2,gpos2[:,1],':x',color='gray')
# plt.ylim(0,2.2)
plt.ylabel('y');plt.xlabel("$\\Delta GMT$")

# plt.figure()
# plt.plot(pars1,[gSeq1[i]["duration"] for i in range(len(gSeq1))],'-o',color='gray')
# # plt.ylim(0,2.2)
# plt.ylabel('trapping time');plt.xlabel("$\\Delta GMT$")



#%% ######################
# VdP 2G           #######
##########################

eps=0.1 # time scale separation
alpha=3.145-0.01 #2 ghosts

para_model = [eps,alpha]

def flow_model(t,z):
    return vanDerPol_2g(t,z,para_model)

# define grid
xmin=-2.5;xmax=2.5
ymin=-2.1;ymax=2.1

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# Q-values on grid
Q, coords = fun.qOnGrid(vanDerPol_2g,para_model,coords=[x_range,y_range], jit=True)

x0s = [jnp.array([-2,-1]), jnp.array([2,1])]

qminima = [fun.find_local_Qminimum(vanDerPol_2g, x0, para_model,tol_glob=1e-6,method="BFGS",verbose=False)[0] for x0 in x0s]



#%% phase space: Qmin, eigenvectors, ICs and trjs
plt.figure(figsize=(16*inCm,14*inCm))
# plt.title(f"$\\tau_1 = {para_model['Taus'][0]},\\tau_2 = {para_model['Taus'][1]}$")
ax = plt.gca()

U,V=funOld.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,0.5],arrowsize=1.2,linewidth=1.1)


ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)

vmin = np.min(Q)   # Avoid zero or negative values
vmax = np.max(Q)
im = plt.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')
 
for i in range(2):
    plt.scatter(*qminima[i], color='w')  # qminima


for i in range(len(qminima)):
    
    ic, eigVals, eigVecs = utils.icAtQmin(qminima[i], -0.1,1, vanDerPol_2g, para_model)
    
    for ii in range(2):
        plt.quiver(*qminima[i], *eigVecs[:,ii], scale=5, angles='xy', scale_units='xy', color=f'C{2*i+ii}', label=f'λ={eigVals[ii]:.7f}')
        

    plt.scatter(*ic, marker='x',color='white')
    
    dt = 0.01
    t_eval = np.asarray(np.arange(0, 20, dt), dtype=np.float64)
    tF = t_eval[-1]

    sol = solve_ivp(vanDerPol_2g, [0,tF], np.real(ic),  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)
    ax.plot(sol.y[0,:],sol.y[1,:],'--w',lw=1)
    
    
plt.legend(loc='lower right')



#%% ghost ID from ICs

ghostSeqs=[]

for i in range(len(qminima)):
    
    ic, eigVals, eigVecs = utils.icAtQmin(qminima[i], -0.1,1, vanDerPol_2g, para_model)
    
    dt = 0.01
    t_eval = np.asarray(np.arange(0, 20, dt), dtype=np.float64)
    tF = t_eval[-1]

    sol = solve_ivp(vanDerPol_2g, [0,tF], np.real(ic),  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)
    Trj=sol.y.T
    ghostSeq = fun.ghostID(vanDerPol_2g,para_model,dt,Trj,0.05,peak_kwargs={"prominence":0.5,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"log","ctrl_evplot":True}) #

    ghostSeqs.append(ghostSeq)
    
ghost_contStart = ghostSeq[0]

#%% numerical test of parameter range

plt.figure()


dt = 0.01
t_eval = np.asarray(np.arange(0, 20, dt), dtype=np.float64)
tF = t_eval[-1]

for i in range (1,10):

    para_model = [eps,alpha-i*0.05]
    
    sol = solve_ivp(vanDerPol_2g, [0,tF], np.real(ic),  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)
    plt.plot(sol.t,sol.y[1,:],'-',lw=1,label=f"$\\alpha$={alpha-i*0.02}")



#%% ghostContinuation

gpos, pars, gSeq =  fun.track_ghost_branch(ghost_contStart, vanDerPol_2g, para_model, 1, 20, -0.1, 10, 0.01, delta=0.2, icStep=0.6, mode="closest", 
                             epsilon_gid=0.1,solve_ivp_method='LSODA', qmin_method="BFGS",qmin_tol=1e-6,peak_kwargs={"prominence":0.5,"width":0},ctrlOutputs={"ctrl_qplot":False,"qplot_xscale":"linear","ctrl_evplot":False})

#%% '

plt.figure()
plt.plot(pars,gpos[:,1],'-o',color='gray')
plt.ylim(0,2.2)
plt.ylabel('y');plt.xlabel("$\\alpha$")

plt.figure()
plt.plot(pars,[gSeq[i]["duration"] for i in range(len(gSeq))],'-o',color='gray')
# plt.ylim(0,2.2)
plt.ylabel('trapping time');plt.xlabel("$\\alpha$")

#%%

# #%%######################
# # Ghost networks        #
# #########################

# define network topology

rnd_network = True # set to 'False' to analyse a simple linear chain leading to a ghost channel
rndNet = 0 # 0 no seed, 1: simple ghost structure, 2: complex ghost structure with oscillatory transients

if rndNet == 0:
    np.random.seed()
    n = 6
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

plt.figure(figsize=(9, 4))

for i in range(n):
    plt.plot(time,sol1.y[i,:])

plt.ylim(-0.1,8)
plt.xlabel('time'); plt.ylabel('value');
#%% ghost ID
Trj1 = sol1.y.T

ghostSeq1 = fun.ghostID(ghostNetwork,paraGNet,dt,Trj1,0.05,peak_kwargs={"prominence":6,"width":500},ctrlOutputs={"ctrl_qplot":True,"ctrl_evplot":False})

for i in range(n):
    plt.plot(time,sol1.y[i,:])
for i in range(len(ghostSeq1)):
    plt.text(ghostSeq1[i]["time"]-10, 7, ghostSeq1[i]["id"])
plt.ylim(-0.1,8)
plt.xlabel('time'); plt.ylabel('value');

#%% resample

start = ghostSeq1[0]["position"]

ic, eigVals, eigVecs = utils.icAtQmin(start, 1,2, ghostNetwork,paraGNet)

dt = 0.02
t_eval = np.asarray(np.arange(0, 100, dt), dtype=np.float64)
tF = t_eval[-1]
   
sol = solve_ivp(ghostNetwork, [0,tF], np.real(ic),  method='LSODA', 
                                    args=(paraGNet,), t_eval=t_eval)
Trj=sol.y.T
ghostSeq = fun.ghostID(ghostNetwork,paraGNet,dt,Trj,0.05,peak_kwargs={"prominence":6,"width":100},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False}) #
   
#%% Ghost continuation SN nf

        # model_params_[par_nr] = parNext
    

def saddleNodeBif_NF(t,z,para): # normal form of saddle-node bifurcation in 2D

    a=para[0]

    dx= a + z[0]*z[0]
    dy= - z[1]
         
    return jnp.array([dx, dy])


    

para_model = jnp.array([0.01])

def flow_model(t,z):
    return saddleNodeBif_NF(t,z,para_model)

# Define grid
xmin=-1.5;xmax=1.5
ymin=-1.5;ymax=1.5

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# Q-values on grid
Q, coords = fun.qOnGrid(saddleNodeBif_NF,para_model,coords=[x_range,y_range], jit=True)

x0s = [jnp.array([-0.05,0.05])]

qminima = [fun.find_local_Qminimum(saddleNodeBif_NF, x0, para_model,tol_glob=1e-6,method="BFGS",verbose=False)[0] for x0 in x0s]

#%% phase space: Qmin, eigenvectors, ICs and trjs
plt.figure(figsize=(16*inCm,14*inCm))
# plt.title(f"$\\tau_1 = {para_model['Taus'][0]},\\tau_2 = {para_model['Taus'][1]}$")
ax = plt.gca()

U,V=funOld.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,0.5],arrowsize=1.2,linewidth=1.1)


ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)

vmin = np.min(Q)   # Avoid zero or negative values
vmax = np.max(Q)
im = plt.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')
 
for i in range(1):
    plt.scatter(*qminima[i], color='w')  # qminima



for i in range(len(qminima)):
    
    ic, eigVals, eigVecs = utils.icAtQmin(qminima[i], -0.1,1, saddleNodeBif_NF, para_model)
    
    for ii in range(2):
        plt.quiver(*qminima[i], *eigVecs[:,ii], scale=5, angles='xy', scale_units='xy', color=f'C{2*i+ii}', label=f'λ={eigVals[ii]:.7f}')
        

    plt.scatter(*ic, marker='x',color='white')
    
    dt = 0.1
    t_eval = np.asarray(np.arange(0, 22, dt), dtype=np.float64)
    tF = t_eval[-1]

    sol = solve_ivp(saddleNodeBif_NF, [0,tF], np.real(ic),  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)
    ax.plot(sol.y[0,:],sol.y[1,:],'--r',lw=1)
    
    
plt.legend(loc='upper right')


#%% ghost ID from ICs

ghostSeqs=[]

for i in range(len(qminima)):
    
    ic, eigVals, eigVecs = utils.icAtQmin(qminima[i], -0.1,1, saddleNodeBif_NF, para_model)
    
    dt = 0.1
    t_eval = np.asarray(np.arange(0, 22, dt), dtype=np.float64)
    tF = t_eval[-1]

    sol = solve_ivp(saddleNodeBif_NF, [0,tF], np.real(ic),  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)
    Trj=sol.y.T
    ghostSeq = fun.ghostID(saddleNodeBif_NF,para_model,dt,Trj,0.05,peak_kwargs={"prominence":0.5,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True}) #

    ghostSeqs.append(ghostSeq)
    
    
    
    
#%% ghostContinuation

gpos, pars, gSeq =  fun.ghostBranch1D(ghostSeq[0], saddleNodeBif_NF, para_model, 0, 10, 0.02, 20, 0.01, delta=0.2, icStep=0.1,
                             epsilon_gid=0.1,solve_ivp_method='RK45', rtol=1.e-6, atol=1.e-6, qmin_method="BFGS",qmin_tol=1e-6,peak_kwargs={"prominence":0.5,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True})

#%% 

plt.figure()
plt.plot(pars,gpos[:,1],'--',color='gray')
plt.ylim(-0.1,0.1)
plt.ylabel('y');plt.xlabel("$\\alpha$")