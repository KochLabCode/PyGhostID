import os
import sys

# Path to project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add src folder (parent of PyGhostID) to Python path
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)

# Add paper folder
paper_dir = os.path.join(root_dir, "paper")
sys.path.insert(0, paper_dir)

# Import core as part of PyGhostID package
from PyGhostID import core as gid
print(gid.__file__)
import utils_paper as fun
print(fun.__file__)
import models_paper as mod

# other imports
import numpy as np
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jacfwd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pylab


pylab.rcParams.update(fun.get_rcparams())
plt.rcParams.update({'font.family':'Arial'})

inCm = 1/2.54 # convert inch to cm for plotting

## load data for plotting
with open("Fig6_theta_bifurcation.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat_theta_cpld = np.asarray(data)

#################################################################################
#%% Figure 6a - ghost plane in coupled theta neuron model from Augustsson & Martens 2024
#################################################################################

# set parameters
n = 0.01; K = 0.1; pS = np.pi
parameters_theta =  [n,K,pS]

# simulate trajectory 
dt = 0.005
timesteps = np.linspace(0,30,int(30/dt))
sol = solve_ivp(mod.coupledThetaNeurons, (0, 30), [0.25,0.25],
                    t_eval=timesteps, args=(parameters_theta,),method='RK45',rtol=1e-4,atol=1e-6)

# run ghostID
Trj=sol.y.T
ghostSeq, ctrlPlots = gid.ghostID(mod.coupledThetaNeurons,parameters_theta,dt,Trj,peak_kwargs={"prominence":0,"width":50},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

#  plot phase space
xmin=0;xmax=2*np.pi
ymin=0;ymax=2*np.pi

Ng=200
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

Q, coords = gid.qOnGrid(mod.coupledThetaNeurons,parameters_theta,coords=[x_range,y_range], jit=True)

plt.figure(figsize=(5*inCm,7*inCm))
ax = plt.gca()

def flow_model(t,z): 
        return mod.coupledThetaNeurons(t,z,parameters_theta)
U,V=fun.vector_field(flow_model,(Xg, Yg),dim='2D') 

ax.streamplot(
    Xg, Yg,
    U, V,
    density=0.8,
    color=[0.75, 0.75, 0.75, 1],
    arrowsize=0.9,
    linewidth=0.9)

# plot trajectory
ax.plot(sol.y[0],sol.y[1],'-',color='ivory',lw=2)

# plot Q-value
vmin = 1e-2 # Define log scale range 
vmax = 10 # Avoid zero or negative values 
im = ax.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# nullclines
f1,f2 = mod.coupledThetaNeurons(0,jnp.array([Xg,Yg]),parameters_theta)
ax.contour(Xg, Yg, f1, levels=[0], colors='cyan', linewidths=1.5, linestyles='-')
ax.contour(Xg, Yg, f2, levels=[0], colors='deepskyblue', linewidths=1.5, linestyles='--')
ax.plot([], [], color='cyan', lw=1.5, linestyle='-', label=r'$\dot\theta_1 = 0$')
ax.plot([], [], color='deepskyblue', lw=1.5, linestyle='--', label=r'$\dot\theta_2 = 0$')

# plot ghost
gx,gy = ghostSeq[0]["position"]
ax.plot(gx,gy,'ow',mec='m',markersize=8,alpha=0.75,label=f'ghost (dimension={ghostSeq[0]["dimension"]})')

# add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax,label="Q")

#labels, limits, legend
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.set_xlim(xmin,xmax); ax.set_xticks([0,np.pi,2*np.pi]); ax.set_xticklabels(['0',r'$\pi$',r'$2\pi$'])
ax.set_ylim(ymin,ymax); ax.set_yticks([0,np.pi,2*np.pi]); ax.set_yticklabels(['0',r'$\pi$',r'$2\pi$'])
ax.legend(fontsize=7)
# plt.savefig("Figure6a_phaseSpace.svg")
plt.show()

# Plot control outputs from ghostID

# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(5*inCm,6*inCm)
plt.figure(fig)
# plt.savefig("Figure4a_pQ.svg")
plt.show()

# # eigenvalues across trajectories
fig, axes = ctrlPlots[1]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,2.25*inCm)
axes[0].set_xlim(0,3.5)
axes[1].set_xlim(0,3.5)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
# plt.savefig("Figure4a_evsQ1.svg")
plt.show()

#%% ##############################################################
ghost_start = ghostSeq[0]

positions_ghosts, paramVals, ghostSeqs =  gid.track_ghost_branch(ghostSeq[0], mod.coupledThetaNeurons, parameters_theta, 0, 20, 0.05, 25, dt, delta=0.35, icStep=0.2, mode="first", 
                             epsilon_gid=0.2,qmin_method="BFGS",evLimit=0.15,solve_ivp_method='RK45', peak_kwargs={"prominence":0},ctrlOutputs={"ctrl_qplot":False,"qplot_xscale":"linear","ctrl_evplot":False},rtol=1e-4,atol=1e-6,distQminThr=0.3)
# #%%
# positions2, paramVals2, ghostSeqs2 =  gid.track_ghost_branch(ghostSeq[0], mod.coupledThetaNeurons, parameters_theta, 0, 20, 0.05, 25, dt, delta=0.4, icStep=0.3, mode="closest", 
#                              epsilon_gid=0.2,qmin_method="BFGS",evLimit=0.15,solve_ivp_method='RK45', peak_kwargs={"prominence":0},ctrlOutputs={"ctrl_qplot":False,"qplot_xscale":"linear","ctrl_evplot":False},rtol=1e-4,atol=1e-6,distQminThr=0.3)

#%% # Plot
plt.figure(figsize=(5*inCm,4.5*inCm))

# # plot SNs from XPPAUT
id_SN = 0
id_SN_end = 61
plt.plot(dat_theta_cpld[id_SN:id_SN_end,3],dat_theta_cpld[id_SN:id_SN_end,6],'-r')
id_us_end = 117
plt.plot(dat_theta_cpld[id_SN_end:id_us_end,3],dat_theta_cpld[id_SN_end:id_us_end,6],':k')

trapping_durations = [ghostSeqs[i]["duration"] for i in range(len(ghostSeqs))]

plt.plot(paramVals, positions_ghosts[:,0],'-',color='grey', lw=0.5, zorder=1)
sc = plt.scatter(paramVals, positions_ghosts[:,0], c=trapping_durations, marker='o', s=3, norm=LogNorm(), cmap='cool_r', zorder=2)
cb = plt.colorbar(sc, label='Trapping duration')
cb.set_ticks([1, 0.1, 0.5, 5])
cb.set_ticklabels(['1', '0.1', '0.5', '5'])


plt.ylabel(r"$\theta_1$")
plt.xlabel(r"$\eta$")
plt.ylim(0,2*np.pi)
plt.xlim(-1,1)
ax = plt.gca()
ax.set_yticks([0,np.pi,2*np.pi])
ax.set_yticklabels(['0',r'$\pi$',r'$2\pi$']);

#%% Mutual information

#################################################################################
#%% Figure 6b - Ghost structures in tipping element model from Wunderling et al 2024
#################################################################################

# set parameters
d = 0.15
GMT = 1.51
Tcrits = np.array([1.5,1.5])
Taus = np.array([500,5000])
interactions=np.array([[0,1],
                       [1,0]])
parameters_Wunderling = [d,GMT,Tcrits,Taus,interactions]

# # simulate trajectory 
dt = 5
timesteps = np.linspace(0,1e5,int(1e5/dt))
sol = solve_ivp(mod.wunderling_model, (0, 1e5), [-0.55,-1.5],
                    t_eval=timesteps, args=(parameters_Wunderling,),method='RK45',rtol=1e-4,atol=1e-6)

#%% run ghostID
Trj=sol.y.T
# ghostSeq,ctrlPlots = gid.ghostID(mod.wunderling_model,parameters_Wunderling,dt,Trj,peak_kwargs={"prominence":5,"width":50*dt},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #
ghostSeqs = gid.ghostID(mod.wunderling_model,parameters_Wunderling,dt,Trj,0.02,peak_kwargs={"prominence":2,"width":50*dt},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":False},evLimit=0.05) #


#%%
import time
start = time.time()
result_pss = gid.ghostID_phaseSpaceSample(mod.wunderling_model,parameters_Wunderling,0,2e5,dt,
                                          [np.linspace(-1.5,1.5,100),np.linspace(-1.5,1.5,100)],n_samples=200,evLimit=0.05,
                                          peak_kwargs={"prominence":2,"width":50*dt},display_warnings=False,epsilon_gid=0.02,epsilon_unify=0.1)
end = time.time()
print(f"Execution time for ghostID_phaseSpaceSample: {end - start} seconds")

ghostList = gid.unique_ghosts(result_pss)

#%% #  plot phase space
xmin=-1.5;xmax=1.5
ymin=-1.5;ymax=1.5

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

Q, coords = gid.qOnGrid(mod.wunderling_model,parameters_Wunderling,coords=[x_range,y_range], jit=True)

plt.figure(figsize=(5*inCm,7*inCm))
ax = plt.gca()

def flow_model(t,z): 
        return mod.wunderling_model(t,z,parameters_Wunderling)
U,V=fun.vector_field(flow_model,(Xg, Yg),dim='2D') 

ax.streamplot(
    Xg, Yg,
    U, V,
    density=0.8,
    color=[0.75, 0.75, 0.75, 1],
    arrowsize=0.9,
    linewidth=0.9)

# # plot trajectory
ax.plot(sol.y[0],sol.y[1],'-',color='ivory',lw=2)

# plot Q-value
vmin = 1e-9 # Define log scale range 
vmax = 1e-5 # Avoid zero or negative values 
im = ax.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# # # nullclines
# f1,f2 = mod.nullclines_Wunderling(parameters_Wunderling,(-1.5,1.5),(-1.5,1.5))
# ax.contour(Xg, Yg, f1, levels=[0], colors='cyan', linewidths=1.5, linestyles='-')
# ax.contour(Xg, Yg, f2, levels=[0], colors='deepskyblue', linewidths=1.5, linestyles='--')
# ax.plot([], [], color='cyan', lw=1.5, linestyle='-', label=r'$\dot x_1 = 0$')
# ax.plot([], [], color='deepskyblue', lw=1.5, linestyle='--', label=r'$\dot x_2 = 0$')

# # plot ghosts
# for i in range(len(ghostList)):
#     gi = ghostList[i]
#     gx,gy = gi["position"]
#     ax.plot(gx,gy,'ow',mec=f'C{i}',markersize=8,alpha=0.75,label=f'ghost {gi["id"]} (dimension={gi["dimension"]})')

# # add colorbar
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax,label="Q")

# #labels, limits, legend
# ax.set_xlabel(r"$x_1$")
# ax.set_ylabel(r"$x_2$")
# ax.set_xlim(xmin,xmax); ax.set_xticks([-1.5,-1,-0.5,0,0.5,1,1.5]);
# ax.set_ylim(ymin,ymax); ax.set_yticks([-1.5,-1,-0.5,0,0.5,1,1.5]);
# ax.legend(fontsize=7)
# # plt.savefig("Figure6a_phaseSpace.svg")
# plt.show()

#%% 
# plot ghost connections

M, M_labels = gid.ghost_connections([ghostSeqs])

N = M.shape[0]
nodeColors = [(0.8, 0.8, 1, 1.0)] * N # all nodes have same color

#%%
fig = plt.figure()
gid.draw_network(M, nodeColors, M_labels)
# plt.show()

#%%
# # Plot control outputs from ghostID

# # pQ timeseries and Q-minima
# fig, ax = ctrlPlots[0]
# fig.set_size_inches(5*inCm,6*inCm)
# plt.figure(fig)
# # plt.savefig("Figure4a_pQ.svg")
# plt.show()

# # # eigenvalues across trajectories
# fig, axes = ctrlPlots[1]
# suptxt = fig._suptitle.get_text()
# fig.suptitle(suptxt,fontsize=5.5)
# fig.set_size_inches(5*inCm,2.25*inCm)
# # axes[0].set_xlim(0,3.5)
# # axes[1].set_xlim(0,3.5)
# axes[0].set_ylabel('$\\lambda_1$')
# axes[1].set_ylabel('$\\lambda_2$')
# axes[1].set_xlabel('t (along trajectory segment)')
# plt.figure(fig)
# # plt.savefig("Figure4a_evsQ1.svg")
# plt.show()
# %%
