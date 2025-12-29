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

#################################################################################
#%% Figure 4a - single ghost in ecological model from Bieg et al. 2024
#################################################################################


# set parameters
a = 2.0; y = 0.7; m = 0.15; g = 0.4; Nt = 0.53; N0 = 0.5; r = 1.8; c = 0.25
parameters_bieg =  [a,Nt,N0,g,y,r,c,m]

# simulate trajectory 
dt = 0.05
timesteps = np.linspace(0,80,int(80/dt))
sol = solve_ivp(mod.bieg_etal, (0, 80), [0.05,0.8],
                    t_eval=timesteps, args=(parameters_bieg,),method='RK45')

# run ghostID
Trj=sol.y.T
ghostSeq, ctrlPlots = gid.ghostID(mod.bieg_etal,parameters_bieg,dt,Trj,0.05,peak_kwargs={"prominence":0,"width":50},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

# plot phase space
xmin=0;xmax=1
ymin=0;ymax=1

Ng=160
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

Q, coords = gid.qOnGrid(mod.bieg_etal,parameters_bieg,coords=[x_range,y_range], jit=True)

plt.figure(figsize=(5*inCm,7*inCm))
ax = plt.gca()

def flow_model(t,z): 
        return mod.bieg_etal(t,z,parameters_bieg)
U,V=fun.vector_field(flow_model,(Xg, Yg),dim='2D') 

mask = (Yg > 1 - Xg)   # region to exclude
U_masked = np.ma.masked_where(mask, U)
V_masked = np.ma.masked_where(mask, V)

ax.streamplot(
    Xg, Yg,
    U_masked, V_masked,
    density=0.8,
    color=[0.75, 0.75, 0.75, 1],
    arrowsize=0.9,
    linewidth=0.9)

# plot trajectory
ax.plot(sol.y[0],sol.y[1],'-',color='ivory',lw=2)

# plot Q-value
vmin = 1e-5 # Define log scale range 
vmax = 0.1 # Avoid zero or negative values 
Q_masked = np.ma.masked_where(mask, Q)
im = ax.imshow(Q_masked.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# nullclines
y_cNC = mod.bieg_etal_cNC(x_range,parameters_bieg)
y_mNC = mod.bieg_etal_mNC(x_range,parameters_bieg)
ax.plot(x_range,y_cNC,'-b',label='coral n.c.')
ax.plot(x_range,y_mNC,'--b',label='macroalgae n.c.')

# plot ghost
gx,gy = ghostSeq[0]["position"]
ax.plot(gx,gy,'ow',mec='m',markersize=8,alpha=0.75,label='ghost')

# add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax,label="Q")

#labels, limits, legend
ax.set_xlabel('Corals'); ax.set_ylabel('Macroalgae')
ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
ax.legend(fontsize=7)
plt.savefig("Figure4a_phaseSpace.svg")

#%%
# Plot control outputs from ghostID

# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(5*inCm,6*inCm)
plt.figure(fig)
plt.savefig("Figure4a_pQ.svg")
plt.show()

# eigenvalues across trajectories
fig, axes = ctrlPlots[1]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,2.5*inCm)
axes[0].set_xlim(0,7)
axes[1].set_xlim(0,7)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure4a_evsQ1.svg")
plt.show()

fig, axes = ctrlPlots[2]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,2.5*inCm)
axes[0].set_xlim(0,7)
axes[1].set_xlim(0,7)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure4a_evsQ2.svg")
plt.show()

#################################################################################
#%% Figure 4b - Ghost cycle in GRN from Farjami et al. 2021
#################################################################################

# set parameters
g_Farjami=1.5

# simulate trajectory 
t_end = 700
dt = 0.05
timesteps = np.linspace(0,t_end,int(t_end/dt))
sol = solve_ivp(mod.sys_Farjami2021, (0, t_end), [0.6,0.8,0.8],
                    t_eval=timesteps, args=(g_Farjami,),method='RK45')

# run ghostID
Trj=sol.y.T
ghostSeq, ctrlPlots = gid.ghostID(mod.sys_Farjami2021,g_Farjami,dt,Trj,0.05,peak_kwargs={"prominence":2,"width":500},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

# plot trajectory

fig = plt.figure(figsize=(5.5*inCm,7*inCm))
ax = fig.add_subplot(projection='3d')

simX,simY,simZ = sol.y[:,::2]
        
col = fun.euklideanVelocity(sol.y[:,::2].T, 1)
cmBounds = [col.min(), col.max()]
# cmBounds = [1e-6, 0.01]
norm = plt.Normalize(cmBounds[0],cmBounds[1])
cmap=plt.get_cmap('cool')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


for i in range(len(simX)-1): #
    ax.plot3D(simX[i],simY[i],simZ[i] ,'o', ms=2,color=np.asarray(cmap(norm(col[i]))[0:3]))
            
fun.noBackground(ax)
ax.view_init(20,45)

# plot ghost
for g in ghostSeq:
    gx,gy,gz = g["position"]
    ax.plot(gx,gy,gz,'ow',mec='m',markersize=7,alpha=0.75)

# add colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(
    sm,
    ax=ax,
    pad=0.1,
    shrink=0.5,
    aspect=20
)
cbar.set_label("velocity (a.u.)")

# #labels
ax.set_xlabel('x$_1$'); ax.set_ylabel('x$_2$');ax.set_zlabel('x$_3$')

plt.savefig("Figure4b_phaseSpace.svg")

# Plot control outputs from ghostID
# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(5*inCm,6*inCm)
plt.figure(fig)
plt.savefig("Figure4b_pQ.svg")
plt.show()

# eigenvalues across trajectories

fig, axes = ctrlPlots[1]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)#5.5
fig.set_size_inches(5,3.5)
fig.set_size_inches(5*inCm,4.25*inCm)
axes[0].set_ylim(-0.205,-0.195)
for i in range(3): axes[i].set_xlim(0,22)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[2].set_ylabel('$\\lambda_3$')
axes[2].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure4b_evsQ1.svg")
plt.show()

fig, axes = ctrlPlots[2]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)#5.5
fig.set_size_inches(5,3.5)
fig.set_size_inches(5*inCm,4.25*inCm)
axes[2].set_ylim(-0.25,-0.15)
for i in range(3): axes[i].set_xlim(0,22)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[2].set_ylabel('$\\lambda_3$')
axes[2].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure4b_evsQ2.svg")
plt.show()


#################################################################################
#%% Hastings Saddle Crawl-By
#################################################################################

# set parameters
gamma=2.5;h=1;v=0.5;m=0.4;alpha=0.8;K=15;eps=1
parameters_hastings =  [gamma,h,v,m,alpha,K,eps]

# simulate trajectory 
dt = 0.1
timesteps = np.linspace(0,1e3,int(1e3/dt))
sol = solve_ivp(mod.Hastings_etal, (0, 1e3), [0.05,0.1],
                    t_eval=timesteps, args=(parameters_hastings,),method='RK45')

# run ghostID
Trj=sol.y[:,int(900/dt):].T
ghostSeq, ctrlPlots = gid.ghostID(mod.Hastings_etal,parameters_hastings,dt,Trj,0.1,peak_kwargs={"prominence":0,"width":10},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

# plot phase space
xmin=-0.5;xmax=15.5
ymin=-0.5;ymax=15.5

Ng=160
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

Q, coords = gid.qOnGrid(mod.Hastings_etal,parameters_hastings,coords=[x_range,y_range], jit=True)

plt.figure(figsize=(5*inCm,7*inCm))
ax = plt.gca()

def flow_model(t,z): 
        return mod.Hastings_etal(t,z,parameters_hastings)
U,V=fun.vector_field(flow_model,(Xg, Yg),dim='2D') 

ax.streamplot(
    Xg, Yg,
    U,V,
    density=0.8,
    color=[0.75, 0.75, 0.75, 1],
    arrowsize=0.9,
    linewidth=0.9)

# plot trajectory
ax.plot(sol.y[0,int(900/dt):],sol.y[1,int(900/dt):],'-',color='ivory',lw=2)

# plot Q-value
vmin = 1e-1 # Define log scale range 
vmax = 1000 # Avoid zero or negative values 
im = ax.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), aspect=1, origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# # nullclines
y_NNC = mod.Hastings_NNC(x_range,parameters_hastings)
x_PNC = mod.Hastings_PNC(y_range,parameters_hastings)
ax.plot(x_range,y_NNC,'-b',label='Prey n.c.')
ax.plot(x_PNC,y_range,'--b',label='Predator n.c.')

# # add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax,label="Q")

# #labels, limits, legend
ax.set_xlabel('Prey'); ax.set_ylabel('Predator')
ax.set_xlim(xmin,xmax); ax.set_xticks([0,5,10,15])
ax.set_ylim(ymin,ymax); ax.set_yticks([0,5,10,15])
ax.legend(fontsize=7)
plt.savefig("Figure4c_phaseSpace.svg")


#%% # Plot control outputs from ghostID

# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(5*inCm,6*inCm)
plt.figure(fig)
plt.savefig("Figure4c_pQ.svg")
plt.show()

#%% # eigenvalues across trajectories
fig, axes = ctrlPlots[1]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,2.5*inCm)
axes[0].set_xlim(0,17)
axes[1].set_xlim(0,17)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure4c_evsQ1.svg")
plt.show()

fig, axes = ctrlPlots[2]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,2.5*inCm)
axes[0].set_xlim(0,2)
axes[1].set_xlim(0,2)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure4c_evsQ2.svg")
plt.show()
#%% Figure 4d - May Leonard Heteroclinic Cycle
#################################################################################

# set parameters

alpha = 0.8
beta = 1.29
parameters_ML=[alpha,beta]

# simulate trajectory 
t_end = 1200
dt = 0.1
timesteps = np.linspace(0,t_end,int(t_end/dt))
sol = solve_ivp(mod.May_Leonard, (0, t_end), [0.6,0.6,0.1],
                    t_eval=timesteps, args=(parameters_ML,),method='RK45',rtol=1e-10,atol=1e-10)

plt.plot(sol.t,sol.y[1,:])
# run ghostID
Trj=sol.y.T
ghostSeq, ctrlPlots = gid.ghostID(mod.May_Leonard,parameters_ML,dt,Trj,0.05,peak_kwargs={"prominence":2,"width":500},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #


# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(5*inCm,6*inCm)
plt.figure(fig)
# plt.savefig("Figure4b_pQ.svg")
plt.show()

#%% plot trajectory

fig = plt.figure(figsize=(5.5*inCm,7*inCm))
ax = fig.add_subplot(projection='3d')

simX,simY,simZ = sol.y[:,::4]
        
col = fun.euklideanVelocity(sol.y[:,::4].T, 1)
# cmBounds = [col.min(), col.max()]
cmBounds = [1e-4, 1e-1]
# norm = plt.Normalize(cmBounds[0],cmBounds[1])

norm = LogNorm(vmin=cmBounds[0], vmax=cmBounds[1])
cmap=plt.get_cmap('cool')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


for i in range(len(simX)-1): #
    ax.plot3D(simX[i],simY[i],simZ[i] ,'o', ms=2,color=np.asarray(cmap(norm(col[i]))[0:3]))
            
fun.noBackground(ax)
ax.view_init(20,45)

# plot ghost
# for g in ghostSeq:
#     gx,gy,gz = g["position"]
#     ax.plot(gx,gy,gz,'ow',mec='m',markersize=7,alpha=0.75)

# add colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(
    sm,
    ax=ax,
    pad=0.1,
    shrink=0.5,
    aspect=20
)
cbar.set_label("velocity (a.u.)")

# #labels
ax.set_xlabel('N$_1$'); ax.set_ylabel('N$_2$');ax.set_zlabel('N$_3$')

# plt.savefig("Figure4d_phaseSpace.svg")

#%% Plot control outputs from ghostID
# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(5*inCm,6*inCm)
plt.figure(fig)
# plt.savefig("Figure4b_pQ.svg")
plt.show()

#%% eigenvalues across trajectories

fig, axes = ctrlPlots[3]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)#5.5
fig.set_size_inches(5,3.5)
fig.set_size_inches(5*inCm,4.25*inCm)
# axes[0].set_ylim(-0.205,-0.195)
# for i in range(3): axes[i].set_xlim(0,22)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[2].set_ylabel('$\\lambda_3$')
axes[2].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
# plt.savefig("Figure4b_evsQ1.svg")
plt.show()

fig, axes = ctrlPlots[2]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)#5.5
fig.set_size_inches(5,3.5)
fig.set_size_inches(5*inCm,4.25*inCm)
# axes[2].set_ylim(-0.25,-0.15)
# for i in range(3): axes[i].set_xlim(0,22)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[2].set_ylabel('$\\lambda_3$')
axes[2].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
# plt.savefig("Figure4b_evsQ2.svg")
plt.show()

# %%
