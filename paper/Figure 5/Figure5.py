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
#%% Figure 5a, Hastings et al. 2018, saddle crawl-by
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

if len(ghostSeq)==0: print("No ghosts for Hastings model in saddle crawl-by regime identified.")

# plot phase space
xmin=-0.5;xmax=15.5
ymin=-0.5;ymax=15.5

Ng=160
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

Q, coords = gid.qOnGrid(mod.Hastings_etal,parameters_hastings,coords=[x_range,y_range], jit=True)

plt.figure(figsize=(4.25*inCm,6*inCm))
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

# plot Q-value
vmin = 1e-1 # Define log scale range 
vmax = 1000 # Avoid zero or negative values 
im = ax.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), aspect=1, origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# # nullclines
y_NNC = mod.Hastings_NNC(x_range,parameters_hastings)
x_PNC = mod.Hastings_PNC(y_range,parameters_hastings)
ax.plot(x_range,y_NNC,'-b',label='Prey n.c.')
ax.plot(x_PNC,y_range,'--b',label='Predator n.c.')
ax.plot(0*y_range,y_range,'-b')
ax.plot(x_range,0*x_range,'--b') 

# plot trajectory
ax.plot(sol.y[0,int(900/dt):],sol.y[1,int(900/dt):],'-',color='ivory',lw=2)

# # add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax,label="Q")

# #labels, limits, legend
ax.set_xlabel('Prey'); ax.set_ylabel('Predator')
ax.set_xlim(xmin,xmax); ax.set_xticks([0,5,10,15])
ax.set_ylim(ymin,ymax); ax.set_yticks([0,5,10,15])
ax.legend(fontsize=7)
plt.savefig("Figure5a_phaseSpace.svg")
plt.show()

#% Plot control outputs from ghostID

# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(4.5*inCm,5.5*inCm)
plt.figure(fig)
plt.savefig("Figure5a_pQ.svg")
plt.show()

# eigenvalues across trajectories
fig, axes = ctrlPlots[1]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,2*inCm)
axes[0].set_xlim(0,17)
axes[1].set_xlim(0,17)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure5a_evsQ1.svg")
plt.show()

fig, axes = ctrlPlots[2]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,2*inCm)
axes[0].set_xlim(0,2)
axes[1].set_xlim(0,2)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure5a_evsQ2.svg")
plt.show()

#################################################################################
#%% Figure 5b, May-Leonard Heteroclinic Cycle
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


# run ghostID
Trj=sol.y.T
ghostSeq, ctrlPlots = gid.ghostID(mod.May_Leonard,parameters_ML,dt,Trj,1e-5,peak_kwargs={"prominence":2,"width":500},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True},eigval_NN_sorting=True) #
if len(ghostSeq)==0: print("No ghosts for May-Leonard model identified.")

# plot trajectory
fig = plt.figure(figsize=(6*inCm,3*inCm))
plt.plot(sol.t,sol.y[1,:],'-')
plt.ylabel('N$_1$'); plt.xlabel('t')

fig = plt.figure(figsize=(4.5*inCm,6.3*inCm))
ax = fig.add_subplot(projection='3d')

simX,simY,simZ = sol.y[:,::4]
        
col = fun.euklideanVelocity(sol.y[:,::4].T, 1)
cmBounds = [1e-4, 1e-1]

norm = LogNorm(vmin=cmBounds[0], vmax=cmBounds[1])
cmap=plt.get_cmap('cool')
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


for i in range(len(simX)-1): #
    ax.plot3D(simX[i],simY[i],simZ[i] ,'o', ms=2,color=np.asarray(cmap(norm(col[i]))[0:3]))
            
fun.noBackground(ax)
ax.view_init(20,45)

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

plt.savefig("Figure5b_phaseSpace.svg")
plt.show()


# Plot control outputs from ghostID
# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(4.5*inCm,5.5*inCm)
plt.figure(fig)
plt.savefig("Figure5b_pQ.svg")
plt.show()

# eigenvalues across trajectories

fig, axes = ctrlPlots[4]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5,3.5)
fig.set_size_inches(5*inCm,3.75*inCm)
axes[1].set_ylim(-1.01,-0.99)
for i in range(3): axes[i].set_xlim(-1,215)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[2].set_ylabel('$\\lambda_3$')
axes[2].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure5b_evsQ4.svg")
plt.show()


#####################################################################
# %% Figure 5c, Kuehn 2014m slow-fast toy-model
#####################################################################

# set parameters
eps=0.1
parameters_kuehn =  eps

# simulate trajectories
dt = 0.01
timesteps = np.linspace(0,100,int(100/dt))
sol1 = solve_ivp(mod.Kuehn_toyModel, (0, 100), [-0.5,-0.25],
                    t_eval=timesteps, args=(parameters_kuehn,),method='RK45')
sol2 = solve_ivp(mod.Kuehn_toyModel, (0, 100), [0.5,0.15],
                    t_eval=timesteps, args=(parameters_kuehn,),method='RK45')

# run ghostID
Trj1=sol1.y.T
ghostSeq1, ctrlPlots1 = gid.ghostID(mod.Kuehn_toyModel,parameters_kuehn,dt,Trj1,0.005,peak_kwargs={"prominence":0,"width":100},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

Trj2=sol2.y.T
ghostSeq2, ctrlPlots2 = gid.ghostID(mod.Kuehn_toyModel,parameters_kuehn,dt,Trj2,0.005,peak_kwargs={"prominence":0,"width":100},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

if len(ghostSeq1)==0 and len(ghostSeq2)==0: print("No ghosts in Kuehn slow-fast toy model identified.")

# plot phase space
xmin=-0.5;xmax=0.5
ymin=-0.5;ymax=0.5

Ng=160
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

Q, coords = gid.qOnGrid(mod.Kuehn_toyModel,parameters_kuehn,coords=[x_range,y_range], jit=True)

plt.figure(figsize=(4.25*inCm,6*inCm))
ax = plt.gca()

def flow_model(t,z): 
        return mod.Kuehn_toyModel(t,z,parameters_kuehn)
U,V=fun.vector_field(flow_model,(Xg, Yg),dim='2D') 

ax.streamplot(
    Xg, Yg,
    U,V,
    density=0.8,
    color=[0.75, 0.75, 0.75, 1],
    arrowsize=0.9,
    linewidth=0.9)

# plot trajectory
ax.plot(sol1.y[0],sol1.y[1],'-',color='green',lw=2)
ax.plot(sol2.y[0],sol2.y[1],'-',color='magenta',lw=2)

# slow manifold
ax.plot(0*y_range+eps,y_range,':',color='blue',label='slow manifold')

# plot Q-value
vmin = 1e-5 # Define log scale range 
vmax = 0.1 # Avoid zero or negative values 
im = ax.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax,label="Q")

#labels, limits, legend
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(xmin,xmax); ax.set_xticks([-0.5,0,0.5])
ax.set_ylim(ymin,ymax); ax.set_yticks([-0.5,0,0.5])
ax.legend(fontsize=7)
plt.savefig("Figure5c_phaseSpace.svg")
plt.show()

# Plot control outputs from ghostID

# pQ timeseries and Q-minima
fig, ax = ctrlPlots1[0]
fig.set_size_inches(4.5*inCm,2.6*inCm)
plt.figure(fig)
plt.savefig("Figure5c_pQ1.svg")
plt.show()


fig, ax = ctrlPlots2[0]
fig.set_size_inches(4.5*inCm,2.6*inCm)
plt.figure(fig)
plt.savefig("Figure5c_pQ2.svg")
plt.show()

# eigenvalues across trajectories
fig, axes = ctrlPlots2[1]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,3.75*inCm)
axes[0].set_xlim(0,3.1)
axes[1].set_xlim(0,3.1)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure5c_evs.svg")
plt.show()


#####################################################################
# %% Figure 5d, Fitz-Hugh Nagumo
#####################################################################

# set parameters
a=0;b=0.5;eps=0.01
parameters_FHN=  [a,b,eps]

# simulate trajectory 
dt = 0.1
timesteps = np.linspace(0,500,int(700/dt))
sol = solve_ivp(mod.FHN, (0, 500), [0.05,0.1],
                    t_eval=timesteps, args=(parameters_FHN,),method='RK45',rtol=1e-6,atol=1e-6)

# run ghostID
Trj=sol.y[:,int(100/dt):].T
ghostSeq, ctrlPlots = gid.ghostID(mod.FHN,parameters_FHN,dt,Trj,0.1,peak_kwargs={"prominence":0,"width":10},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

if len(ghostSeq)==0: print("No ghosts for FHN model identified.")

# plot phase space
xmin=-1.3;xmax=1.3
ymin=-1.3;ymax=1.3

Ng=160
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

Q, coords = gid.qOnGrid(mod.FHN,parameters_FHN,coords=[x_range,y_range], jit=True)

plt.figure(figsize=(4.25*inCm,6*inCm))
ax = plt.gca()

def flow_model(t,z): 
        return mod.FHN(t,z,parameters_FHN)
U,V=fun.vector_field(flow_model,(Xg, Yg),dim='2D') 

ax.streamplot(
    Xg, Yg,
    U,V,
    density=0.8,
    color=[0.75, 0.75, 0.75, 1],
    arrowsize=0.9,
    linewidth=0.9)

# plot trajectory
ax.plot(sol.y[0,int(100/dt):],sol.y[1,int(100/dt):],'-',color='ivory',lw=2)

# plot Q-value
vmin = 1e-4 # Define log scale range 
vmax = 10 # Avoid zero or negative values 
im = ax.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), aspect=1, origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# # nullclines
v = lambda u: u - u**3 # u-Nullcline # u=x, v=y
u = lambda v: b*v-a # v-Nullcline
ax.plot(x_range,v(x_range),'-b',label='u nullcline')
ax.plot(u(y_range),y_range,'--b',label='v nullcline')

# # add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax,label="Q")

# #labels, limits, legend
ax.set_xlabel('u'); ax.set_ylabel('v')
ax.set_xlim(xmin,xmax); ax.set_xticks([-1,0,1])
ax.set_ylim(ymin,ymax); ax.set_yticks([-1,0,1])
ax.legend(fontsize=7)
plt.savefig("Figure5d_phaseSpace.svg")
plt.show()

# Plot control outputs from ghostID

# pQ timeseries and Q-minima
fig, ax = ctrlPlots[0]
fig.set_size_inches(4.5*inCm,5.5*inCm)
plt.figure(fig)
plt.savefig("Figure5d_pQ.svg")
plt.show()

# eigenvalues across trajectories
fig, axes = ctrlPlots[1]
suptxt = fig._suptitle.get_text()
fig.suptitle(suptxt,fontsize=5.5)
fig.set_size_inches(5*inCm,3.75*inCm)
axes[0].set_xlim(0,29)
axes[1].set_xlim(0,29)
axes[0].set_ylabel('$\\lambda_1$')
axes[1].set_ylabel('$\\lambda_2$')
axes[1].set_xlabel('t (along trajectory segment)')
plt.figure(fig)
plt.savefig("Figure5d_evsQ1.svg")
plt.show()

#####################################################################
# %% Michaelis-Menten Model of Enzymatic Catalysis 
#####################################################################

# # set parameters
# alpha = 0.85
# beta = 0.3
# mu = 0.1
# parameters_MM =  [alpha,beta,mu]

# # simulate trajectories
# dt = 0.005
# timesteps = np.linspace(0,60,int(60/dt))
# sol = solve_ivp(mod.MMenten_slowFast, (0, 60), [0.99,0.01],
#                     t_eval=timesteps, args=(parameters_MM,),method='RK45',rtol=1e-9,atol=1e-9)
# # sol2 = solve_ivp(mod.MMenten_slowFast, (0, 100), [0.5,0.15],
# #                     t_eval=timesteps, args=(parameters_MM,),method='RK45')

# # run ghostID
# Trj=sol.y.T
# ghostSeq, ctrlPlots = gid.ghostID(mod.MMenten_slowFast,parameters_MM,dt,Trj,0.01,peak_kwargs={"prominence":0,"width":150},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

# # Trj2=sol2.y.T
# # ghostSeq2, ctrlPlots2 = gid.ghostID(mod.Kuehn_toyModel,parameters_kuehn,dt,Trj2,0.005,peak_kwargs={"prominence":0,"width":10},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

# # if len(ghostSeq1)==0 and len(ghostSeq2)==0: print("No ghosts in Kuehn slow-fast toy model identified.")

# # plot phase space
# xmin=0;xmax=1.1
# ymin=0;ymax=1.1

# Ng=160
# x_range=np.linspace(xmin,xmax,Ng)
# y_range=np.linspace(ymin,ymax,Ng)
# grid_ss = np.meshgrid(x_range, y_range)
# Xg,Yg=grid_ss

# Q, coords = gid.qOnGrid(mod.MMenten_slowFast,parameters_MM,coords=[x_range,y_range], jit=True)

# plt.figure(figsize=(4.5*inCm,6.3*inCm))
# ax = plt.gca()

# def flow_model(t,z): 
#         return mod.MMenten_slowFast(t,z,parameters_MM)
# U,V=fun.vector_field(flow_model,(Xg, Yg),dim='2D') 

# ax.streamplot(
#     Xg, Yg,
#     U,V,
#     density=0.8,
#     color=[0.75, 0.75, 0.75, 1],
#     arrowsize=0.9,
#     linewidth=0.9)

# # plot trajectory
# ax.plot(sol.y[0],sol.y[1],'-',color='ivory',lw=2)
# # ax.plot(sol2.y[0],sol2.y[1],'-',color='magenta',lw=2)

# # plot Q-value
# vmin = 1e-4 # Define log scale range 
# vmax = 1e2 # Avoid zero or negative values 
# im = ax.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# # add colorbar
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax,label="Q")

# #labels, limits, legend
# ax.set_xlabel('substrate'); ax.set_ylabel('E-S complex')
# ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
# ax.legend(fontsize=7)
# # plt.savefig("Figure4e_phaseSpace.svg")
# plt.show()

# # Plot control outputs from ghostID

# # pQ timeseries and Q-minima
# fig, ax = ctrlPlots[0]
# fig.set_size_inches(5*inCm,6*inCm)
# plt.figure(fig)
# # plt.savefig("Figure4e_pQ1.svg")
# plt.show()

# # eigenvalues across trajectories
# fig, axes = ctrlPlots[1]
# suptxt = fig._suptitle.get_text()
# fig.suptitle(suptxt,fontsize=5.5)
# fig.set_size_inches(5*inCm,2.5*inCm)
# # axes[0].set_xlim(0,1.7)
# # axes[1].set_xlim(0,1.7)
# axes[0].set_ylabel('$\\lambda_1$')
# axes[1].set_ylabel('$\\lambda_2$')
# axes[1].set_xlabel('t (along trajectory segment)')
# plt.figure(fig)
# # plt.savefig("Figure4e_evs.svg")
# plt.show()


# #############################################################
# #%% Hastings slow-fast
# #############################################################

# # set parameters
# gamma=2.5;h=1;v=0.5;m=0.4;alpha=1.8;K=2.2;eps=0.01
# parameters_hastings =  [gamma,h,v,m,alpha,K,eps]

# # simulate trajectory 
# dt = 0.1
# timesteps = np.linspace(0,1e4,int(1e4/dt))
# sol = solve_ivp(mod.Hastings_etal, (0, 1e4), [0.05,0.6],
#                     t_eval=timesteps, args=(parameters_hastings,),method='RK45',rtol=1e-6,atol=1e-6)

# # run ghostID
# Trj=sol.y[:,int(9050/dt):int(9800/dt)].T
# ghostSeq, ctrlPlots = gid.ghostID(mod.Hastings_etal,parameters_hastings,dt,Trj,0.02,peak_kwargs={"prominence":0,"width":10},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":True,"return_ctrl_figs":True}) #

# if len(ghostSeq)==0: print("No ghosts for Hastings model in slow-fast regime identified.")

# # plot phase space
# xmin=0;xmax=1.5
# ymin=0.5;ymax=2

# Ng=160
# x_range=np.linspace(xmin,xmax,Ng)
# y_range=np.linspace(ymin,ymax,Ng)
# grid_ss = np.meshgrid(x_range, y_range)
# Xg,Yg=grid_ss

# Q, coords = gid.qOnGrid(mod.Hastings_etal,parameters_hastings,coords=[x_range,y_range], jit=True)

# plt.figure(figsize=(4.5*inCm,6.3*inCm))
# ax = plt.gca()

# def flow_model(t,z): 
#         return mod.Hastings_etal(t,z,parameters_hastings)
# U,V=fun.vector_field(flow_model,(Xg, Yg),dim='2D') 

# ax.streamplot(
#     Xg, Yg,
#     U,V,
#     density=0.8,
#     color=[0.75, 0.75, 0.75, 1],
#     arrowsize=0.9,
#     linewidth=0.9)

# # plot trajectory
# ax.plot(sol.y[0,int(5000/dt):],sol.y[1,int(5000/dt):],'-',color='green',lw=2)

# # plot Q-value
# vmin = 1e-5 # Define log scale range 
# vmax = 1 # Avoid zero or negative values 
# im = ax.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), aspect=1, origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 

# # # nullclines
# y_NNC = mod.Hastings_NNC(x_range,parameters_hastings)
# x_PNC = mod.Hastings_PNC(y_range,parameters_hastings)
# ax.plot(x_range,y_NNC,'-b',label='Prey n.c.')
# ax.plot(x_PNC,y_range,'--b',label='Predator n.c.')

# # # add colorbar
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax,label="Q")

# # #labels, limits, legend
# ax.set_xlabel('Prey'); ax.set_ylabel('Predator')
# ax.set_xlim(xmin,xmax); #ax.set_xticks([0,5,10,15])
# ax.set_ylim(ymin,ymax); #ax.set_yticks([0,5,10,15])
# ax.legend(fontsize=7)
# # plt.savefig("Figure4c_phaseSpace.svg")
# plt.show()

# #% Plot control outputs from ghostID

# # pQ timeseries and Q-minima
# fig, ax = ctrlPlots[0]
# fig.set_size_inches(5*inCm,6*inCm)
# plt.figure(fig)
# # plt.savefig("Figure4c_pQ.svg")
# plt.show()

# # eigenvalues across trajectories
# fig, axes = ctrlPlots[1]
# suptxt = fig._suptitle.get_text()
# fig.suptitle(suptxt,fontsize=5.5)
# fig.set_size_inches(5*inCm,2.5*inCm)
# # axes[0].set_xlim(0,17)
# # axes[1].set_xlim(0,17)
# axes[0].set_ylabel('$\\lambda_1$')
# axes[1].set_ylabel('$\\lambda_2$')
# axes[1].set_xlabel('t (along trajectory segment)')
# plt.figure(fig)
# # plt.savefig("Figure4c_evsQ1.svg")
# plt.show()

# fig, axes = ctrlPlots[2]
# suptxt = fig._suptitle.get_text()
# fig.suptitle(suptxt,fontsize=5.5)
# fig.set_size_inches(5*inCm,2.5*inCm)
# # axes[0].set_xlim(0,2)
# # axes[1].set_xlim(0,2)
# axes[0].set_ylabel('$\\lambda_1$')
# axes[1].set_ylabel('$\\lambda_2$')
# axes[1].set_xlabel('t (along trajectory segment)')
# plt.figure(fig)
# # plt.savefig("Figure4c_evsQ2.svg")
# plt.show()
# # %%
