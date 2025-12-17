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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jacfwd
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax
import jax.numpy as jnp

import matplotlib.pylab as pylab
pylab.rcParams.update(fun.get_rcparams())
plt.rcParams.update({'font.family':'Arial'})

inCm = 1/2.54 # convert inch to cm for plotting

#%% bifurcation diagram

mu_values = [-0.2, 0, 0.05] # Saddle & Stable FP

plt.figure(figsize=(10*inCm,6*inCm))
mu_range = np.linspace(-0.5,0,200)
x_stable = -np.sqrt(-mu_range)
x_saddle = np.sqrt(-mu_range)
plt.vlines(mu_values,-np.ones(3),np.ones(3),colors=['k','m','m'],alpha=[0.5,1,0.5])
plt.plot(mu_range,x_stable,'-r',lw=2,label='stable fixed point')
plt.plot(mu_range,x_saddle,'--k',lw=2,label='saddle')
plt.legend()
plt.xlabel("$\\mu$"); plt.ylabel("x")
plt.xlim(-0.5,0.15);plt.xticks([-0.5,-0.25,0,0.1])
plt.ylim(-0.75,0.75);plt.yticks([-0.75,-0.5,-0.25,0,0.25,0.5,0.75])
# plt.tight_layout()
plt.savefig("bif_1d.svg")
#%%
print("Perform manual analyses for saddle model...")

# Define grid
xmin=-1;xmax=1
ymin=-1;ymax=1

Ng=160
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# Eigenvalues on grid
print("Eigenvalue calculations")

def eigValsOnGrid_system1(X_grid, Y_grid):

    lambda1 = np.zeros_like(X_grid)
    lambda2 = np.zeros_like(X_grid)

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            x,y = [X_grid[i, j], Y_grid[i, j]]
            lambda1[i,j]= 2*x
            lambda2[i,j]= -1

    return [lambda1, lambda2]

lambdas = eigValsOnGrid_system1(Xg, Yg)

Qgrids = []
print("Q calculation")
for mu in mu_values:
    print("mu = ", mu)
    Q, coords = gid.qOnGrid(mod.snb_nf,[mu],coords=[x_range,y_range], jit=True)
    Qgrids.append(Q)
  

#%% plot

for i in range (3):
    

    mu = mu_values[i]

    dt = 0.01
    timesteps = np.linspace(0,100,int(100/dt))
    sol = solve_ivp(mod.snb_nf, (0, 100), [-4,-2],
                        t_eval=timesteps, args=([mu],),method='RK45')

    plt.figure(figsize=(18*inCm,10*inCm))

    plt.subplot(1,3,1)
    ax = plt.gca()
    def flow_model(t,z): 
            return mod.snb_nf(t,z,[mu_values[i]]) 
    U,V=fun.vector_field(flow_model,grid_ss,dim='2D') 
    ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=0.9,linewidth=0.9) 
    ax.plot(sol.y[0],sol.y[1],'-',color='ivory',lw=2.5)
    # Define log scale range 
    vmin = 1e-3 
    # Avoid zero or negative values 
    vmax = 1 
    im = ax.imshow(Qgrids[i].T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax)) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax,label="Q")
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)

    if mu < 0:
            ax.scatter(
                [-np.sqrt(-mu)], [0],
                s=14,          # dot size
                c='black',
                edgecolors='black',
                marker='o',
                alpha=1
            )
            ax.scatter(
                [np.sqrt(-mu)], [0],
                s=14,          # dot size
                c='white',
                edgecolors='black',
                marker='o',
                alpha=1
            )
        
    if mu == 0:
        ax.scatter(
            [np.sqrt(-mu)], [0],
            s=14,          # dot size
            c='m',
            edgecolors='black',
            marker='o',
            alpha=1
        )
    

    vmin=float(np.min(np.asarray(lambdas)))*1.05
    vmax=float(np.max(np.asarray(lambdas)))*1.05
    for ii in range(2):
    
        plt.subplot(1,3,2+ii)
        ax = plt.gca()
        ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.8,0.8,0.8,1],arrowsize=0.85,linewidth=0.85) 
        im = ax.imshow(lambdas[ii], extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax) 
        
        # Overlay contour for Q < 0.01
        level = 0.01
        cs = ax.contour(
            x_range,             # x coordinates
            y_range,             # y coordinates
            Qgrids[i].T,         # match imshow orientation
            levels=[level],      # single contour at 0.01
            colors='cyan',       # choose any color
            linewidths=1.2
        )
        
        if mu < 0:
            ax.scatter(
                [-np.sqrt(-mu)], [0],
                s=14,          # dot size
                c='black',
                edgecolors='black',
                marker='o',
                alpha=1
            )
            ax.scatter(
                [np.sqrt(-mu)], [0],
                s=14,          # dot size
                c='white',
                edgecolors='black',
                marker='o',
                alpha=1
            )
        
        if mu == 0:
            ax.scatter(
                [np.sqrt(-mu)], [0],
                s=14,          # dot size
                c='m',
                edgecolors='black',
                marker='o',
                alpha=1
            )
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label = f"$\\lambda_{ii+1}$")
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)

    plt.tight_layout(pad=-2.5)
    plt.savefig(f"q_ev_1d_{i+1}.svg")
# %%

plt.figure(figsize=(2*inCm,0.8*inCm))
plt.plot(sol.t,sol.y[0],'-',color='k',lw=1.5)
plt.xlim(-0.66,13.85); plt.ylim(-4,4)
plt.xticks([0,2,4,6,8,10,12],['0', '', '4', '', '8', '', '12'])
plt.yticks([-4,0,4])
plt.xlabel('t'),plt.ylabel('x')
plt.savefig("inset.svg")
# %%
