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

#%%
print("Perform manual analyses for saddle model...")

alpha_values = [-0.2, 0, 0.05] # Saddle & Stable FP

# Define grid
xmin=-1;xmax=1
ymin=-1;ymax=1

Ng=160
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

Qgrids = []
EVgrids = []
for alpha in alpha_values:
    print(alpha)
    F = lambda x: mod.snb_nf(0,x,[alpha]) # for calculation of eigenvalues
    # # Q-values on grid
    # # Q = gid.qOnGrid(mod.snb_nf,alpha,Xg,Yg)

    # # Q-values on grid
    Q, coords = gid.qOnGrid(mod.snb_nf,[alpha],coords=[x_range,y_range], jit=True)
    # E = eigValsOnGrid(mod.snb_nf,[alpha],coords=[x_range,y_range], jit=True)
    #

    print("Q calculation")
    # Q-values on grid
    # Q = fun.qOnGrid(mod.snb_nf,[alpha],Xg,Yg)

    print("ev calculation")
    # Eigenvalues on grid
    ev1, ev2 = fun.eigValsOnGrid(F, Xg, Yg)

    Qgrids.append(Q)
    EVgrids.append((ev1,ev2))

# # Eigenvalues on grid
# ev1, ev2 = fun.eigValsOnGrid(F, Xg, Yg)


#%% plot

for i in range (3):

    plt.figure(figsize=(18*inCm,10*inCm))

    plt.subplot(1,3,1)
    ax = plt.gca()
    def flow_model(t,z): 
            return mod.snb_nf(t,z,[alpha_values[i]]) 
    U,V=fun.vector_field(flow_model,grid_ss,dim='2D') 
    ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=0.9,linewidth=0.9) 
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
    

    vmin=float(np.min(np.real(np.asarray(EVgrids[i]))))*1.05
    vmax=float(np.max(np.real(np.asarray(EVgrids[i]))))*1.05
    for ev in range(2):
    
        plt.subplot(1,3,2+ev)
        ax = plt.gca()
        ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.8,0.8,0.8,1],arrowsize=0.85,linewidth=0.85) 
        im = ax.imshow(np.real( EVgrids[i][ev]), extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax) 
        
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

        # ax.clabel(cs, inline=False, fontsize=8)   # optional label
        
        
        mask = Qgrids[i].T < 2e-5

        # Coordinates of each (x,y) pixel center
        Xg_, Yg_ = np.meshgrid(x_range, y_range, indexing="xy")

        ax.scatter(
            Xg_[mask], Yg_[mask],
            s=3,          # dot size
            c='black',
            marker='o',
            alpha=1
        )
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label = f"$\\lambda_{ev+1}$")
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)

    plt.tight_layout(pad=-2.5)

        



# %%
