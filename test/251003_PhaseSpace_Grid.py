# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:35:38 2025

@author: Daniel Koch
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import functions_v07 as fun
import functions_v08 as fun2
from jax import jacfwd
import jax.numpy as jnp
import os
import sys
from scipy.signal import find_peaks
from matplotlib.colors import LogNorm
inCm = 1/2.54 # convert inch to cm for plotting

#paths
sys.path.append(os.path.join( os.path.dirname( __file__ ), '..' ))
os.chdir(os.path.dirname(os.path.abspath(__file__)))



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
    print(np.round(t,3), end = '\r')
    
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


def wunderling_model_vectorized(t, Z, para_model):
    """
    Vectorized version using explicit broadcasting
    """
    Z = np.asarray(Z, dtype=np.float64)
    
    # Unpack parameters
    d = para_model['d']
    GMT = para_model['GMT']
    Tcrits = para_model['Tcrits'] 
    Taus = para_model['Taus'] 
    mat_inter = para_model['mat_inter']
    
    # Ensure Z is at least 2D for consistent processing
    if Z.ndim == 1:
        Z = Z[np.newaxis, :]  # Convert to 2D with shape (1, n_vars)
        return_single = True
    else:
        return_single = False
    
    # Calculate intrinsic term (works for any number of points)
    intrinsic = -Z**3 + Z + np.sqrt(4/27) * GMT / Tcrits
    
    # Calculate coupling term using batch matrix multiplication
    # (Z + 1) has shape (n_points, n_vars)
    # mat_inter has shape (n_vars, n_vars)
    # Result should have shape (n_points, n_vars)
    coupling = d/10 * (Z + 1) @ mat_inter.T  # Equivalent to mat_inter @ (Z + 1)^T
    
    dZdt = (intrinsic + coupling) / Taus
    
    # Return to original shape if input was single point
    if return_single:
        return dZdt[0]  # Return 1D array
    else:
        return dZdt

F = lambda x: wunderling_modeljx(0,x,para_model) # for calculation of eigenvalues


def plot_nullclines(ax, para_model, x_range=(-2, 2), y_range=(-2, 2), resolution=200):
    """
    Plot x- and y-nullclines of the 2D reduced wunderling model.
    """
    # Unpack parameters
    d = para_model['d']
    GMT = para_model['GMT']
    Tcrit = para_model['Tcrits']  # assume scalar, not array
    mat_inter = np.array(para_model['mat_inter'])
    
    # Define constants
    gamma1 = np.sqrt(4/27) * GMT / Tcrit[0]
    gamma2 = np.sqrt(4/27) * GMT / Tcrit[1]
    
    # Build gridf
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Nullclines (dotx = 0, doty = 0)
    f1 = -X**3 + X + gamma1 + d/10 * (mat_inter[0,0]*(X+1) + mat_inter[0,1]*(Y+1))
    f2 = -Y**3 + Y + gamma2 + d/10 * (mat_inter[1,0]*(X+1) + mat_inter[1,1]*(Y+1))
    
    # Plot
    ax.contour(X, Y, f1, levels=[0], colors='g', linewidths=2, linestyles='-',label="x-nullcline",alpha=0.7)
    ax.contour(X, Y, f2, levels=[0], colors='g', linewidths=2, linestyles='--',label="y-nullcline",alpha=0.7)
 

para_model = {
    "d": 0.2,
    "GMT": 1.51,
    "mat_inter": np.array([[0,0],[1,0]]),
    "Tcrits": np.array([1.5,1.5]),
    "Taus": np.array([50,5000])
    }

# para_model = {
#     "d": 0.15,
#     "GMT": 1.52,
#     "mat_inter": np.array([[0,1],[1,0]]),
#     "Tcrits": np.array([1.5,1.5]),
#     "Taus": np.array([5000,5000])
#     }



def flow_model(t,z):
    return wunderling_model(t,z,para_model)

# Define grid
xmin=-1.5;xmax=1.5
ymin=-1.5;ymax=1.5

Ng=250
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# Q-values on grid
Q = fun2.qOnGrid(wunderling_model,para_model,Xg,Yg)


# FP as last pt in trajectory:
dt = 20#para_model['dt'] 
t_eval = np.asarray(np.arange(0, 1e9, dt), dtype=np.float64)
tF = t_eval[-1]

Z0 = np.array([-1.2,-1])
sol0 = solve_ivp(wunderling_model, [0,tF], Z0,  method='LSODA', 
                                    args=(para_model,), t_eval=t_eval)

fixedPt = sol0.y[:,sol0.y.shape[1]-1]
 
dt = 5
t_eval = np.asarray(np.arange(0, 1e6, dt), dtype=np.float64)
tF = t_eval[-1]

Z0 = np.array([-1.2,-1])
sol1 = solve_ivp(wunderling_model, [0,tF], Z0,  method='LSODA', 
                                    args=(para_model,), t_eval=t_eval)



#%% Q-values  along trj

def qAtPt_vectorized(F, p, Pt_array):
    """
    Vectorized version - requires F to support vectorized operations
    """
    # Evaluate F for all points at once (if F supports vectorization)
    X = F(0, Pt_array, p)
    
    # Calculate sum of squares along the appropriate axis
    # Assuming X has shape (n_points, n_variables)
    return np.sum(X**2, axis=1) / 2


idcsQmin = []

Q_ts = qAtPt_vectorized( wunderling_model_vectorized,para_model,sol1.y.T)



pQ = -np.log(Q_ts)

dur_ghost_min = 0 # set small to make saddle candidate of being identified as ghost

idx_minima, widths = find_peaks(pQ,width=dur_ghost_min/dt)

#%%

# def qAtPt_vectorized2(F, p, Pt_array):
#     """
#     Vectorized version - requires F to support vectorized operations
#     """
#     # Evaluate F for all points at once (if F supports vectorization)
#     X = F(0, Pt_array, p)
    
#     # Calculate sum of squares along the appropriate axis
#     # Assuming X has shape (n_points, n_variables)
#     return np.sum(X**2, axis=1) / 2

Q =  qAtPt_vectorized(wunderling_model_vectorized,para_model,Xg,Yg)
  
#%% ######## PLOT 1: flow, Q-values and trajectory

plt.figure(figsize=(13*inCm,11*inCm))
plt.title(f"$\\tau_1 = {para_model['Taus'][0]},\\tau_2 = {para_model['Taus'][1]}$")
ax = plt.gca()

# Flow
U,V=fun2.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1.2,linewidth=1.1)


# trajectory
ax.plot(sol1.y[0,:],sol1.y[1,:],lw=3,color='magenta')
ax.plot(sol1.y[0,idx_minima],sol1.y[1,idx_minima],'xr',lw=3,markersize=12) # position of Q-minima

#Nullclines
plot_nullclines(ax,para_model, x_range=(xmin, xmax), y_range=(ymin, ymax))

ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
# Define log scale range
vmin = np.min(Q)   # Avoid zero or negative values
vmax = np.max(Q)
im = plt.imshow(Q, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')


#%% time-course
plt.figure(figsize=(10*inCm,5*inCm))

plt.plot(sol1.t,sol1.y[0,:],lw=2,label='$x_1$')
plt.plot(sol1.t,sol1.y[1,:],lw=2,label='$x_2$')
plt.xscale('log')
plt.xlabel('time (yrs)'); plt.ylabel('value')
plt.legend()


#%% pQ along trajectory
# plt.figure(figsize=(10*inCm,10*inCm))

# plt.subplot(2,1,1)
# plt.plot(sol1.t[::5],pQ[::5],'-k',lw=1)
# plt.plot(sol1.t[idx_minima],pQ[idx_minima],'xr',lw=2,ms=10)
# plt.ylabel('$Q$',fontsize=14)
# plt.yscale("log")

# plt.subplot(2,1,2)
# plt.plot(sol1.t[::5],pQ[::5],'-k',lw=1)
# plt.plot(sol1.t[idx_minima],pQ[idx_minima],'xr',lw=2,ms=10)
# plt.ylabel('$Q$',fontsize=14)
# plt.xlabel('$time$',fontsize=14)
# plt.yscale("log")
# plt.xscale('log')


########## plot Q including distance to FP
plt.figure(figsize=(12*inCm,25*inCm))

plt.subplot(3,1,1)
plt.plot(sol1.t,Q_ts,'-k',lw=1)
plt.plot(sol1.t[idx_minima],Q_ts[idx_minima],'xr',lw=2,ms=10)
plt.ylabel('$Q$',fontsize=14)
plt.yscale("log")

plt.subplot(3,1,2)
plt.plot(sol1.t,Q_ts,'-k',lw=1)
plt.plot(sol1.t[idx_minima],Q_ts[idx_minima],'xr',lw=2,ms=10)
plt.ylabel('$Q$',fontsize=14)
plt.yscale("log")
plt.xscale('log')

plt.subplot(3,1,3)

dists = np.linalg.norm(sol1.y - fixedPt[:, None], axis=0)

plt.plot(sol1.t[::5],dists[::5],'-k',lw=1)
plt.plot(sol1.t[idx_minima],dists[idx_minima],'xr',lw=2,ms=10)
plt.ylabel('$ED(FP)$',fontsize=14)
plt.xlabel('$time$',fontsize=14)
plt.tight_layout()

#%% Eigenvalues on Grid:
   
ev1, ev2 = fun.eigValsOnGrid(F, Xg, Yg)

def plot_ev_subplot(ax, ev, Xg, Yg, sol, idx_minima, title, flow_model, grid_ss, x_range, y_range, xmin, xmax, ymin, ymax, tick_rotation=45):
    # Plot streamlines
    U, V = fun.vector_field(flow_model, grid_ss, dim='2D')
    ax.streamplot(Xg, Yg, U, V, density=0.8, color=[0.75,0.75,0.75,1], arrowsize=1.2, linewidth=1.1)

    # Plot heatmap
    l = np.max(np.abs(ev))/2
    vmin, vmax = -l, l
    im = ax.imshow(ev, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                    origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    # Plot minima positions
    ax.plot(sol.y[0, idx_minima[2:]], sol.y[1, idx_minima[2:]], 'xr', lw=3, markersize=12)
    ax.plot(sol.y[0, idx_minima[3:]], sol.y[1, idx_minima[3:]], lw=3, color='magenta')

    # Colorbar on top
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.5)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(title, fontsize=16, labelpad=10, loc='center')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.tick_top()
    
    # Rotate tick labels
    cbar.ax.tick_params(labelrotation=tick_rotation)

    # Labels and limits
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')  # 1:1 aspect ratio

# Figure setup
plt.figure(figsize=(30*inCm, 13*inCm))

# Subplot 1
ax1 = plt.subplot(1,2,1)
plot_ev_subplot(ax1, ev1, Xg, Yg, sol1, idx_minima, r'$\lambda_1$', flow_model, grid_ss, x_range, y_range, xmin, xmax, ymin, ymax)
ax1.plot(sol1.y[0,idx_minima],sol1.y[1,idx_minima],'xr',lw=3,markersize=12) # position of Q-minima
ax1.plot(sol1.y[0,:],sol1.y[1,:],lw=3,color='magenta')

# Subplot 2
ax2 = plt.subplot(1,2,2)
plot_ev_subplot(ax2, ev2, Xg, Yg, sol1, idx_minima, r'$\lambda_2$', flow_model, grid_ss, x_range, y_range, xmin, xmax, ymin, ymax)
ax2.plot(sol1.y[0,idx_minima],sol1.y[1,idx_minima],'xr',lw=3,markersize=12) # position of Q-minima
ax2.plot(sol1.y[0,:],sol1.y[1,:],lw=3,color='magenta')

# plt.tight_layout()
plt.show()
#%%

ev_steps = 100

J_fun = jacfwd(F)

n = 2
evLimit = 0.1
slopeLimits = [1e-9,1e-2]
taus = para_model["Taus"]

for i in idx_minima:

    eigVals = []        
    for pt in sol1.y.T[i-ev_steps:i+ev_steps]:
        
        jac = J_fun(jnp.asarray(pt))
        
        eigVals.append(np.linalg.eigvals(jac))
        
    eigVals = np.asarray(eigVals)
        
    #############
    
    idcs_evSignChanges = np.array([ii for ii in range(n) if fun.sign_change(np.real(eigVals[:,ii]))])
    
    signChangeCount = len(idcs_evSignChanges)
    
    idcs_qualifyingSlopes = []
    
    ev_slope = []
    r2s = []
    
    for ii in range(n):
        
        coeffs = np.polyfit(np.asarray(range(0,2*ev_steps))*dt, eigVals[:,ii], 1)
    
        arr_pred = np.polyval(coeffs, np.asarray(range(0,2*ev_steps))*dt)
        
        # R² calculation
        ss_res = np.sum((eigVals[:,ii] - arr_pred) ** 2)
        ss_tot = np.sum((eigVals[:,ii] - np.mean(eigVals[:,ii])) ** 2)
        r2 = 1 - ss_res/ss_tot
        
        ev_slope.append(coeffs[0])
        r2s.append(r2)
    
    for ii in range(n):
        if np.abs(np.average(np.real(eigVals[:,ii])))<evLimit:
            if ev_slope[ii]*taus[ii]>slopeLimits[0] and ev_slope[ii]*taus[ii]<slopeLimits[1]: 
                idcs_qualifyingSlopes.append(ii)
    
    qualSlopeCount = len(idcs_qualifyingSlopes)

    idcs_ = np.union1d(idcs_evSignChanges, idcs_qualifyingSlopes).astype(int)
    
    if len(idcs_) > 0:
        dimSmallestTau = np.argmin(taus[idcs_])
        ghostCheck=True
        print(f"Ghost at t={sol1.t[i]}. Element with smallest timescale is: ",dimSmallestTau, " with ", taus[dimSmallestTau], "yrs.")
    else:
        print(f"At pQ-peak at t={sol1.t[i]} the criteria for ghosts are not fulfilled.")  
        ghostCheck=False
    ##############

    fig, ax1 = plt.subplots()
    
    plt.suptitle(f'eigenvalues at pQ peak t={sol1.t[i]}, ghost: {ghostCheck}, \n evSignChanges: {idcs_evSignChanges}, slopes: {ev_slope*taus}, \n r²: {r2s}')

    # First line on left y-axis
    pl1, = ax1.plot(sol1.t[i-ev_steps:i+ev_steps], eigVals[:, 0], '-', lw=2, color='C0')
    ax1.set_ylabel(r'$\lambda_1$', fontsize=14, color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    
    # Second line on right y-axis
    ax2 = ax1.twinx()
    pl2, = ax2.plot(sol1.t[i-ev_steps:i+ev_steps], eigVals[:, 1], '-', lw=2, color='C1')
    ax2.set_ylabel(r'$\lambda_2$', fontsize=14, color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    
    # Common x-label
    ax1.set_xlabel(r'$time$', fontsize=14)
    
    plt.tight_layout()
    
    
#%% 2025-09-19 zoom into dynamics close to stable FP
# Eigenvalues on Grid:

delta1 = 1e-3
delta2 = 1e-3
# Define grid
xmin=fixedPt[0]-delta1
xmax=fixedPt[0]+delta1
ymin=fixedPt[1]-delta2
ymax=fixedPt[1]+delta2

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

ev1, ev2 = fun.eigValsOnGrid(F, Xg, Yg)


#%%

def plot_ev_subplot(ax, ev, Xg, Yg, sol, idx_minima, title, flow_model, grid_ss, x_range, y_range, xmin, xmax, ymin, ymax, tick_rotation=45):
    # Plot streamlines
    U, V = fun.vector_field(flow_model, grid_ss, dim='2D')
    ax.streamplot(Xg, Yg, U, V, density=0.8, color=[0.75,0.75,0.75,1], arrowsize=1.2, linewidth=1.1)

    # Plot heatmap
    vmin, vmax = np.min(ev), np.max(ev)
    im = ax.imshow(ev, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                    origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    # Plot minima positions
    ax.plot(sol.y[0, idx_minima[2:]], sol.y[1, idx_minima[2:]], 'xr', lw=3, markersize=12)
    ax.plot(sol.y[0, idx_minima[3:]], sol.y[1, idx_minima[3:]], lw=3, color='magenta')

    # Colorbar on top
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.5)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(title, fontsize=16, labelpad=10, loc='center')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.tick_top()
    
    # Rotate tick labels
    cbar.ax.tick_params(labelrotation=tick_rotation)

    # Labels and limits
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')  # 1:1 aspect ratio

# # Figure setup
plt.figure(figsize=(30*inCm, 13*inCm))

# Subplot 1
ax1 = plt.subplot(1,2,1)
plot_ev_subplot(ax1, ev1, Xg, Yg, sol1, idx_minima, r'$\lambda_1$', flow_model, grid_ss, x_range, y_range, xmin, xmax, ymin, ymax)
ax1.plot(sol1.y[0,idx_minima],sol1.y[1,idx_minima],'xr',lw=3,markersize=12) # position of Q-minima
ax1.plot(sol1.y[0,:],sol1.y[1,:],lw=3,color='magenta')

# Subplot 2
ax2 = plt.subplot(1,2,2)
plot_ev_subplot(ax2, ev2, Xg, Yg, sol1, idx_minima, r'$\lambda_2$', flow_model, grid_ss, x_range, y_range, xmin, xmax, ymin, ymax)
ax2.plot(sol1.y[0,idx_minima],sol1.y[1,idx_minima],'xr',lw=3,markersize=12) # position of Q-minima
ax2.plot(sol1.y[0,:],sol1.y[1,:],lw=3,color='magenta')

# plt.tight_layout()
plt.show()

#%% 
from scipy.spatial import cKDTree
import jax
import jax.numpy as jnp

# ---------------- JAX version of wunderling_model ----------------
def wunderling_model_jx(t, Z, para_model):
    # Unpack parameters
    d = para_model['d']
    GMT  = para_model['GMT']
    Tcrits = para_model['Tcrits'] 
    Taus = para_model['Taus'] 
    mat_inter = para_model['mat_inter'] 

    intrinsic = -Z**3 + Z + jnp.sqrt(4/27)*GMT/Tcrits
    coupling = d/10 * (mat_inter @ (Z + 1))
    dZdt = (intrinsic + coupling)/Taus
    return dZdt


# ---------------- JAX batch version ----------------
def wunderling_model_batch_jx(Zs, para_model):
    """
    Vectorized JAX version of wunderling_model for batch evaluation.
    Zs: shape (num_points, n)
    Returns dZdt: shape (num_points, n)
    """
    d = para_model['d']
    GMT  = para_model['GMT']
    Tcrits = para_model['Tcrits'] 
    Taus = para_model['Taus'] 
    mat_inter = para_model['mat_inter'] 
    
    intrinsic = -Zs**3 + Zs + jnp.sqrt(4/27)*GMT/Tcrits
    coupling = d/10 * (mat_inter @ (Zs.T + 1)).T
    dZdt = (intrinsic + coupling)/Taus
    return dZdt


#%%
import time
     
Trj = sol1.y.T


start = time.time()
seq = fun2.ghostID_v08(wunderling_modeljx,para_model,dt,Trj,0.05,plot_ctrl=True)
end = time.time()

print(f"Execution time: {end - start} seconds")
# start = time.time()
# seq2 = fun.ghostID_v07(wunderling_modeljx,para_model,dt,Trj,ev_steps=50,checkOscTr=False,dur_ghost_min=0,evLimit=0.1,thr_dtEigVal=0.0,plot_ctrl=True)
# end = time.time()

# print(f"Execution time: {end - start} seconds")
# # ghostID_v07(model,params,dt,trajectory,ev_steps=10,checkOscTr=False,dur_ghost_min=5,evLimit=0.1,thr_dtEigVal=0.005,epsilon_SN_ghosts=0.1,sr_windowsize=10,dOscT_min=20,epsilon_osc=0.6, **kwargs):
#     # 