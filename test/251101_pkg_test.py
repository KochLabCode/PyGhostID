import PyGhostID as gid
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
import functions_v08 as funOld

# def make_jacfun(model, params):
#     F = lambda x: model(0, x, params)
#     J_fun = jax.jacfwd(F)
#     return jax.jit(J_fun)


def icAtQmin(qmin,step,nlowest,model,params):
    
    J_fun = make_jacfun(model, params)
    eig = jnp.linalg.eig(J_fun(qmin))
    eigVals = jnp.real(eig[0])
    eigVecs = eig[1]
    
    idcs = np.argsort(np.abs(eigVals))
 
    direction = eigVecs[:, idcs[0]]
    
    if nlowest > 1:
        for i in range(1,nlowest):
            direction += eigVecs[:, idcs[i]]
    
    direction_norm = direction/jnp.linalg.norm(direction)
            
    pos = qmin + step * direction_norm
    
    return pos,eigVals,eigVecs

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
Q, coords = gid.qOnGrid(wunderling_modeljx,para_model,coords=[x_range,y_range], jit=True)

x0s = [jnp.array([-0.7,-0.7]), jnp.array([1,-0.6]), jnp.array([-0.6,1])]

qminima = [gid.find_local_Qminimum(wunderling_modeljx, x0, para_model,tol_glob=1e-6,method="BFGS",verbose=False)[0] for x0 in x0s]

#%% phase space: Qmin, eigenvectors, ICs and trjs
plt.figure(figsize=(16*inCm,14*inCm))
plt.title(f"$\\tau_1 = {para_model['Taus'][0]},\\tau_2 = {para_model['Taus'][1]}$")
ax = plt.gca()

# U,V=funOld.vector_field(flow_model,grid_ss,dim='2D')    
# ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,0.5],arrowsize=1.2,linewidth=1.1)


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
    
    ic, eigVals, eigVecs = icAtQmin(qminima[i], -0.1,2, wunderling_modeljx, para_model)
    
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
    
    ic, eigVals, eigVecs = icAtQmin(qminima[i], -0.1,2,  wunderling_modeljx, para_model)
    
    dt = 10
    t_eval = np.asarray(np.arange(0, 1e3, dt), dtype=np.float64)
    tF = t_eval[-1]

    sol = solve_ivp(wunderling_modeljx, [0,tF], np.real(ic),  method='LSODA', 
                                        args=(para_model,), t_eval=t_eval)
    Trj=sol.y.T
    ghostSeq = gid.ghostID(wunderling_modeljx,para_model,dt,Trj,0.05,peak_kwargs={"prominence":0.5,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False}) #

    ghostSeqs.append(ghostSeq)
    

#%% 
gS_unified = gid.unify_IDs(ghostSeqs)

# gS_unified = fun.unifyIDs(ghostSeqs)


uniqueGhosts = gid.unique_ghosts(gS_unified)

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


gpos0, pars0, gSeq0 =  gid.track_ghost_branch(uniqueGhosts[0], wunderling_modeljx_listpara, pars, 1, 20, 0.1, 1e3, 5, delta=0.2, icStep=0.4, mode="closest", 
                             epsilon_gid=0.1,solve_ivp_method='LSODA', qmin_method="BFGS",qmin_tol=1e-6,peak_kwargs={"prominence":0.1,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False})

gpos1, pars1, gSeq1 =  gid.track_ghost_branch(uniqueGhosts[1], wunderling_modeljx_listpara, pars, 1, 20, 0.1, 1e3, 5, delta=0.2, icStep=0.4, mode="closest", 
                             epsilon_gid=0.1,solve_ivp_method='LSODA', qmin_method="BFGS",qmin_tol=1e-6,peak_kwargs={"prominence":0.1,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False})

gpos2, pars2, gSeq2 =  gid.track_ghost_branch(uniqueGhosts[2], wunderling_modeljx_listpara, pars, 1, 20, 0.1, 1e3, 5, delta=0.2, icStep=0.4, mode="closest", 
                             epsilon_gid=0.1,solve_ivp_method='LSODA', qmin_method="BFGS",qmin_tol=1e-6,peak_kwargs={"prominence":0.1,"width":0},ctrlOutputs={"ctrl_qplot":True,"qplot_xscale":"linear","ctrl_evplot":False})



#%% '

plt.figure()
plt.plot(pars0,gpos0[:,1],'--o',color='gray')
plt.plot(pars1,gpos1[:,1],'-d',color='gray')
plt.plot(pars2,gpos2[:,1],':x',color='gray')
# plt.ylim(0,2.2)
plt.ylabel('y');plt.xlabel("$\\Delta GMT$")
# %%
