# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 22:39:35 2025

@author: dkoch
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 23 21:57:06 2025

@author: dkoch
"""

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.optimize import approx_fprime
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
# import nolds
import matplotlib.pylab as pylab
import warnings
from scipy.integrate import solve_ivp
import networkx as nx
import os
import sys
import jax
import jax.numpy as jnp

from tqdm import tqdm

#paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join( os.path.dirname( __file__ ), '..' ))
import functions_v08 as fun

from sklearn.decomposition import PCA
n_components=3
pca = PCA(n_components)
options = {"node_color": "lightblue", "node_size": 220, "linewidths": 0.1, "width": 0.3}


# def FHN_coupledjx(t, x, para_model):
#     a, b, eps, K, A = para_model
#     n = A.shape[0]
#     u = x[:n]
#     v = x[n:]

#     D = jnp.sum(A, axis=1)
#     L = jnp.diag(D) - A

#     du = u - u**3 - v - K * (L @ u)
#     dv = eps * (u - b * v + a)

#     return np.concatenate([np.array(du), np.array(dv)])  # return as numpy for solve_ivp


def FHN_coupledjx(t, x, para_model):
    a, b, eps, K, A = para_model
    n = A.shape[0]
    u = x[:n]
    v = x[n:]

    du = u - u**3 - v + K * (A @ v)
    dv = eps * (u - b * v + a)

    return np.concatenate([np.array(du), np.array(dv)])  # return as numpy for solve_ivp



# def FHN_coupledjx(t, x, para_model):
#     a, b, eps, K, A = para_model
#     n = A.shape[0]
#     u = x[:n]
#     v = x[n:]
#     phi=jnp.pi/2-1

#     Buu=jnp.cos(phi)
#     Buv=jnp.sin(phi)
#     Bvu=-jnp.sin(phi)
#     Bvv=jnp.cos(phi)

#     D = jnp.sum(A, axis=1)
#     L = jnp.diag(D) - A

#     du = u - u**3 - v - K * (L @ (Buu*u + Buv*v))
#     dv = eps * (u - b * v + a) - K * (L @ (Bvu*u + Bvv*v))

#     return np.concatenate([np.array(du), np.array(dv)])  # return as numpy for solve_ivp


#%%%

# Seeds, interesting ones are as follows

# N = 8
# 3: single ghost and 3-ghost channel

# rs = 2
# np.random.seed(rs)
np.random.seed()

# define network topology
# n = 20
# G = nx.erdos_renyi_graph(n, 0.1, directed=True)
# A = nx.to_numpy_array(G)

# Parameters
n = 10          # number of nodes (neurons)
k = 6           # mean degree (number of neighbors in the ring lattice)
p = 0.1          # rewiring probability

# Step 1: Generate undirected Watts–Strogatz small-world network
G_undirected = nx.watts_strogatz_graph(n, k, p)

# Step 2: Convert to directed network
G_directed = nx.DiGraph()

# Copy nodes
G_directed.add_nodes_from(G_undirected.nodes())

# Step 3: For each undirected edge, assign one or both directions
for u, v in G_undirected.edges():
    # 80% chance of being unidirectional (biological asymmetry)
    if np.random.rand() < 0.8:
        if np.random.rand() < 0.5:
            G_directed.add_edge(u, v)
        else:
            G_directed.add_edge(v, u)
    else:
        # 20% chance to keep bidirectional connection
        G_directed.add_edge(u, v)
        G_directed.add_edge(v, u)


A = nx.to_numpy_array(G_directed)

p_inhibitory = 0.4

# introduce inhibitory links
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if A[i,j] == 1:
            if np.random.rand() < p_inhibitory:
                A[i,j] = -1


# Plot the network
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G_directed, seed=1)  # layout for visualization
nx.draw_networkx(
    G_directed,
    pos=pos,
    with_labels=False,
    node_color="darkblue",
    node_size=100,
    edge_color="gray",
    font_size=10,
)
plt.title(f"Erdős–Rényi Graph (n={n}, p={1/np.sqrt(2*n):.3f})")
plt.axis("off")
plt.show()


# #%%
# np.random.seed()

#%% set some parameters
K = 0.1
dt = 0.02; t_end = 500; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)


params = [0.815,3.5,0.2,K,A]

# params = [0.2,0,0.05,K,A]

#run simulation; available methods: RK23,RK45,DOP853,Radau, BDF, LSODA

for i in range(8):
    ICs = np.random.rand(2*n)
    solution = solve_ivp(FHN_coupledjx, (0,t_end),ICs, rtol=1.e-6, atol=1.e-6,t_eval=time,args=([params]), method='RK45') 

    # plot timecourses
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_title('rs: '+ str(rs))


    u_sol = solution.y[:n,:]
    v_sol = solution.y[n:,:]
    for ii in range(n):
        ax.plot(time,u_sol[ii,:],lw=0.75)

    ax.set_xlim(0,t_end);     
    ax.set_xlabel('time'); ax.set_ylabel('value');

#%% plot PCA

# sol_pca = pca.fit_transform(solution.y[:,::4].T).T 

# #%%
# fig = plt.figure(figsize=(12,4))
# ax = fig.add_subplot(1,1,1, projection='3d')
# fun.plot_tc_phasespace_colored(ax,sol_pca[:,1000:],[0,1,2],fileName='test',axlabels=['PCA1','PCA2','PCA3'],mode='velocity',colormap='cool',saveAnimation=False)

# ax.set_xlim(np.min([np.min(sol_pca[0,:]),0]),np.max([np.max(sol_pca[0,:]),1]));
# ax.set_ylim(np.min([np.min(sol_pca[1,:]),0]),np.max([np.max(sol_pca[1,:]),1]));
# ax.set_zlim(np.min([np.min(sol_pca[2,:]),0]),np.max([np.max(sol_pca[2,:]),1]));

# plt.tight_layout()

# %%
