# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 23:09:19 2025

@author: Daniel Koch

"""
# Import packages
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib as mpl
from scipy.optimize import approx_fprime
from scipy.signal import find_peaks
# from scipy.signal import argrelextrema
# import nolds
# import matplotlib.pylab as pylab
# import warnings
# from scipy.integrate import solve_ivp
import networkx as nx
import os
import sys
import jax.numpy as jnp
# from jax.nn import sigmoid
from scipy.spatial import cKDTree
import jax 
from jax import jacfwd
# from matplotlib.colors import LogNorm
# from scipy.optimize import approx_fprime 
# import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.patches import FancyArrowPatch

#paths

sys.path.append(os.path.join( os.path.dirname( __file__ ), '..' ))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

options = {"node_color": "lightblue", "node_size": 220, "linewidths": 0.1, "width": 0.3}
inCm = 1/2.54

# Auxilliary functions needed for the algorithm

# def qAtPt(F,p,Pt): # F(t,xi,parameters) is your model (ODE system)
#     X = F(0,Pt,p) 
#     return np.sum(X**2)/2 #Q

# def qAtPt_vectorized(F, p, Pt_array):
#     """
#     Vectorized version - requires F to support vectorized operations
#     """
#     # Evaluate F for all points at once (if F supports vectorization)
#     X = F(0, Pt_array, p)
    
#     # Calculate sum of squares along the appropriate axis
#     # Assuming X has shape (n_points, n_variables)
#     return np.sum(X**2, axis=1) / 2


def sign_change(arr):
    """
    Returns True if and only if:
      1. arr has at least two elements
      2. arr[0] < 0  (starts negative)
      3. arr[-1] > 0 (ends positive)
      AND 
      4. Values are increasing (arr[i] >= arr[i-1])
      5. There is exactly one transition where prev < 0 and curr > 0
      OR
      6. The slope of the fitted values is positive and R² >= 95

    Zero values or held-constant segments will cause it to return False.
    """
    if len(arr) < 2:
        return False

    if arr[0] >= 0 or arr[-1] <= 0:
        return False


    sign_change_occured = False
    tryFit = False
    for prev, curr in zip(arr, arr[1:]):
        if curr < prev:
            print("Error: monotonicity violated. Trying linear fit.")
            tryFit = True
            break         
        # 2) detect the one negative→positive jump
        if prev < 0 and curr > 0:
            if not sign_change_occured:
                sign_change_occured = True
            else:
                print("Error: more than one sign changes occured. Trying linear fit.")
                sign_change_occured = False
                tryFit = True
                break

    if tryFit:
        
        x = np.asarray(range(0,len(arr)))
        
        coeffs = np.polyfit(x, arr, 1)
        arr_pred = np.polyval(coeffs, x)
        
        # R² calculation
        ss_res = np.sum((arr - arr_pred) ** 2)
        ss_tot = np.sum((arr - np.mean(arr)) ** 2)
        r2 = 1 - ss_res/ss_tot

        if r2 >= 0.95 and coeffs[0] > 0:
            sign_change_occured = True
        
    return sign_change_occured


def trjSegment(idcs,iq):

    start = np.searchsorted(idcs, iq)

    a,b=iq,iq
    
    for i in range(start-1,-1,-1):
        d = idcs[i+1]-idcs[i]
        if d != 1:
            break
        a = i
    
    for i in range(start+1,len(idcs)):
        d = idcs[i]-idcs[i-1]
        if d != 1:
            break
        b = i
        
    return idcs[np.arange(a, b+1, 1, dtype=int)]

def make_batch_model(model, params):
    """
    Wrap a single-point model into a batch version using vmap.
    
    model: function (t, z, params) -> dz/dt
    params: model parameters (passed unchanged)
    
    Returns: function (Zs, params) -> dZs/dt for batch input
             where Zs has shape (num_points, n).
    """
    def single(z):
        return model(0, z, params)   # ignore t (or pass if needed)
    
    batched = jax.vmap(single)
    return batched

# ---------------- JAX Jacobian utility ----------------
def make_jacfun(model, params):
    F = lambda x: model(0, x, params)
    J_fun = jax.jacfwd(F)
    return jax.jit(J_fun)

# ---------------- Fast slope calculation ----------------
def slope_and_r2(y, dt):
    """
    Compute slope and R² of linear regression of y vs time
    y: shape (N,)
    dt: time step
    """
    N = len(y)
    x = np.arange(N) * dt
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    y_pred = slope * (x - x_mean) + y_mean
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0
    return slope, r2

def get_ctrl_plot_settings(kwargs, key, default_x="linear", default_y="linear"):
    ctrl = kwargs.get("ctrlOutputs", {})
    flag = ctrl.get(f"ctrl_{key}", False)

    if not isinstance(flag, bool):
        raise TypeError(f"ctrl_{key} must be boolean, got {type(flag).__name__}")
    if not flag:
        return False, None, None

    xscale = ctrl.get(f"{key}_xscale", default_x)
    yscale = ctrl.get(f"{key}_yscale", default_y)

    if xscale not in ("linear", "log"):
        raise ValueError(f"{key}_xscale must be 'linear' or 'log', got {xscale!r}")
    if yscale not in ("linear", "log"):
        raise ValueError(f"{key}_yscale must be 'linear' or 'log', got {yscale!r}")

    return True, xscale, yscale

# ---------------- ghostID algorithm ----------------
def ghostID_v08(model, params, dt, trajectory, epsilon_Qmin, evLimit=0.1, epsilon_SN_ghosts=0.1, **kwargs):
    
    """ HYPERPARAMETERS OF THE ALGORITHM
    
    #####################################################################################################
    epsilon_Qmin         - distance around Q-minima in which a trajectory segment is evaluated
                           along which to evaluate eigenvalues
    evLimit              - maximum value below which the absolute averaged eigenvalues of a trajectory
                           are considered to be close enough to 0    
    epsilon_SN_ghosts    - distance below which ghosts are considered to be the same
    
    OPTIONAL PARAMETERS
    #####################################################################################################
    slopeLimits          - upper and lower limits for positive eigenvalue slopes
    ctrlOutputs          - dict, enables control outputs for various quantities calculated by the algortihm
    peak_kwargs          - dict, additional arguments for scipy.signal.find_peaks
    """
    
         
    slopeLimits = [0, np.inf]
    if "slopeLimits" in kwargs:
        if np.all(kwargs["slopeLimits"] >= 0) and kwargs["slopeLimits"][0] < kwargs["slopeLimits"][1]:
            slopeLimits = kwargs["slopeLimits"]
        else:
            print("Invalid values for \"slopeLimits\".")
            return
        
    # Peak detection kwargs
    peak_kwargs = kwargs.get("peak_kwargs", {})
    if "width" not in peak_kwargs:  
        peak_kwargs["width"] = 10 * dt   # default if not supplied
    
    if "batchModel" in kwargs:
        Xs = kwargs["batchModel"](trajectory,params)
    else:
        model_batch = make_batch_model(model, params)
        Xs = model_batch(trajectory)  

    ctrl_qplot, qplot_xscale, qplot_yscale = get_ctrl_plot_settings(kwargs, "qplot")
    ctrl_evplot, evplot_xscale, evplot_yscale = get_ctrl_plot_settings(kwargs, "evplot")
            
    n = trajectory.shape[1]  # dimension of trajectory
    fullTransientSeq = []  # list of visited transient states to be filled
              
    ############# STEP 1 - identify non-oscillatory saddle-node ghosts #############################
    
    
    ### Identify minima in Q-values along trajectory using batch model output
    Q_ts = 0.5 * np.sum(Xs**2, axis=1)
    pQ = -np.log(Q_ts)

    idx_minima, pk_props = find_peaks(pQ, **peak_kwargs)  # positions of Q-minima
        
    # Precompile JAX Jacobian function
    J_fun = make_jacfun(model, params)

    # Build KD-tree once for trajectory
    kdtree = cKDTree(trajectory)

    if len(idx_minima) > 0:
        ghostSeq = []             # sequence of visited ghosts 
        ghostTimes = []           # times at which ghosts have been visited - important for sorting ghosts and oscillatory transients
        ghostCoordinates = []     # unique phase-space positions of all ghosts visited
        
        if ctrl_qplot:
            t_axis = np.arange(len(Q_ts)) * dt
        
            plt.figure()
            plt.plot(t_axis, pQ, label="pQ(t)")
            plt.plot(idx_minima * dt, pQ[idx_minima], 'xr', label="Q-minima")
        
            # Plot prominence lines
            if "prominences" in pk_props:
                prominences = pk_props["prominences"]
                for idx, prom in zip(idx_minima, prominences):
                    x = idx * dt
                    peak_val = pQ[idx]
                    base_val = peak_val - prom
                    # vertical line showing prominence
                    plt.vlines(x, base_val, peak_val, color="gray", linestyle="--")
                    # add text next to line
                    plt.text(x, base_val - 0.05 * prom, f"{prom:.2f}",
                             ha="center", va="top", fontsize=9, color="gray")
        
            plt.xlim(0, len(Q_ts) * dt)
            plt.ylabel("pQ(t)")
            plt.xlabel("t")
            plt.xscale(qplot_xscale)
            plt.yscale(qplot_yscale)
            plt.legend()
            plt.title("Detected Q-minima with prominences")
            plt.tight_layout()
    
        for i in idx_minima:
            
            ghostCheck = False
            
            t_ghost = i * dt  # time at which a potential ghost was found
            dur_ghost = pk_props["widths"][np.where(idx_minima==i)[0][0]] # trapping time
            
            qmin_xyz = trajectory[i] # position of Q_minimum in phase space 
            
            # KD-tree neighborhood query
            idcs_Ueps_qmin = kdtree.query_ball_point(qmin_xyz, epsilon_Qmin)
            idcs_Ueps_qmin = np.sort(np.asarray(idcs_Ueps_qmin, dtype=int))
            idcs_segment = trjSegment(idcs_Ueps_qmin, i)
            
            # Batch Jacobian + eigenvalue evaluation for segment
            pts_segment = jnp.asarray(trajectory[idcs_segment])        # JAX array
            J_batch = jax.vmap(J_fun)(pts_segment)                     # batch Jacobians
            eigVals = jax.vmap(jnp.linalg.eigvals)(J_batch)            # eigenvalues
            eigVals_real = np.real(np.asarray(eigVals))                # back to numpy for analysis
                
            # Determine eigenvalue crossings along the segment
            ev_signChanges = [sign_change(eigVals_real[:, ii]) for ii in range(n)]
            crossings = sum(ev_signChanges)
            
            # determine eigenvalue slopes
            qualifyingSlopes = []
            
            if ctrl_evplot: r2s = []
            for ii in range(n):
                # Only consider eigenvalues with small mean real part along segment
                if np.abs(np.average(eigVals_real[:, ii])) < evLimit:
                    slope, r2 = slope_and_r2(eigVals_real[:, ii], dt)
                    if ctrl_evplot: r2s.append(r2)
                    if np.all((slope > slopeLimits[0]) & (slope < slopeLimits[1]) & (r2 >= 0.99)):
                        qualifyingSlopes.append(ii)

            # Check for ghost
            leaves_eps_qmin_i = False
            if crossings > 0 or len(qualifyingSlopes) > 0:
                # Check if trajectory leaves the epsilon environment
                dists = np.linalg.norm(trajectory[i:] - qmin_xyz, axis=1)
                if np.any(dists > epsilon_Qmin):
                    ghostCheck = True
                    leaves_eps_qmin_i = True
            
            if ctrl_evplot:
                n_eig = eigVals_real.shape[1]
                fig, axes = plt.subplots(n_eig, 1, figsize=(8, 2*n_eig), sharex=True)
                if n_eig == 1: axes = [axes]  # ensure axes is iterable
                
                t_seg = np.arange(len(eigVals_real)) * dt
                
                for j, ax in enumerate(axes):
                    ax.plot(t_seg, eigVals_real[:, j], '-', lw=2)
                    ax.set_ylabel(f'$\\lambda_{j+1}$', fontsize=12)
                
                axes[-1].set_xlabel('Time along segment', fontsize=12)
                plt.suptitle(f'Eigenvalues near Q-min at t={t_ghost:.3f}, ghost: {ghostCheck}, leaves U$_\\epsilon$: {leaves_eps_qmin_i}\n'
                              f'evSignChanges: {ev_signChanges}, qualifying slopes: {len(qualifyingSlopes)}, R²: {r2s}')
                plt.xscale(evplot_xscale)
                plt.yscale(evplot_yscale)
                plt.tight_layout()
                
            # If ghost found, characterize its dimension and check if it has been found previously
            if ghostCheck:
                ghostTimes.append(t_ghost)
                gdim = max([crossings, len(qualifyingSlopes)])

                if len(ghostCoordinates) > 0:
                    # Calculate distances to all previously found ghosts
                    distances = np.asarray([np.linalg.norm(g - trajectory[i]) for g in ghostCoordinates])
                
                    if not any(d < epsilon_SN_ghosts for d in distances):  # current ghost has not been found previously
                        ghostCoordinates.append(trajectory[i])
                        ghost = { 
                            "id": "G" + str(len(ghostCoordinates)),  # assign ID to the new ghost
                            "time": t_ghost,
                            "duration": dur_ghost,
                            "position": trajectory[i],
                            "dimension": gdim,
                            "Q-value:": Q_ts[i],
                            "Crossing_eigenvalues": np.where(np.array(ev_signChanges)==True)[0],
                            "Qualifying slopes:":qualifyingSlopes
                            }
                    else:  # current ghost has already been found previously
                        gidx = np.where(distances < epsilon_SN_ghosts)[0][0] + 1
                        ghost = {
                            "id": "G" + str(gidx),
                            "time": t_ghost,
                            "duration": dur_ghost,
                            "position": trajectory[i],
                            "dimension": gdim,
                            "Q-value:": Q_ts[i],
                            "Crossing_eigenvalues": np.where(np.array(ev_signChanges)==True)[0],
                            "Qualifying slopes:":qualifyingSlopes
                            }
                    ghostSeq.append(ghost)
                else:  # No ghost previously found yet
                    ghost = {
                        "id": "G" + str(len(ghostCoordinates) + 1),
                        "time": t_ghost,
                        "duration": dur_ghost,
                        "position": trajectory[i],
                        "dimension": gdim,
                        "Q-value:": Q_ts[i],
                        "Crossing_eigenvalues": np.where(np.array(ev_signChanges)==True)[0],
                        "Qualifying slopes:":qualifyingSlopes
                        }
                    ghostSeq.append(ghost)
                    ghostCoordinates.append(trajectory[i])

        ############# STEP 2 - identify oscillatory transients #############################
        oscSeq = []
        oscTimes = []
                             
        # Merge transient state lists
        allTimes = np.asarray(ghostTimes + oscTimes)
        allTimes.sort()
        ghostTimes = np.asarray(ghostTimes)
        oscTimes = np.asarray(oscTimes)
        
        for t in allTimes:
            i_t = np.where(ghostTimes == t)[0]
            if len(i_t) == 1:
                fullTransientSeq.append(ghostSeq[i_t[0]])
            else:
                i_t = np.where(oscTimes == t)[0]
                fullTransientSeq.append(oscSeq[i_t[0]])
        
    return fullTransientSeq



def unifyIDs(Seqs, epsilon_SN_gs=0.1, epsilon_osc=0.66):
    """
    Unify ids across multiple simulation sequences of transient state objects.

    Arguments:
        Seqs: list of lists of dicts, each dict with keys 'position' (np.ndarray) and 'id' (str: 'Gn' or 'Om').
        epsilon_SN_gs: distance threshold for equating SN ghosts (G).
        epsilon_osc: distance threshold for equating oscillatory states (O).

    Returns:
        The same list Seqs with updated, unified 'id' strings.
    """
    # Initialize known objects from first sequence
    first = Seqs[0]
    known_G = {}  # id_str -> position (representative)
    known_O = {}  # id_str -> position

    # Track maximum numeric index seen
    max_g = 0
    max_o = 0

    for obj in first:
        pid = obj['id']
        if pid.startswith('G'):
            idx = int(pid[1:])
            known_G[pid] = obj['position'].copy()
            max_g = max(max_g, idx)
        elif pid.startswith('O'):
            idx = int(pid[1:])
            known_O[pid] = obj['position'].copy()
            max_o = max(max_o, idx)

    # Process subsequent sequences
    for seq in Seqs[1:]:
        for obj in seq:
            pos = obj['position']
            orig = obj['id']
            if orig.startswith('G'):
                # check against known_G
                matched = False
                for pid, refpos in known_G.items():
                    if np.linalg.norm(pos - refpos) < epsilon_SN_gs:
                        obj['id'] = pid
                        matched = True
                        break
                if not matched:
                    max_g += 1
                    new_id = f'G{max_g}'
                    obj['id'] = new_id
                    known_G[new_id] = pos.copy()
            elif orig.startswith('O'):
                # check against known_O
                matched = False
                for pid, refpos in known_O.items():
                    if np.linalg.norm(pos - refpos) < epsilon_osc:
                        obj['id'] = pid
                        matched = True
                        break
                if not matched:
                    max_o += 1
                    new_id = f'O{max_o}'
                    obj['id'] = new_id
                    known_O[new_id] = pos.copy()
            else:
                # Unknown id format, leave unchanged or raise error
                raise ValueError(f"Unrecognized id '{orig}'")
    return Seqs

# Functions for drawing nice networks

def draw_custom_edges(G, pos, edgelist, color="red", head_length=10, head_width=5, width=1, rad=0.1, trim_fraction=0.1):
    """
    Draws edges using FancyArrowPatch. Instead of drawing an arrow from the exact start to end points,
    the arrow is drawn only along the inner portion of the edge. The arrow starts at:
      (x1, y1) + trim_fraction * (x2 - x1, y2 - y1)
    and ends at:
      (x1, y1) + (1 - trim_fraction) * (x2 - x1, y2 - y1)
    
    mutation_scale is set to 1.
    """
    ax = plt.gca()  # Get current axis
    for u, v in edgelist:
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            tf = trim_fraction
            # Compute trimmed start and end points
            l = tf * np.sqrt((y2-y1)**2+(x2-x1)**2)
            if l > 30000: 
                tf = 28000/np.sqrt((y2-y1)**2+(x2-x1)**2)
                
            # if l < 10000: 
            #     tf = 10000/np.sqrt((y2-y1)**2+(x2-x1)**2)
            start = (x1 + tf * (x2 - x1), y1 + tf * (y2 - y1))
            end = (x1 + (1 - tf) * (x2 - x1), y1 + (1 - tf) * (y2 - y1))
            
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
                color=color,
                linewidth=width,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=1  # Set mutation_scale as requested
            )
            ax.add_patch(arrow)

def drawNetwork(adj_matrix, nodeColMap, nlbls):

    nw_dim = adj_matrix.shape[0]
    G = nx.from_numpy_array(adj_matrix.transpose(), create_using=nx.DiGraph)

    # Define edges
    inhEdges, actEdges = [], []
    for i in range(nw_dim):
        for ii in range(nw_dim):
            if adj_matrix[i, ii] == 1:
                G.add_edge(i, ii, weight=1)
                actEdges.append((i, ii))
            elif adj_matrix[i, ii] == -1:
                G.add_edge(i, ii, weight=-1)
                inhEdges.append((i, ii))

    # Apply Graphviz 'dot' layout
    graphviz_args = "-Nwidth=350 -Nheight=350 -Nfixedsize=true -Goverlap=scale -Gnodesep=5000 -Granksep=200 -Nshape=oval -Nfontsize=14 -Econstraint=true"
    pos = graphviz_layout(G, prog='fdp',args=graphviz_args)

    # Draw nodes
    node_opts = {"node_size": 1800, "edgecolors": "lightgrey"}
    nx.draw_networkx_nodes(G, pos, node_color=nodeColMap, **node_opts, alpha=0.90)
    
    # Draw custom edges using the modified arrow approach.
    # Adjust head_length, head_width, etc. as desired.
    draw_custom_edges(G, pos, actEdges, color="k", head_length=8, head_width=4, width=1.1, trim_fraction=0.2)
    draw_custom_edges(G, pos, inhEdges, color="red", head_length=8, head_width=4, width=1.1, trim_fraction=0.2)
    
    # Draw labels
    labels = {i: nlbls[i] for i in range(nw_dim)}
    nx.draw_networkx_labels(G, pos, labels, font_size=16.5, font_weight="bold")

    plt.tight_layout()
    plt.axis("off")
    
    
# functions for phase space analyses

def vector_field(reaction_terms,grid,dim):
    
    '''
    This function returns the local reaction rates at grid points of interest.
    Used then for plotting phase space flows
    
    inputs
    ----------
    
    reaction_terms : callable function that returns the ode 
    grid: spatial grid of aread of interest. This region must include the slow point region
    dim: dimensionality of the system. Works for 2D and 3D systems
    
    returns
    ----------
    
    multidimensional array of velocity components
    
    '''
  
    if dim=='3D':
        Xg,Yg,Zg=grid
        
        x_range=Xg[0]
        y_range=Yg[:,0]
        z_range=Zg[0]
        
        Lx,Ly,Lz=len(x_range),len(y_range),len(z_range)
        U=np.zeros((Lx,Ly,Lz));V=np.zeros((Lx,Ly,Lz));W=np.zeros((Lx,Ly,Lz))
        
        for i in range(Lx):
            for j in range(Ly): 
                for k in range(Lz): 
                    U[i,j,k],V[i,j,k],W[i,j,k]=reaction_terms(0,[Xg[i,j,k],Yg[i,j,k],Zg[i,j,k]])
        
        return U,V,W
    
    elif dim=='2D':
        
        Xg,Yg=grid
        
        x_range=Xg[0]
        y_range=Yg[:,0]
        
        Lx,Ly=len(x_range),len(y_range)
        # U=np.zeros((Lx,Ly));V=np.zeros((Lx,Ly))
        
        U=np.empty((Lx,Ly),np.float64);V=np.empty((Lx,Ly),np.float64)
        
        for i in range(Lx):
            for j in range(Ly):  
                U[i,j],V[i,j]=reaction_terms(0,[Xg[i,j],Yg[i,j]])
        return U,V

def qOnGrid(F, p, X_grid, Y_grid):
    Q_grid = np.zeros_like(X_grid)

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            Pt = np.array([X_grid[i, j], Y_grid[i, j]])
            X = F(0, Pt, p)
            Q_grid[i, j] = np.sum(X**2) / 2

    return Q_grid

def eigValsOnGrid(F, X_grid, Y_grid):

        ev1 = np.zeros_like(X_grid)
        ev2 = np.zeros_like(X_grid)

        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                Pt = np.array([X_grid[i, j], Y_grid[i, j]])
                
                jac = approx_fprime(Pt,F,epsilon=1e-4)
                ev1[i,j],ev2[i,j] = np.linalg.eigvals(jac)

        return ev1, ev2

# def qAtPt_vectorized(F, p, Pt_array):
#     """
#     Vectorized version - requires F to support vectorized operations
#     """
#     # Evaluate F for all points at once (if F supports vectorization)
#     X = F(0, Pt_array, p)
    
#     # Calculate sum of squares along the appropriate axis
#     # Assuming X has shape (n_points, n_variables)
#     return np.sum(X**2, axis=1) / 2
