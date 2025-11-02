# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:51:22 2025

@author: dkoch
"""

# Import packages
import numpy as np
import jax 
import jax.numpy as jnp
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from _utils import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os
import sys
from tqdm import tqdm

#######################################

def ghostID(model, params, dt, trajectory, epsilon_Qmin, evLimit=0.1, epsilon_SN_ghosts=0.1, **kwargs):
    
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
    
    Version 0.8
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
        peak_kwargs["width"] = 5 * dt   # default if not supplied
    
    if "batchModel" in kwargs:
        Xs = kwargs["batchModel"](trajectory)
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
        
    if ctrl_qplot:
        if not len(idx_minima)>0:
            t_axis = np.arange(len(Q_ts)) * dt
            plt.figure()
            plt.plot(t_axis, pQ, label="pQ(t)")        
            plt.ylabel("pQ(t)")
            plt.xlabel("t")
            plt.xscale(qplot_xscale)
            plt.yscale(qplot_yscale)
            plt.legend()
            plt.title("Detected Q-minima with prominences")
            plt.tight_layout()
            plt.show()
    
    # Precompile JAX Jacobian function
    J_fun = make_jacfun(model, params)

    # Build KD-tree once for trajectory
    kdtree = cKDTree(trajectory)

    if len(idx_minima) > 0:
        ghostSeq = []             # sequence of visited ghosts 
        ghostTimes = []           # times at which ghosts have been visited - important for sorting ghosts and oscillatory transients
        ghostCoordinates = []     # unique phase-space positions of all ghosts visited
        
        if ctrl_qplot:
            plt.figure()
            t_axis = np.arange(len(Q_ts)) * dt
            plt.plot(t_axis, pQ, label="pQ(t)")        
    
            plt.title("Detected Q-minima with prominences")
            
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
                    
            plt.ylabel("pQ(t)")
            plt.xlabel("t")
            plt.xscale(qplot_xscale)
            plt.yscale(qplot_yscale)
            plt.tight_layout()
            plt.legend()
            plt.show()
            
        for i in idx_minima:
            
            ghostCheck = False
            
            t_ghost = i * dt  # time at which a potential ghost was found
            dur_ghost = pk_props["widths"][np.where(idx_minima==i)[0][0]]*dt # trapping time
            
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
                    ax.plot(t_seg, eigVals_real[:, j], '-ok', lw=1)
                    ax.set_ylabel(f'λ{j+1}', fontsize=12)
    
                axes[-1].set_xlabel('Time along segment', fontsize=12)
                plt.suptitle(f'Eigenvalues near Q-min at t={t_ghost:.3f}, ghost: {ghostCheck}, leaves Uɛ: {leaves_eps_qmin_i}\n'
                              f'evSignChanges: {ev_signChanges}, qualifying slopes: {len(qualifyingSlopes)}, R²: {r2s}')
                plt.xscale(evplot_xscale)
                plt.yscale(evplot_yscale)
                plt.tight_layout()
                plt.show()
                
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
                        "Q-value": Q_ts[i],
                        "Crossing_eigenvalues": np.where(np.array(ev_signChanges)==True)[0],
                        "Qualifying slopes":qualifyingSlopes
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

def ghostID_phaseSpaceSample(model, model_params, t_start, t_end, dt, state_ranges,
                             epsilon_gid=0.1, epsilon_uni=0.1, n_samples=50,
                             method='RK45', rtol=1.e-6, atol=1.e-6,
                             n_workers=None, **kwargs):
    """
    Adaptive parallel version:
      - Uses threads when run in Spyder/Jupyter (no pickling issues)
      - Uses processes when run as a standalone script for full CPU utilization
    """

    # ---- Parameters ----
    peak_kwargs = kwargs.get("peak_kwargs", {})
    ctrlOutputs = kwargs.get("ctrlOutputs", {})
    model_batch = kwargs.get("batchModel", make_batch_model(model, model_params))
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    npts = int(t_end / dt)
    t_eval = np.linspace(0, t_end, npts + 1)
    ICs = phaseSpaceLHS(state_ranges, n_samples)

    # ---- Choose backend automatically ----
    in_spyder_or_jupyter = (
        "SPYDER" in sys.modules or
        "spyder_kernels" in sys.modules or
        "ipykernel" in sys.modules
    )
    Executor = ThreadPoolExecutor if in_spyder_or_jupyter else ProcessPoolExecutor
    mode = "threads" if Executor is ThreadPoolExecutor else "processes"
    print(f"[ghostID_phaseSpaceSample] Running with {mode} ({n_workers} workers)")

    # ---- Define worker ----
    def process_ic(ic):
        sol = solve_ivp(model, (t_start, t_end), ic,
                        t_eval=t_eval, args=(model_params,),
                        method=method, rtol=rtol, atol=atol)
        ghostSeq = ghostID(model, model_params, dt, sol.y.T,
                           epsilon_gid, peak_kwargs=peak_kwargs,
                           batchModel=model_batch,ctrlOutputs=ctrlOutputs)
        return ghostSeq if ghostSeq else None

    # ---- Parallel execution ----
    ghostSeqs = []
    with Executor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_ic, ic) for ic in ICs]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="Processing ICs", unit="IC"):
            res = f.result()
            if res is not None:
                ghostSeqs.append(res)

    # ---- Unify results ----
    ghostSeqs_unified = unify_IDs(ghostSeqs, epsilon_uni)
    return ghostSeqs_unified


def find_local_Qminimum(F, x0, p, delta=0.5, method='L-BFGS-B',
                               n_global_iter=1000, tol_glob=0.01,tol_grad=1e-6, max_iter_local=500,
                               verbose=False):
    """
    Finds a local minimum of Q(x) = 0.5 * ||F||^2 near x0 in arbitrary dimensions.
    Performs radius-constrained global search using differential evolution,
    followed by SciPy local refinement.

    Parameters
    ----------
    F : callable
        Vector field F(t, x, p)
    x0 : array_like
        Initial point in phase space (any dimension)
    p : array_like
        Model parameters
    delta : float
        Maximum distance from x0 for global search and final refinement
    method : str
        SciPy local optimization method ('BFGS', 'L-BFGS-B', 'CG')
    n_global_iter : int
        Number of iterations for differential evolution
    tol_grad : float
        Gradient tolerance for local refinement
    max_iter_local : int
        Maximum iterations for local refinement
    verbose : bool
        Print progress messages

    Returns
    -------
    x_min : np.ndarray
        Coordinates of the local minimum
    Q_min : float
        Q value at the local minimum
    res_local : OptimizeResult
        SciPy local minimization result
    """

    x0 = jnp.array(x0)
    dim = x0.shape[0]

    # Scalar field Q(x) = 0.5 * ||F||²
    def Q_func(x):
        z = jnp.array(x)
        return float(0.5 * jnp.sum(F(0.0, z, p)**2))

    # Gradient using JAX
    grad_Q = lambda x: np.array(jax.grad(lambda z: 0.5*jnp.sum(F(0.0, jnp.array(z), p)**2))(x))

    # -----------------------------
    # Global search using differential evolution
    # -----------------------------
    if verbose:
        print(f"Running global search with differential evolution (radius {delta}) ...")

    # Bounds per dimension for radius constraint
    bounds = [(float(x0[i]-delta), float(x0[i]+delta)) for i in range(dim)]

    result_global = differential_evolution(Q_func, bounds, maxiter=n_global_iter, tol=tol_glob, disp=verbose, polish=False)
    x_global_best = result_global.x
    Q_global_best = Q_func(x_global_best)

    if verbose:
        print(f"Global search best Q = {Q_global_best:.3e}")

    # -----------------------------
    # Optional: local refinement using SciPy minimize
    # -----------------------------
    if method not in ['None']:
        if verbose:
            print(f"Refining local minimum with {method} ...")
    
        res_local = minimize(Q_func, x_global_best, jac=grad_Q, method=method,
                             tol=tol_grad, options={'maxiter': max_iter_local, 'disp': verbose},
                             bounds=bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None)
    
        x_min = res_local.x
        Q_min = Q_func(x_min)
    
        if verbose:
            print(f"Refined local minimum Q = {Q_min:.3e} at x = {x_min}")

        return x_min, Q_min, res_local
    
    return x_global_best, Q_global_best, result_global

def qOnGrid(F, p, coords=None, dim=None, n_points=50, ranges=None, overrides=None, indexing="ij", jit=False):
    if coords is None:
        if dim is None:
            test = F(0.0, jnp.zeros(1), p)
            dim = test.shape[0]

        if ranges is None:
            ranges = [(-2.0, 2.0)]*dim
        elif isinstance(ranges[0], (int,float)):
            ranges = [ranges]*dim

        if isinstance(n_points,int):
            n_points = [n_points]*dim

        coords = []
        for d in range(dim):
            n = n_points[d] if d<len(n_points) else n_points[-1]
            r = ranges[d] if d<len(ranges) else ranges[-1]
            if overrides and d in overrides:
                if "n" in overrides[d]:
                    n = overrides[d]["n"]
                if "range" in overrides[d]:
                    r = overrides[d]["range"]
            coords.append(jnp.linspace(r[0], r[1], n))

    meshes = jnp.meshgrid(*coords, indexing=indexing)
    grid_points = jnp.stack(meshes, axis=-1)

    def core(grid_points):
        flat_pts = grid_points.reshape(-1, grid_points.shape[-1])
        F_vmapped = jax.vmap(lambda pt: F(0.0, pt, p))
        values = F_vmapped(flat_pts)
        Q_flat = 0.5*jnp.sum(values**2, axis=-1)
        return Q_flat.reshape(grid_points.shape[:-1])

    core = jax.jit(core) if jit else core
    return core(grid_points), grid_points
###############

def track_ghost_branch(ghost, model, model_params, par_nr, par_steps, dpar, t_end, dt, delta=0.5, icStep=0.1, mode="first",
                             epsilon_gid=0.1,solve_ivp_method='RK45', rtol=1.e-3, atol=1.e-6, qmin_method="BFGS",qmin_tol=1e-6,**kwargs):
    
    # ---- Parameters ----
    peak_kwargs = kwargs.get("peak_kwargs", {})
    ctrlOutputs = kwargs.get("ctrlOutputs", {})
    model_batch = kwargs.get("batchModel", make_batch_model(model, model_params))
    
    if "distQminThr" in kwargs:
        distQminThr = kwargs["distQminThr"]
    else:
        distQminThr = np.inf 
    
    ghostSeq_p = []
    parSeq = []
    
    parNext = model_params[par_nr] 
    try:
        model_params_ = np.asarray(model_params).copy()
    except:
        model_params_ = model_params.copy()

    ghost_ = ghost.copy()

    i = 0
    with tqdm(total=par_steps + 1) as pbar:
        while i < par_steps+1:
            pct = 100 * i / par_steps
            pbar.set_description(f"Progress: {pct:6.2f}% | param value={parNext:.5f}")
            
            x0 = ghost_["position"]
            
            qmin = find_local_Qminimum(model, x0, model_params_, delta, tol_glob=qmin_tol,method=qmin_method)[0]
                    
            ic_plus, _ , _ = icAtQmin(qmin, icStep ,ghost_["dimension"],model,model_params_)
            ic_minus, _ , _ = icAtQmin(qmin, -icStep ,ghost_["dimension"],model,model_params_)
            
            sol_plus = solve_ivp(model, (0,5*dt), jnp.real(ic_plus), t_eval=np.asarray(np.arange(0, 5*dt, dt)), rtol=rtol, atol=atol, args=([model_params_]), method=solve_ivp_method) 
            sol_minus = solve_ivp(model, (0,5*dt), jnp.real(ic_minus), t_eval=np.asarray(np.arange(0, 5*dt, dt)), rtol=rtol, atol=atol, args=([model_params_]), method=solve_ivp_method) 
                
            dist_ic_plus = np.linalg.norm(qmin-ic_plus)
            dist_sol_plus = np.linalg.norm(qmin-sol_plus.y[:,-1])
            
            dist_ic_minus = np.linalg.norm(qmin-ic_minus)
            dist_sol_minus= np.linalg.norm(qmin-sol_minus.y[:,-1])
                    
            if dist_sol_plus<dist_ic_plus:
                ic_pick = ic_plus
            elif dist_sol_minus<dist_ic_minus:
                ic_pick = ic_minus
            else: 
                print("Error in chosing initial conditions around qmin.")
                return
            
            ic_pick = jnp.real(ic_pick)
        
            sol = solve_ivp(model, (0,t_end), ic_pick, t_eval=np.asarray(np.arange(0, t_end, dt)), rtol=rtol, atol=atol, args=([model_params_]), method=solve_ivp_method)
            
            ghostSeq = ghostID(model, model_params_, dt, sol.y.T,
                            epsilon_gid, peak_kwargs=peak_kwargs,
                            batchModel=model_batch,ctrlOutputs=ctrlOutputs)
            
            if len(ghostSeq)>0:
                
                #append
                if mode=="first":
                    distance = np.linalg.norm(ghostSeq[0]["position"]-qmin)
                    if distance < distQminThr:
                        ghostSeq_p.append(ghostSeq[0])
                        parSeq.append(parNext)
                elif mode == "closest":
                    positions = np.array([ghostSeq[ii]["position"] for ii in range(len(ghostSeq))])
                    distances = np.linalg.norm(positions-qmin,axis=1)
                    idx_min = np.argmin(distances)
                    if distances[idx_min]<distQminThr:
                        ghostSeq_p.append(ghostSeq[idx_min])
                        parSeq.append(parNext)
                else:
                    print("Unknown mode argument. Use default mode instead.")
                    mode = "first"
                    continue
                    
                #update
                parNext = parNext + dpar
                model_params_[par_nr] = parNext
                pbar.update(1)
                i+=1
                
            else: 
                print("No further ghosts found.")
                break
        
    ghostPositions = np.asarray([ghostSeq_p[ii]["position"] for ii in range(len(ghostSeq_p))])
    
    return ghostPositions, np.asarray(parSeq), ghostSeq_p
    

###############

def ghost_connections(gSeq):  
    """
    ##############################################################################################
    Takes list of ghost sequences and turns it into an adjacency matrix.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Input: 
        - gSeq: list of ghost sequences that have been generated by ghostID
    Output:
        - adjM: adjecency matrix representing connections between identified ghosts in phase space
        - labels: labels of matrix rows/columns
    ##############################################################################################
    """
    
    labels = []
    
    for s in gSeq:
        for i in s:
            if i["id"][:1]=="G" and not i["id"] in labels:
                labels.append(i["id"])
    
    ng = len(labels)
    adjM = np.zeros((ng,ng))
    
    seqIDs = [[g["id"] for g in s] for s in gSeq]
    
    for s in seqIDs:
        for i in range(len(s)-1):
            e_out = labels.index(s[i])
            e_in = labels.index(s[i+1])
            if adjM[e_out,e_in]==0:
                adjM[e_out,e_in]=1
            
    return adjM, labels



def unique_ghosts(gSeq):  
    """
    ##############################################################################################
    Takes list of unified ghost sequences and returns a list of all unique ghosts
    ##############################################################################################
    """
    
    ghostIDs = []
    ghostsUnique = []
    
    for s in gSeq:
        for i in s:
            if i["id"][:1]=="G" and not i["id"] in ghostIDs:
                ghostIDs.append(i["id"])
                ghostsUnique.append(i)
    
    return ghostsUnique
                
    
    
            
#############



def unify_IDs(Seqs, epsilon_SN_ghosts=0.1):
    """
    Unify ids across multiple sequences of transient ghost state objects found in timecourse simulations.

    Arguments:
        Seqs: list of lists of dicts, each dict with keys 'position' (np.ndarray) and 'id' (str: 'G{i}').
        epsilon_SN_gs: distance threshold for equating SN ghosts.

    Returns:
        The same list Seqs with updated, unified 'id' strings.
    """
    # Initialize known objects from first sequence
    first = Seqs[0]
    known_G = {}  # id_str -> position (representative)

    # Track maximum numeric index seen
    max_g = 0

    for obj in first:
        pid = obj['id']
        if pid.startswith('G'):
            idx = int(pid[1:])
            known_G[pid] = obj['position'].copy()
            max_g = max(max_g, idx)
    
    # Process subsequent sequences
    for seq in Seqs[1:]:
        for obj in seq:
            pos = obj['position']
            orig = obj['id']
            if orig.startswith('G'):
                # check against known_G
                matched = False
                for pid, refpos in known_G.items():
                    if np.linalg.norm(pos - refpos) < epsilon_SN_ghosts:
                        obj['id'] = pid
                        matched = True
                        break
                if not matched:
                    max_g += 1
                    new_id = f'G{max_g}'
                    obj['id'] = new_id
                    known_G[new_id] = pos.copy()   
            else:
                # Unknown id format, leave unchanged or raise error
                raise ValueError(f"Unrecognized id '{orig}'")
    return Seqs


######################################

def draw_network(adj_matrix, nodeColMap, nlbls):

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