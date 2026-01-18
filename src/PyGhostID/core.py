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
from scipy.optimize import (
    minimize,
    differential_evolution,
    dual_annealing,
    basinhopping,
)
from scipy.stats import qmc

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from ._utils import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os
import sys
from tqdm import tqdm

#######################################

def ghostID(model, params, dt, trajectory, epsilon_Qmin=0.05, **kwargs):
    
    """ HYPERPARAMETERS OF THE ALGORITHM
    
    #####################################################################################################
    epsilon_Qmin            - distance around Q-minima in which a trajectory segment is evaluated
                              along which to evaluate eigenvalues  
    
    
    OPTIONAL PARAMETERS (kwargs)
    #####################################################################################################
    epsilon_SN_ghosts       - distance below which ghosts are considered to be the same
    peak_kwargs             - dict, additional arguments for scipy.signal.find_peaks
    evLimit                 - maximum value below which the absolute averaged eigenvalues of a trajectory
                              are considered to be close enough to 0. Set to >0 if you want to enable indirect
                              identification of ghosts.
    slopeLimits             - upper and lower limits for positive eigenvalue slopes. Will be ignored if evLimit = 0.
    eigval_NN_sorting       - Nearest-neigbor reconstruction of eigenvalue timeseries. 
                              Use if eigenvalue timeseries appear scattered/discontinues.
    ev_outlier_removal      - Boolean, whether to apply outlier removal to eigenvalue timeseries before further processing.
    ev_outlier_removal_ws   - Size of sliding window for outlier removal in eigenvalue timeseries. Will be ignored if ev_outlier_removal=False.
    ev_outlier_removal_k    - Size of the filter for outlier removal in eigenvalue timeseries. 
                              Values outsisde +/- k*interquartile ranges are removed. Will be ignored if ev_outlier_removal=False.

    Version 0.9
    """

    # Parse and validate kwargs
    config = parse_kwargs(**kwargs)
    
    # Extract parameters from config
    display_warnings = config['display_warnings']
    epsilon_SN_ghosts = config['epsilon_SN_ghosts']
    peak_kwargs = config['peak_kwargs']
    if "width" not in peak_kwargs:  
        peak_kwargs["width"] = 5 * dt   # default if not supplied
    batchModel = config['batchModel']
    eigval_NN_sorting = config['eigval_NN_sorting']
    ev_outlier_removal = config['ev_outlier_removal']
    ev_outlier_removal_ws = config['ev_outlier_removal_ws']
    ev_outlier_removal_k = config['ev_outlier_removal_k']
    evLimit = config['evLimit']
    slopeLimits = config['slopeLimits']
    
    # Plotting control settings
    return_ctrl_figs = config['return_ctrl_figs']
    ctrl_qplot = config['ctrl_qplot']
    qplot_xscale = config['qplot_xscale']
    qplot_yscale = config['qplot_yscale']
    ctrl_evplot = config['ctrl_evplot']
    evplot_xscale = config['evplot_xscale']
    evplot_yscale = config['evplot_yscale']
          
    ####################################

    # Handle batch model
    if batchModel is not None:
        Xs = batchModel(trajectory)
    else:
        model_batch = make_batch_model(model, params)
        Xs = model_batch(trajectory)

    if return_ctrl_figs:
        ctrl_figures = []
            
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
            fig, ax = plt.subplots()
            fig.set_size_inches(17/(2*2.54),17/(3*2.54))
            ax.plot(t_axis, pQ, '-k', lw=0.8, label="pQ(t)")        
            ax.set_ylabel("pQ(t)")
            ax.set_xlabel("t")
            ax.set_xscale(qplot_xscale)
            ax.set_yscale(qplot_yscale)
            ax.legend(fontsize = 9)
            ax.set_title("Detected Q-minima with prominences",fontsize=12)
            plt.tight_layout()
            if return_ctrl_figs:
                ctrl_figures.append((fig,ax))
                plt.close(fig)
            else:
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

            t_axis = np.arange(len(Q_ts)) * dt
            fig, ax = plt.subplots()
            fig.set_size_inches(17/(2*2.54),17/(3*2.54))
            ax.plot(t_axis, pQ, '-k', lw=0.8, label="pQ(t)")       
            ax.plot(idx_minima * dt, pQ[idx_minima], 'xr', label="Q-minima")
            # Plot prominence lines
            if "prominences" in pk_props:
                prominences = pk_props["prominences"]
                for idx, prom in zip(idx_minima, prominences):
                    x = idx * dt
                    peak_val = pQ[idx]
                    base_val = peak_val - prom
                    # vertical line showing prominence
                    ax.vlines(x, base_val, peak_val, color="gray", linestyle="--")
                    # add text next to line
                    ax.text(x, base_val - 0.05 * prom, f"{prom:.2f}",
                              ha="center", va="top", fontsize=7.5, color="gray")
                    
            ax.set_ylabel("pQ(t)")
            ax.set_xlabel("t")
            ax.set_xscale(qplot_xscale)
            ax.set_yscale(qplot_yscale)
            ax.legend(fontsize = 9)
            ax.set_title("Detected Q-minima with prominences",fontsize=12)
            plt.tight_layout()
            if return_ctrl_figs:
                ctrl_figures.append((fig,ax))
                plt.close(fig)
            else:
                plt.show()
    
            
        for i in idx_minima:
            
            ghostCheck = False
            
            t_ghost = i * dt  # time at which a potential ghost was found
            dur_ghost = pk_props["widths"][np.where(idx_minima==i)[0][0]]*dt # trapping time
            
            qmin_xyz = trajectory[i] # position of Q_minimum in phase space 
            
            # KD-tree neighborhood query
            idcs_Ueps_qmin = kdtree.query_ball_point(qmin_xyz, epsilon_Qmin)
            idcs_Ueps_qmin = np.sort(np.asarray(idcs_Ueps_qmin, dtype=int))
            
            # print(len(idcs_Ueps_qmin), np.min(idcs_Ueps_qmin), np.max(idcs_Ueps_qmin))
            if len(idcs_Ueps_qmin)<5:
                print("ghostID error: insuffienct number of points in Ueps!")
                if not(return_ctrl_figs):    
                    return []
                else:
                    return [], ctrl_figures

            idcs_segment = trjSegment(idcs_Ueps_qmin, i)

            # Check if trajectory leaves the epsilon environment
            leaves_eps_qmin_i = False
            dists = np.linalg.norm(trajectory[i:] - qmin_xyz, axis=1)
            if np.any(dists > epsilon_Qmin):
                leaves_eps_qmin_i = True
                
            if leaves_eps_qmin_i:
                # Batch Jacobian + eigenvalue evaluation for segment
                pts_segment = jnp.asarray(trajectory[idcs_segment])        # JAX array
                J_batch = jax.vmap(J_fun)(pts_segment)                     # batch Jacobians
                eigVals = jax.vmap(jnp.linalg.eigvals)(J_batch)            # eigenvalues
                eigVals_real_ = np.real(np.asarray(eigVals))                # back to numpy for analysis

                if eigval_NN_sorting:
                    eigVals_real = sort_NN(eigVals_real_.T).T
                else:
                    eigVals_real = eigVals_real_
                    
                # Determine eigenvalue crossings along the segment
                ev_signChanges = [sign_change(eigVals_real[:, ii],ev_outlier_removal,ev_outlier_removal_ws,ev_outlier_removal_k,display_warnings=display_warnings) for ii in range(n)]
                crossings = sum(ev_signChanges)
                
                # determine eigenvalue slopes
                qualifyingSlopes = []
                
                if ctrl_evplot: r2s = []
                for ii in range(n):
                    # Only consider eigenvalues with small median real part along segment
                    if np.abs(np.median(eigVals_real[:, ii])) < evLimit:
                        slope, r2 = slope_and_r2(eigVals_real[:, ii], dt,ev_outlier_removal,ev_outlier_removal_ws,ev_outlier_removal_k)
                        if ctrl_evplot: r2s.append(r2)
                        if np.all((slope > slopeLimits[0]) & (slope < slopeLimits[1]) & (r2 >= 0.99)):
                            qualifyingSlopes.append(ii)

                # Check for ghost
                if crossings > 0 or len(qualifyingSlopes) > 0:
                        ghostCheck = True
                
                if ctrl_evplot:
                    n_eig = eigVals_real.shape[1]
                    fig, axes = plt.subplots(n_eig, 1, figsize=(17/(2*2.54), 17/(4*2.54)*n_eig), sharex=True)
                    if n_eig == 1: axes = [axes]  # ensure axes is iterable
                    
                    t_seg = np.arange(len(eigVals_real)) * dt
                    
                    for j, ax in enumerate(axes):
                        ax.plot(t_seg, eigVals_real[:, j], '-ok',  markersize=1.5, lw=0.75)
                        ax.set_ylabel(f'λ{j+1}')
        
                    axes[-1].set_xlabel('Time along segment')
                    if evLimit > 0:
                        qsl = len(qualifyingSlopes)
                    else:
                        qsl = 'N/A'

                    Q_mami = np.max(Q_ts[idcs_segment])/Q_ts[i]

                    plt.suptitle(
                    f"Eig.vals near Qmin at t={t_ghost:.2f}, ghost: {str(ghostCheck)[0]}, leaves Uɛ: {str(leaves_eps_qmin_i)[0]}, Qmami: {Q_mami:.1e} \n"
                    f"sign changes λi: " + "".join(np.where(ev_signChanges, 'T ', 'F '))+f", qualifying slopes: {qsl}, "
                    f"R²: {[f'{ri:.3f}' for ri in r2s]}", fontsize=9)
                    # plt.suptitle(
                    # f"Eigenvalues near Q-min at t={t_ghost:.2f}, ghost: {str(ghostCheck)[0]}, leaves Uɛ: {str(leaves_eps_qmin_i)[0]}, "
                    # f"Qmami: {Q_mami:.1e}" if Q_mami >= 1000 else f"Qmami: {Q_mami:.1f}\n"
                    # f"sign changes λi: " + "".join(np.where(ev_signChanges, 'T ', 'F ')) + f", qualifying slopes: {qsl}, "
                    # f"R²: {[f'{ri:.3f}' for ri in r2s]}", fontsize=9)
                    ax.set_xscale(evplot_xscale)
                    ax.set_yscale(evplot_yscale)
                    plt.tight_layout()
                    if return_ctrl_figs:
                        ctrl_figures.append((fig,axes))
                        plt.close(fig)
                    else:
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
                                "q-value": Q_ts[i],
                                "crossing_eigenvalues": np.where(np.array(ev_signChanges)==True)[0],
                                "qualifying_slopes":qualifyingSlopes
                                }
                        else:  # current ghost has already been found previously
                            gidx = np.where(distances < epsilon_SN_ghosts)[0][0] + 1
                            ghost = {
                                "id": "G" + str(gidx),
                                "time": t_ghost,
                                "duration": dur_ghost,
                                "position": trajectory[i],
                                "dimension": gdim,
                                "q-value": Q_ts[i],
                                "crossing_eigenvalues": np.where(np.array(ev_signChanges)==True)[0],
                                "qualifying_slopes":qualifyingSlopes
                                }
                        ghostSeq.append(ghost)
                    else:  # No ghost previously found yet
                        ghost = {
                            "id": "G" + str(len(ghostCoordinates) + 1),
                            "time": t_ghost,
                            "duration": dur_ghost,
                            "position": trajectory[i],
                            "dimension": gdim,
                            "q-value": Q_ts[i],
                            "crossing_eigenvalues": np.where(np.array(ev_signChanges)==True)[0],
                            "qualifying_slopes":qualifyingSlopes
                            }
                        ghostSeq.append(ghost)
                        ghostCoordinates.append(trajectory[i])
            else:
                if display_warnings:
                    print("GhostID: Trajectory does not leave U_eps - stopping ghostID.")
                break


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

    if not(return_ctrl_figs):    
        return fullTransientSeq
    else:
        return fullTransientSeq, ctrl_figures

def ghostID_phaseSpaceSample(model, model_params, t_start, t_end, dt, state_ranges,
                             method='RK45', rtol=1.e-3, atol=1.e-6, n_workers=None, **kwargs):
    """
    Adaptive parallel version:
      - Uses threads when run in Spyder/Jupyter (no pickling issues)
      - Uses processes when run as a standalone script for full CPU utilization
    """

    # # ---- Parameters ----
    # peak_kwargs = kwargs.get("peak_kwargs", {})
    # ctrlOutputs = kwargs.get("ctrlOutputs", {})
    # model_batch = kwargs.get("batchModel", make_batch_model(model, model_params))
    
    # Parse and validate kwargs
    config = parse_kwargs(**kwargs)

    display_warnings = config['display_warnings']
    
    # Extract parameters from config
    epsilon_SN_ghosts = config['epsilon_SN_ghosts']
    peak_kwargs = config['peak_kwargs']
    if "width" not in peak_kwargs:  
        peak_kwargs["width"] = 5 * dt   # default if not supplied
    batchModel = config['batchModel']
    eigval_NN_sorting = config['eigval_NN_sorting']
    ev_outlier_removal_ws = config['ev_outlier_removal_ws']
    ev_outlier_removal_k = config['ev_outlier_removal_k']
    evLimit = config['evLimit']
    slopeLimits = config['slopeLimits']
    epsilon_gid = config['epsilon_gid']
    epsilon_unify = config['epsilon_unify']
    n_samples = config['n_samples']
    seed = config['seed']

    # Plotting control settings
    return_ctrl_figs = config['return_ctrl_figs']
    ctrlOutputs = kwargs.get("ctrlOutputs", {})

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)

    npts = int(t_end / dt)
    t_eval = np.linspace(0, t_end, npts + 1)

    ICs = phaseSpaceLHS(state_ranges, n_samples, seed)
    
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
        ghostSeq_ = ghostID(model, model_params, dt, sol.y.T,
                           epsilon_gid, epsilon_SN_ghosts=epsilon_SN_ghosts,peak_kwargs=peak_kwargs,
                           batchModel=batchModel,return_ctrl_figs=return_ctrl_figs,ctrlOutputs=ctrlOutputs,
                           evLimit=evLimit,slopeLimits=slopeLimits,eigval_NN_sorting=eigval_NN_sorting,
                           ev_outlier_removal_ws=ev_outlier_removal_ws,ev_outlier_removal_k=ev_outlier_removal_k,
                           display_warnings=display_warnings)
        if return_ctrl_figs == False:
                ghostSeq = ghostSeq_
        else:
            ghostSeq, ctrl_figures = ghostSeq_ #ignore ctrl_figures for now
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
    ghostSeqs_unified = unify_IDs(ghostSeqs, epsilon_unify)
    return ghostSeqs_unified

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

# def find_local_Qminimum(F, x0, p, delta=0.5, method='L-BFGS-B',
#                                n_global_iter=1000, tol_glob=0.01,tol_grad=1e-6, max_iter_local=500,
#                                verbose=False):
#     """
#     Finds a local minimum of Q(x) = 0.5 * ||F||^2 near x0 in arbitrary dimensions.
#     Performs radius-constrained global search using differential evolution,
#     followed by SciPy local refinement.

#     Parameters
#     ----------
#     F : callable
#         Vector field F(t, x, p)
#     x0 : array_like
#         Initial point in phase space (any dimension)
#     p : array_like
#         Model parameters
#     delta : float
#         Maximum distance from x0 for global search and final refinement
#     method : str
#         SciPy local optimization method ('BFGS', 'L-BFGS-B', 'CG')
#     n_global_iter : int
#         Number of iterations for differential evolution
#     tol_grad : float
#         Gradient tolerance for local refinement
#     max_iter_local : int
#         Maximum iterations for local refinement
#     verbose : bool
#         Print progress messages

#     Returns
#     -------
#     x_min : np.ndarray
#         Coordinates of the local minimum
#     Q_min : float
#         Q value at the local minimum
#     res_local : OptimizeResult
#         SciPy local minimization result
#     """

#     x0 = jnp.array(x0)
#     dim = x0.shape[0]

#     # Scalar field Q(x) = 0.5 * ||F||²
#     def Q_func(x):
#         z = jnp.array(x)
#         return float(0.5 * jnp.sum(F(0.0, z, p)**2))

#     # Gradient using JAX
#     grad_Q = lambda x: np.array(jax.grad(lambda z: 0.5*jnp.sum(F(0.0, jnp.array(z), p)**2))(x))

#     # -----------------------------
#     # Global search using differential evolution
#     # -----------------------------
#     if verbose:
#         print(f"Running global search with differential evolution (radius {delta}) ...")

#     # Bounds per dimension for radius constraint
#     bounds = [(float(x0[i]-delta), float(x0[i]+delta)) for i in range(dim)]

#     result_global = differential_evolution(Q_func, bounds, maxiter=n_global_iter, tol=tol_glob, disp=verbose, polish=False)
#     x_global_best = result_global.x
#     Q_global_best = Q_func(x_global_best)

#     if verbose:
#         print(f"Global search best Q = {Q_global_best:.3e}")

#     # -----------------------------
#     # Optional: local refinement using SciPy minimize
#     # -----------------------------
#     if method not in ['None']:
#         if verbose:
#             print(f"Refining local minimum with {method} ...")
    
#         res_local = minimize(Q_func, x_global_best, jac=grad_Q, method=method,
#                              tol=tol_grad, options={'maxiter': max_iter_local, 'disp': verbose},
#                              bounds=bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None)
    
#         x_min = res_local.x
#         Q_min = Q_func(x_min)
    
#         if verbose:
#             print(f"Refined local minimum Q = {Q_min:.3e} at x = {x_min}")

#         return x_min, Q_min, res_local
    
#     return x_global_best, Q_global_best, result_global

def find_local_Qminimum(
    F,
    x0,
    p,
    delta,
    *,
    global_method="lhs",
    local_method="L-BFGS-B",
    global_options=None,
    local_options=None,
    verbose=False,
):
    """
    Find a local minimum of Q(x) = 0.5 * ||F||^2 near x0.
    """

    x0 = np.asarray(x0, dtype=float)
    dim = x0.size
    bounds = [(x0[i] - delta, x0[i] + delta) for i in range(dim)]

    global_options = {} if global_options is None else dict(global_options)
    local_options = {} if local_options is None else dict(local_options)


    # --------------------------------------------------
    # Define Q and grad Q
    # --------------------------------------------------
    def Q_func(x):
        z = jnp.asarray(x)
        return float(0.5 * jnp.sum(F(0.0, z, p) ** 2))

    grad_Q = jax.grad(lambda z: 0.5 * jnp.sum(F(0.0, z, p) ** 2))

    def grad_Q_np(x):
        return np.asarray(grad_Q(jnp.asarray(x)))

    # --------------------------------------------------
    # Global search
    # --------------------------------------------------
    if verbose:
        print(f"[Q-min] Global search method: {global_method}")

    candidate_points = []

    # ---- LHS ------------------------------------------------
    if global_method == "lhs":
         # Compute dimension-aware defaults if None
        n_samples = global_options.get("n_samples", None)
        if n_samples is None:
            n_samples = min(2000, max(200, 20 * dim))

        k_seeds = global_options.get("k_seeds", None)
        if k_seeds is None:
            k_seeds = min(5, max(2, int(np.sqrt(dim))))
            
        seed = global_options.get("seed", None)

        if verbose:
            print(
                f"[Q-min] LHS: n_samples={n_samples}, "
                f"k_seeds={k_seeds}, seed={seed}"
            )

        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        samples = sampler.random(n=n_samples)
        points = qmc.scale(
            samples,
            [b[0] for b in bounds],
            [b[1] for b in bounds],
        )

        Q_vals = np.array([Q_func(x) for x in points])
        best_idx = np.argsort(Q_vals)[:k_seeds]
        candidate_points = points[best_idx]

    # ---- Differential Evolution -----------------------------
    elif global_method == "differential_evolution":
        res = differential_evolution(
            Q_func,
            bounds,
            disp=verbose,
            **global_options,
        )
        candidate_points = [res.x]

    # ---- Dual Annealing -------------------------------------
    elif global_method == "dual_annealing":
        res = dual_annealing(
            Q_func,
            bounds,
            **global_options,
        )
        candidate_points = [res.x]

    # ---- Basin Hopping --------------------------------------
    elif global_method == "basin_hopping":
        minimizer_kwargs = {
            "method": local_method,
            "jac": grad_Q_np,
            "bounds": bounds if local_method in {"L-BFGS-B", "TNC", "SLSQP"} else None,
        }
        res = basinhopping(
            Q_func,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            **global_options,
        )
        candidate_points = [res.x]

    else:
        raise ValueError(f"Unknown global_method '{global_method}'")

    # --------------------------------------------------
    # Local refinement
    # --------------------------------------------------
    if local_method is None:
        Q_vals = [Q_func(x) for x in candidate_points]
        best = int(np.argmin(Q_vals))
        return candidate_points[best], Q_vals[best], None

    results_local = []

    for x_start in candidate_points:
        res = minimize(
            Q_func,
            x_start,
            jac=grad_Q_np,
            method=local_method,
            bounds=bounds if local_method in {"L-BFGS-B", "TNC", "SLSQP"} else None,
            options={
                "disp": verbose,
                **local_options,
            },
        )
        results_local.append(res)

    best_res = min(results_local, key=lambda r: r.fun)

    if verbose:
        print(f"[Q-min] Final Q = {best_res.fun:.3e}")
        print(f"[Q-min] x = {best_res.x}")

    return best_res.x, best_res.fun, best_res


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
                             epsilon_gid=0.1,solve_ivp_method='RK45', rtol=1.e-3, atol=1.e-6,**kwargs):
    
    # # ---- Parameters ----
    # evLimit = kwargs.get("evLimit", None)
    # slopeLimits = kwargs.get("slopeLimits", None)
    # peak_kwargs = kwargs.get("peak_kwargs", {})
    # ctrlOutputs = kwargs.get("ctrlOutputs", {})
    # model_batch = kwargs.get("batchModel", make_batch_model(model, model_params))

    # if "ctrlOutputs" in kwargs:
    #     if "return_ctrl_figs" in kwargs["ctrlOutputs"]:
    #         return_ctrl_figs = kwargs["ctrlOutputs"]["return_ctrl_figs"]
    #     else:
    #         return_ctrl_figs = False
    # else:
    #     return_ctrl_figs = False
    
    # if "distQminThr" in kwargs:
    #     distQminThr = kwargs["distQminThr"]
    # else:
    #     distQminThr = np.inf 

    # Parse and validate kwargs
    config = parse_kwargs(**kwargs)
    
    display_warnings = config['display_warnings']

    # Extract parameters from config
    epsilon_SN_ghosts = config['epsilon_SN_ghosts']
    peak_kwargs = config['peak_kwargs']
    if "width" not in peak_kwargs:  
        peak_kwargs["width"] = 5 * dt   # default if not supplied
    batchModel = config['batchModel']
    eigval_NN_sorting = config['eigval_NN_sorting']
    ev_outlier_removal_ws = config['ev_outlier_removal_ws']
    ev_outlier_removal_k = config['ev_outlier_removal_k']
    evLimit = config['evLimit']
    slopeLimits = config['slopeLimits']
    
    # Plotting control settings
    return_ctrl_figs = config['return_ctrl_figs']
    ctrlOutputs = kwargs.get("ctrlOutputs", {})

    # Q-min search related kwargs
    distQminThr = config['distQminThr'] 
    qmin_glob_method = config['qmin_glob_method']
    qmin_loc_method = config['qmin_loc_method']
    qmin_glob_options = config['qmin_glob_options'] 
    min_loc_options = config['qmin_loc_options'] 
    
    ghostSeq_p = []
    ctrl_figures_p = []
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
            
            # qmin = find_local_Qminimum(model, x0, model_params_, delta, tol_glob=qmin_tol,method=qmin_method)[0]
            qmin = find_local_Qminimum(model,x0,model_params_,delta,
                                       global_method=qmin_glob_method,
                                       local_method=qmin_loc_method,
                                       global_options=qmin_glob_options, 
                                       local_options=min_loc_options,
                                       verbose=False)[0]
                    
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
                print("Terminating track_ghost_branch: Error in chosing initial conditions around qmin (both trajectories are diverging). Try different global/local method/options for finding qmin or different icStep size.")
                if return_ctrl_figs == False:
                    return None, None, None
                else:
                    return None, None, None, None
            
            ic_pick = jnp.real(ic_pick)
        
            sol = solve_ivp(model, (0,t_end), ic_pick, t_eval=np.asarray(np.arange(0, t_end, dt)), rtol=rtol, atol=atol, args=([model_params_]), method=solve_ivp_method)
            
            gid_output = ghostID(model, model_params, dt, sol.y.T,
                           epsilon_gid, epsilon_SN_ghosts=epsilon_SN_ghosts,peak_kwargs=peak_kwargs,
                           batchModel=batchModel,return_ctrl_figs=return_ctrl_figs,ctrlOutputs=ctrlOutputs,
                           evLimit=evLimit,slopeLimits=slopeLimits,eigval_NN_sorting=eigval_NN_sorting,
                           ev_outlier_removal_ws=ev_outlier_removal_ws,ev_outlier_removal_k=ev_outlier_removal_k,
                           display_warnings=display_warnings)
            # ghostID(model, model_params_, dt, sol.y.T,
            #                 epsilon_gid, evLimit=evLimit, slopeLimits=slopeLimits, peak_kwargs=peak_kwargs,
            #                 batchModel=model_batch,ctrlOutputs=ctrlOutputs)
            
            if return_ctrl_figs == False:
                ghostSeq = gid_output
            else:
                ghostSeq, ctrl_figures = gid_output
                ctrl_figures_p.append(ctrl_figures)
            
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
    
    if return_ctrl_figs == False:
        return ghostPositions, np.asarray(parSeq), ghostSeq_p
    else:
        return ghostPositions, np.asarray(parSeq), ghostSeq_p, ctrl_figures_p



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

# def unify_IDs(Seqs, epsilon_SN_ghosts=0.1):
#     """
#     Unify ids across multiple sequences of transient ghost state objects found in timecourse simulations.

#     Arguments:
#         Seqs: list of lists of dicts, each dict with keys 'position' (np.ndarray) and 'id' (str: 'G{i}').
#         epsilon_SN_gs: distance threshold for equating SN ghosts.

#     Returns:
#         The same list Seqs with updated, unified 'id' strings.
#     """
#     # Initialize known objects from first sequence
#     first = Seqs[0]
#     known_G = {}  # id_str -> position (representative)

#     # Track maximum numeric index seen
#     max_g = 0

#     for obj in first:
#         pid = obj['id']
#         if pid.startswith('G'):
#             idx = int(pid[1:])
#             known_G[pid] = obj['position'].copy()
#             max_g = max(max_g, idx)
    
#     # Process subsequent sequences
#     for seq in Seqs[1:]:
#         for obj in seq:
#             pos = obj['position']
#             orig = obj['id']
#             if orig.startswith('G'):
#                 # check against known_G
#                 matched = False
#                 for pid, refpos in known_G.items():
#                     if np.linalg.norm(pos - refpos) < epsilon_SN_ghosts:
#                         obj['id'] = pid
#                         matched = True
#                         break
#                 if not matched:
#                     max_g += 1
#                     new_id = f'G{max_g}'
#                     obj['id'] = new_id
#                     known_G[new_id] = pos.copy()   
#             else:
#                 # Unknown id format, leave unchanged or raise error
#                 raise ValueError(f"Unrecognized id '{orig}'")
#     return Seqs

def unify_IDs(Seqs, epsilon_SN_ghosts=0.1, update=True):
    """
    Unify ids across multiple sequences of transient ghost state objects
    found in timecourse simulations.

    Parameters
    ----------
    Seqs : list[list[dict]]
        Each inner list corresponds to one simulation run.
        Each dict represents a ghost state and must contain:
            - 'position'  : np.ndarray, phase-space position of the ghost
            - 'id'        : str of the form 'G{i}'
            - 'q-value'   : float, scalar quality / stability measure
            - 'dimension' : int, dimension associated with the ghost

    epsilon_SN_ghosts : float, optional
        Distance threshold below which two ghost states are considered
        identical (i.e. the same ghost across runs).

    update : bool, optional (default=True)
        If True, perform a second pass after ID unification that
        synchronizes properties (position, q-value, dimension)
        across all ghosts sharing the same ID.

    Returns
    -------
    Seqs : list[list[dict]]
        The same list structure, with unified IDs and (optionally)
        updated ghost properties.
    """

    # ------------------------------------------------------------------
    # STEP 1: Initialize reference ghosts from the first sequence
    # ------------------------------------------------------------------

    first = Seqs[0]

    # Dictionary mapping ghost ID -> representative position
    known_G = {}

    # Track the highest numerical ghost index encountered so far
    max_g = 0

    for obj in first:
        pid = obj['id']
        if pid.startswith('G'):
            idx = int(pid[1:])            # extract numerical part of ID
            known_G[pid] = obj['position'].copy()
            max_g = max(max_g, idx)
        else:
            raise ValueError(f"Unrecognized id '{pid}'")

    # ------------------------------------------------------------------
    # STEP 2: Unify IDs across all subsequent sequences
    # ------------------------------------------------------------------

    for seq in Seqs[1:]:
        for obj in seq:
            pos = obj['position']
            orig = obj['id']

            if not orig.startswith('G'):
                raise ValueError(f"Unrecognized id '{orig}'")

            # Try to match this ghost against previously known ghosts
            matched = False
            for pid, refpos in known_G.items():
                # Compare Euclidean distance in phase space
                if np.linalg.norm(pos - refpos) < epsilon_SN_ghosts:
                    obj['id'] = pid         # reuse existing ID
                    matched = True
                    break

            # If no match was found, register a new ghost ID
            if not matched:
                max_g += 1
                new_id = f'G{max_g}'
                obj['id'] = new_id
                known_G[new_id] = pos.copy()

    # ------------------------------------------------------------------
    # STEP 3 (optional): Update ghost properties across identical IDs
    # ------------------------------------------------------------------

    if update:
        # Collect all ghosts grouped by ID
        ghosts_by_id = {}
        for seq in Seqs:
            for obj in seq:
                gid = obj['id']
                ghosts_by_id.setdefault(gid, []).append(obj)

        # For each ghost ID, synchronize properties
        for gid, ghosts in ghosts_by_id.items():
            # ----------------------------------------------------------
            # (a) Find ghost with minimal q-value
            # ----------------------------------------------------------
            missing = [o for o in ghosts if 'q-value' not in o]
            if missing:
                print(f"Ghosts missing q-value for id {gid}:")
                for m in missing:
                    print(m.keys())
            ref = min(ghosts, key=lambda o: o['q-value'])
            ref_pos = ref['position'].copy()
            ref_q   = ref['q-value']

            # ----------------------------------------------------------
            # (b) Find maximal dimension across ghosts with this ID
            # ----------------------------------------------------------
            max_dim = max(o['dimension'] for o in ghosts)

            # ----------------------------------------------------------
            # (c) Update all ghosts with synchronized values
            # ----------------------------------------------------------
            for o in ghosts:
                o['position']  = ref_pos.copy()
                o['q-value']   = ref_q
                o['dimension'] = max_dim

    return Seqs


######################################

# def draw_network(adj_matrix, nodeColMap, nlbls):

#     # Note: using this function requies pygraphviz to be installed
#     # https://pygraphviz.github.io/documentation/stable/install.html

#     nw_dim = adj_matrix.shape[0]
#     G = nx.from_numpy_array(adj_matrix.transpose(), create_using=nx.DiGraph)

#     # Define edges
#     inhEdges, actEdges = [], []
#     for i in range(nw_dim):
#         for ii in range(nw_dim):
#             if adj_matrix[i, ii] == 1:
#                 G.add_edge(i, ii, weight=1)
#                 actEdges.append((i, ii))
#             elif adj_matrix[i, ii] == -1:
#                 G.add_edge(i, ii, weight=-1)
#                 inhEdges.append((i, ii))

#     # Apply Graphviz 'dot' layout
#     graphviz_args = "-Nwidth=350 -Nheight=350 -Nfixedsize=true -Goverlap=scale -Gnodesep=5000 -Granksep=200 -Nshape=oval -Nfontsize=14 -Econstraint=true"
#     pos = graphviz_layout(G, prog='fdp',args=graphviz_args)

#     # Draw nodes
#     node_opts = {"node_size": 1800, "edgecolors": "lightgrey"}
#     nx.draw_networkx_nodes(G, pos, node_color=nodeColMap, **node_opts, alpha=0.90)
    
#     # Draw custom edges using the modified arrow approach.
#     # Adjust head_length, head_width, etc. as desired.
#     draw_custom_edges(G, pos, actEdges, color="k", head_length=8, head_width=4, width=1.1, trim_fraction=0.2)
#     draw_custom_edges(G, pos, inhEdges, color="red", head_length=8, head_width=4, width=1.1, trim_fraction=0.2)
    
#     # Draw labels
#     labels = {i: nlbls[i] for i in range(nw_dim)}
#     nx.draw_networkx_labels(G, pos, labels, font_size=16.5, font_weight="bold")

#     plt.axis("off")

def draw_network(
    adj_matrix,
    nodeColMap,
    nlbls,
    layout="fdp",
    graphviz_args=None,
    layout_kwargs=None,
    rankdir="TB",
    node_size=1800,           
    label_font_size=16.5,     
    font="Arial"
):
    """
    layout options
    --------------
    Graphviz:
        'fdp', 'dot', 'neato', 'sfdp', 'circo'
    Semantic aliases:
        'hierarchical' -> Graphviz 'dot'
    NetworkX:
        any nx.*_layout function (NOT nx.draw_*)
    """

    if layout_kwargs is None:
        layout_kwargs = {}

    nw_dim = adj_matrix.shape[0]
    G = nx.from_numpy_array(adj_matrix.transpose(), create_using=nx.DiGraph)

    # --- Define edges ------------------------------------------------------
    inhEdges, actEdges = [], []
    for i in range(nw_dim):
        for ii in range(nw_dim):
            if adj_matrix[i, ii] == 1:
                G.add_edge(i, ii, weight=1)
                actEdges.append((i, ii))
            elif adj_matrix[i, ii] == -1:
                G.add_edge(i, ii, weight=-1)
                inhEdges.append((i, ii))

    # --- Layout handling ---------------------------------------------------
    if graphviz_args is None:
        graphviz_args = (
            f"-Grankdir={rankdir} "
            "-Nwidth=350 -Nheight=350 -Nfixedsize=true "
            "-Goverlap=scale -Gnodesep=5000 -Granksep=200 "
            "-Nshape=oval -Nfontsize=14 -Econstraint=true"
        )

    if layout == "hierarchical":
        layout = "dot"

    if isinstance(layout, str):
        pos = graphviz_layout(G, prog=layout, args=graphviz_args)

    elif callable(layout):
        if layout.__name__.startswith("draw_"):
            raise ValueError(
                f"{layout.__name__} is a drawing function. "
                "Use a layout function like nx.spectral_layout instead."
            )
        pos = layout(G, **layout_kwargs)

    else:
        raise ValueError("layout must be a string or a callable")

    # --- Draw nodes --------------------------------------------------------
    node_opts = {
        "node_size": node_size,
        "edgecolors": "white",
    }
    nx.draw_networkx_nodes(
        G, pos, node_color=nodeColMap, **node_opts, alpha=0.90
    )

    # --- Draw edges --------------------------------------------------------
    draw_custom_edges(
        G, pos, actEdges,
        color="k", head_length=8, head_width=4,
        width=1.1, trim_fraction=0.2
    )
    draw_custom_edges(
        G, pos, inhEdges,
        color="red", head_length=8, head_width=4,
        width=1.1, trim_fraction=0.2
    )

    # --- Draw labels -------------------------------------------------------
    labels = {i: nlbls[i] for i in range(nw_dim)}
    nx.draw_networkx_labels(
    G,
    pos,
    labels,
    font_size=label_font_size,
    font_weight="bold",
    font_family=font,
)
