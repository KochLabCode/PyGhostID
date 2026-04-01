# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:50:48 2025

@author: dkoch
"""

import numpy as np
import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import qmc

def iqr_sliding_filter(x, windowsize, k):
    x = np.asarray(x, dtype=float)
    y = x.copy()
    n = len(x)

    for i in range(n):
        i0 = max(0, i - windowsize // 2)
        i1 = min(n, i + windowsize // 2 + 1)

        w = x[i0:i1]
        q1, q3 = np.percentile(w, [25, 75])
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        if x[i] < lower or x[i] > upper:
            w_no_outlier = w[w != x[i]]
            if w_no_outlier.size > 0:
                y[i] = w_no_outlier.mean()

    return y

def sort_NN(x): 
    x_ = np.zeros(x.shape)
    x_[:,0] = x[:,0]
    for t in range(0,x.shape[1]-1):
        idcs_used = []

        for j in range(0,x.shape[0]):

            if t < 2:
                d = np.abs(x[:,t+1]-x_[j,t])
                idcs_sorted = np.argsort(d)
            else:
                    x_pred = x_[j, t] + (x_[j, t] - x_[j, t-1])
                    d = np.abs(x[:, t+1] - x_pred)
                    idcs_sorted = np.argsort(d)
            for i in range(len(idcs_sorted)):
                if idcs_sorted[i] not in idcs_used:
                    idx = idcs_sorted[i]
                    idcs_used.append(idx)
                    break
            x_[j,t+1] = x[idx,t+1]
    return x_


def sign_change(arr,OR,OR_ws,OR_k,**kwargs):
    """
    Returns True if and only if:
      1. arr has at least two elements
      2. arr[0] < 0  (starts negative)
      3. arr[-1] > 0 (ends positive)
      AND 
      4. Values are increasing (arr[i] >= arr[i-1])
      5. There is exactly one transition where prev < 0 and curr > 0

    Zero values or held-constant segments will cause it to return False.
    """

    display_warnings = kwargs.get("display_warnings", True)

    if len(arr) < 2:
        return False

    if arr[0] >= 0 or arr[-1] <= 0:
        return False

    sign_change_occured = False
    tryOR = False
    for prev, curr in zip(arr, arr[1:]):
        if curr < prev:
            if display_warnings:
                print(f"Error in evaluating sign change of eigenvalues: monotonicity violated. {"Trying outlier removal..." if OR else ""}")
            tryOR = True
            break         
        # 2) detect the one negative→positive jump
        if prev < 0 and curr > 0:
            if not sign_change_occured:
                sign_change_occured = True
            else:
                if display_warnings:
                    print(f"Error in evaluating sign change of eigenvalues: more than one sign changes detected. {"Trying outlier removal..." if OR else ""}")
                sign_change_occured = False
                tryOR = True
                break

    if OR and tryOR:
               
        arr_ = iqr_sliding_filter(arr,OR_ws,OR_k)
        for prev, curr in zip(arr_, arr_[1:]):
            if curr < prev:
                if display_warnings:
                    print("... unsuccessful.")
                break         
            # 2) detect the one negative→positive jump
            if prev < 0 and curr > 0:
                if not sign_change_occured:
                    sign_change_occured = True
                    if display_warnings:
                        print("... success.")
                else:
                    if display_warnings:
                        print("... unsuccessful.")
                    sign_change_occured = False
                    break

    return sign_change_occured

def phaseSpaceLHS(ranges, n_samples, seed):
    """
    Latin-hypercube sample points from a phase-space region defined by np.linspace ranges.

    Parameters
    ----------
    ranges : list of np.ndarray
        Each array defines the range (e.g., np.linspace(...)) for one dimension.
    n_samples : int
        Number of points to sample.

    Returns
    -------
    samples : np.ndarray of shape (n_samples, n_dims)
        The sampled points within the specified state-space region.
    """
    n_dims = len(ranges)
    sampler = qmc.LatinHypercube(d=n_dims,seed=seed)
    unit_samples = sampler.random(n=n_samples)

    # Scale to each dimension’s bounds
    bounds = np.array([[r[0], r[-1]] for r in ranges])
    samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
    return samples


def trjSegment(idcs,iq):

    start = np.searchsorted(idcs, iq)

    # a,b=iq,iq

    a = start
    b = start

    
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

# ---------------- JAX Jacobian utility ----------------
def make_jacfun(model, params):
    F = lambda x: model(0, x, params)
    J_fun = jax.jacfwd(F)
    return jax.jit(J_fun)

# ---------------- Fast slope calculation ----------------
def slope_and_r2(y_, dt, ev_outlier_removal, ev_outlier_removal_ws, ev_outlier_removal_k):
    """
    Compute slope and R² of linear regression of y vs time
    y: shape (N,)
    dt: time step
    """

    if ev_outlier_removal:
        y = iqr_sliding_filter(y_, ev_outlier_removal_ws, ev_outlier_removal_k)
    else:
        y = y_
    
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


def parse_kwargs(**kwargs):
    """Parse and validate kwargs for ghostID and dependent functions.
    
    Returns a dictionary with all parameters and their validated values.
    """
    config = {}
    
    # ghostID parameters
    
    config['delta_gid'] = kwargs.get("delta_gid", 0.1)
    
    # Peak detection parameters
    config['peak_kwargs'] = kwargs.get("peak_kwargs", {})
    
    # Model and batch processing
    config['batchModel'] = kwargs.get("batchModel", None)
    
    # Control outputs and plotting
    config['ctrlOutputs'] = kwargs.get("ctrlOutputs", {})
    config['return_ctrl_figs'] = config['ctrlOutputs'].get('return_ctrl_figs', False)
    config['display_warnings'] = kwargs.get("display_warnings", True)

    # Plotting control settings
    ctrl = config['ctrlOutputs']
    
    # Q-plot settings
    config['ctrl_qplot'] = ctrl.get("ctrl_qplot", False)
    if not isinstance(config['ctrl_qplot'], bool):
        raise TypeError(f"ctrl_qplot must be boolean, got {type(config['ctrl_qplot']).__name__}")
    if config['ctrl_qplot']:
        config['qplot_xscale'] = ctrl.get("qplot_xscale", "linear")
        config['qplot_yscale'] = ctrl.get("qplot_yscale", "linear")
        if config['qplot_xscale'] not in ("linear", "log"):
            raise ValueError(f"qplot_xscale must be 'linear' or 'log', got {config['qplot_xscale']!r}")
        if config['qplot_yscale'] not in ("linear", "log"):
            raise ValueError(f"qplot_yscale must be 'linear' or 'log', got {config['qplot_yscale']!r}")
    else:
        config['qplot_xscale'] = None
        config['qplot_yscale'] = None
    
    # Eigenvalue plot settings
    config['ctrl_evplot'] = ctrl.get("ctrl_evplot", False)
    if not isinstance(config['ctrl_evplot'], bool):
        raise TypeError(f"ctrl_evplot must be boolean, got {type(config['ctrl_evplot']).__name__}")
    if config['ctrl_evplot']:
        config['evplot_xscale'] = ctrl.get("evplot_xscale", "linear")
        config['evplot_yscale'] = ctrl.get("evplot_yscale", "linear")
        if config['evplot_xscale'] not in ("linear", "log"):
            raise ValueError(f"evplot_xscale must be 'linear' or 'log', got {config['evplot_xscale']!r}")
        if config['evplot_yscale'] not in ("linear", "log"):
            raise ValueError(f"evplot_yscale must be 'linear' or 'log', got {config['evplot_yscale']!r}")
    else:
        config['evplot_xscale'] = None
        config['evplot_yscale'] = None
    
    # Eigenvalue processing
    config['eigval_NN_sorting'] = kwargs.get("eigval_NN_sorting", False)
    config['ev_outlier_removal'] = kwargs.get("ev_outlier_removal", False)
    config['ev_outlier_removal_ws'] = kwargs.get("ev_outlier_removal_ws", 7)
    config['ev_outlier_removal_k'] = kwargs.get("ev_outlier_removal_k", 1.5)
    config['evLimit'] = kwargs.get("evLimit", 0)
  
    # Slope limits with validation
    sl = kwargs.get("slopeLimits", None)
    if sl is not None:
        sl = np.asarray(sl, dtype=float)
        if sl.shape == (2,) and np.all(sl >= 0) and sl[0] < sl[1]:
            config['slopeLimits'] = sl
        else:
            raise ValueError(
                f"slopeLimits must be a 2-element nonnegative interval [min,max], got {sl}"
            )
    else:
        config['slopeLimits'] = np.array([0, np.inf])
    
    #############################################################
    
    # ghostID_phaseSpaceSample specific parameters
    config['epsilon_gid'] = kwargs.get("epsilon_gid", 0.1)
    config['epsilon_unify'] = kwargs.get("epsilon_unify", 0.1)
    # config['n_samples'] = kwargs.get("n_samples", 50)
    config['seed'] = kwargs.get("seed", None)

    # track_ghost_branch specific parameters
    config['distQminThr'] = kwargs.get("distQminThr", np.inf)

    # --- Define method-aware defaults ---------------------
    DEFAULT_QMIN_GLOB_OPTIONS = {
        "lhs": {
            "n_samples": None,   # auto = max(200, 20*dim) inside optimizer
            "k_seeds": None,     # auto = min(5, sqrt(dim))
            "seed": None,        # reproducible if set
        },
        "differential_evolution": {
            "maxiter": 1000,
            "tol": 1e-2,
            "seed": None,        # reproducible if set
        },
        "dual_annealing": {
            "maxiter": 1000,
        },
        "basin_hopping": {
            "niter": 100,
            "stepsize": 0.5,
        },
    }

    DEFAULT_QMIN_LOC_OPTIONS = {
        "L-BFGS-B": {"maxiter": 500, "gtol": 1e-6},
        "BFGS": {"maxiter": 500, "gtol": 1e-6},
        "CG": {"maxiter": 500, "gtol": 1e-6},
        "TNC": {"maxiter": 500, "gtol": 1e-6},
        "SLSQP": {"maxiter": 500, "ftol": 1e-9},
        None: {},
    }

    # --- Pick methods from kwargs with fallback -----------
    config['qmin_glob_method'] = kwargs.get("qmin_glob_method", "lhs")
    config['qmin_loc_method']  = kwargs.get("qmin_loc_method", "L-BFGS-B")

    glob_method = config['qmin_glob_method']
    loc_method  = config['qmin_loc_method']

    # --- Merge user-specified options with defaults -------
    user_glob_opts = kwargs.get("qmin_glob_options", {})
    default_glob_opts = DEFAULT_QMIN_GLOB_OPTIONS.get(glob_method, {})
    config['qmin_glob_options'] = {**default_glob_opts, **user_glob_opts}

    user_loc_opts = kwargs.get("qmin_loc_options", {})
    default_loc_opts = DEFAULT_QMIN_LOC_OPTIONS.get(loc_method, {})
    config['qmin_loc_options'] = {**default_loc_opts, **user_loc_opts}

    return config