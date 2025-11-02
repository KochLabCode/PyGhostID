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


def phaseSpaceLHS(ranges, n_samples):
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
    sampler = qmc.LatinHypercube(d=n_dims)
    unit_samples = sampler.random(n=n_samples)

    # Scale to each dimension’s bounds
    bounds = np.array([[r[0], r[-1]] for r in ranges])
    samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
    return samples


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

