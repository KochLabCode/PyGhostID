import jax
import jax.numpy as jnp
from scipy.ndimage import minimum_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

# -----------------------------
# 1. Q-value on grid
# -----------------------------
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


def generate_grid(dim, n_points=25, ranges=None, indexing="ij"):
    """
    Generate a regular N-D grid.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    n_points : int or list of int
        Number of points per dimension (default 50).
    ranges : tuple or list of tuples
        Default ranges for each dimension. If single tuple, used for all dims.
    overrides : dict
        Per-dimension overrides: {dim_index: {"range": (min,max), "n": int}}
    indexing : "ij" or "xy"
        Indexing convention for meshgrid.

    Returns
    -------
    coords : list of 1D arrays
        Coordinate vectors for each dimension.
    grid_points : array
        Full stacked N-D grid of shape (..., dim)
    """
    # Default range
    if ranges is None:
        ranges = [(-2.0, 2.0)] * dim
    elif isinstance(ranges[0], (int,float)):
        ranges = [ranges] * dim

    # Default n_points
    if isinstance(n_points, int):
        n_points = [n_points]*dim

    coords = []
    for d in range(dim):
        n = n_points[d] if d < len(n_points) else n_points[-1]
        r = ranges[d] if d < len(ranges) else ranges[-1]
        coords.append(jnp.linspace(r[0], r[1], n))

    # Generate meshgrid and stack
    meshes = jnp.meshgrid(*coords, indexing=indexing)
    grid_points = jnp.stack(meshes, axis=-1)

    return coords, grid_points


# -----------------------------
# 2. Local minima finder (SciPy)
# -----------------------------
def find_local_minima(Q_grid, coords=None, mode="strict"):
    neighborhood_min = minimum_filter(Q_grid, size=3, mode='constant', cval=np.inf)
    mask = Q_grid < neighborhood_min if mode=="strict" else Q_grid <= neighborhood_min
    minima_indices = np.argwhere(mask)
    minima_values = Q_grid[tuple(minima_indices.T)]
    minima_coords = None
    if coords is not None:
       if isinstance(coords, jnp.ndarray) or isinstance(coords, np.ndarray):
           # stacked grid: shape (..., dim)
           minima_coords = coords[tuple(minima_indices.T)]
       else:
           # list of 1D arrays
           minima_coords = np.stack([coords[d][minima_indices[:,d]] for d in range(len(coords))], axis=1)
    return minima_indices, minima_coords, minima_values

# -----------------------------
# 3. Van der Pol oscillator (new version)
# -----------------------------
def vdp(t, z, p):
    eps = p[0]
    dx = (1/eps)*(z[1] - (1/3)*z[0]**3 + z[0])
    dy = -z[0]
    return jnp.array([dx, dy])

# # Grid
x = np.linspace(-2.5,2.5,50)
y = np.linspace(-2.1,2.1,50)


# Compute Q
Q_vdp, coords_vdp = qOnGrid(vdp, (0.1,), coords=[jnp.linspace(-3,3,50) for i in range(2)], jit=True)
# indices_vdp, coords_min_vdp, values_vdp = find_local_minima(np.array(Q_vdp), coords_vdp)
indices_vdp, coords_min_vdp, values_vdp = find_local_minima(np.array(Q_vdp), coords_vdp, mode="nonstrict")


# -----------------------------
# 4. Compute nullclines
# -----------------------------
# dx/dt = 0 -> y = x^3/3 - x
# dy/dt = 0 -> x = 0
# X, Y = np.meshgrid(np.array(x), np.array(y), indexing='ij')
# nullcline_dx = X**3/3 - X
# nullcline_dy = np.zeros_like(X)

# -----------------------------
# 5. Plot Q-values with log scale and nullclines
# -----------------------------


vmin = 1e-2   # Avoid zero or negative values
vmax = 1000

Q_plot = np.clip(np.array(Q_vdp), vmin, vmax)

plt.figure(figsize=(8,6))
cf = plt.contourf(x, y, Q_plot.T, levels=50, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='magma_r')
plt.colorbar(cf, label='Q')
# plt.plot(x, nullcline_dx, 'r--', label='dx/dt=0')
# plt.plot(x, nullcline_dy, 'b--', label='dy/dt=0')
plt.scatter(coords_min_vdp[:,0], coords_min_vdp[:,1], color='red', s=20, label='minima')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Van der Pol Q-value (log scale) with nullclines and minima')
# plt.legend()
plt.show()

#%% Define log scale range
inCm = 1/2.54 # convert inch to cm for plotting

plt.figure(figsize=(13*inCm,11*inCm))

im = plt.imshow(Q_vdp.T, extent=(x.min(), x.max(), y.min(), y.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')

#%%
from scipy.optimize import minimize, differential_evolution

def find_local_Qminimum(F, x0, p, delta=0.5, method='L-BFGS-B',
                               n_global_iter=1000, tol_grad=1e-6, max_iter_local=500,
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

    result_global = differential_evolution(Q_func, bounds, maxiter=n_global_iter, disp=verbose, polish=False)
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

para_model = {
    "d": 0.2,
    "GMT": 1.51,
    "mat_inter": np.array([[0,0],[1,0]]),
    "Tcrits": np.array([1.5,1.5]),
    "Taus": np.array([50,5000])
    }


# Define grid
xmin=-1.5;xmax=1.5
ymin=-1.5;ymax=1.5

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# Q-values on grid
Q, coords = qOnGrid(wunderling_modeljx,para_model,coords=[x_range,y_range], jit=True)


# minima
# indices, coords_min, values = find_local_minima(np.array(Q), coords, mode="nonstrict")

#%%
x0 = jnp.array([-0.4, 1.0])
x_min, Q_min, mode = find_local_Qminimum(wunderling_modeljx, x0, para_model, delta=0.1)



plt.figure(figsize=(13*inCm,11*inCm))
plt.title(f"$\\tau_1 = {para_model['Taus'][0]},\\tau_2 = {para_model['Taus'][1]}$")
ax = plt.gca()

ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
# Define log scale range
vmin = np.min(Q)   # Avoid zero or negative values
vmax = np.max(Q)
im = plt.imshow(Q.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()),
                origin='lower', cmap='magma_r', norm=LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar(im, label='Q value (log scale)')
# Found minimum
ax.plot(float(x0[0]), float(x0[1]), 'bo', markersize=8, markeredgecolor='k', label='start')

ax.plot(float(x_min[0]), float(x_min[1]), 'wo', markersize=8, markeredgecolor='k', label='Local minimum')
# ax.plot(float(x_min2[0]), float(x_min2[1]), 'ro', markersize=8, markeredgecolor='k', label='Local minimum, gd')

#%% higher n

# import time



# n = 6
# Ng = 30

# para_model = {
#     "d": 0.2,
#     "GMT": 1.51,
#     "mat_inter": np.ones((n,n)),
#     "Tcrits": np.ones(n)*1.5,
#     "Taus":  np.ones(n)*500
#     }


# start = time.time()
# # Q-values on grid
# Q, coords = qOnGrid(wunderling_modeljx,para_model,coords=[np.linspace(-1.5,1.5,Ng) for i in range(n)], jit=True)


# # minima
# indices, coords_min, values = find_local_minima(np.array(Q), coords, mode="nonstrict")


# end = time.time()

# print(f"Execution time: {end - start} seconds")

# #%%