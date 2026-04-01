
import numpy as np
# from scipy.optimize import approx_fprime
import jax 
import jax.numpy as jnp


def get_rcparams():
    params = {'legend.fontsize': 10,
              'axes.labelsize': 10,
              'axes.labelpad' : 15,
              'axes.titlesize':12,
              'xtick.labelsize':7,
              'ytick.labelsize':7,
               'text.usetex': False
              }
    return params

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

        # ev1 = np.zeros_like(X_grid)
        # ev2 = np.zeros_like(X_grid)
        # J_fun = jax.jacfwd(F)
        # J_fun = jax.jit(J_fun)

              
        # # Batch Jacobian + eigenvalue evaluation for segment
        # pts_segment = jnp.asarray(trajectory[idcs_segment])        # JAX array
        # J_batch = jax.vmap(J_fun)(pts_segment)                     # batch Jacobians
        # eigVals = jax.vmap(jnp.linalg.eigvals)(J_batch)            # eigenvalues
        # eigVals_real = np.real(np.asarray(eigVals))                # back to numpy for analysis

        # for i in range(X_grid.shape[0]):
        #     for j in range(X_grid.shape[1]):
        #         Pt = np.array([X_grid[i, j], Y_grid[i, j]])
                
        #     # F = lambda x: model(0, x, params)
                
        #         ev1[i,j],ev2[i,j] = np.linalg.eigvals(jac)
        #         # jac = approx_fprime(Pt,F,epsilon=1e-6)
        #         # ev1[i,j],ev2[i,j] = np.linalg.eigvals(jac)

        #         # try:
        #         #     jac = approx_fprime(Pt, F, epsilon=1e-8)

        #         #     # Check Jacobian validity
        #         #     if not np.isfinite(jac).all():
        #         #         continue

        #         #     eigs = np.linalg.eigvals(jac)
        #         #     ev1[i, j], ev2[i, j] = eigs

        #         # except Exception:
        #         #     # catches LinAlgError, ValueError, etc.
        #         #     continue

        # return ev1, ev2
    # Convert grids to JAX arrays
    X = jnp.asarray(X_grid)
    Y = jnp.asarray(Y_grid)

    Nx, Ny = X.shape

    # Flatten grid → (Npoints, 2)
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Jacobian function
    J_fun = jax.jacfwd(F)
    J_fun = jax.jit(J_fun)

    # Batched Jacobians and eigenvalues
    J_batch = jax.vmap(J_fun)(pts)                    # (N, 2, 2)
    eigvals = jax.vmap(jnp.linalg.eigvals)(J_batch)   # (N, 2)

    # Reshape back to grid
    # eigvals = eigvals.reshape(Nx, Ny, 2)
    # ev1, ev2 = np.asarray(eigvals)

    eigvals = np.asarray(eigvals).reshape(Nx, Ny, 2)
    # Return as NumPy array
    ev1 = eigvals[..., 0]
    ev2 = eigvals[..., 1]

    return ev1, ev2

def noBackground(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.grid(False)

    
def euklideanVelocity(x,dt):
    v = np.array([])
    n = x.shape[0]
    for i in range(1,n):
        d = np.linalg.norm(x[i,:]-x[i-1,:])
        v = np.append(v, d/dt)
    return v

def generate_peak_series(
    total_duration=10.0,   # Total time (arbitrary units)
    dt=0.001,              # Time step (arbitrary units)
    amp_mean=1.0,          # Mean peak amplitude
    amp_std=0.3,           # STD of peak amplitudes
    peak_width=0.4,        # Width of each peak (same time units)
    n=4,                   # Sharpness parameter (higher = sharper)
    allow_negative=True,   # If True, peaks can be negative with 50% probability
    **kwargs
):
    # Choose IPI distribution
    if kwargs["ipi_distr"] == "normal":
        if "ipi_params" in kwargs:
            ipi_mean, ipi_std = kwargs["ipi_params"]
        else:
            ipi_mean, ipi_std = 1.0, 0.2

        peak_times = [0.0]
        while peak_times[-1] < total_duration:
            ipi = np.random.normal(ipi_mean, ipi_std)
            ipi = max(ipi, peak_width)
            peak_times.append(peak_times[-1] + ipi)

    elif kwargs["ipi_distr"] == "exponential":
        if "ipi_params" in kwargs:
            ipi_scale = kwargs["ipi_params"][0]
        else:
            ipi_scale = 1.0

        peak_times = [0.0]
        while peak_times[-1] < total_duration:
            ipi = np.random.exponential(ipi_scale)
            ipi = max(ipi, peak_width)
            peak_times.append(peak_times[-1] + ipi)

    elif kwargs["ipi_distr"] == "poisson":
        if "ipi_params" in kwargs:
            ipi_lam = kwargs["ipi_params"][0]
        else:
            ipi_lam = 1.0

        peak_times = [0.0]
        while peak_times[-1] < total_duration:
            ipi = np.random.poisson(ipi_lam)
            ipi = max(ipi, peak_width)
            peak_times.append(peak_times[-1] + ipi)

    # Generate amplitudes
    amplitudes = np.random.normal(amp_mean, amp_std, len(peak_times))
    amplitudes = np.clip(amplitudes, 0.1, None)

    if allow_negative:
        amplitudes *= np.random.choice([-1, 1], size=len(amplitudes))

    # Create time array using timestep
    t = np.arange(0.0, total_duration, dt)
    signal = np.zeros_like(t)

    # Add peaks
    for t_peak, a in zip(peak_times, amplitudes):
        if t_peak > total_duration:
            continue

        tau = t - t_peak
        mask = np.abs(tau) <= peak_width / 2

        peak = a * (1 + np.cos(2 * np.pi * tau[mask] / peak_width))**n / (2**n)
        signal[mask] += peak

    return t, signal


def rankOrdering(arr):
    arr = np.asarray(arr)
    # Flatten, argsort twice to get ranks
    ranks = arr.argsort().argsort().astype(float) + 1  # ranks start at 1
    N = len(ranks)
    # Map to (0, 1)
    uniform = ranks / (N + 1)
    # Shift to (-0.5, 0.5)
    uniform_zero_mean = uniform - 0.5
    return uniform_zero_mean.reshape(arr.shape)

def mutualInformation(x,y,nbins,rank=False):
    # Naive algorithm as described in Selbig et al. (2002):
    # The mutual information: Detecting and evaluating dependencies between variables.
    # Bioinformatics, Vol. 18 Suppl. 2, S231–S240
    N = len(x)
    
    if rank == True:
        x_=rankOrdering(x)
        y_=rankOrdering(y)
    else:
        x_=x
        y_=y
        
    x_range=[np.min(x_),np.max(x_)]
    y_range=[np.min(y_),np.max(y_)]
    
    if N == len(y):
        if N<10*nbins: print("Warning: Very few samples compared to bin numbers. Calculated mutual information may not be accurate!")
        kh, xedges, yedges = np.histogram2d(x_,y_,nbins,range=[x_range,y_range])
        sum_p = 0
        for i in range(nbins):
            for j in range(nbins):
                if all([kh[i,j]!=0,np.nansum(kh[i,:])!=0,np.nansum(kh[:,j]!=0)]): # exclude empty bins
                    sum_p += kh[i,j]*np.log2(kh[i,j]/(np.nansum(kh[i,:])*np.nansum(kh[:,j])))
        MI = np.log2(N) + sum_p/N
        
        err_finSize = (nbins**2 - 2*nbins * 1)/(2*N) # estimated deviation from true MI
        
        return MI - err_finSize
    else:
        print("Error: x and y do not have the same size.")
        return None, None
    
def spike_rate(binary_series, dt, window_size, step_size):
    """
    Compute spike rate over time using a moving window.

    Parameters:
        binary_series : 1D numpy array of 0s and 1s
        dt            : time step between samples (in seconds)
        window_size   : size of window in samples
        step_size     : step size in samples

    Returns:
        times         : array of center times for each window
        rates         : array of spike rates in Hz for each window
    """
    rates = []
    times = []
    for start in range(0, len(binary_series) - window_size + 1, step_size):
        end = start + window_size
        window = binary_series[start:end]
        rate = np.sum(window) / (window_size * dt)
        rates.append(rate)
        times.append((start + end) / 2 * dt)
    return np.array(times), np.array(rates)