
import numpy as np
import random as rnd
import jax 
import jax.numpy as jnp
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

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

def fromArray(t, p):
    t_end, dt, arr = p
    if t < t_end:
        ni = int(t / dt)
        return arr[ni]

def OU(tau,sigma,t0,t_end,dt):
    c = 2*sigma**2/tau
    n =  int((t_end-t0)/dt)
    s = [0]
    for t in range(0,n-1):
        s.append(s[t]-s[t]*dt/tau + np.random.normal()*c**0.5*dt**0.5)
    return np.asarray(s)


def generatePulseSignals(nr_of_pulses, pulse_dur, pause_dur, p_shuffle, t_first, t_end, p1 = 1, mode = 'binary',random_seed = None):
    s = []
    if random_seed is not None:
        rnd.seed(random_seed)
    for i in range(1,nr_of_pulses+1):
        dice = rnd.random()
        if dice < p_shuffle:
            s_start = t_first+(i-1)*(pulse_dur+pause_dur)
            s_stop = t_first+(i-1)*(pulse_dur+pause_dur)+pulse_dur
            s.append(s_start)
            s.append(s_stop)
        else:
            new_pos = rnd.randint(0, t_end)
            s_start = new_pos 
            s_stop = new_pos+pulse_dur
            s.append(s_start)
            s.append(s_stop)           
    s.sort()
    
    if p1 == 1 and mode == 'binary':
        return np.array(s)
    elif p1 != 1 and mode == 'binary':
        a = []
        for i in range(nr_of_pulses):
            dice = rnd.random()
            if dice < p1:
                a.append(1)
            else:
                a.append(-1)
        return [np.array(s), np.array(a)]
    elif mode == 'continuous':
        a = []
        for i in range(nr_of_pulses):
            dice = rnd.random()
            if dice < p1:
                a.append(rnd.random())
            else:
                a.append(-rnd.random())
        return [np.array(s), np.array(a)]


def signal2array(sig,t_end,sz):
    intvs, amps = sig
    length = int(t_end/sz)
    a = np.array([])
    for i in range(length):
        s = 0
        for ii in range(0,len(intvs),2):
            if i*sz >= intvs[ii] and i*sz <= intvs[ii+1]:
                s = amps[int(ii/2)]
        a = np.append(a,s)
    return a


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


def calc_rate(x, threshold, dt, k_rate=10):
 
        N = x.shape[0]
        peak_idxs = find_peaks(x, height=threshold,prominence=1)[0]

        bin_arr = np.zeros(N)
        bin_arr[peak_idxs] = 1

        rate = gaussian_filter(bin_arr, sigma=1/k_rate/dt)        
        rate *= 1/dt
        
        return rate

def calc_MI_xy(x, x_bins, y, y_bins): 
    # From https://github.com/emonetlab/bifurcation-temporal-information/blob/main/models/utils.py 
    # See also Choi et al. 2024, https://doi.org/10.1103/PRXLife.2.043011

    num_x_bins = len(x_bins)
    dx = x_bins[1] - x_bins[0]
    dy = y_bins[1] - y_bins[0]

    # Bin the rates (Y) for a given stimulus (X = x) to get P(Y|X = x)
    # H(Y|X) = sum_X P(X) sum_Y H(Y|X = x)
    #          = sum_X P(X) sum_Y p(Y|X = x) log p(Y|X = x)
    H_Y_X = 0
    H_Y = 0
    p_x, _ = np.histogram(x, x_bins, density=True)
    p_x += 1e-8

    # For each time in the stimulus, digitizes the stimulus into stim_bins
    bin_vals = np.digitize(x, x_bins, right=True) - 1
    
    for iS in range(num_x_bins - 1):

        # Get all times where the stimulus is in the iS'th bin
        idxs_in_bin = np.where(bin_vals == iS)[0]
        # idxs_in_bin = (bin_vals == iS)
        
        # Get rates for all times at which stim is in iS'th bin, histogram
        if len(idxs_in_bin) > 0:
            p_Y_x, _ = np.histogram(y[idxs_in_bin], y_bins, density=True)
            H_y_x = -np.nansum(p_Y_x*np.log(p_Y_x)/np.log(2))*dy
            if np.isfinite(H_y_x):
                H_Y_X += p_x[iS]*H_y_x*dx
    
    # H(Y)
    p_Y, _ = np.histogram(y, y_bins, density=True)
    H_Y = -np.nansum(p_Y*np.log(p_Y)/np.log(2))*dy
    MI = H_Y - H_Y_X
    # print(MI)

    return MI


    
def RK4_na_noisy(f,p,ICs,t0,dt,t_end, noiseVector, sigma=0, naFun = None,naFunParams = None,**kwargs):     # args: ODE system, parameters, initial conditions, starting time t0, dt, number of steps
        
        # using Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
        
        if 'multiplicative_noise' in kwargs:
            multNoise = kwargs['multiplicative_noise']
            # print(multNoise)
        else:
            multNoise = False

        steps = int((t_end-t0)/dt)
        dims = tuple([steps]+list(ICs.shape))
        
        x = np.zeros(dims)
        t = np.zeros(steps,dtype=float)
        x[0] = ICs
        t[0] = t0
        
        if naFun != None and naFunParams != None:
            for i in range(1,steps):
                
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1],t[i-1],p,naFun,naFunParams)*dt
                k2 = f(x[i-1]+k1/2,t[i-1],p,naFun,naFunParams)*dt
                k3 = f(x[i-1]+k2/2,t[i-1],p,naFun,naFunParams)*dt
                k4 = f(x[i-1]+k3,t[i-1],p,naFun,naFunParams)*dt
                x_next = x[i-1] + (k1+2*k2+2*k3+k4)/6
                if multNoise == False:
                    dW=sigma*np.sqrt(dt)*np.random.normal(size=x_next.shape)
                    x[i,:] = x_next + dW*noiseVector 
                else:
                    dW=sigma*np.sqrt(dt*x[i-1])*np.random.normal(size=x_next.shape)
                    # dW=sigma*np.sqrt(dt)*x[i-1]*np.random.normal(size=x_next.shape)
                    x[i,:] = x_next + dW*noiseVector 
                    
        else:
            for i in range(1,steps):
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1],t[i-1],p)*dt
                k2 = f(x[i-1]+k1/2,t[i-1],p)*dt
                k3 = f(x[i-1]+k2/2,t[i-1],p)*dt
                k4 = f(x[i-1]+k3,t[i-1],p)*dt
                x_next = x[i-1] + (k1+2*k2+2*k3+k4)/6
                if multNoise == False:
                    dW=sigma*np.sqrt(dt)*np.random.normal(size=x_next.shape)
                    x[i,:] = x_next + dW*noiseVector
                else:
                    dW=sigma*np.sqrt(dt*x[i-1])*np.random.normal(size=x_next.shape)
                    # dW=sigma*np.sqrt(dt)*x[i-1]*np.random.normal(size=x_next.shape)
                    x[i,:] = x_next + dW*noiseVector 
            
        return t,x.T    