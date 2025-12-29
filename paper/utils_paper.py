
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