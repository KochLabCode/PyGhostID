
import numpy as np
from scipy.optimize import approx_fprime


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

        ev1 = np.zeros_like(X_grid)
        ev2 = np.zeros_like(X_grid)

        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                Pt = np.array([X_grid[i, j], Y_grid[i, j]])
                
                jac = approx_fprime(Pt,F,epsilon=1e-4)
                ev1[i,j],ev2[i,j] = np.linalg.eigvals(jac)

        return ev1, ev2

def noBackground(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.grid(False)