import numpy as np
import jax.numpy as jnp

def snb_nf(t,z,para): # normal form of saddle-node bifurcation in 2D

    mu=para[0]
    dx= mu + z[0]**2
    dy= - z[1]
         
    return jnp.stack([dx, dy])

def snb_2dnf(t,z,para): # parameter-coupled saddle-node bifurcation normal forms

    mu=para[0]
    dx= mu + z[0]**2
    dy= mu + z[1]**2
         
    return jnp.stack([dx, dy])

def bieg_etal(t,z,para): # from  https://doi.org/10.1007/s10021-023-00892-8

    a,Nt,N0,g,y,r,c,m = para
    B = 1-z[0]-z[1]
    dC = z[0]*(r*(1-c*Nt)*B-m-a*z[1]*(Nt/(N0+Nt)))
    dM = z[1]*(a*z[0]*(Nt/(N0+Nt))-g/(z[1]+B)+y*B)

    return jnp.array([dC, dM])

def bieg_etal_cNC(C,para):
    a,Nt,N0,g,y,r,c,m = para
    M = (N0+Nt)*(m-r*(1-c*Nt)*(1-C))/(r*(N0+Nt)*(c*Nt-1)-a*Nt)
    return M

def bieg_etal_mNC(C,para):
    a,Nt,N0,g,y,r,c,m = para
    M = 1-C+a*C/y*(Nt/(N0+Nt))-g/(y*(1-C))
    return M

def sys_Farjami2021(t,x,p):
    # doi: 10.1098/rsif.2021.0442 
    g = p
    
    g1 = g; g2 = g; g3 = g
    
    b1 = 1e-5
    b2 = 1e-5
    b3 = 1e-5
    
    alpha1 = 9
    alpha2 = 9
    alpha3 = 9
    
    beta1 = 0.1
    beta2 = 0.1
    beta3 = 0.1
    
    h = 3
    
    d1 = 0.2
    d2 = 0.2
    d3 = 0.2
    
    dx1 = b1 + g1 / ((1+alpha1*(x[1]**h))*(1+beta1*(x[2]**h))) - d1*x[0]
    dx2 = b2 + g2 / ((1+alpha2*(x[2]**h))*(1+beta2*(x[0]**h))) - d2*x[1]
    dx3 = b3 + g3 / ((1+alpha3*(x[0]**h))*(1+beta3*(x[1]**h))) - d3*x[2]
    
    return jnp.array([dx1, dx2, dx3])

def Hastings_etal(t,z,para): # from  

    gamma,h,v,m,alpha,K,eps=para

    dN = alpha*z[0]*(1-z[0]/K) - gamma*z[0]*z[1]/(z[0]+h)
    dP = eps*(v*gamma*z[0]*z[1]/(z[0]+h) - m*z[1])
    
    return jnp.array([dN, dP])

def Hastings_EVs_real(z,para):

    gamma,h,v,m,alpha,K,eps=para

    dFdN = alpha - 2*alpha*z[0]/K - gamma*z[1]*h/(z[0]+h)**2
    dFdP = -gamma*z[0]/(z[0]+h)
    dGdN = eps*v*gamma*z[1]*h/(z[0]+h)**2
    dGdP = eps*v*gamma*z[0]/(z[0]+h)-eps*m

    tr = dFdN + dGdP
    det = dFdN*dGdP-dFdP*dGdN

    lambda1 = jnp.real((tr+np.sqrt(tr**2-4*det))/2)
    lambda2 = jnp.real((tr-np.sqrt(tr**2-4*det))/2)

    return jnp.array([lambda1,lambda2])

def Hastings_NNC(N,para):

    gamma,h,v,m,alpha,K,eps=para

    P = alpha/gamma*(1-N/K)*(N+h)

    return P

def Hastings_PNC(P,para):

    gamma,h,v,m,alpha,K,eps=para

    N=-m*h/(m-gamma*v)

    return N*jnp.ones(len(P))

def May_Leonard(t,z,para): # from  

    alpha,beta=para

    dN1 = z[0]*(1-z[0]-alpha*z[1]-beta*z[2])
    dN2 = z[1]*(1-beta*z[0]-z[1]-alpha*z[2])
    dN3 = z[2]*(1-alpha*z[0]-beta*z[1]-z[2])

    return jnp.array([dN1, dN2, dN3])

def Kuehn_toyModel(t,z,para): # from  Kuehn 2017

    eps=para

    dx = eps - z[0]
    dy = eps*z[1]**2

    return jnp.array([dx,dy])

def FHN(t, z, para): # from ...
    a, b, eps = para

    du = z[0] - z[0]**3 - z[1] 
    dv = eps * (z[0] - b * z[1] + a)

    return jnp.array([du,dv])

def MMenten_slowFast(t, z, para): # from ...
    alpha, beta, mu = para
    ds = beta*z[1]*(1-alpha) - z[0]*(1-alpha*z[1])
    dc = (z[0]*(1-alpha*z[1])-z[1]*(1-alpha))/mu
    return jnp.array([ds,dc])
    

# def vanDerPol(t,z,para):
#     eps=para
#     dx=(1/eps)*(z[1]-(1/3)*z[0]**3+z[0])
#     dy = -z[0]
#     return jnp.array([dx, dy])

def coupledThetaNeurons(t, z, para): # from Augustsson & Martens 2024, doi: 10.1063/5.0226338
    n,K,pS = para
    theta1_dot = 1-jnp.cos(z[0]+pS)+(1+jnp.cos(z[0]+pS))*(n+K*(1-jnp.cos(z[1]+pS))) 
    theta2_dot = 1-jnp.cos(z[1]+pS)+(1+jnp.cos(z[1]+pS))*(n+K*(1-jnp.cos(z[0]+pS))) 
    return jnp.array([theta1_dot,theta2_dot])

def fromArray(t,p):
    t_end,dt,arr = p
    if t<t_end:
        ni = int(t/dt)
        return arr[ni]
    return 0

def coupledThetaNeurons_na(t, z, para): # from Augustsson & Martens 2024, doi: 10.1063/5.0226338
    n,K,pS,p_inp = para

    inp = fromArray(t,p_inp)
  
    theta1_dot = 1-jnp.cos(z[0]+pS)+(1+jnp.cos(z[0]+pS))*(n+K*(1-jnp.cos(z[1]+pS)) + inp) 
    theta2_dot = 1-jnp.cos(z[1]+pS)+(1+jnp.cos(z[1]+pS))*(n+K*(1-jnp.cos(z[0]+pS))) 
    
    return jnp.array([theta1_dot,theta2_dot])

def singleThetaNeuron_na(t, z, para): # from Augustsson & Martens 2024, doi: 10.1063/5.0226338
    n,pS,p_inp = para

    inp = fromArray(t,p_inp)
  
    theta1_dot = 1-jnp.cos(z[0]+pS)+(1+jnp.cos(z[0]+pS))*(n + inp) 
    
    return jnp.array([theta1_dot])

def wunderling_model(t, Z, para):
    Z = jnp.asarray(Z, dtype=jnp.float32)

    d,GMT,Tcrits,Taus,mat_inter=para

    intrinsic = -Z**3 + Z + np.sqrt(4/27)*GMT/Tcrits
    coupling = d/10* mat_inter @ (Z + 1) # Coupling effects: sum over j of C_ij * x_j

    # Total derivative
    dZdt = (intrinsic+coupling)/Taus

    return jnp.array(dZdt) 


# def wunderling_model_vectorized(t, Z, para):
#     """
#     Vectorized version using explicit broadcasting
#     """
#     Z = np.asarray(Z, dtype=np.float64)
    
#     # Unpack parameters
#     d,GMT,Tcrits,Taus,mat_inter=para
    
#     # Ensure Z is at least 2D for consistent processing
#     if Z.ndim == 1:
#         Z = Z[np.newaxis, :]  # Convert to 2D with shape (1, n_vars)
#         return_single = True
#     else:
#         return_single = False
    
#     # Calculate intrinsic term (works for any number of points)
#     intrinsic = -Z**3 + Z + np.sqrt(4/27) * GMT / Tcrits
    
#     # Calculate coupling term using batch matrix multiplication
#     # (Z + 1) has shape (n_points, n_vars)
#     # mat_inter has shape (n_vars, n_vars)
#     # Result should have shape (n_points, n_vars)
#     coupling = d/10 * (Z + 1) @ mat_inter.T  # Equivalent to mat_inter @ (Z + 1)^T
    
#     dZdt = (intrinsic + coupling) / Taus
    
#     # Return to original shape if input was single point
#     if return_single:
#         return dZdt[0]  # Return 1D array
#     else:
#         return dZdt

# def wunderling_model_vectorized(t, Z, para):
#     """
#     Vectorized version using explicit broadcasting
#     """
#     Z = np.asarray(Z, dtype=np.float64)
    
#     # Unpack parameters
#     d,GMT,Tcrits,Taus,mat_inter=para
    
#     # Ensure Z is at least 2D for consistent processing
#     if Z.ndim == 1:
#         Z = Z[np.newaxis, :]  # Convert to 2D with shape (1, n_vars)
#         return_single = True
#     else:
#         return_single = False
    
#     # Calculate intrinsic term (works for any number of points)
#     intrinsic = -Z**3 + Z + np.sqrt(4/27) * GMT / Tcrits
    
#     # Calculate coupling term using batch matrix multiplication
#     # (Z + 1) has shape (n_points, n_vars)
#     # mat_inter has shape (n_vars, n_vars)
#     # Result should have shape (n_points, n_vars)
#     coupling = d/10 * (Z + 1) @ mat_inter.T  # Equivalent to mat_inter @ (Z + 1)^T
    
#     dZdt = (intrinsic + coupling) / Taus
    
#     # Return to original shape if input was single point
#     if return_single:
#         return dZdt[0]  # Return 1D array
#     else:
#         return dZdt


def nullclines_Wunderling(para, x_range=(-2, 2), y_range=(-2, 2), resolution=200):
    """
    Nullclines of the 2D wunderling model.
    """
    # Unpack parameters
    d,GMT,Tcrits,Taus,mat_inter=para
    
    # Define constants
    gamma1 = np.sqrt(4/27) * GMT / Tcrits[0]
    gamma2 = np.sqrt(4/27) * GMT / Tcrits[1]
    
    # Build grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Nullclines (dotx = 0, doty = 0)
    f1 = -X**3 + X + gamma1 + d/10 * (mat_inter[0,0]*(X+1) + mat_inter[0,1]*(Y+1))
    f2 = -Y**3 + Y + gamma2 + d/10 * (mat_inter[1,0]*(X+1) + mat_inter[1,1]*(Y+1))
    
    return f1, f2

# def GRN_net(t, x, para):

#     a, b, K, Ka, Ki, A = para

#     N = len(x)
#     xdot = np.zeros(N)

#     for i in range(N):
#         # Self activation
#         self_act = a * x[i]**2 / (x[i]**2 + K**2)

#         # Excitatory inputs
#         exc_idx = np.where(A[i] == 1)[0]
#         exc = np.sum(x[exc_idx]**2 / (x[exc_idx]**2 + Ka**2)) if exc_idx.size > 0 else 0.0
#         exc *= b

#         # Inhibitory inputs (multiplicative)
#         inh_idx = np.where(A[i] == -1)[0]
#         inh = np.prod(Ki**2 / (x[inh_idx]**2 + Ki**2)) if inh_idx.size > 0 else 1.0

#         xdot[i] = (self_act + exc)*inh - x[i]

#     return jnp.asarray(xdot)

def GRN_net(t, x, para):
    a, b, K, Ka, Ki, A = para
    x = jnp.asarray(x, dtype=jnp.float32)

    # Nonlinear transforms
    x2 = x**2
    f_self = a * x2 / (x2 + K**2)                 # shape (N,)
    f_exc  = x2 / (x2 + Ka**2)                    # shape (N,)
    f_inh  = Ki**2 / (x2 + Ki**2)                 # shape (N,)

    # Masks
    exc_mask = (A == 1).astype(float)             # shape (N,N)
    inh_mask = (A == -1).astype(float)            # shape (N,N)

    # Excitatory sum
    exc = b * (exc_mask @ f_exc)                  # shape (N,)

    # Inhibitory product
    # prod_j f_inh[j]^{inh_mask[i,j]}
    inh = jnp.exp(inh_mask @ jnp.log(f_inh))        # shape (N,)

    # Final ODE
    xdot = (f_self + exc) * inh - x

    return jnp.asarray(xdot)
