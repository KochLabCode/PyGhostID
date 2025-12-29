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
# def vanDerPol_2g(t,z,para):
#     eps,alpha=para
#     dx=(1/eps)*(z[1]-(1/3)*z[0]**3+z[0])
#     dy = -z[0] + alpha*(z[1] - (1/3)*z[1]**3)
#     return jnp.array([dx, dy])

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