import numpy as np
import jax.numpy as jnp

def snb_nf(t,z,para): # normal form of saddle-node bifurcation in 2D

    mu=para[0]
    dx= mu + z[0]**2
    dy= - z[1]
         
    return jnp.stack([dx, dy])


def snb_2dnf(t,z,para): # normal form of saddle-node bifurcation in 2D

    mu=para[0]
    dx= mu + z[0]**2
    dy= mu + z[1]**2
         
    return jnp.stack([dx, dy])