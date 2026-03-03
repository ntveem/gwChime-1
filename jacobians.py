import jax.numpy as jnp
from jax import jit 

@jit
def m1_m2_to_m1s_q(m1_source, z):
    """
    (m1, m2) -> (m1_source, q)
    returns log(|det(J)|)
    """
    return jnp.log(m1_source) + 2*jnp.log(1+z)

@jit
def m1_m2_to_m1s_m2s(z):
    """
    (m1, m2) -> (m1_source, m2_source)
    returns log(|det(J)|)
    """
    return 2*jnp.log(1+z)

@jit
def m1s_q_to_m1s_m2s(m1_source):
    """
    (m1_source, q) -> (m1_source, m2_source)
    returns log(|det(J)|)
    """
    return -jnp.log(m1_source)

@jit
def m1_m2_to_mchirp_q(mchirp, q):
    """
    detector frame chirp mass
    (m1, m2) -> (mchirp, q) 
    returns log(|det(J)|)
    """
    return (jnp.log(mchirp) 
            + (2./5.)*jnp.log(1+q) 
            - (6./5.)*jnp.log(q))

@jit
def m1_m2_to_mchirp_source_q(mchirp_source, q, z):
    """
    source frame chirp mass
    (m1, m2) -> (mchirp_source, q)
    returns log(|det(J)|)
    """
    return (jnp.log(mchirp_source) 
            + (2./5.)*jnp.log(1+q) 
            - (6./5.)*jnp.log(q) 
            + 2*jnp.log(1+z))