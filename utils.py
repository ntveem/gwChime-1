import jax
import jax.numpy as jnp
from jax import jit

@jit
def compute_chieff(s1z, s2z, q):
    return (s1z + q*s2z)/(1+q)

@jit
def spin_norm_from_cartesian_spins(sx, sy, sz):
    """
    returns norm of spin with cartesian components 
    (sx, sy, sz)
    """
    s = jnp.sqrt(sx**2 + sy**2 + sz**2)
    return s

@jit
def costheta_ls_from_cartesian_spins(m1, m2, s1x, s1y, s1z,
                                     s2x, s2y, s2z):
    """
    returns costhetals
    """
    
    s_vec = jnp.array([m1**2*s1x + m2**2*s2x, 
                       m1**2*s1y + m2**2*s2y, 
                       m1**2*s1z + m2**2*s2z])
    s_mag = spin_norm_from_cartesian_spins(s_vec[0], s_vec[1], s_vec[2])
    s_hat = s_vec/s_mag
    l_hat = jnp.array([0.0, 0.0, 1.0])
    
    costheta_ls = jnp.dot(l_hat, s_hat)
    return costheta_ls

@jit
def costheta_from_cartesian_spins(sx, sy, sz):
    """
    returns cosine of spin tilt angle
    """
    costheta = sz/spin_norm_from_cartesian_spins(sx, sy, sz)
    return costheta

@jit
def compute_chirp_mass(m1, m2):
    """
    returns detector frame chirp mass
    """
    mchirp = (m1*m2)**(3/5) / (m1+m2)**(1/5)
    return mchirp

@jit
def compute_chirp_mass_source(m1_source, m2_source):
    """
    returns source frame chirp mass
    """
    mchirp_source = (m1_source*m2_source)**(3/5) / (m1_source+m2_source)**(1/5)
    return mchirp_source