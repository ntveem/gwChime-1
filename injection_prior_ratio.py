from pdfs import *
from jacobians import m1_m2_to_m1s_m2s

from astropy.cosmology import FlatwCDM
from astropy import units as u
import jax.numpy as jnp

cosmo = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)
def lvc_redshift_lnp(z, pow_z):
    dVc_dz = cosmo.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value * 4 * jnp.pi
    return jnp.log((1 + z)**(pow_z - 1)* dVc_dz)

def lvc_spin_lnp(s1x, s1y, s1z,
                 s2x, s2y, s2z, 
                 max_spin = 0.998):
    """
    defined in params: (s1x, s1y, s1z, s2x, s2y, s2z)
    isotropic in spin direction,
    uniform in spin magnitude: U(0,max_spin)
    """
    lnp = (- jnp.log(4*jnp.pi*(s1x**2 + s1y**2 + s1z**2)*max_spin)
                  - jnp.log(4*jnp.pi*(s2x**2 + s2y**2 + s2z**2)*max_spin))
    return lnp

def lvc_pe_to_injection_lnp(s1x, s1y, s1z, 
                            s2x, s2y, s2z, 
                            z, ln_sampling_pdf):
    """
    returns log prior ratio of pe to 
    -- for now returning 1/pinj
    """
    log_mass_pe = m1_m2_to_m1s_m2s(z)
    log_spin_pe = lvc_spin_lnp(s1x, s1y, s1z, 
                               s2x, s2y, s2z)
    log_redshift_pe = lvc_redshift_lnp(z, 0.0)
    log_pe = (log_mass_pe 
              + log_spin_pe 
              + log_redshift_pe)
    return log_pe-ln_sampling_pdf