from pdfs import *
from jacobians import m1_m2_to_m1s_q, m1s_q_to_m1s_m2s

import jax.numpy as jnp
from jax import jit

@jit
def powerlaw_peak_mass_q_to_lvc_pe(m1_source, q, z,
                                   lambda_peak, alpha, beta,
                                   m_min, m_max, m_mean, m_std):
    """
    returns log prior ratio
    """
    # numerator
    smoothing_m1 = 0.9
    log_p_m1 = jnp.log((1-lambda_peak)*smoothed_powerlaw(m1_source, alpha, m_min, m_max, smoothing_m1)
                      + lambda_peak*smoothed_truncated_gaussian(m1_source, m_min, m_max, m_mean, m_std, smoothing_m1))
    
    q_min = jnp.minimum(m_min/m1_source, 0.99)
    q_max = jnp.ones_like(q_min)
    smoothing_q = 0.03
    log_p_q = jnp.log(smoothed_powerlaw(q, beta, q_min, q_max, smoothing_q))
    
    log_p_mass = (log_p_m1 
                  + log_p_q)

    log_pe = m1_m2_to_m1s_q(m1_source, z)

    return log_p_mass-log_pe

@jit
def powerlaw_mass_q_to_lvc_pe(m1_source, q, z,
                              alpha, beta,
                              m_min, m_max):
    """
    returns log prior ratio
    """
    # numerator
    smoothing_m1 = 0.9
    log_p_m1 = jnp.log(smoothed_powerlaw(m1_source, alpha, m_min, m_max, smoothing_m1))
    
    q_min = jnp.minimum(m_min/m1_source, 0.99)
    q_max = jnp.ones_like(q_min)
    smoothing_q = 0.03
    log_p_q = jnp.log(smoothed_powerlaw(q, beta, q_min, q_max, smoothing_q))
    
    log_p_mass = (log_p_m1 
                  + log_p_q)

    log_pe = m1_m2_to_m1s_q(m1_source, z)

    return log_p_mass-log_pe

@jit
def gaussian_plus_flat_costheta_ls(s1, s2, costheta_ls,
                                   f_isotropic, mu_theta_ls, sigma_theta_ls,
                                   s_mean, s_sigma):
    """
    returns log prior ratio
    """
    # smoothing = 0.2
    log_p_costheta_ls = jnp.log((1-f_isotropic)*jitted_truncated_gaussian(costheta_ls, -1.0, 1.0, mu_theta_ls, sigma_theta_ls)
                               + f_isotropic*0.5)
    log_p_s1_s2 = jnp.log(jitted_truncated_gaussian(s1, 0.0, 1.0, s_mean, s_sigma)
                          *jitted_truncated_gaussian(s2, 0.0, 1.0, s_mean, s_sigma))
    
    log_p_spin = (log_p_costheta_ls 
                  + log_p_s1_s2)
    log_pe = jnp.log(0.5) # spin magnitude is just 2*log(1)
    
    return log_p_spin-log_pe

@jit
def spin_up_gaussian_plus_flat_costheta_ls(s1, s2, costheta_ls,
                                           f_isotropic, mu_theta_ls, sigma_theta_ls,
                                           chi_mean_a, chi_std_a, chi_mean_i, chi_std_i):
    """
    returns log prior ratio
    """
    p_spun_up_s1s2 =  (jitted_truncated_gaussian(s1, 0.0, 1.0, chi_mean_a, chi_std_a)*
                       jitted_truncated_gaussian(s2, 0.0, 1.0, chi_mean_a, chi_std_a))
    
    p_isotropic_s1s2 = (jitted_truncated_gaussian(s1, 0.0, 1.0, chi_mean_i, chi_std_i)*
                       jitted_truncated_gaussian(s2, 0.0, 1.0, chi_mean_i, chi_std_i))
    
    log_p_spin = jnp.log((1-f_isotropic)*p_spun_up_s1s2*jitted_truncated_gaussian(costheta_ls, -1.0, 1.0, mu_theta_ls, sigma_theta_ls)
                               + f_isotropic*p_isotropic_s1s2*0.5)
    log_pe = jnp.log(0.5) # spin magnitude is just 2*log(1)
    
    return log_p_spin-log_pe

@jit
def combined_mass_spin_up_costheta_ls(m1_source, q, z, s1, s2, costheta_ls,
                                      alpha, beta, m_min, m_max,
                                      f_isotropic, mu_theta_ls, sigma_theta_ls,
                                      chi_mean_a, chi_std_a, chi_mean_i, chi_std_i):
    """
    returns log prior ratio
    """
    log_pr_mass = powerlaw_mass_q_to_lvc_pe(m1_source, q, z,
                                            alpha, beta, m_min, m_max)
    log_pr_spin = spin_up_gaussian_plus_flat_costheta_ls(s1, s2, costheta_ls,
                                           f_isotropic, mu_theta_ls, sigma_theta_ls,
                                           chi_mean_a, chi_std_a, chi_mean_i, chi_std_i)
    return log_pr_mass + log_pr_spin

@jit
def combined_mass_costheta_ls(m1_source, q, z, s1, s2, costheta_ls,
                             alpha, beta, m_min, m_max, 
                              f_isotropic, mu_theta_ls, sigma_theta_ls,
                              s_mean, s_sigma):
    """
    returns log prior ratio
    """
    log_pr_mass = powerlaw_mass_q_to_lvc_pe(m1_source, q, z,
                                            alpha, beta, m_min, m_max)
    log_pr_spin = gaussian_plus_flat_costheta_ls(s1, s2, costheta_ls,
                                                 f_isotropic, mu_theta_ls, sigma_theta_ls,
                                                 s_mean, s_sigma)
    return log_pr_mass+log_pr_spin

@jit
def planck_taper(x, x_min, delta_x):
    x_prime = x - x_min
    f_x = jnp.exp(delta_x / x_prime + delta_x / (x_prime - delta_x))
    low_filter = jnp.power(f_x+1, -1)
    low_filter = jnp.where(jnp.logical_and(x>x_min, x<x_min+delta_x),
                           low_filter ,1.)
    low_filter = jnp.where(x<=x_min, 
                           EPSILON,
                           low_filter)
    return low_filter

@jit
def fixed_smoothing(x, x_max, smoothing):
    high_start = x_max - smoothing
    high_filter = jnp.exp(-(x - high_start)**2/(2.*smoothing**2))
    high_filter = jnp.where(x > high_start,high_filter,1.)
    return high_filter

@jit
def compute_norm(p_x, x_min, x_max):
    x_grid = jnp.linspace(x_min, x_max, 1000)
    norm = jnp.trapezoid(p_x, x_grid)
    return norm

@jit
def broken_powerlaw_two_peaks(m1_source, lambda0, lambda1, lambda2,
                              alpha1, alpha2, mu1, std1, mu2, std2,
                              m_break, m1_min, m_max, delta_m1):
    # define the powerlaw
    mask_lower_powerlaw = jnp.logical_and(m1_source>m1_min, m1_source<=m_break)
    mask_upper_powerlaw = jnp.logical_and(m1_source>m_break, m1_source<m_max)
    p_lowerpl = jnp.where(mask_lower_powerlaw, 
                          (m1_source/m_break)**(-alpha1),
                          EPSILON)
    p_bpl = jnp.where(mask_upper_powerlaw,
                      (m1_source/m_break)**(-alpha2),
                      p_lowerpl)
    
    # compute bpl norm
    first_pl_term = (1 - (m1_min/m_break)**(1-alpha1))/(1-alpha1)
    second_pl_term = ((m_max/m_break)**(1-alpha2) - 1)/(1-alpha2)
    norm_bpl = m_break*(first_pl_term + second_pl_term)
    p_bpl /= norm_bpl
    
    # do first peak
    p_peak_lower = jitted_truncated_gaussian(m1_source, m1_min, m_max,
                                             mu1, std1)

    # do second peak
    p_peak_higher = jitted_truncated_gaussian(m1_source, m1_min, m_max,
                                             mu2, std2)

    smoothing = 0.9 # fixed smoothing for max mass
    # full m1 mixture
    log_p_mass1 = (jnp.log(lambda0*p_bpl 
                          + lambda1*p_peak_lower 
                          + lambda2*p_peak_higher)
                  + jnp.log(planck_taper(m1_source, m1_min, delta_m1))
                  + jnp.log(fixed_smoothing(m1_source, m_max, smoothing)))
    
    return log_p_mass1 

@jit
def smoothed_powerlaw_q(m1_source, q, beta_q, m2_min, m_max, delta_m2):
    smoothing = 0.9
    log_p_q = (beta_q*jnp.log(q)
           + jnp.log(planck_taper(m1_source*q, m2_min, delta_m2))
           + jnp.log(fixed_smoothing(m1_source*q, m_max, smoothing)))
    return log_p_q

@jit
def broken_powerlaw_two_peaks_to_pe(m1_source, q, z, lambda0, lambda1, lambda2, 
                                    alpha1, alpha2, beta_q,
                                    mu1, std1, mu2, std2,
                                    m_break, m1_min, m2_min,
                                    m_max, delta_m1, delta_m2):

    # compute pdf
    unnormed_log_p_m1 = broken_powerlaw_two_peaks(m1_source, lambda0, lambda1, lambda2, 
                                                alpha1, alpha2, mu1, std1, mu2, std2,
                                                m_break, m1_min, m_max, delta_m1)
    unormed_log_p_q = smoothed_powerlaw_q(m1_source, q, beta_q, m2_min, m_max, delta_m2)

    # compute norm
    m1_grid = jnp.linspace(m1_min, m_max, 10000)
    m2_grid = jnp.linspace(m2_min, m_max, 10000)
    q_grid = m2_grid/m1_grid
    norm_m1 = jnp.trapezoid(jnp.exp(broken_powerlaw_two_peaks(m1_grid, lambda0, lambda1, lambda2, 
                                                alpha1, alpha2, mu1, std1, mu2, std2,
                                                m_break, m1_min, m_max, delta_m1)),
                            m1_grid)
    norm_q = jnp.trapezoid(jnp.exp(smoothed_powerlaw_q(m1_grid, q_grid, beta_q, m2_min, m_max, delta_m2)),
                           q_grid)

    log_p_mass = (unnormed_log_p_m1 - jnp.log(norm_m1)
                  + unormed_log_p_q -jnp.log(norm_q))
    # compute jacobian & pe prior
    log_pe = m1_m2_to_m1s_q(m1_source, z)
    
    return log_p_mass - log_pe

@jit
def combined_bpl_mass_spin_up_costheta_ls(m1_source, q, z, s1, s2, costheta_ls,
                                          lambda0, lambda1, lambda2, alpha1, alpha2, beta_q,
                                          mu1, std1, mu2, std2,
                                          m_break, m1_min, m2_min,
                                          m_max, delta_m1, delta_m2,
                                          f_isotropic, mu_theta_ls, sigma_theta_ls,
                                          s_mean, s_sigma):
    """
    returns log prior ratio
    """
    log_pr_mass = broken_powerlaw_two_peaks_to_pe(m1_source, q, z, lambda0, lambda1, lambda2,
                                                alpha1, alpha2, beta_q,
                                                mu1, std1, mu2, std2,
                                                m_break, m1_min, m2_min,
                                                m_max, delta_m1, delta_m2)
    log_pr_spin = gaussian_plus_flat_costheta_ls(s1, s2, costheta_ls,
                                                 f_isotropic, mu_theta_ls, sigma_theta_ls,
                                                 s_mean, s_sigma)
    return log_pr_mass + log_pr_spin