import jax
import jax.numpy as jnp
from jax import jit

from jax.scipy.special import erf

EPSILON=1e-35

def powerlaw(x, alpha, x_min, x_max):
    """
    Returns f(x| alpha, x_min, x_max) = A * x^alpha 
    where A = (1+alpha)/(x_max^(1+alpha) - x_min^(1+alpha)).

    Seems to be faster than powerlaw2 on cpu 
    (on gpu they are equally fast)
    """
    log_x_alpha = alpha*jnp.log(x)
    f_x = jnp.exp(log_x_alpha)
    
    log_x_max_alpha = (1+alpha)*jnp.log(x_max)
    log_x_min_alpha = (1+alpha)*jnp.log(x_min)
    A = (1+alpha)/(jnp.exp(log_x_max_alpha) - jnp.exp(log_x_min_alpha))

    mask_non_zero = jnp.logical_and(x>=x_min, x<=x_max)
    normed_f_x = jnp.clip(
        jnp.where(mask_non_zero, A*f_x, EPSILON),
        EPSILON, None)
    
    return normed_f_x

def powerlaw2(x, alpha, x_min, x_max):
    """
    Returns f(x| alpha, x_min, x_max) = A * x^alpha 
    where A = (1+alpha)/(x_max^(1+alpha) - x_min^(1+alpha)).
    """
    f_x = jnp.power(x, alpha)
    
    x_max_alpha = jnp.power(x_max,(1+alpha))
    x_min_alpha = jnp.power(x_min, (1+alpha))
    A = (1+alpha)/(x_max_alpha - x_min_alpha)

    mask_non_zero = jnp.logical_and(x>=x_min, x<=x_max)
    normed_f_x = jnp.clip(
        jnp.where(mask_non_zero, A*f_x, EPSILON), 
        EPSILON, None)
    
    return normed_f_x

def truncated_gaussian(x, x_min, x_max, mean, std):
    """
    evaluates the gaussian pdf described by mean, std, x_min, x_max
    at array of values x.
    f_x  = A * exp(-(x-mean)/std**2),
        A = [sqrt(2*pi*sigma**2) * (Phi((x_max-mu)/sqrt(2)*sigma) - Phi((x_min-mu)/sqrt(2)*sigma)]^-1
    where Phi(xi) = 0.5*(1 + erf(xi/sqrt(2))).
    """
    f_x = jnp.exp(-(x-mean)**2/(2*std**2))
    
    xi_max = (x_max-mean)/std
    xi_min = (x_min-mean)/std
    Phi_x_max = 0.5*(1+erf(xi_max/jnp.sqrt(2)))
    Phi_x_min = 0.5*(1+erf(xi_min/jnp.sqrt(2)))
    A = (jnp.sqrt(2*jnp.pi*std**2)*(Phi_x_max-Phi_x_min))**-1
    # treat the extreme case
    trapz_norm = 1/((x_max-x_min)*f_x)
    A = jnp.clip(jnp.where(Phi_x_max==Phi_x_min, trapz_norm, A), 
                 EPSILON, None)
    
    mask_non_zero = jnp.logical_and(x>=x_min, x<=x_max)
    normed_f_x = jnp.clip(
        jnp.where(mask_non_zero, A*f_x, EPSILON),
        EPSILON, None)
    
    return normed_f_x

@jit
def smoothed_truncated_gaussian(x, x_min, x_max, mean, std, smoothing):
    """
    evaluates the gaussian pdf described by mean, std, x_min, x_max
    at array of values x.
    f_x  = A * exp(-(x-mean)/std**2),
        A = [sqrt(2*pi*sigma**2) * (Phi((x_max-mu)/sqrt(2)*sigma) - Phi((x_min-mu)/sqrt(2)*sigma)]^-1
    where Phi(xi) = 0.5*(1 + erf(xi/sqrt(2))).
    """
    low_start = x_min + smoothing
    high_start = x_max - smoothing
    low_filter = jnp.exp(-(x - low_start)**2/(2.*smoothing**2))
    low_filter = jnp.where(x < low_start,low_filter,1.)
    high_filter = jnp.exp(-(x - high_start)**2/(2.*smoothing**2))
    high_filter = jnp.where(x > high_start,high_filter,1.)

    f_x = jnp.clip(jnp.exp(-(x-mean)**2/(2*std**2)), 
                   EPSILON, None)
    
    xi_max = (x_max-mean)/std
    xi_min = (x_min-mean)/std
    Phi_x_max = 0.5*(1+erf(xi_max/jnp.sqrt(2)))
    Phi_x_min = 0.5*(1+erf(xi_min/jnp.sqrt(2)))
    A = (jnp.sqrt(2*jnp.pi*std**2)*(Phi_x_max-Phi_x_min))**-1
    # treat the extreme case
    trapz_norm = 1/((x_max-x_min)*f_x)
    A = jnp.clip(jnp.where(Phi_x_max==Phi_x_min, trapz_norm, A), 
                 EPSILON, None)

    mask_non_zero = jnp.logical_and(x>=x_min, x<=x_max)
    normed_f_x = jnp.clip(
        jnp.where(mask_non_zero, A*f_x*low_filter*high_filter, EPSILON),
        EPSILON, None)
    
    return normed_f_x

def gaussian(x, mean, std):
    """
    evaluates the gaussian pdf described by mean, std,
    at array of values x.
    f_x  = A * exp(-(x-mean)/std**2),
        A = [sqrt(2*pi*sigma**2)]^-1
    where Phi(xi) = 0.5*(1 + erf(xi/sqrt(2))).
    """
    f_x = jnp.exp(-(x-mean)**2/(2*std**2))
    A = 1./jnp.sqrt(2*jnp.pi*std**2)
    normed_f_x = jnp.clip(
        A*f_x,
        EPSILON, None)
    
    return normed_f_x


def uniform(x, x_min, x_max):
    """
    Returns U(x_min, x_max) = 1/(x_max - x_min)
    for x_min < x < x_max
    """
    norm = 1/(x_max - x_min)
    mask_non_zero = jnp.logical_and(x>=x_min, x<=x_max)
    f_x = jnp.clip(
        jnp.where(mask_non_zero, norm, EPSILON),
        EPSILON, None)
    return f_x

@jit
def right_smoothed_uniform(x, x_min, x_max, smoothing):
    """
    Returns U(x_min, x_max) = 1/(x_max - x_min)
    for x_min < x < x_max with smoothing to 0 at x_max
    """
    high_start = x_max - smoothing
    high_filter = jnp.exp(-(x - high_start)**2/(2.*smoothing**2))
    high_filter = jnp.where(x > high_start,high_filter,1.)
    f_x = jnp.clip(
        uniform(x, x_min, x_max) * high_filter, 
        EPSILON, None)
    return f_x

@jit
def left_smoothed_uniform(x, x_min, x_max, smoothing):
    """
    Returns U(x_min, x_max) = 1/(x_max - x_min)
    for x_min < x < x_max with smoothing to 0 at x_min
    """
    low_start = x_min + smoothing
    low_filter = jnp.exp(-(x - low_start)**2/(2.*smoothing**2))
    low_filter = jnp.where(x < low_start,low_filter,1.)
    f_x = jnp.clip(
        uniform(x, x_min, x_max) * low_filter, 
        EPSILON, None)
    return f_x

@jit
def smoothed_powerlaw_unnormed(x,slope,minimum,maximum,smoothing):
    low_start = minimum + smoothing
    high_start = maximum - smoothing
    low_filter = jnp.exp(-(x - low_start)**2/(2.*smoothing**2))
    low_filter = jnp.where(x < low_start,low_filter,1.)
    high_filter = jnp.exp(-(x - high_start)**2/(2.*smoothing**2))
    high_filter = jnp.where(x > high_start,high_filter,1.)
    f_x = jnp.clip(
        jnp.power(x,slope) * low_filter * high_filter,
        EPSILON, None)
    return f_x

@jit
def smoothing_window_function(x, x_min, x_max, smoothing):
    low_start = x_min + smoothing
    high_start = x_max - smoothing
    low_filter = jnp.exp(-(x - low_start)**2/(2.*smoothing**2))
    low_filter = jnp.where(x < low_start,low_filter,1.)
    high_filter = jnp.exp(-(x - high_start)**2/(2.*smoothing**2))
    high_filter = jnp.where(x > high_start,high_filter,1.)
    f_x = jnp.clip(low_filter * high_filter,
                   EPSILON, None)
    return f_x

@jit
def smoothed_powerlaw(x,slope,minimum,maximum,smoothing):
    """a power law distribution that is smoothed at the edges (for HMC happiness).
    p(x|slope,minimum,maximum) propto x^slope between minimum and maximum.
    Approximately normalized, (for small smoothing lengths).
    """
    approxnorm = (1 + slope) / jnp.array(jnp.power(maximum, 1 + slope) - jnp.power(minimum, 1 + slope))
    mask_non_zero = jnp.logical_and(x>=minimum, x<=maximum)
    f_x = jnp.where(mask_non_zero, 
                    smoothed_powerlaw_unnormed(x,slope,minimum,maximum,smoothing) * approxnorm, 
                    EPSILON)
    f_x = jnp.clip(
        f_x, EPSILON, None)
    return f_x

@jit
def gaussian_2d(x, y, mu_x, mu_y, 
                sigma_x, sigma_y, r):
    """ a correlated 2d gaussian in x and y"""
    x_vals = jnp.array([x, y])
    mu_vals = jnp.array([mu_x, mu_y])
    cov_mat = jnp.array([[sigma_x**2, r*sigma_x*sigma_y],
                           [r*sigma_x*sigma_y, sigma_y**2]])
    f_x = jax.scipy.stats.multivariate_normal.pdf(x_vals, mu_vals, cov_mat)
    return f_x

# Jit the functions
jitted_powerlaw = jit(powerlaw)
jitted_truncated_gaussian = jit(truncated_gaussian)
jitted_gaussian = jit(gaussian)
jitted_uniform = jit(uniform)