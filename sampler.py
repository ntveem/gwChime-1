import jax.random as random
import numpyro
import numpyro.distributions as dist
import time
import dynesty

from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)
import pickle

def run_dynesty(likelihood_obj, hp_range_dic,
                sampler_kwargs={'dlogz':0.01, 'nlive':1000},
                output_fname=None):
    """
    run using dynesty
    """
    def make_loglike_function(likelihood_object, hyperparams_list):
        def loglike_function(u):
            pars_dict = {key: u[i] 
                         for i, key in enumerate(hyperparams_list)}
            
            if jnp.logical_or(
                jnp.any(likelihood_object.lnlike(pars_dict)[2]<153), 
                likelihood_object.lnlike(pars_dict)[1]<4*153):
                return -jnp.inf
            else:
                return likelihood_object.lnlike(pars_dict)[0]
        return loglike_function

    def make_ptform(bounds_dict):
        def ptform(u):
            # u is a 1D array of 10 values in [0, 1]
            return [a + (b - a) * ui for ui, (a, b) in zip(u, bounds_dict.values())]
        return ptform

    hp_list = list(hp_range_dic.keys())
    print(hp_list)
    loglike_func = make_loglike_function(likelihood_obj, hp_list)
    ptform_func = make_ptform(hp_range_dic)
    ndim = len(hp_list)
    
    sampler = dynesty.NestedSampler(loglike_func, ptform_func,
                                    ndim, bound='single',
                                    nlive=sampler_kwargs['nlive'])
    sampler.run_nested(dlogz=sampler_kwargs['dlogz'], 
                       print_progress=True)
    res = sampler.results
    res.summary()
    posterior_samples_dict = {key:res.samples[i] 
                              for i, key in enumerate(hp_list)}
    posterior_samples_dict.update({'weights': res.importance_weights()})
    # posterior_samples_dict.update({'logz':})
    
    # Save to a pickle file
    if output_fname:
        with open(output_fname, "wb") as f:
            pickle.dump(posterior_samples_dict, f)
    else:
        print("No output_fname provided so not saving samples")
    return res


# def run_numpyro_nuts(likelihood_obj, sampler_kwargs={},
#                      save_output=True, output_fname=None):
#     """
#     run using HMC numpyro NUTS
#     """
#     return