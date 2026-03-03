import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import copy
from jax.scipy.special import logsumexp

from typing import List, Dict

class JittedLikelihood:
    def __init__(self, ninj, tobs, injections, events_posteriors, 
                 pop_to_pe_func, pe_to_inj_func,
                 params_for_injs = ['s1x', 's1y', 's1z', 
                                    's2x', 's2y', 's2z', 
                                    'z', 'ln_sampling_pdf'],
                 population_params = ['m1_source', 'q', 'z' , 's1', 
                                      'costheta1', 's2', 'costheta2'],
                 debug=False, norm_val=2000):
        self.ninj = ninj
        self.tobs = tobs
        self.pop_to_pe_func = pop_to_pe_func
        self.pe_to_inj_func = pe_to_inj_func
        self.nevents = len(events_posteriors)
        self.debug = debug
        # self.norm_val=norm_val

        # add log pe to pinj ratio column for injections
        self.params_for_injs = params_for_injs
        self.ln_pe_to_pinj = self.pe_to_inj_func(
            **self._get_subset_dict(injections, self.params_for_injs))

        # population_params
        self.population_params = population_params
        
        # only save injection and events samples subset columns
        self.injections = self._get_subset_dict(injections, self.population_params)
        self.events_posteriors = [self._get_subset_dict(event, self.population_params)
            for event in events_posteriors]
        self.events_posteriors_2darr = self._list_of_dicts_to_dict_of_arrays(
            self.events_posteriors)

        # weights -> already including ninj and nsamps here!!
        self.log_weights_injections = (jnp.log(injections['mixture_weights'])
                                      - jnp.log(self.ninj))
        self.log_weights_events = self._list_of_dicts_to_dict_of_arrays(
            events_posteriors)['log_weights']

    @partial(jax.jit, static_argnames=['self'])
    def compute_neff(self, weights):
        return jnp.sum(weights, axis=-1)**2/jnp.sum(weights**2, axis=-1)

    @staticmethod
    def _get_subset_dict(original_dict, list_keynames):
        subset_dict = {
            key: original_dict[key]
            for key in list_keynames
            if key in original_dict
        }
        return subset_dict

    @staticmethod
    def _list_of_dicts_to_dict_of_arrays(data_list: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
        if not data_list:
            raise ValueError("Input list is empty")

        # Get key order from first dict (assumes all dicts have same keys)
        keys = list(data_list[0].keys())

        # Stack each key's values across the list
        out = {}
        for key in keys:
            stacked = jnp.stack([d[key] for d in data_list])  # shape (N, M)
            out[key] = stacked
        return out

    @partial(jax.jit, static_argnames=['self'])
    def _compute_vt(self, shape_hyperparams_dict):
        log_pop_to_pe = self.pop_to_pe_func(**self.injections,
                                     **shape_hyperparams_dict)
        log_vt = (jnp.log(self.tobs) 
                  # - jnp.log(ninj) #included in log_weights
                  + logsumexp(
                      log_pop_to_pe
                      + self.ln_pe_to_pinj 
                      + self.log_weights_injections)
                 )
        neff = self.compute_neff(jnp.exp(log_pop_to_pe 
                               + self.ln_pe_to_pinj 
                               + self.log_weights_injections))
        return jnp.exp(log_vt), neff
    
    @partial(jax.jit, static_argnames=['self'])
    def _compute_log_warr(self, shape_hyperparams_dict):

        log_pop_to_pe = self.pop_to_pe_func(**self.events_posteriors_2darr,
                           **shape_hyperparams_dict)
        logsum = jax.scipy.special.logsumexp(
            log_pop_to_pe
            + self.log_weights_events,
            axis=1)
        
        all_neff = self.compute_neff(
            jnp.exp(log_pop_to_pe + self.log_weights_events))
        return logsum, all_neff

    @partial(jax.jit, static_argnames=['self'])
    def lnlike(self, hyperparams_dict):
        hyperparams_dict = hyperparams_dict.copy()
        rate = hyperparams_dict.pop('rate')
        
        vt, vt_neff = self._compute_vt(hyperparams_dict)
        log_warr, all_pe_neff = self._compute_log_warr(hyperparams_dict)
        
        lnlike = (-(rate * vt)
                  + self.nevents*jnp.log(rate)
                  + jnp.sum(log_warr)
                 )
        return lnlike, vt_neff, all_pe_neff