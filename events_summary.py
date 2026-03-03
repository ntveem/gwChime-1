"""
Contains all the events information required to
for population inference
"""
import h5py
import numpy as np
import os
import pandas as pd
import warnings

class EventsSummary:
    def __init__(self, pe_samples, pastros, ifars,
                 obs_run, pe_prior, ifar_threshold=1.0):
        """
        Parameters
        ----------
        pe_samples: dict of pd.DataFrames
            dictionary whose keys are event names
            and values are a dataframe of samples
        
        pastros: dict
            dictionary with event names
            and their pastro

        ifars: dict
            dictionary with event names
            and their ifars

        obs_run: str
            Example: 'O3a'

        pe_prior: str
            prior used to do PE

        ifar_threshold: float
            threshold on ifar, same as the
            threshold used on injections

        """
        self.obs_run = obs_run
        self.pe_prior = pe_prior
        self.events = list(pe_samples.keys())
        self.ifar_threshold = ifar_threshold
        
        self.pe_samples = pe_samples
        self.ifars = ifars
        self.pastros = pastros

        self.pe_samples_array, self.pastros_array = \
            self._get_samples_and_pastro_arrays()

        warnings.warn("In progress..." +
            "The ifar threshold is not actually being applied." +
            "Not a problem if doing LVC analysis.")

    def to_hdf5(self, file_path):
        """
        Write object to hdf5 file
        """
        with h5py.File(file_path, 'w') as f:
            f.attrs["obs_run"] = self.obs_run
            f.attrs["prior"] = self.pe_prior
            for event_name, samples_df in self.pe_samples.items():
                event_group = f.create_group(event_name)
                event_group.attrs["ifar"] = self.ifars[event_name]
                event_group.attrs["pastro"] = self.pastros[event_name]
                for key in samples_df.keys():
                    event_group.create_dataset(str(key),
                                               data=samples_df[key])

    @classmethod
    def from_hdf5(cls, file_path,
                  ifar_threshold=1.0):
        """
        Read from hdf5 file and return
        EventsSummary object.
        """
        pe_samples_all = {} # dict of PEs indexed by event
        ifars_dict = {} # dict of ifars indexed by event
        pastros_dict = {} # dict of pastros indexed by event

        with h5py.File(file_path, 'r') as f:
            obs_run = f.attrs["obs_run"]
            prior_name = f.attrs["prior"]
            for event_name in f.keys():
                event_group = f[event_name]
                ifars_dict[event_name] = event_group.attrs["ifar"]
                pastros_dict[event_name] = event_group.attrs["pastro"]
                event_samples = {}
                for key in event_group.keys():
                    event_samples[key] = event_group[key][:]
                pe_samples_all[event_name] = pd.DataFrame(event_samples)

        return cls(pe_samples_all, pastros_dict, ifars_dict,
                 obs_run, prior_name, ifar_threshold=ifar_threshold)

    def downsample_posteriors(self, max_samples, rs=1):
        """
        modifies pe_samples dictionary such that
        the len(samples)<= max_samples
        """
        for key, samples_df in self.pe_samples.items():
            if len(samples_df)>max_samples:
                self.pe_samples[key] = samples_df.sample(
                    max_samples, random_state=rs).reset_index(drop=True)
        self.pe_samples_array, self.pastros_array = \
            self._get_samples_and_pastro_arrays()
        
    def _get_samples_and_pastro_arrays(self):
        """
        returns an array of pd.Dataframes
        and pastros in the same order
        """
        pastros_list = []
        pe_samples_list = []
        for evname in self.events:
            pastros_list.append(self.pastros[evname])
            pe_samples_list.append(self.pe_samples[evname])
        pastros_array = np.array(pastros_list)
        pe_samples_array = np.empty(len(pe_samples_list), dtype=object)
        for i, df in enumerate(pe_samples_list):
            pe_samples_array[i] = df
        # pe_samples_array = np.array(pe_samples_list, dtype=object)
        return (pe_samples_array, pastros_array)

    # def _apply_ifar_cut(self, ifar_threshold=1.0):
    #     '''
    #     Apply ifar cut such that only events
    #     above  
    #     '''
    