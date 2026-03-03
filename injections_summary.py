"""
Class that reads the injections summary from -
"utils.DATA_ROOT/injections/O3(a or b)/injection_loader/injections_summary.hdf5"
"""
import h5py
import numpy as np
import os
import pandas as pd

# These are the directories where latest injections are for O3a and O3b
# INJECTION_ROOT_DIRS = {'O3a': '/home/isha/O3a_data/injections/O3a',
#                  'O3b': '/home/isha/O3a_data/injections/O3b'}
# SUMMARY_FILE_PATHS = {'O3a': os.path.join('data',
                                # "injections_summary_pastro_added.hdf5")}
                     # 'O3b': os.path.join(INJECTION_ROOT_DIRS['O3b'], "injection_loader",
                     #            "injections_summary.hdf5")}
Z = 2.15 # 2.1455 Gpc^3 # same for O3a, O3b

class InjectionsSummary:
    def __init__(self, n_inj, t_obs, z, obs_run, 
                 recovered_injections, ifar_threshold=1.0,
                 using_lvc_injections=True, apply_lvc_pastro_cut=False,
                 ifar_column_name='ifar'):
        """
        Parameters
        ----------
        n_inj: int
            Number of injected waveforms 

        t_obs: float
            duration of observing run (in yrs)

        recovered_injections: pandas.DataFrame 
            parameters of injections recovered by search
            in injection campaign

        obs_run: str
            Example: 'O3a' or 'O3b'

        ifar_threshold: float
            threshold on ifar, same as the
            threshold used on events

        ifar_column_name: str

        """
        self.n_inj = n_inj
        self.t_obs = t_obs
        self.z = z
        self.obs_run = obs_run

        if using_lvc_injections:
            if 'name' in recovered_injections.keys():
                print('using name to apply cut')
                mask_ifar_ge_one = np.logical_or.reduce((recovered_injections['ifar_gstlal']>=ifar_threshold, 
                                                        recovered_injections['ifar_pycbc_bbh']>=ifar_threshold,
                                                        recovered_injections['ifar_pycbc_hyperbank']>=ifar_threshold,
                                                        recovered_injections['ifar_mbta']>=ifar_threshold))
                mask_snr_ge_ten = recovered_injections['optimal_snr_net']>=10
                mask_selected_injections = np.where(recovered_injections['name']==b'o3', mask_ifar_ge_one, mask_snr_ge_ten)
                recovered_injections = recovered_injections.drop('name', axis=1)
                self.recovered_injections = recovered_injections[mask_selected_injections].copy()
                self.recovered_injections.reset_index(drop=True, inplace=True)
            elif 'snr' in recovered_injections.keys():
                print("doing runs that contain semianalytic injections")
                #using O1 + O2 + O3 injections so need to apply cut separately
                snr_thr = 10
                far_thr = 1.0
                mask_selected_injections = (recovered_injections['snr'] > snr_thr) | (recovered_injections['far'] < far_thr)
                self.recovered_injections = recovered_injections[mask_selected_injections].copy()
                self.recovered_injections.reset_index(drop=True, inplace=True)
            else:
                print("doing single run")
                #using from single run
                mask_ifar_ge_one = np.logical_or.reduce((recovered_injections['ifar_gstlal']>=ifar_threshold, 
                                                        recovered_injections['ifar_pycbc_bbh']>=ifar_threshold,
                                                        recovered_injections['ifar_pycbc_hyperbank']>=ifar_threshold,
                                                        recovered_injections['ifar_mbta']>=ifar_threshold))
                mask_ifar_ge_one_indices = np.where(mask_ifar_ge_one)[0]
                self.recovered_injections = recovered_injections.iloc[mask_ifar_ge_one_indices].copy()
                self.recovered_injections.reset_index(drop=True, inplace=True)
        elif apply_lvc_pastro_cut:
            mask_pastro_ge_point5 = np.logical_or.reduce((recovered_injections['pastro_cwb']>=0.5, 
                                                recovered_injections['pastro_gstlal']>=0.5,
                                                recovered_injections['pastro_mbta']>=0.5,
                                                recovered_injections['pastro_pycbc_bbh']>=0.5,
                                                recovered_injections['pastro_pycbc_broad']>=0.5))
            mask_pastro_ge_point5_indices = np.where(mask_pastro_ge_point5)[0]
            self.recovered_injections = recovered_injections.iloc[mask_pastro_ge_point5_indices].copy()
            self.recovered_injections.reset_index(drop=True, inplace=True)
        else:
            mask_ifar_threshold = np.where(recovered_injections[ifar_column_name]>=ifar_threshold)[0]
            self.recovered_injections = recovered_injections.iloc[mask_ifar_threshold].copy()
            self.recovered_injections.reset_index(drop=True, inplace=True)

    @classmethod
    def from_hdf5(cls, file_path, ifar_threshold=1.0,
                  using_lvc_injections=True, apply_lvc_pastro_cut=False):
        """
        Reads injection information from file and 
        returns summary object.
        """
        recovered_injections_h5 = pd.DataFrame()
        try:
            with h5py.File(file_path, 'r') as f:
                n_inj_h5 = f.attrs['N_inj']
                t_obs_h5 = f.attrs['TOBS']
                z_h5 = f.attrs['Z']
                obs_run_h5 = f.attrs['obs_run']
                for name, dataset in f.items():
                    recovered_injections_h5[name] = dataset[:]
        except KeyError as e:
            print(e)
            print(f"{file_path} does not contain all the information needed to create this object")
        return cls(n_inj=n_inj_h5, t_obs=t_obs_h5, z=z_h5, obs_run=obs_run_h5,
                   recovered_injections=recovered_injections_h5,
                   ifar_threshold=ifar_threshold,
                   using_lvc_injections=using_lvc_injections,
                   apply_lvc_pastro_cut=apply_lvc_pastro_cut)

    def to_hdf5(self, file_path):
        """
        Create h5 file with summary
        """
        with h5py.File(file_path, 'w') as f:
            f.attrs['N_inj'] = self.n_inj
            f.attrs['TOBS'] = self.t_obs
            f.attrs['Z'] = self.z
            f.attrs['obs_run'] = self.obs_run
            for key, row in self.recovered_injections.items():
                # print(f"==> {key} dtypes:\n", row.dtypes)
                # print(row.head())
                f.create_dataset(key, data=row)