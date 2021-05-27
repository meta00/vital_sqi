"""
Implementation of SQIs for ECG raw signals based on DiMarco2012.
"""
import scipy.signal as sn
import numpy as np


def band_energy_sqi(signal, sampling_rate, band=None):
    """

    Parameters
    ----------
    signal :
        
    sampling_rate :
        
    band :
         (Default value = None)

    Returns
    -------

    """
    f, t, spec = sn.stft(signal, fs = sampling_rate,
                         window = 'hann', nperseg = 2048, noverlap = 1838,
                         detrend = False, return_onesided = False,
                         boundary = 'zeros',
                         padded = True)
    idx = np.where(np.logical_and(f > band[0], f <= band[1]))
    max_time_marginal = max(np.sum(spec[idx, :], axis = 1)[0, :]).real
    return max_time_marginal


def lf_energy_sqi(signal, sampling_rate, band=[0, 0.5]):
    """

    Parameters
    ----------
    signal :
        
    sampling_rate :
        
    band :
         (Default value = [0)
    0.5] :
        

    Returns
    -------

    """
    return band_energy_sqi(signal, sampling_rate, band)


def qrs_energy_sqi(signal, sampling_rate, band=[5, 25]):
    """

    Parameters
    ----------
    signal :
        
    sampling_rate :
        
    band :
         (Default value = [5, 25] :
        

    Returns
    -------

    """
    return band_energy_sqi(signal, sampling_rate, band)


def hf_energy_sqi(signal, sampling_rate, band=[100, np.Inf]):
    """

    Parameters
    ----------
    signal :
        
    sampling_rate :
        
    band :
         (Default value = [100, np.Inf] :
        

    Returns
    -------

    """
    return band_energy_sqi(signal, sampling_rate, band)


def vhf_norm_power_sqi(signal, sampling_rate, band =[150, np.Inf]):
    """

    Parameters
    ----------
    signal :
        
    sampling_rate :
        
    band :
         (Default value = [150, np.Inf] :
        

    Returns
    -------

    """
    f, t, spec = sn.stft(signal, fs = sampling_rate,
                         window = 'hann', nperseg = 2048, noverlap = 1838,
                         detrend = False, return_onesided = False,
                         boundary = 'zeros',
                         padded = True)
    idx = np.where(np.logical_and(spec > band[0], spec <= band[1]))
    freq_marginal = np.sum(spec[idx, :], axis = 0)[0, :]
    np_vhf = (np.median(freq_marginal)/max(freq_marginal)).real
    return np_vhf


def qrs_amplitude_sqi():
    """ """
    return


import os, tempfile
from vital_sqi.data.signal_io import *

file_in = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
                          '/vital_sqi/tests/test_data/example.edf')
out = ECG_reader(file_in, 'edf')
vhf = vhf_norm_power_sqi(out.signals[:, 0], out.sampling_rate)
