"""
Implementation of SQIs for waveform:
 - For ECG based on DiMarco2012.
 - For PPG to be coded (Khoa check with Stefan)
 Doi lai tham so signal
"""
import scipy.signal as sn
import numpy as np
from vital_sqi.common.rpeak_detection import *
from vital_sqi.common.utils import *


def band_energy_sqi(s, sampling_rate=100, band=None):
    """
    Compute the peak value of the time marginal of the energy distribution in a
    frequency band (DiMarco et al., 2012).

    Parameters
    ----------
<<<<<<< Updated upstream
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.
=======
    s : array_like
        Time series of measurement values
>>>>>>> Stashed changes
    sampling_rate : float, optional
        Sampling rate of the signal.
        (Default value = 100)
    band : list
        Frequency band. If None, the whole spectrum is used.
        (Default value = None)

    Returns
    -------
    max_time_marginal : float
        Maximum time marginal power in the frequency band

    Raises
    ------
    AssertionError
        when invalid band
    """
    assert np.isreal(sampling_rate), "Invalid sampling rate value."
    f, t, spec = sn.stft(s, fs=sampling_rate,

                         window='hann', nperseg=2048, noverlap=1838,
                         detrend=False, return_onesided=False,
                         boundary='zeros',
                         padded=True)
    if band is None:
        max_time_marginal = max(np.sum(spec, axis=0)).real
    if band is not None:
        assert isinstance(band, list) and band[0] <= band[1], "Invalid band " \
                                                            "values"
        idx = np.where(np.logical_and(f > band[0], f <= band[1]))
        max_time_marginal = max(np.sum(spec[idx], axis=0)).real
    return max_time_marginal


def lf_energy_sqi(s, sampling_rate, band=[0, 0.5]):
    """

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.

    sampling_rate :

    band :
        (Default value = [0)
    0.5] :


    Returns
    -------


    """
    return band_energy_sqi(s, sampling_rate, band)


def qrs_energy_sqi(s, sampling_rate, band=[5, 25]):
    """

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.

    sampling_rate :

    band :
        (Default value = [5, 25] :

    Returns
    -------


    """
    return band_energy_sqi(s, sampling_rate, band)


def hf_energy_sqi(s, sampling_rate, band=[100, np.Inf]):
    """

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.

    sampling_rate :

    band :
         (Default value = [100, np.Inf] :

    Returns
    -------


    """
    return band_energy_sqi(s, sampling_rate, band)


def vhf_norm_power_sqi(s, sampling_rate, band=[150, np.Inf]):
    """

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.

    sampling_rate :

    band :
         (Default value = [150, np.Inf] :

    Returns
    -------


    """
    f, t, spec = sn.stft(s.iloc[:, 1], fs=sampling_rate,
                         window='hann', nperseg=2048, noverlap=1838,
                         detrend=False, return_onesided=False,
                         boundary='zeros',
                         padded=True)
    idx = np.where(np.logical_and(spec > band[0], spec <= band[1]))
    freq_marginal = np.sum(spec[idx], axis=0)[0, :]
    np_vhf = (np.median(freq_marginal)/max(freq_marginal)).real
    return np_vhf


def qrs_a_sqi(s, sampling_rate):
    """QRS_A or qrs amplitude is defined as the median value of the
    peak-to-nadir amplitude difference of the QRS complexes detected,
    in a segment of 10s. Beat detection is done by Pan and Tompkins's
    algorithm. A threshold of 5 mV is set for the peak-to-nadir amplitude
    difference. For each beat, the fiducial point is set to the dominant peak
    of the QRS complex.

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as
        float.

    sampling_rate :

    Returns
    -------

    """
    detector = PeakDetector(wave_type='ecg', fs=sampling_rate)
    peaks, troughs, nadirs = detector.ecg_detector(s.iloc[:, 1],
                                                   detector_type=7)
    peak_to_nadir = np.array(peaks) - np.array(nadirs)
    peak_to_nadir = np.delete(peak_to_nadir, np.where(peak_to_nadir > 5))
    qrs_a = np.median(peak_to_nadir)
    return qrs_a


# import os, tempfile
# from vital_sqi.data.signal_io import *
#
# file_in = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                           '/vital_sqi/tests/test_data/example.edf')
# out = ECG_reader(file_in, 'edf')
# # vhf = band_energy_sqi(out.signals[:, 0], out.sampling_rate, [0, 0.5])
# f = band_energy_sqi(out.signals[:, 0], sampling_rate='100')
