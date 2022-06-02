"""Signal quality indexes based on R peak detection
This module capitalises on different R peak detectors for ECG and PGG,
using the resulted NN series to evaluate the raw signal quality.
- Ratio of ectopic beats
- Correlogram
- MSQ: Evaluate the consistency of two NN intervals by two peak detectors.
- Interpolation: Comparison of interpolated and non-interpolated NN intervals
    evaluate gap in the non-interpolated NN interval.
"""
import numpy as np
import scipy.interpolate
from scipy import signal
from hrvanalysis.preprocessing import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from statsmodels.tsa.stattools import acf
from vital_sqi.common.rpeak_detection import PeakDetector


def ectopic_sqi(data_sample, sample_rate=100, rpeak_detector=0,
                            low_rri=300,
                            high_rri=2000):
    """

    Parameters
    ----------
    data_sample :
        
    sample_rate :
        (Default value = 100)
    rpeak_detector :
        (Default value = 0)
        To explain other detector options
    low_rri :
        (Default value = 300)
    high_rri :
        (Default value = 2000)

    Returns
    -------

    
    """
    rules = ["malik", "karlsson", "kamath", "acar"]
    try:
        wd, m = hp.process(data_sample, sample_rate, calc_freq=True)
    except:
        try:
            wd, m = hp.process(data_sample, sample_rate)
        except:
            error_dict = {rule+"_error": np.nan for rule in rules}
            error_dict["outlier_error"] = np.nan
            return error_dict

    if rpeak_detector in [1, 2, 3, 4]:
        detector = PeakDetector(wave_type='ecg')
        peak_list = detector.ppg_detector(data_sample, rpeak_detector,
                                          preprocess=False)[0]
        wd["peaklist"] = peak_list
        wd = calc_rr(peak_list, sample_rate, working_data=wd)
        wd = check_peaks(wd['RR_list'], wd['peaklist'], wd['ybeat'],
                         reject_segmentwise=False, working_data=wd)
        wd = clean_rr_intervals(working_data=wd)

    rr_intervals = wd["RR_list"]

    rr_intervals_cleaned = remove_outliers(rr_intervals, low_rri=low_rri,
                                           high_rri=high_rri)
    number_outliers = len(np.where(np.isnan(rr_intervals_cleaned))[0])
    outlier_ratio = number_outliers/(len(rr_intervals_cleaned)-number_outliers)

    error_sqi = {}
    error_sqi['outlier_error'] = outlier_ratio

    interpolated_rr_intervals = interpolate_nan_values(rr_intervals_cleaned)

    for rule in rules:
        nn_intervals = remove_ectopic_beats(interpolated_rr_intervals,
                                            method=rule)
        number_ectopics = len(np.where(np.isnan(nn_intervals))[0])
        ectopic_ratio = number_ectopics/(len(nn_intervals)-number_ectopics)
        error_sqi[rule+"_error"] = ectopic_ratio

    return error_sqi


def correlogram_sqi(data_sample, sample_rate=100, time_lag=3, n_selection=3):
    """The method is based on the paper 'Classification of the Quality of
    Wristband-based Photoplethysmography Signals'

    Parameters
    ----------
    data_sample :
        Raw data
    sample_rate :
        (Default value = 100)
    time_lag :
        (Default value = 3)
    n_selection :
        (Default value = 3)

    Returns
    -------

    
    """
    nlags = time_lag*sample_rate
    corr = acf(data_sample, nlags=nlags)
    corr_peaks_idx = signal.find_peaks(corr)[0]
    corr_peaks_value = corr[corr_peaks_idx]
    if n_selection > len(corr_peaks_value):
        n_selection = len(corr_peaks_value)

    corr_sqi = [i for i in corr_peaks_idx[:n_selection]]+\
          [i for i in corr_peaks_value[:n_selection]]
    return corr_sqi


def interpolation_sqi(sample_data):
    """

    Parameters
    ----------
    sample_data :
        

    Returns
    -------

    """

    return None


def msq_sqi(y, peaks_1, peak_detect2=6):
    """
    MSQ SQI as defined in Elgendi et al
    "Optimal Signal Quality Index for Photoplethysmogram Signals"
    with modification of the second algorithm used.
    Instead of Bing's, a SciPy built-in implementation is used.
    The SQI tracks the agreement between two peak detectors
    to evaluate quality of the signal.

    Parameters
    ----------
    x : sequence
        A signal with peaks.

    peaks_1 : array of int
        Already computed peaks arry from the primary peak_detector

    peak_detect2 : int
        Type of the second peak detection algorithm, default = Scipy

    Returns
    -------
    msq_sqi : number
        MSQ SQI value for the given signal

    """
    # Viet lai cho ecg va ppg, input la ten 2 peaks
    detector = PeakDetector(wave_type='ppg')
    peaks_2 = detector.ppg_detector(y, detector_type=peak_detect2, preprocess=False)
    if len(peaks_1)==0 or len(peaks_2)==0:
        return 0.0
    peak1_dom = len(np.intersect1d(peaks_1,peaks_2))/len(peaks_1)
    peak2_dom = len(np.intersect1d(peaks_2,peaks_1))/len(peaks_2)
    return min(peak1_dom, peak2_dom)