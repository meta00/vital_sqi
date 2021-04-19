"""Signal quality indexes based on R peak detection"""
import numpy as np
from scipy import signal
from scipy import interpolate

import heartpy as hp
from heartpy.datautils import rolling_mean
from hrvanalysis import get_time_domain_features,get_frequency_domain_features,\
    get_nn_intervals,get_csi_cvi_features,get_geometrical_features
from hrvanalysis.preprocessing import remove_outliers,remove_ectopic_beats,interpolate_nan_values
from heartpy.analysis import calc_ts_measures, calc_rr, calc_fd_measures,\
    clean_rr_intervals,calc_poincare,calc_breathing
from heartpy.peakdetection import check_peaks, detect_peaks

from vital_sqi.common.rpeak_detection import PeakDetector

def get_all_features_hrva(data_sample,sample_rate=100,rpeak_method=0):
    """
    :param data_sample:
    :param sample_rate:
    :param rpeak_method:
    :return:
    """

    if rpeak_method in [1,2,3,4]:
        detector = PeakDetector()
        peak_list = detector.ppg_detector(data_sample,rpeak_method)[0]
    else:
        rol_mean = rolling_mean(data_sample, windowsize=0.75, sample_rate=100.0)
        peaks_wd = detect_peaks(data_sample,rol_mean,ma_perc = 20, sample_rate = 100.0)
        peak_list = peaks_wd["peaklist"]

    rr_list = np.diff(peak_list) * (1000/sample_rate) #1000 milisecond

    nn_list = get_nn_intervals(rr_list)
    nn_list_non_na = np.copy(nn_list)
    nn_list_non_na[np.where(np.isnan(nn_list_non_na))[0]] = -1

    time_domain_features = get_time_domain_features(rr_list)
    frequency_domain_features = get_frequency_domain_features(rr_list)
    geometrical_features = get_geometrical_features(rr_list)
    csi_cvi_features = get_csi_cvi_features(rr_list)

    return time_domain_features,frequency_domain_features,\
           geometrical_features,csi_cvi_features

def get_all_features_heartpy(data_sample,sample_rate=100,rpeak_detector = 0):
    # time domain features
    td_features = ["bpm", "ibi", "sdnn", "sdsd", "rmssd",
                   "pnn20", "pnn50", "hr_mad", "sd1", "sd2",
                   "s", "sd1/sd2", "breathingrate"]
    # frequency domain features
    fd_features = ["lf", "hf", "lf/hf"]
    try:
        wd, m = hp.process(data_sample, sample_rate,calc_freq = True)
    except Exception as e:
        try:
            wd, m = hp.process(data_sample, sample_rate)
        except:
            time_domain_features = {k: np.nan for k in td_features}
            frequency_domain_features = {k: np.nan for k in fd_features}
            return time_domain_features,frequency_domain_features
    if rpeak_detector in [1,2,3,4]:
        detector = PeakDetector(wave_type='ecg')
        peak_list = detector.ppg_detector(data_sample,rpeak_detector,preprocess=False)[0]
        wd["peaklist"] = peak_list
        wd = calc_rr(peak_list,sample_rate,working_data=wd)
        wd = check_peaks(wd['RR_list'], wd['peaklist'], wd['ybeat'],
                                   reject_segmentwise=False, working_data=wd)
        wd = clean_rr_intervals(working_data=wd)
        rr_diff = wd['RR_list']
        rr_sqdiff = np.power(rr_diff, 2)
        wd, m = calc_ts_measures(wd['RR_list'], rr_diff, rr_sqdiff,working_data=wd)
        m = calc_poincare(wd['RR_list'], wd['RR_masklist'], measures=m,
                                 working_data=wd)
        try:
            measures, working_data = calc_breathing(wd['RR_list_cor'], data_sample, sample_rate,
                                                    measures=m,
                                                    working_data=wd)
        except:
            measures['breathingrate'] = np.nan

        wd, m = calc_fd_measures(measures=measures,working_data=working_data)

    time_domain_features = {k:m[k] for k in td_features}

    frequency_domain_features = {}
    for k in fd_features:
        if k in m.keys():
            frequency_domain_features[k] = m[k]
        else:
            frequency_domain_features[k] = np.nan
    # frequency_domain_features = {k:m[k] for k in fd_features if k in m.keys}
    # frequency_domain_features = {k:np.na for k in fd_features if k not in m.keys}

    return time_domain_features,frequency_domain_features

def get_peak_error_features(data_sample,sample_rate=100,rpeak_detector = 0,low_rri=300,
                            high_rri=2000):
    rules = ["malik", "karlsson", "kamath", "acar"]
    try:
        wd, m = hp.process(data_sample, sample_rate, calc_freq=True)
    except:
        try:
            wd, m = hp.process(data_sample, sample_rate)
        except:
            error_dict = {rule+"_error":np.nan  for rule in rules}
            error_dict["outlier_error"] = np.nan
            return error_dict

    if rpeak_detector in [1, 2, 3, 4]:
        detector = PeakDetector(wave_type='ecg')
        peak_list = detector.ppg_detector(data_sample, rpeak_detector, preprocess=False)[0]
        wd["peaklist"] = peak_list
        wd = calc_rr(peak_list, sample_rate, working_data=wd)
        wd = check_peaks(wd['RR_list'], wd['peaklist'], wd['ybeat'],
                         reject_segmentwise=False, working_data=wd)
        wd = clean_rr_intervals(working_data=wd)

    rr_intervals = wd["RR_list"]

    rr_intervals_cleaned = remove_outliers(rr_intervals, low_rri=low_rri, high_rri=high_rri)
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

def calculate_Spectrum(rr_intervals, method='welch',
                           sampling_frequency=4,
                           interpolation_method="linear",
                           power_type='density',
                           vlf_min= 0.003, vlf_max= 0.04,
                           hf_min= 0.15, hf_max= 0.4,
                           ):
    """
    Returns the frequency and power of the signal.

    Parameters
    ---------
    rr_intervals : array-like
        list of RR interval (in ms)
    method : str
        Method used to calculate the psd or powerband or spectrogram.
        available methods are:
        'welch': apply welch method to compute PSD
        'lomb': apply lomb method to compute PSD
        'ar': method to compute the periodogram - if compute PSD then power_type = 'density'
        'spectrogram': method to compute the spectrogram, output an extra list represent timestamp
        'powerband': compute the power density at 4 power band.
            The min-max boundary of power need to be defined

    sampling_frequency : int
        Frequency of the spectrum need to be observed. Common value range from 1 Hz to 10 Hz,
        by default set to 4 Hz. Detail can be found from the ECG book
    interpolation_method : str
        Kind of interpolation as a string, by default "linear". Applicable for welch
    power_type: str
        'density':
        'spectrogram':

    Returns
    ---------
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    """

    # create timestamp to do the interpolation
    ts_rr = [np.sum(rr_intervals[:i]) / 1000 for i in range(len(rr_intervals))]
    # convert each RR interval (unit ms) to bpm - representative
    bpm_list = (1000 * 60) / rr_intervals

    first = ts_rr[0]
    last = ts_rr[-1]
    # interpolate the data

    if method == 'welch':
        # ---------- Interpolation of signal ---------- #
        # funct = interpolate.interp1d(x=timestamp_list, y=nn_intervals, kind=interpolation_method)
        interpolator = interpolate.interp1d(ts_rr, bpm_list, kind=interpolation_method)
        # create timestamp for the interpolate rr
        time_offset = 1 / sampling_frequency
        # ts_interpolate = [np.sum(RR[:i]) / 100 for i in range(len(IRR))]
        ts_interpolate = np.arange(0, last - first, time_offset)

        # timestamps_interpolation = _create_interpolated_timestamp_list(nn_intervals, sampling_frequency)
        nni_interpolation = interpolator(ts_interpolate)

        # ---------- Remove DC Component ---------- #
        nni_normalized = nni_interpolation - np.mean(nni_interpolation)

        #  --------- Compute Power Spectral Density  --------- #
        freq, psd = signal.welch(x=nni_normalized, fs=sampling_frequency, window='hann',
                                 nfft=4096)

    elif method == 'lomb':
        freq, psd = signal.lombscargle(ts_rr, bpm_list, sampling_frequency)
        # freq, psd = LombScargle(timestamp_list, nn_intervals,
        #                         normalization='psd').autopower(minimum_frequency=vlf_band[0],
        #                                                        maximum_frequency=hf_band[1])
    elif method == 'ar':
        freq, psd = signal.periodogram(bpm_list, sampling_frequency, window='boxcar',
                                       nfft=None, detrend='constant',
                                       return_onesided=True,
                                       scaling=power_type, axis=- 1)
    elif method == 'powerband':
        # TODO
        freq, psd, t = signal.spectrogram(bpm_list, sampling_frequency)

    elif method == 'spectrogram':
        freq, psd, t = signal.spectrogram(bpm_list, sampling_frequency)
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

    return freq, psd

#TODO remove test file
import pandas as pd
import  os
if __name__ == "__main__":
    filename = "24EI-011-PPG-day1-0237.csv"
    df = pd.read_csv(os.path.join(os.getcwd(), "../../../..", "Work", "data",
                                  "peak_detection_ds", filename))
    y = np.array(df).reshape(-1)
    # time_domain_features, frequency_domain_features, \
    # geometrical_features, csi_cvi_features = get_all_features_hrva(y,rpeak_method=2)
    #
    # time_domain_features_heartpy, frequency_domain_features_heartpy = \
    #     get_all_features_heartpy(y,rpeak_detector=1)

    peak_error_features = get_peak_error_features(y,rpeak_detector=1)

    print("")