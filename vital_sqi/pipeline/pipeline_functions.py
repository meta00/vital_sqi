import numpy as np
import vital_sqi.preprocess.preprocess_signal as sqi_pre
import heartpy as hp
from heartpy.analysis import calc_ts_measures, calc_rr, calc_fd_measures,\
    clean_rr_intervals, calc_poincare, calc_breathing
from heartpy.peakdetection import check_peaks, detect_peaks
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.common.utils import get_nn
import pandas as pd
from vital_sqi.data.signal_io import PPG_reader
import warnings
import inspect


def get_all_features_heartpy(data_sample, sample_rate=100, rpeak_detector=0):
    """

    Parameters
    ----------
    data_sample :
        Raw signal

    sample_rate :
        (Default value = 100)
    rpeak_detector :
        (Default value = 0)

    Returns
    -------


    """
    # time domain features
    td_features = ["bpm", "ibi", "sdnn", "sdsd", "rmssd", "pnn20", "pnn50",
                   "hr_mad", "sd1", "sd2", "s", "sd1/sd2", "breathingrate"]
    # frequency domain features
    fd_features = ["lf", "hf", "lf/hf"]
    try:
        wd, m = hp.process(data_sample, sample_rate, calc_freq=True)
    except Exception as e:
        try:
            wd, m = hp.process(data_sample, sample_rate)
        except:
            time_domain_features = {k: np.nan for k in td_features}
            frequency_domain_features = {k: np.nan for k in fd_features}
            return time_domain_features, frequency_domain_features
    if rpeak_detector in [1, 2, 3, 4]:
        detector = PeakDetector(wave_type='ecg')
        peak_list = \
        detector.ppg_detector(data_sample, rpeak_detector, preprocess=False)[0]
        wd["peaklist"] = peak_list
        wd = calc_rr(peak_list, sample_rate, working_data=wd)
        wd = check_peaks(wd['RR_list'], wd['peaklist'], wd['ybeat'],
                         reject_segmentwise=False, working_data=wd)
        wd = clean_rr_intervals(working_data=wd)
        rr_diff = wd['RR_list']
        rr_sqdiff = np.power(rr_diff, 2)
        wd, m = calc_ts_measures(wd['RR_list'], rr_diff, rr_sqdiff,
                                 working_data=wd)
        m = calc_poincare(wd['RR_list'], wd['RR_masklist'], measures=m,
                          working_data=wd)
        try:
            measures, working_data = calc_breathing(wd['RR_list_cor'],
                                                    data_sample, sample_rate,
                                                    measures=m, working_data=wd)
        except:
            measures['breathingrate'] = np.nan

        wd, m = calc_fd_measures(measures=measures, working_data=working_data)

    time_domain_features = {k: m[k] for k in td_features}

    frequency_domain_features = {}
    for k in fd_features:
        if k in m.keys():
            frequency_domain_features[k] = m[k]
        else:
            frequency_domain_features[k] = np.nan
    # frequency_domain_features = {k:m[k] for k in fd_features if k in m.keys}
    # frequency_domain_features = {k:np.na for k in fd_features if k not in m.keys}

    return time_domain_features, frequency_domain_features


def per_beat_sqi(sqi_func, troughs, signal, taper, **kwargs):
    """
    Perform a per-beat application of the selected SQI function on the signal
    segment

    Parameters
    ----------
    sqi_func : function
        An SQI function to be performed.

    troughs : array of int
        Idices of troughs in the signal provided by peak detector to be able to extract individual beats

    signal :
        Signal array containing one segment of the waveform

    taper : bool
        Is each beat need to be tapered or not before executing the SQI function

    **kwargs : dict
        Additional positional arguments that needs to be fed into the SQI function

    Returns
    -------
    calculated_SQI : array
        An array with SQI values for each beat of the signal

    """
    #Remove first and last trough as they might be on the edge
    troughs = troughs[1:-1]
    if len(troughs) > 2:
        sqi_vals = []
        for idx, beat_start in enumerate(troughs[:-1]):
            single_beat = signal[beat_start:troughs[idx+1]]
            if taper:
                single_beat = sqi_pre.taper_signal(single_beat)
            if len(kwargs) != 0:
                args = tuple(kwargs.values())
                sqi_vals.append(sqi_func(single_beat, *args))
            else:
                sqi_vals.append(sqi_func(single_beat))
        return sqi_vals

    else:
        return -np.inf
        raise Exception("Not enough peaks in the signal to generate per beat SQI")


def get_sqi_dict(sqis, sqi_name):
    """

    :param sqis:
    :param sqi_name:
    :return:
    """
    if isinstance(sqis, dict):
        return sqis

    if isinstance(sqis, (float, int)):
        return {sqi_name: sqis}

    if isinstance(sqis, list):
        SQI_dict = {}
        variations_stats = ['_mean', '_median', '_std']
        SQI_dict[sqi_name + variations_stats[1]] = np.mean(sqis)
        SQI_dict[sqi_name + variations_stats[2]] = np.median(sqis)
        SQI_dict[sqi_name + variations_stats[3]] = np.std(sqis)
        return SQI_dict

    if sqi_name == 'correlogram':
        SQI_dict = {}
        variations_acf = ['_peak1', '_peak2', '_peak3', '_value1', '_value2', '_value3']
        for idx, variations in enumerate(variations_acf):
            SQI_dict['correlogram' + variations] = sqis[idx]

    if isinstance(sqis, tuple):
        SQI_dict = {}
        for features_dict in sqis:
            SQI_dict = {**SQI_dict, **features_dict}
        return SQI_dict

def get_sqi(sqi_func, s, per_beat=False,
            wave_type='ppg',peak_detector=7,
            **kwargs):
    signal_arg = inspect.getfullargspec(sqi_func)[0][0]
    if signal_arg == 'nn_interval':
        warnings.warn("Using a SQI requires NN interval input")
    if per_beat:
        # Prepare primary peak detector and perform peak detection
        detector = PeakDetector()
        if wave_type =='ppg':
            peak_list, trough_list = detector.ppg_detector(s,
                                                    peak_detector)
        else:
            peak_list, trough_list = detector.ecg_detector(s,
                                                    peak_detector)
        sqi_scores = per_beat_sqi(sqi_func, trough_list, s, **kwargs)
    else:
        if 'wave_type' in inspect.getfullargspec(sqi_func)[0]:
            kwargs['wave_type'] = wave_type
        sqi_scores = sqi_func(s,**kwargs)
        sqi_name = sqi_func.__name__
    sqi_score_dict = get_sqi_dict(sqi_scores,sqi_name)
    #===================================================================
    #   Output the dictionary -> then use dict.update(dict)
    #===================================================================
    return sqi_score_dict


def segment_PPG_SQI_extraction(sig,sqi_list,nn_sqi_list,nn_sqi_arg_list,sqi_arg_list):
    """
    # sqi_list = [
    #         dtw_sqi,
    #         euclidean_sqi,
    #         get_all_features_hrva,
    #         ectopic_sqi,
    #         correlogram_sqi,
    #         interpolation_sqi,
    #         msq_sqi,
    #         perfusion_sqi,
    #         kurtosis_sqi, skewness_sqi, entropy_sqi, signal_to_noise_sqi,
    #         zero_crossings_rate_sqi, mean_crossing_rate_sqi,
    #         band_energy_sqi, lf_energy_sqi, qrs_energy_sqi, hf_energy_sqi, vhf_norm_power_sqi, qrs_a_sqi
    #     ]
    #
    #     nn_sqi_list = [
    #         nn_mean_sqi, sdnn_sqi, sdsd_sqi, rmssd_sqi, cvsd_sqi, cvnn_sqi,
    #         mean_nn_sqi, median_nn_sqi, pnn_sqi,
    #         hr_mean_sqi, hr_median_sqi, hr_min_sqi, hr_max_sqi, hr_range_sqi,
    #         peak_frequency_sqi, absolute_power_sqi, log_power_sqi, relative_power_sqi, normalized_power_sqi,
    #         lf_hf_ratio_sqi
    #         # , poincare_feature_sqi
    #     ]
    #
    #     nn_sqi_arg_list = [
    #                           {}
    #                       ] * len(nn_sqi_list)
    :param sig:
    :param sqi_list:
    :param nn_sqi_list:
    :param nn_sqi_arg_list:
    :param sqi_arg_list:
    :return:
    """
    s = sig.iloc[:,1]
    sqi_score = {}
    for sqi_ in sqi_list:
        try:
            sqi_score = {**sqi_score,**get_sqi(sqi_,s)}
        except Exception as err:
            print(err)
            continue
    print("*------------------------------HRV---------------------------------*")
    for (sqi_,args_) in zip(nn_sqi_list, nn_sqi_arg_list):
        try:
            nn_intervals = get_nn(s)
            sqi_score = {**sqi_score,**get_sqi(sqi_,nn_intervals,**args_)}
        except Exception as err:
            print(err)
            continue

def compute_SQI(signal, segment_length='30s', primary_peakdet=7, secondary_peakdet=6, wave_type='ppg', sampling_rate=100, template_type=1):
    if wave_type == 'ppg':
        try:
            sqis = signal.groupby(pd.Grouper(freq=segment_length)).apply(segment_PPG_SQI_extraction)
        except Exception as e:
            return None
    # elif wave_type == 'ecg':
    #     sqis = signal.groupby(pd.Grouper(freq=segment_length)).apply(segment_ECG_SQI_extraction, sampling_rate, primary_peakdet, secondary_peakdet, (1, 1), (20, 4), template_type)
    else:
        raise Exception("Wrong type of waveform supplied. Only accepts 'ppg' or 'ecg'.")
    return sqis


def pipeline(file_name):
    out = PPG_reader(file_name,
                     timestamp_idx=['TIMESTAMP_MS'], signal_idx=['PLETH'], info_idx=['PULSE_BPM',
                                                                                     'SPO2_PCT', 'PERFUSION_INDEX'],
                     start_datetime='2020-04-12 10:00:00')

    ppg_stable = out.signals

    ppg_stable.index = pd.to_timedelta(ppg_stable.index / 100, unit='s')
    ppg_stable = ppg_stable[["timestamps", "PLETH"]]

    sqis = compute_SQI(ppg_stable, '30s')
    print(sqis)
    return sqis