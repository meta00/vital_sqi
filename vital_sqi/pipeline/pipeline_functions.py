import numpy
import numpy as np
import pandas as pd
import vital_sqi.preprocess.preprocess_signal as sqi_pre
import os
import  json
from tqdm import tqdm
from hrvanalysis import get_nn_intervals
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.common.utils import create_rule_def
from vital_sqi.rule import RuleSet, Rule, update_rule
import inspect

sqi_nn_list = ["nn_mean_sqi",
               "sdnn_sqi", "sdsd_sqi",
               "rmssd_sqi", "cvsd_sqi",
               "cvnn_sqi", "pnn_sqi",
               "mean_nn_sqi", "median_nn_sqi",
               "hr_mean_sqi", "hr_median_sqi"
                              "hr_min_sqi", "hr_max_sqi", "hr_range_sqi",
               "peak_frequency_sqi", "absolute_power_sqi",
               "log_power_sqi", "relative_power_sqi",
               "normalized_power_sqi", "lf_hf_ratio_sqi",
               "poincare_sqi", "hrva"
               ]


def classify_segments(sqis, rule_dict, ruleset_order):
    ruleset = {}
    for i in ruleset_order:
        rule = Rule(ruleset_order[i])
    rule.load_def(rule_dict)
    ruleset[i] = rule
    sqis['decision'] = ruleset.execute(sqis)
    return ruleset, sqis


# Khoa
def get_reject_segments(segments, wave_type, milestones=None, info=None):
    if wave_type == 'ppg':
        out = pd.Series()
    if wave_type == 'ecg':
        out = pd.Series()
    return out


def map_decision(i):
    if i == 'accept':
        return 0
    if i == 'reject':
        return 1


def get_decision_segments(segments, decision, reject_decision):
    decision = map(map_decision, decision)
    reject_decision = map(map_decision, reject_decision)
    decision = [a + b for a, b in zip(decision + reject_decision)]
    a_segments, r_segments = []
    for i in decision:
        if decision[i] == 0:
            a_segments.append(segments[i])
        if decision[i] == 1:
            r_segments.append(segments[i])
    return a_segments, r_segments


def per_beat_sqi(sqi_func, troughs, signal, taper=False, **kwargs):
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
        return [-np.inf]
        raise Exception("Not enough peaks in the signal to generate per beat SQI")


def get_sqi_dict(sqis, sqi_name):
    """

    :param sqis:
    :param sqi_name:
    :return:
    """
    if sqi_name == 'correlogram_sqi':
        SQI_dict = {}
        variations_acf = ['_peak1', '_peak2', '_peak3', '_value1', '_value2', '_value3']
        for idx, variations in enumerate(variations_acf):
            SQI_dict['correlogram' + variations+"_sqi"] = sqis[idx]
        return SQI_dict

    if isinstance(sqis, dict):
        return sqis

    if isinstance(sqis, (float, int)):
        return {sqi_name: sqis}

    if isinstance(sqis,numpy.ndarray):
        if len(sqis.shape) == 0:
            return {sqi_name: -1}
        return {sqi_name: sqis[0]}

    if isinstance(sqis, list):
        SQI_dict = {}
        variations_stats = ['_mean', '_median', '_std']
        SQI_dict[sqi_name.split("_")[0] + variations_stats[0]+"_sqi"] = np.mean(sqis)
        SQI_dict[sqi_name.split("_")[0] + variations_stats[1]+"_sqi"] = np.median(sqis)
        SQI_dict[sqi_name.split("_")[0] + variations_stats[2]+"_sqi"] = np.std(sqis)
        return SQI_dict

    if isinstance(sqis, tuple):
        SQI_dict = {}
        for features_dict in sqis:
            features_dict_ = dict((key+"_sqi", value) for (key, value) in features_dict.items())
            SQI_dict = {**SQI_dict, **features_dict_}
        return SQI_dict


def get_sqi(sqi_func, s, per_beat=False,
            wave_type='ppg',peak_detector=7,
            **kwargs):
    signal_arg = inspect.getfullargspec(sqi_func)[0][0]
    if signal_arg == 'nn_intervals':
        s = get_nn(s.iloc[:, 1])
    else:
        s = s.iloc[:,1]
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
    return sqi_score_dict


def segment_SQI_extraction(sig,sqi_list,sqi_arg_list,wave_type):
    """

    :param sig:
    :param sqi_list: list of sqi as in MASTERDICT
    :param nn_sqi_list: list of sqi using nn_intervals as in 'HRV' MASTER_DICT
    :param nn_sqi_arg_list:
    :param sqi_arg_list:
    :return:
    """
    sqi_score = {}
    for (sqi_,args_) in zip(sqi_list,sqi_arg_list):
        try:
            args_["wave_type"] = wave_type
            sqi_score = {**sqi_score, **get_sqi(sqi_, sig, **args_)}
        except Exception as err:
            print(sqi_)
            print(err)
            continue
    return pd.Series(sqi_score)


def extract_sqi(segments, sqi_list, file_name,arg_path=None,wave_type='ppg'):
    if arg_path == None:
        arg_path = os.path.join(os.getcwd(),"../resource/sqi_args.json")
    with open(arg_path, 'r') as arg_file:
        sqi_arg_list = json.loads(arg_file.read())
    # nn_sqi_arg_list = sqi_list
    df_sqi = pd.DataFrame()
    for segment_idx in tqdm(range(len(segments))):
        segment = segments[segment_idx]
        sqi_arg_list['perf'] = {'y':segment.iloc[:,1]}
        # sqi_arg_list['zc'] = {'y': ppg_stable.iloc[:, 1]}
        # sqi_arg_list['mc'] = {'y': ppg_stable.iloc[:, 1]}
        sqis = segment_SQI_extraction(segment, sqi_list.values(),sqi_arg_list.values(),wave_type)
        segment_name_list = file_name.split("/")[-1] + "_" +str(segment_idx)
        sqis['id'] = segment_name_list
        df_sqi = df_sqi.append(sqis, ignore_index=True)
    return df_sqi


def generate_rule(rule_name, rule_def):
    rule_def, boundaries, label_list = update_rule(rule_def, is_update=False)
    rule_detail = {'def': rule_def,
                     'boundaries': boundaries,
                     'labels': label_list}
    rule = Rule(rule_name,rule_detail)
    return rule


def get_decision(df_sqi, selected_rule, json_rule_dict):
    rule_list = {}
    for (i, selected_sqi) in zip(range(len(selected_rule)), selected_rule):
        rule = generate_rule(selected_sqi, json_rule_dict[selected_sqi]['def'])

        rule_list[i + 1] = rule
    ruleset = RuleSet(rule_list)

    decision_list = []
    for idx in range(len(df_sqi)):
        row_data = pd.DataFrame(dict(df_sqi[selected_rule].iloc[idx]), index=[0])
        decision_list.append(ruleset.execute(row_data))

    return decision_list


def get_nn(s,wave_type='ppg',sample_rate=100,rpeak_method=7,remove_ectopic_beat=False):
    if wave_type=='ppg':
        detector = PeakDetector(wave_type='ppg')
        peak_list, trough_list = detector.ppg_detector(s, detector_type=rpeak_method)
    else:
        detector = PeakDetector(wave_type='ecg')
        peak_list, trough_list = detector.ecg_detector(s, detector_type=rpeak_method)

    rr_list = np.diff(peak_list) * (1000 / sample_rate)
    if not remove_ectopic_beat:
        return rr_list
    nn_list = get_nn_intervals(rr_list)
    nn_list_non_na = np.copy(nn_list)
    nn_list_non_na[np.where(np.isnan(nn_list_non_na))[0]] = -1
    return nn_list_non_na
