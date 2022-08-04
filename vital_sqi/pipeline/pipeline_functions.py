import numpy
import numpy as np
import pandas as pd
import vital_sqi.preprocess.preprocess_signal as sqi_pre
import json
from tqdm import tqdm
from scipy.signal import resample

from vital_sqi.common.rpeak_detection import PeakDetector
import vital_sqi.sqi as sq
from vital_sqi.rule import RuleSet, Rule, update_rule
from vital_sqi.common.utils import get_nn
import inspect


def classify_segments(sqis, rule_dict_filename, ruleset_order):
    """
    Get the output decision of each segment (accept or reject)
    by evaluating the threshold of the selected rules
    with the sqi scores from the sqis dataframe
    Parameters
    ----------
    sqis : DataFrame
        A nxm table contains the list of sqis of each segment.
        n is the number of segment
        m is the number of sqi

    rule_dict_filename : string
        Path to the json file that define the threshold  of sqis
        Sample format of the json file can be found at 'vital_sqi/resource/rule_dict.json'

    ruleset_order : dict
        a dictionary specifies the order of the selected rules.
        The key indicates the order of that rule
        The value map with the rule name as defines in 'rule_dict_filename' json

    Returns
    -------
    rule_list : dict
        A dictionary contains the rule-name as the key and the rule object as the value
    sqis: DataFrame
        The same sqis table with 1 extended column indicates the decision ('accept' or 'reject')
         of each segment
    """
    with open(rule_dict_filename) as rule_file:
        rule_dict = json.loads(rule_file.read())
    rule_list = {}
    for rule_order, rule_name in ruleset_order.items():
        rule = generate_rule(rule_name, rule_dict[rule_name]['def'])
        rule_list[rule_order] = rule
    ruleset = RuleSet(rule_list)
    selected_sqi = list(ruleset_order.values())
    for i in range(len(sqis)):
        sqi_df = sqis[i]
        decision_list = []
        for idx in range(len(sqi_df)):
            row_data = pd.DataFrame(dict(sqi_df[selected_sqi].iloc[idx]),
                                    index=[0])
            decision_list.append(ruleset.execute(row_data))
        sqi_df['decision'] = decision_list
        sqis[i] = sqi_df

    return rule_list, sqis


def get_reject_segments(segments, wave_type, milestones=None, info=None, dict=None):
    """
    Handy function to eliminate any invalid signals by other criteria besides SQI

    Parameters
    ----------
    segments : list
        The list of signal (in dataframe)

    wave_type : str
        Either 'ppg' or 'ecg'

    milestones : list
        The start indices and end indices of each segment.
         (Default value = None)
    info : list
        The list of criteria column to reject invalid signal
         (Default value = None)

    dict : dict
        The dictionary map the info column with rejected threshold

    Returns
    -------
    out : Series
        A Series with accept/reject decision
    """
    if wave_type == 'ppg':
        out = pd.Series(['accept']*len(segments))
    if wave_type == 'ecg':
        out = pd.Series(['accept']*len(segments))
    return out


def map_decision(i):
    """
    Handy function to transform decision value.

    Parameters
    ----------
    i :


    Returns
    -------

    """
    if i == 'accept':
        return 0
    if i == 'reject':
        return 1


def get_decision_segments(segments, decision, reject_decision):
    """
    Return the lists of accepted and rejected segment after evaluating the classification
    Parameters
    ----------
    segments : list
        The list of all segments

    decision : array like
        An array with rejected and accepted result of the relevant segment.

    reject_decision : list
        A list indicates the rejection by other criteria besides sqi

    Returns
    -------
    a_segments : list
        List of accepted segments
    r_segments : list
        List of rejected segments
    """
    decision = list(map(map_decision,decision))
    reject_decision = list(map(map_decision, reject_decision))
    decision = [a + b for a, b in zip(decision,reject_decision)]
    a_segments= []
    r_segments = []
    for i in decision:
        if decision[i] == 0:
            a_segments.append(segments[i])
        if decision[i] == 1:
            r_segments.append(segments[i])
    return a_segments, r_segments


def per_beat_sqi(sqi_func, troughs, signal, use_mean_beat,
            mean_resample_size, taper=False, **kwargs):
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

    use_mean_beat : bool
        If is True, each beat will be resampled and get the mean.
        The mean beat is used as the representative of a single_beat for sqi computation
        (Default value = True)

    mean_resample_size : int
        The resample size if use mean_beat computation
        (Default value = 100)

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
    beat_list = []
    # sqi_scores = sqi_func(s, **kwargs)
    if len(troughs) > 2:
        sqi_vals = []
        for idx, beat_start in enumerate(troughs[:-1]):
            single_beat = signal[beat_start:troughs[idx+1]]
            if taper:
                single_beat = sqi_pre.taper_signal(single_beat)
            if use_mean_beat:
                beat_list.append(resample(single_beat, mean_resample_size))
            else:
                sqi_vals.append(sqi_func(single_beat, **kwargs))
            # if len(kwargs) != 0:
            #     args = tuple(kwargs.values())
            #     sqi_vals.append(sqi_func(single_beat, *args))
            # else:
            #     sqi_vals.append(sqi_func(single_beat))
        if use_mean_beat:
            beat_list = np.array(beat_list)
            beat = np.apply_along_axis(np.mean, axis=0, arr=beat_list)
            sqi_vals.append(sqi_func(beat, **kwargs))
        return sqi_vals

    else:
        return [-np.inf]
        raise Exception("Not enough peaks in the signal to generate per beat SQI")


def get_sqi_dict(sqis, sqi_name):
    """
    Handy function to map the sqi name with its output.

    When the SQI computes with per-beat option,
    the suffices of 3-criteria (_mean, _std, _median) is appended.

    When the SQI computes with per-segment option,
    a suffices _sqi is appended.

    Parameters
    ----------
    sqis :  dict, numeric, list
        The sqi scores

    sqi_name : str
        The name of the current sqi

    Returns
    -------
    SQI_dict: dict
        A dictionary with the keys indicate the sqi names and the value is the relevant scores
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

    if isinstance(sqis, numpy.ndarray):
        if len(sqis.shape) == 0:
            return {sqi_name: -1}
        return {sqi_name: sqis[0]}

    if isinstance(sqis, list):
        SQI_dict = {}
        variations_stats = ['_mean', '_median', '_std']
        if len(sqis)>1:
            SQI_dict[sqi_name + variations_stats[0]+"_sqi"] = np.mean(sqis)
            SQI_dict[sqi_name + variations_stats[1]+"_sqi"] = np.median(sqis)
            SQI_dict[sqi_name + variations_stats[2]+"_sqi"] = np.std(sqis)
        else:
            SQI_dict[sqi_name] = sqis[0]
        return SQI_dict

    if isinstance(sqis, tuple):
        SQI_dict = {}
        for features_dict in sqis:
            features_dict_ = dict((key+"_sqi", value) for (key, value) in features_dict.items())
            SQI_dict = {**SQI_dict, **features_dict_}
        return SQI_dict


def get_sqi(sqi_func, sqi_name, s, per_beat=False,
            use_mean_beat=True,
            mean_resample_size = 100,
            wave_type='ppg', peak_detector=7,
            **kwargs):
    """
    Generic function to invoke the computation of SQI

    Parameters
    ----------
    sqi_func : function
        The sqi function in vital_sqi/sqi
        (or user can define their function)

    s : array like
        The signal values of the examining segment

    use_mean_beat : bool
        If is True, each beat will be resampled and get the mean.
        The mean beat is used as the representative of a single_beat for sqi computation
        (Default value = True)

    mean_resample_size : int
        The resample size if use mean_beat computation
        (Default value = 100)

    per_beat :
        Compute the segment with per_beat option
         (Default value = False)

    wave_type : str
        Either 'ppg' or 'ecg'
         (Default value = 'ppg')

    peak_detector : int
        The peak detector mode (from 1 - 7)
         (Default value = 7)
    **kwargs :


    Returns
    -------
    sqi_score_dict : dict
        The dictionary with the keys indicate the sqi name and the values indicates the scores

    """
    signal_arg = inspect.getfullargspec(sqi_func)[0][0]
    if signal_arg == 'nn_intervals':
        s = get_nn(s.iloc[:, 1])
    else:
        s = s.iloc[:, 1]
    if per_beat:
        # Prepare primary peak detector and perform peak detection
        detector = PeakDetector()
        if wave_type =='ppg':
            peak_list, trough_list = detector.ppg_detector(s,
                                                    peak_detector)
        else:
            peak_list, trough_list = detector.ecg_detector(s,
                                                    peak_detector)
        sqi_scores = per_beat_sqi(sqi_func, trough_list, s, use_mean_beat,
            mean_resample_size,**kwargs)
    else:
        if 'wave_type' in inspect.getfullargspec(sqi_func)[0]:
            kwargs['wave_type'] = wave_type
        sqi_scores = sqi_func(s, **kwargs)
    # sqi_name = sqi_func.__name__
    sqi_score_dict = get_sqi_dict(sqi_scores, sqi_name)
    return sqi_score_dict


def extract_segment_sqi(s, sqi_list, sqi_names, sqi_arg_list, wave_type):
    """

    Parameters
    ----------
    s :
        param sqi_list: list of sqi as in MASTERDICT
    nn_sqi_list :
        list of sqi using nn_intervals as in 'HRV' MASTER_DICT
    nn_sqi_arg_list :
        param sqi_arg_list:
    sqi_list :

    sqi_arg_list :

    wave_type :


    Returns
    -------

    """
    sqi_score = {}
    sqi_type = list(sqi_arg_list.keys())
    # for (sqi_,args_) in zip(sqi_list,sqi_arg_list):
    for idx in range(len(sqi_list)):
        try:
            args_ = sqi_arg_list[sqi_type[idx]]
            sqi_name = sqi_names[idx]
            sqi_ = sqi_list[idx]
            args_["wave_type"] = wave_type
            if sqi_.__name__ == "perfusion_sqi":
                args_ = {'y': np.array(s.iloc[:,1])}
            sqi_score = {**sqi_score, **get_sqi(sqi_, sqi_name, s, **args_)}
        except Exception as err:
            print('Error')
            print(sqi_)
            print(err)
            continue
    return pd.Series(sqi_score)


def extract_sqi(segments, milestones, sqi_dict_filename, wave_type='ppg'):
    """
    Extract SQIs requested in SQI dictionary for a list of  segments

    Parameters
    ----------
    segments : list
        List of segments to compute SQIs

    milestones : pandas dataframe
        Dataframe with 2 columns for start and end milestones of segments.

    sqi_dict_filename :
        Path to SQI dictionary json file which contains the requested SQIs and
        their corresponding parameters.

    wave_type :
         (Default value = 'ppg')

    Returns
    -------

    """
    sqi_mapping_list = {
        # Standard SQI
        'perfusion_sqi': sq.perfusion_sqi,
        'kurtosis_sqi': sq.kurtosis_sqi,
        'skewness_sqi': sq.skewness_sqi,
        'entropy_sqi': sq.entropy_sqi,
        'signal_to_noise_sqi': sq.signal_to_noise_sqi,
        'zero_crossings_rate_sqi': sq.zero_crossings_rate_sqi,
        'mean_crossing_rate_sqi': sq.mean_crossing_rate_sqi,  # should be merged with zero crossing
        # Peaks SQI
        'ectopic_sqi': sq.ectopic_sqi,
        'correlogram_sqi': sq.correlogram_sqi,
        'interpolation_sqi': sq.interpolation_sqi,
        'msq_sqi': sq.msq_sqi,
        # 'poincare_feature_sqi': sq.poincare_feature_sqi,
        # Waveform SQI
        'band_energy_sqi': sq.band_energy_sqi,
        'lfe_sqi': sq.lf_energy_sqi,
        'qrse_sqi': sq.qrs_energy_sqi,
        'hfe_sqi': sq.hf_energy_sqi,
        'vhfp_sqi': sq.vhf_norm_power_sqi,
        'qrsa_sqi': sq.qrs_a_sqi,
        # DTW SQI
        'dtw_sqi': sq.dtw_sqi,
        # ======================
        'nn_mean_sqi': sq.nn_mean_sqi,
        'sdnn_sqi': sq.sdnn_sqi,
        'sdsd_sqi': sq.sdsd_sqi,
        'rmssd_sqi': sq.rmssd_sqi,
        'cvsd_sqi': sq.cvsd_sqi,
        'cvnn_sqi': sq.cvnn_sqi,
        'mean_nn_sqi': sq.mean_nn_sqi,
        'median_nn_sqi': sq.median_nn_sqi,
        'pnn_sqi': sq.pnn_sqi,
        'hr_mean_sqi': sq.hr_mean_sqi,
        'hr_median_sqi': sq.hr_median_sqi,
        'hr_min_sqi': sq.hr_min_sqi,
        'hr_max_sqi': sq.hr_max_sqi,
        'hr_range_sqi': sq.hr_range_sqi,
        'peak_frequency_sqi': sq.peak_frequency_sqi,
        'absolute_power_sqi': sq.absolute_power_sqi,
        'log_power_sqi': sq.log_power_sqi,
        'relative_power_sqi': sq.relative_power_sqi,
        'normalized_power_sqi': sq.normalized_power_sqi,
        'lf_hf_ratio_sqi': sq.lf_hf_ratio_sqi,
        'poincare_sqi': sq.poincare_features_sqi,
        'hrv_sqi': sq.get_all_features_hrva
    }
    assert sqi_dict_filename is not None,  'Expected an sqi_dict json file. ' \
                                            'Template could be found in ' \
                                            'vita_sqi/resource.'
    assert segments is not None, 'Expected a list of segments.'
    assert milestones is not None, 'Expected milestones of split segments.'

    arg_path = sqi_dict_filename
    with open(arg_path, 'r') as arg_file:
        sqi_dict = json.loads(arg_file.read())
    sqi_list = []
    sqi_names = []
    sqi_arg_list = {}
    for item_key, item_value in sqi_dict.items():
        # sqi_name, sqi_arg
        sqi_names.append(item_key)
        sqi_list.append(sqi_mapping_list[item_value['sqi']])
        sqi_arg_list[item_key] = item_value['args']
    df_sqi = pd.DataFrame()
    for segment_idx in tqdm(range(len(segments))):
        segment = segments[segment_idx]
        # if
        sqi_arg_list['perfusion_sqi'] = {'y':segment.iloc[:, 1]}
        sqis = extract_segment_sqi(segment, sqi_list,sqi_names, sqi_arg_list,wave_type)
        # segment_name_list = file_name.split("/")[-1] + "_" +str(segment_idx)
        # sqis['id'] = segment_name_list
        df_sqi = df_sqi.append(sqis, ignore_index=True)
    df_sqi['start_idx'] = milestones.iloc[:, 0]
    df_sqi['end_idx'] = milestones.iloc[:, 1]
    return df_sqi


def generate_rule(rule_name, rule_def):
    """

    Parameters
    ----------
    rule_name :

    rule_def :


    Returns
    -------

    """
    rule_def, boundaries, label_list = update_rule(rule_def, is_update=False)
    rule_detail = {'def': rule_def,
                     'boundaries': boundaries,
                     'labels': label_list}
    rule = Rule(rule_name, rule_detail)
    return rule
