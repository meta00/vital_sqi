import vital_sqi
from sklearn.metrics import auc,brier_score_loss,f1_score,roc_auc_score,fbeta_score,jaccard_score,hamming_loss
import numpy as np
import vital_sqi.sqi as sq
from vital_sqi.data import *
from vital_sqi.preprocess.segment_split import split_segment
from vital_sqi.pipeline.pipeline_functions import *
from vital_sqi import RuleSet,Rule
import warnings
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from vital_sqi.common import update_rule
from tqdm import tqdm
from vital_sqi.data.signal_io import PPG_reader
import numpy as np

from vital_sqi import utils
from datetime import datetime
from vital_sqi.data.signal_io import PPG_reader

sqi_list = {
    # Standard SQI
    'perf': sq.perfusion_sqi,
    'kurt':     sq.kurtosis_sqi,
    'skew':     sq.skewness_sqi,
    'ent':      sq.entropy_sqi,
    'kurt_seg':     sq.kurtosis_sqi,
    'skew_seg':     sq.skewness_sqi,
    'ent_seg':      sq.entropy_sqi,
    'snr': sq.signal_to_noise_sqi,
    'zc': sq.zero_crossings_rate_sqi,
    'mc': sq.mean_crossing_rate_sqi,  # should be merged with zero crossing
    # Peaks SQI
    'ect': sq.ectopic_sqi,
    'corr': sq.correlogram_sqi,
    'intp': sq.interpolation_sqi,
    'msq': sq.msq_sqi,

    # 'poincare_feature_sqi': sq.poincare_feature_sqi,
    # Waveform SQI
    'be': sq.band_energy_sqi,
    'lfe': sq.lf_energy_sqi,
    'qrse': sq.qrs_energy_sqi,
    'hfe': sq.hf_energy_sqi,
    'vhfp': sq.vhf_norm_power_sqi,
    'qrsa': sq.qrs_a_sqi,
    # DTW SQI
    'dtw': sq.dtw_sqi
}
nn_sqi_list = {
    # HRV SQI
    # TODO
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
    'hrva': sq.get_all_features_hrva
}
sqi_arg_list = {
    # Standard SQI
    'perf': {},
    'kurt':     {'per_beat':True},
    'skew':     {'per_beat':True},
    'ent':      {'per_beat':True},
    'kurt_seg':     {'per_beat':False},
    'skew_seg':     {'per_beat':False},
    'ent_seg':      {'per_beat':False},
    'snr': {},
    'zc': {},
    'mc': {},  # should be merged with zero crossing
    # Peaks SQI
    'ect': {},
    'corr': {},
    'intp': {},
    'msq': {},

    # 'poincare_feature_sqi': sq.poincare_feature_sqi,
    # Waveform SQI
    'be': {},
    'lfe': {'sampling_rate': 100},
    'qrse': {'sampling_rate': 100},
    'hfe': {'sampling_rate': 100},
    'vhfp': {'sampling_rate': 100},
    'qrsa': {'sampling_rate': 100},
    # DTW SQI
    'eucl': {'template_type': 0, 'per_beat': True},
    'dtw': {'per_beat': True}
}
nn_sqi_arg_list = {
    'nn_mean_sqi': {},
    'sdnn_sqi': {},
    'sdsd_sqi': {},
    'rmssd_sqi': {},
    'cvsd_sqi': {},
    'cvnn_sqi': {},
    'mean_nn_sqi': {},
    'median_nn_sqi': {},
    'pnn_sqi': {},
    'hr_mean_sqi': {},
    'hr_median_sqi': {},
    'hr_min_sqi': {},
    'hr_max_sqi': {},
    'hr_range_sqi': {},
    'peak_frequency_sqi': {},
    'absolute_power_sqi': {},
    'log_power_sqi': {},
    'relative_power_sqi': {},
    'normalized_power_sqi': {},
    'lf_hf_ratio_sqi': {},
    'poincare_sqi':{},
    'hrva': {}
}

from vital_sqi.pipeline.pipeline_functions import *

# D:\Workspace\oucru\classification_sqi
PPG_ALL_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/ppg_all"
G_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/good"
NG_1_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/NG_1"
NG_2_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/NG_2"
NG_OUTPUT_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/NG_output"
G_OUTPUT_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/G_output"

good_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(G_FOLDER), (None, None, []))[2]]
ng_1_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(NG_1_FOLDER), (None, None, []))[2]]
ng_2_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(NG_2_FOLDER), (None, None, []))[2]]

def pipeline(file_name):
    try:
        ppg_stable = PPG_reader(os.path.join(os.getcwd(), file_name))
        # ppg_stable = pd.read_csv(os.path.join(os.getcwd(), file_name), header=None)
        # ppg_stable = pd.DataFrame(ppg_stable)
        # ppg_stable.index = pd.to_timedelta(ppg_stable.index / 100, unit='s')
        # ppg_stable["idx"] = pd.to_timedelta(ppg_stable.index / 100, unit='s')
        # ppg_stable = ppg_stable[["idx", 0]]
        #
        # sqi_arg_list['perf'] = {'y':ppg_stable.iloc[:,1]}
        # # sqi_arg_list['zc'] = {'y': ppg_stable.iloc[:, 1]}
        # # sqi_arg_list['mc'] = {'y': ppg_stable.iloc[:, 1]}
        # sqis = segment_PPG_SQI_extraction(ppg_stable, sqi_list.values(), nn_sqi_list.values(),
        #                                   nn_sqi_arg_list.values(), sqi_arg_list.values())
        #
        # # segment_name_list = [file_name.split("/")[-1] + "_" + str(i) for i in range(len(sqis))]
        # segment_name_list = file_name.split("/")[-1]
        # sqis['id'] = segment_name_list
        sqis = None
    except Exception as e:
        print(e)
        print(file_name)
        return pd.DataFrame([e])
    return sqis

# all_ng_file_sqi_1 = pd.DataFrame()
# for file_name in tqdm(ng_1_files):
#     file_path = os.path.join(PPG_ALL_FOLDER,file_name)
#     sqis = pipeline(file_path)
#     if file_path is not None:
#         all_ng_file_sqi_1 = all_ng_file_sqi_1.append(sqis,ignore_index=True)
#         # sqis.to_csv(os.path.join(NG_OUTPUT_FOLDER,"sqi_"+file_name))
#
# col = all_ng_file_sqi_1.pop("id")
# sqis = all_ng_file_sqi_1.insert(0, col.name, col)
# all_ng_file_sqi_1.to_csv(os.path.join(NG_OUTPUT_FOLDER,"NG_sqi_1.csv"))

# all_ng_file_sqi_2 = pd.DataFrame()
# for file_name in tqdm(ng_2_files):
#     file_path = os.path.join(PPG_ALL_FOLDER,file_name)
#     sqis = pipeline(file_path)
#     if file_path is not None:
#         all_ng_file_sqi_2 = all_ng_file_sqi_2.append(sqis,ignore_index=True)
#         # sqis.to_csv(os.path.join(NG_OUTPUT_FOLDER,"sqi_"+file_name))
#
# col = all_ng_file_sqi_2.pop("id")
# sqis = all_ng_file_sqi_2.insert(0, col.name, col)
# all_ng_file_sqi_2.to_csv(os.path.join(NG_OUTPUT_FOLDER,"NG_sqi_2.csv"))


#==========================================================================================

sqi_list = {
    # Standard SQI
    'perf': sq.perfusion_sqi,
    'kurt':     sq.kurtosis_sqi,
    'skew':     sq.skewness_sqi,
    'ent':      sq.entropy_sqi,
    'kurt_seg':     sq.kurtosis_sqi,
    'skew_seg':     sq.skewness_sqi,
    'ent_seg':      sq.entropy_sqi,
    'snr': sq.signal_to_noise_sqi,
    'zc': sq.zero_crossings_rate_sqi,
    'mc': sq.mean_crossing_rate_sqi,  # should be merged with zero crossing
    # Peaks SQI
    'ect': sq.ectopic_sqi,
    'corr': sq.correlogram_sqi,
    'intp': sq.interpolation_sqi,
    'msq': sq.msq_sqi,

    # 'poincare_feature_sqi': sq.poincare_feature_sqi,
    # Waveform SQI
    'be': sq.band_energy_sqi,
    'lfe': sq.lf_energy_sqi,
    'qrse': sq.qrs_energy_sqi,
    'hfe': sq.hf_energy_sqi,
    'vhfp': sq.vhf_norm_power_sqi,
    'qrsa': sq.qrs_a_sqi,
    # DTW SQI
    'dtw': sq.dtw_sqi,
    #======================
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
    'hrva': sq.get_all_features_hrva
}

nn_sqi_list = {
    # HRV SQI
    # TODO
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
    'hrva': sq.get_all_features_hrva
}


def get_ppg_sqis(file_name, signal_idx, timestamp_idx,info_idx,sqi_dict,
                 timestamp_unit='ms',sampling_rate=100, start_datetime=None,
                 split_type=0, duration=30, overlapping=None, peak_detector=7):
    """
    This function takes input signal, computes a number of SQIs, and outputs
    a table (row - signal segments, column SQI values.

    Step 1: Read data to make SignalSQI object. (filename, other ppg
    parameter for PPG_reader()
    - PPG_reader > signalSQI obj
    Step 2: Produce segments. (split_segment parameters).
    - split_segments (s) > signalSQI obj with attribute segments.
    Step 3: Extract sqis for each segments (sqi_dict, signalSQI obj)
    - sqi_extract > signalSQI obj with attribute sqis
        - read sqi_dict > function calls
        - run function calls
        - append results into df > attribute sqis
    Return signalSQI obj: signals, sqis, segments
    """
    signal_obj = PPG_reader(file_name=file_name,
                     signal_idx=signal_idx,
                     timestamp_idx=timestamp_idx,
                     info_idx=info_idx,
                     timestamp_unit=timestamp_unit,
                     sampling_rate=sampling_rate,
                     start_datetime=start_datetime,)
    segments, milestones = split_segment(pd.DataFrame(signal_obj.signals.iloc[:,0:2]),

                                    sampling_rate=signal_obj.sampling_rate,
                                    split_type=split_type, duration=duration,
                                    overlapping=overlapping,
                                    peak_detector=peak_detector,
                                    wave_type='ppg')
    signal_obj.signals = pd.DataFrame()
    signal_obj.sqis = extract_sqi(segments, milestones, sqi_dict,file_name,None)
    return segments, signal_obj


def get_ecg_sqis(file_name, file_type, channel_num, channel_name,
                 sampling_rate,  start_datetime, split_type, duration,
                 overlapping, peak_detector, sqi_dict):
    """
    multiple channels ecgs: signals  = df multiple columns
    sqis[channel 1 - df, chanel 2 - df]
    segments [channel 1 - series, channel 2 - series]

    return signalSQI obj
    Parameters
    ----------
    file

    Returns
    -------

    """
    signal_obj = ECG_reader(file_name=file_name, file_type=file_type,
                            channel_num=channel_num,
                            channel_name=channel_name,
                            sampling_rate=sampling_rate,
                            start_datetime=start_datetime)

    segments_lst = []
    milestones_lst = []
    for i in range(1, len(signal_obj.signals.columns)-1):
        signals = signal_obj.signals.iloc[:, [0, i]]
        segments, milestones = split_segment(signals, split_type=split_type,
                                             duration=duration,
                                             overlapping=overlapping,
                                             peak_detector=peak_detector,
                                             wave_type='ecg')
        segments_lst.append(segments)
        milestones_lst.append(milestones)
    signal_obj.signals = None
    signal_obj.sqis = []
    for i in range(0, len(segments_lst)):
        signal_obj.sqis.append(extract_sqi(segments, milestones, sqi_dict))
    return segments_lst, signal_obj


def get_qualified_ppg(file_name, signal_idx, timestamp_idx,
                 timestamp_unit, sampling_rate, start_datetime,
                 split_type, duration, overlapping, peak_detector,
                 sqi_dict, ruleset_order, rule_dict, segment_name,
                 save_image, output_dir):
    """
    Step 1: Read data to make SignalSQI object. (filename, other ppg
    parameter for PPG_reader()
    - PPG_reader > signalSQI obj
    Step 2: Produce segments. (split_segment parameters).
    - split_segments (s) > signalSQI obj with sqis columns start end segments.
    Step 3: Extract sqis for each segments (sqi_dict, signalSQI obj)
    - sqi_extract > signalSQI obj with attribute sqis
        - read sqi_dict > function calls
        - run function calls
        - append results into df > attribute sqis > signalSQI obj: signals,
        sqis, segments.

    - classify > signalSQI ojb with attribute sqis with decision column
        - make_ruleset: update signalSQI with attribute rules, ruleset.
        - execute (sqi) >> update sqis decision column >>> signalSQI.
    - cut (signalSQI obj):
        - cut_segment
        - save(output_dir)
    >> return signalSQI obj
    """
    assert(os.path.exists(output_dir)) is True
    segments, signal_obj = get_ppg_sqis(file_name, signal_idx, timestamp_idx,
                 timestamp_unit, sampling_rate, start_datetime,
                 split_type, duration, overlapping, peak_detector,
                 sqi_dict)
    signal_obj.ruleset, signal_obj.sqis = classify_segments(signal_obj.sqis,
                                                     rule_dict, ruleset_order)
    a_segments, r_segments = get_decision_segments(segments,
                                     signal_obj.sqis.iloc[:, 'decision'])
    os.makedirs(os.path.join(output_dir, 'accept'))
    save_segment(a_segments, file_name=segment_name,
                 save_file_folder=os.path.join(output_dir, 'accept'),
                 save_image=save_image)
    os.makedirs(os.path.join(output_dir, 'reject'))
    save_segment(a_segments, file_name=segment_name,
                 save_file_folder=os.path.join(output_dir, 'reject'),
                 save_image=save_image)
    return signal_obj


def get_qualified_ecg(file_name, file_type, channel_num, channel_name,
                      sampling_rate, start_datetime, split_type, duration,
                      overlapping, peak_detector, sqi_dict, ruleset_order,
                      rule_dict, segment_name, save_image, output_dir):
    """
    Extract intended SQI
    Build ruleset
    Classify
    Cut from original
    Write to original file with bad segments removed.
    """
    assert(os.path.exists(output_dir)) is True
    segment_lst, signal_obj = get_ecg_sqis(file_name, file_type, channel_num,
                                           channel_name, sampling_rate,
                                           start_datetime, split_type,
                                           duration, overlapping,
                                           peak_detector, sqi_dict)
    sqi_lst = []
    for i in segment_lst:
        signal_obj.ruleset, sqis = classify_segments(signal_obj.sqis,
                                                     rule_dict, ruleset_order)

        a_segments, r_segments = get_decision_segments(segment_lst[i],
                                                       sqis.iloc[:, 'decision'])
        sqi_lst.append(sqis)
        os.makedirs(os.path.join(output_dir, i, 'accept'))
        os.makedirs(os.path.join(output_dir, i, 'reject'))
        save_segment(a_segments, file_name=segment_name,
                     save_file_folder=os.path.join(output_dir, i, 'accept'),
                     save_image=save_image)
        save_segment(a_segments, file_name=segment_name,
                     save_file_folder=os.path.join(output_dir, i, 'reject'),
                     save_image=save_image)

    return signal_obj


def signal_preprocess():
    return
def write_ecg():
	return


file_name = "../../tests/test_data/ppg_smartcare.csv"

# PPG_reader('/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests/test_data/ppg_smartcare.csv',
#                 timestamp_idx = ['TIMESTAMP_MS'], signal_idx = ['PLETH'], info_idx = ['PULSE_BPM',
#                                                         'SPO2_PCT','PERFUSION_INDEX'],

segments, signal_obj = get_ppg_sqis(file_name,timestamp_idx = ['TIMESTAMP_MS'],
                                    signal_idx = ['PLETH'],
                                    info_idx = ['PULSE_BPM','SPO2_PCT','PERFUSION_INDEX'],sqi_dict=sqi_list)


