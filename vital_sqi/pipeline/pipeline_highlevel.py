import sys

import vital_sqi
# from sklearn.metrics import auc,brier_score_loss,f1_score,roc_auc_score,fbeta_score,jaccard_score,hamming_loss
# import numpy as np
# import vital_sqi.sqi as sq
from vital_sqi.data.signal_io import PPG_reader, ECG_reader
from vital_sqi.preprocess.segment_split import split_segment, save_segment
from vital_sqi.pipeline.pipeline_functions import *
# import json
import os
import pandas as pd

# D:\Workspace\oucru\classification_sqi

# PPG_ALL_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/ppg_all"
# G_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/good"
# NG_1_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/NG_1"
# NG_2_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/NG_2"
# NG_OUTPUT_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/NG_output"
# G_OUTPUT_FOLDER = "D:/Workspace/oucru/classification_sqi/dataset/G_output"
#
# good_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(G_FOLDER), (None, None, []))[2]]
# ng_1_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(NG_1_FOLDER), (None, None, []))[2]]
# ng_2_files = [".".join(x.split(".")[:-1])+".csv" for x in next(os.walk(NG_2_FOLDER), (None, None, []))[2]]


def get_ppg_sqis(file_name, signal_idx, timestamp_idx, sqi_dict_filename,
                 info_idx=[],
                 timestamp_unit='ms', sampling_rate=None, start_datetime=None,
                 split_type=0, duration=30, overlapping=None, peak_detector=7):
    """This function takes input signal, computes a number of SQIs, and outputs
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

    Parameters
    ----------
    file_name :
        
    sqi_dict :
        
    signal_idx :
        
    timestamp_idx :
        
    info_idx :
         (Default value = [])
    timestamp_unit :
         (Default value = 'ms')
    sampling_rate :
         (Default value = None)
    start_datetime :
         (Default value = None)
    split_type :
         (Default value = 0)
    duration :
         (Default value = 30)
    overlapping :
         (Default value = None)
    peak_detector :
         (Default value = 7)

    Returns
    -------

    
    """
    signal_obj = PPG_reader(file_name=file_name,
                     signal_idx=signal_idx,
                     timestamp_idx=timestamp_idx,
                     info_idx=info_idx,
                     timestamp_unit=timestamp_unit,
                     sampling_rate=sampling_rate,
                     start_datetime=start_datetime,)

    segments, milestones = split_segment(signal_obj.signals.iloc[:, 0:2],
                                    sampling_rate=signal_obj.sampling_rate,
                                    split_type=split_type, duration=duration,
                                    overlapping=overlapping,
                                    peak_detector=peak_detector,
                                    wave_type='ppg')
    signal_obj.signals = pd.DataFrame()
    sqi_lst = [
        extract_sqi(segments, milestones, sqi_dict_filename, wave_type='ppg')]
    signal_obj.sqis = sqi_lst
    return segments, signal_obj


def get_qualified_ppg(file_name, sqi_dict_filename, signal_idx, timestamp_idx,
                      rule_dict_filename, ruleset_order,
                      predefined_reject=False, info_idx=[],
                      timestamp_unit='ms', sampling_rate=None,
                      start_datetime=None, split_type=0, duration=30,
                      overlapping=None, peak_detector=7, segment_name=None,
                      save_image=False, output_dir=None):
    """Step 1: Read data to make SignalSQI object. (filename, other ppg
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

    Parameters
    ----------
    file_name :
        
    sqi_dict_filename :
        
    signal_idx :
        
    timestamp_idx :
        
    rule_dict_filename :
        
    ruleset_order :
        
    predefined_reject :
         (Default value = False)
    info_idx :
         (Default value = [])
    timestamp_unit :
         (Default value = 'ms')
    sampling_rate :
         (Default value = None)
    start_datetime :
         (Default value = None)
    split_type :
         (Default value = 0)
    duration :
         (Default value = 30)
    overlapping :
         (Default value = None)
    peak_detector :
         (Default value = 7)
    segment_name :
         (Default value = None)
    save_image :
         (Default value = False)
    output_dir :
         (Default value = None)

    Returns
    -------

    """
    if output_dir is None:
        output_dir = os.getcwd()
    assert(os.path.exists(output_dir)) is True
    segments, signal_obj = get_ppg_sqis(file_name, signal_idx,
                                        timestamp_idx, sqi_dict_filename,
                                        info_idx,
                                        timestamp_unit, sampling_rate,
                                        start_datetime, split_type, duration,
                                        overlapping, peak_detector)
    signal_obj.ruleset, signal_obj.sqis = classify_segments(signal_obj.sqis,
                                                            rule_dict_filename,
                                                            ruleset_order)
    if predefined_reject is True:
        milestones = signal_obj.sqis[0].iloc['start', 'end']
        reject_decision = get_reject_segments(segments, wave_type='ppg',
                                              milestones=milestones,
                                              info=signal_obj.info)
    else:
        reject_decision = ['accept']*len(signal_obj.sqis[0])
    a_segments, r_segments = get_decision_segments(segments,
                                        signal_obj.sqis[0]['decision'],
                                        reject_decision)
    if save_image:
        os.makedirs(os.path.join(output_dir, 'accept', 'img'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reject', 'img'), exist_ok=True)
    else:
        os.makedirs(os.path.join(output_dir, 'accept'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reject'), exist_ok=True)
    save_segment(a_segments, segment_name=segment_name,
                 save_file_folder=os.path.join(output_dir, 'accept'),
                 save_image=save_image,
                 save_img_folder=os.path.join(output_dir, 'accept', 'img'))
    save_segment(r_segments, segment_name=segment_name,
                 save_file_folder=os.path.join(output_dir, 'reject'),
                 save_image=save_image,
                 save_img_folder=os.path.join(output_dir, 'reject', 'img'))
    return signal_obj


def get_ecg_sqis(file_name, sqi_dict_filename, file_type, channel_num=None,
                 channel_name=None, sampling_rate=None, start_datetime=None,
                 split_type=0, duration=30, overlapping=None, peak_detector=7):
    """multiple channels ecgs: signals  = df multiple columns
    sqis[channel 1 - df, chanel 2 - df]
    segments [channel 1 - series, channel 2 - series]
    
    return signalSQI obj

    Parameters
    ----------
    file_name :
        
    sqi_dict_filename :
        
    file_type :
        
    channel_num :
         (Default value = None)
    channel_name :
         (Default value = None)
    sampling_rate :
         (Default value = None)
    start_datetime :
         (Default value = None)
    split_type :
         (Default value = 0)
    duration :
         (Default value = 30)
    overlapping :
         (Default value = None)
    peak_detector :
         (Default value = 7)

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
    for i in range(1, len(signal_obj.signals.columns)):
        signals = signal_obj.signals.iloc[:, [0, i]]
        segments, milestones = split_segment(signals, split_type=split_type,
                                             sampling_rate=
                                             signal_obj.sampling_rate,
                                             duration=duration,
                                             overlapping=overlapping,
                                             peak_detector=peak_detector,
                                             wave_type='ecg')
        segments_lst.append(segments)
        milestones_lst.append(milestones)
    signal_obj.signals = pd.DataFrame()
    signal_obj.sqis = []
    for i in range(0, len(segments_lst)):
        signal_obj.sqis.append(extract_sqi(segments, milestones, sqi_dict_filename))
    return segments_lst, signal_obj


def get_qualified_ecg(file_name, file_type, sqi_dict_filename,
                      rule_dict_filename, ruleset_order,
                      channel_num=None, channel_name=None,
                      predefined_reject=False,
                      sampling_rate=None, start_datetime=None, split_type=0,
                      duration=30, overlapping=None, peak_detector=7,
                      segment_name=None, save_image=False, output_dir=None):
    """Extract intended SQI

    Build ruleset
    Classify
    Cut from original
    Write to original file with bad segments removed.

    Parameters
    ----------
    file_name :
        
    file_type :
        
    sqi_dict_filename :
        
    ruleset_order :
        
    rule_dict_filename :
        
    channel_num :
         (Default value = None)
    channel_name :
         (Default value = None)
    predefined_reject :
         (Default value = False)
    sampling_rate :
         (Default value = None)
    start_datetime :
         (Default value = None)
    split_type :
         (Default value = 0)
    duration :
         (Default value = 30)
    overlapping :
         (Default value = None)
    peak_detector :
         (Default value = 7)
    segment_name :
         (Default value = None)
    save_image :
         (Default value = False)
    output_dir :
         (Default value = None)

    Returns
    -------

    
    """
    if output_dir is None:
        output_dir = os.getcwd()
    assert(os.path.exists(output_dir)) is True
    segment_lst, signal_obj = get_ecg_sqis(file_name, sqi_dict_filename, file_type,
                                           channel_num, channel_name,
                                           sampling_rate, start_datetime,
                                           split_type, duration, overlapping,
                                           peak_detector)
    sqi_lst = []
    for i in range(0, len(segment_lst)):
        signal_obj.ruleset, sqis = classify_segments(signal_obj.sqis,
                                                     rule_dict_filename,
                                                     ruleset_order)
        if predefined_reject is True:
            reject_decision = get_reject_segments(segment_lst[i],
                                                  wave_type='ecg')
        else:
            reject_decision = []
        a_segments, r_segments = get_decision_segments(segment_lst[i],
                                                sqis[i].loc[:, 'decision'],
                                                reject_decision)
        if save_image:
            os.makedirs(os.path.join(output_dir, str(i), 'accept', 'img'),
                        exist_ok=True)
            os.makedirs(os.path.join(output_dir, str(i), 'reject', 'img'),
                        exist_ok=True)
        else:
            os.makedirs(os.path.join(output_dir, str(i), 'accept'),
                        exist_ok=True)
            os.makedirs(os.path.join(output_dir, str(i), 'reject'),
                        exist_ok=True)
        save_segment(a_segments, segment_name=segment_name,
                     save_file_folder=os.path.join(output_dir, str(i),
                                                   'accept'),
                     save_image=save_image,
                     save_img_folder=os.path.join(output_dir, str(i),
                                                  'accept', 'img'))
        save_segment(r_segments, segment_name=segment_name,
                     save_file_folder=os.path.join(output_dir, str(i),
                                                   'reject'),
                     save_image=save_image,
                     save_img_folder=os.path.join(output_dir, str(i),
                                                  'reject', 'img'))
    signal_obj.sqis = sqi_lst
    return signal_obj


def signal_preprocess():
    """ """
    return


def write_ecg():
    """ """
    return