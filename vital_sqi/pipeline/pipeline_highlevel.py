from vital_sqi.data import *
from vital_sqi.preprocess.segment_split import split_segment
from vital_sqi.pipeline.pipeline_functions import *


def get_ppg_sqis(file_name, signal_idx, timestamp_idx,
                 timestamp_unit, sampling_rate, start_datetime,
                 split_type, duration, overlapping, peak_detector,
                 sqi_dict):
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
                     timestamp_unit=timestamp_unit,
                     sampling_rate=sampling_rate,
                     start_datetime=start_datetime)
    segments, milestones = split_segment(signal_obj.signals.iloc[:, 0:1],
                                    sampling_rate=signal_obj.sampling_rate,
                                    split_type=split_type, duration=duration,
                                    overlapping=overlapping,
                                    peak_detector=peak_detector,
                                    wave_type='ppg')
    signal_obj.signals = None
    signal_obj.sqis = extract_sqi(segments, milestones, sqi_dict)
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