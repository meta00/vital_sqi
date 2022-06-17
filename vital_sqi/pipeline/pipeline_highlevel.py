from vital_sqi.data import *
from vital_sqi.preprocess.segment_split import split_segment


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
    segments, milestones = split_segment(signal_obj.signals[:, 0:1],
                                    sampling_rate=signal_obj.sampling_rate,
                                    split_type=split_type, duration=duration,
                                    overlapping=overlapping,
                                    peak_detector=peak_detector,
                                    wave_type='ppg')
    signal_obj.signals = None
    signal_obj.sqis = extract_sqi(segments, milestones, sqi_dict)
    return segments, signal_obj


def get_ecg_sqis(file_name, channel,):
    # option for channels
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
    return

def get_qualified_ppg(output_dir):
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
    return

def get_qualified_ecg():
    """
    Extract intended SQI
    Build ruleset
    Classify
    Cut from original
    Write to original file with bad segments removed.
    """
    return

def signal_preprocess():
    return
def write_ecg():
	return