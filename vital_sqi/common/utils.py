import numpy as np
import os
import datetime as dt
import json
import pandas as pd
from pandas.core.dtypes.common import is_number
from vital_sqi.common.rpeak_detection import PeakDetector
from hrvanalysis import get_nn_intervals
import dateparser

OPERAND_MAPPING_DICT = {
    ">": 5,
    ">=": 4,
    "=": 3,
    "<=": 2,
    "<": 1
}


def get_nn(s,wave_type='ppg',sample_rate=100,rpeak_method=7,
           remove_ectopic_beat=False):

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


def check_valid_signal(x):
    """Check whether signal is valid, i.e. an array_like numeric, or raise errors.

    Parameters
    ----------
    x :
        array_like, array of signal

    Returns
    -------


    """
    if isinstance(x, dict) or isinstance(x, tuple):
        raise ValueError("Expected array_like input, instead found {"
                         "0}:".format(type(x)))
    if len(x) == 0:
        raise ValueError("Empty signal")
    types = []
    x = list(x)
    for i in range(len(x)):
        types.append(str(type(x[i])))
    type_unique = np.unique(np.array(types))
    if len(type_unique) != 1 and (type_unique[0].find("int") != -1 or
                                  type_unique[0].find("float") != -1):
        raise ValueError("Invalid signal: Expect numeric array, instead found "
                         "array with types {0}: ".format(type_unique))
    if type_unique[0].find("int") == -1 and type_unique[0].find("float") == -1:
        raise ValueError("Invalid signal: Expect numeric array, instead found "
                         "array with types {0}: ".format(type_unique))
    return True


def calculate_sampling_rate(timestamps):
    """

    Parameters
    ----------
    x : array_like of timestamps, float (unit second)

    Returns
    -------
    float : sampling rate
    """
    if isinstance(timestamps[0], float):
        timestamps_second = timestamps
    elif isinstance(timestamps[0], pd.Timestamp):
        timestamps_second = []
        for i in range(0, len(timestamps)):
            timestamps_second.append((timestamps[i] -
                                      timestamps[0]).total_seconds())
    else:
        try:
            v_parse_datetime = np.vectorize(parse_datetime)
            timestamps = v_parse_datetime(timestamps)
            timestamps_second = []
            timestamps_second.append(0)
            for i in range(1, len(timestamps)):
                timestamps_second.append((timestamps[i] - timestamps[
                    i - 1]).total_seconds())
        except Exception:
            sampling_rate = None
            return sampling_rate
    steps = np.diff(timestamps_second)
    sampling_rate = round(1 / np.min(steps[steps != 0]), 3)
    return sampling_rate


def generate_timestamp(start_datetime, sampling_rate, signal_length):
    """

    Parameters
    ----------
    start_datetime :

    sampling_rate : float or int

    signal_length : int


    Returns
    -------
    list : list of timestamps with length equal to signal_length.
    """
    assert np.isreal(sampling_rate), 'Sampling rate is expected to be a int ' \
                                     'or float.'
    number_of_seconds = signal_length / sampling_rate
    if start_datetime is None:
        start_datetime = dt.datetime.now()
    timestamps = []
    try:
        timestamps.append(dt.datetime.timestamp(start_datetime))
    except Exception as e:
        print(e)
    # end_datetime = start_datetime + dt.timedelta(seconds=number_of_seconds)
    # time_range = DateTimeRange(start_datetime, end_datetime)
    # timestamps = []
    # for value in time_range.range(dt.timedelta(seconds=1 / sampling_rate)):
    for value in range(1, signal_length):
            timestamps.append(timestamps[value-1] + 1 /
                                           sampling_rate)
    for x in range(0, len(timestamps)):
        timestamps[x] = pd.Timestamp(timestamps[x], unit = 's')
    return timestamps


def parse_datetime(string, type='datetime'):
    """
    A simple dateparser that detects common  datetime formats

    Parameters
    ----------
    string : str
        a date string in format as denoted below.

    Returns
    -------
    datetime.datetime
        datetime object of a time.

    """
    # some common formats.
    date_formats = ['%Y-%m-%d',
                    '%d-%m-%Y',
                    '%d.%m.%Y',
                    '%Y.%m.%d',
                    '%d %b %Y',
                    '%Y/%m/%d',
                    '%d/%m/%Y']
    datime_formats = ['%Y-%m-%d %H:%M:%S.%f',
                      '%d-%m-%Y %H:%M:%S.%f',
                      '%d.%m.%Y %H:%M:%S.%f',
                      '%Y.%m.%d %H:%M:%S.%f',
                      '%d %b %Y %H:%M:%S.%f',
                      '%Y/%m/%d %H:%M:%S.%f',
                      '%d/%m/%Y %H:%M:%S.%f',
                      '%Y-%m-%d %I:%M:%S.%f',
                      '%d-%m-%Y %I:%M:%S.%f',
                      '%d.%m.%Y %I:%M:%S.%f',
                      '%Y.%m.%d %I:%M:%S.%f',
                      '%d %b %Y %I:%M:%S.%f',
                      '%Y/%m/%d %I:%M:%S.%f',
                      '%d/%m/%Y %I:%M:%S.%f']
    if type == 'date':
        formats = date_formats
    if type == 'datetime':
        formats = datime_formats
    for f in formats:
        try:
            return dt.datetime.strptime(string, f)
        except:
            pass
    try:
        return dateparser.parse(string)
    except:
        raise ValueError('Datetime string must be of standard Python format '
                         '(https://docs.python.org/3/library/time.html), '
                         'e.g., `%d-%m-%Y`, eg. `24-01-2020`')


def parse_rule(name, source):
    assert os.path.isfile(source) is True, 'Source file not found'
    with open(source) as json_file:
        all = json.load(json_file)
        try:
            sqi = all[name]
        except:
            raise Exception("SQI {0} not found".format(name))
        rule_def, boundaries, label_list = update_rule(sqi['def'],
                                                       is_update=False)
    return rule_def, \
           boundaries, \
           label_list


def update_rule(rule_def, threshold_list=[], is_update=True):
    if rule_def is None or is_update:
        all_rules = []
    else:
        all_rules = list(np.copy(rule_def))
    for threshold in threshold_list:
        all_rules.append(threshold)
    df = sort_rule(all_rules)
    df = decompose_operand(df.to_dict('records'))
    boundaries = np.sort(df["value"].unique())
    inteveral_label_list = get_inteveral_label_list(df, boundaries)
    value_label_list = get_value_label_list(df, boundaries, inteveral_label_list)

    label_list = []
    for i in range(len(value_label_list)):
        label_list.append(inteveral_label_list[i])
        label_list.append(value_label_list[i])
    label_list.append(inteveral_label_list[-1])
    return all_rules, boundaries, label_list


def sort_rule(rule_def):
    df = pd.DataFrame(rule_def)
    df["value"] = pd.to_numeric(df["value"])
    df['operand_order'] = df['op'].map(OPERAND_MAPPING_DICT)
    df.sort_values(by=['value', 'operand_order'],
                   inplace=True,
                   ascending=[True, True],
                   ignore_index=True)

    return df


def decompose_operand(rule_dict):
    df = pd.DataFrame(rule_dict)
    df["value"] = pd.to_numeric(df["value"])
    df['operand_order'] = df['op'].map(OPERAND_MAPPING_DICT)

    single_operand = df[(df["operand_order"] == 5)
                        | (df["operand_order"] == 3)
                        | (df["operand_order"] == 1)].to_dict('records')

    df_gte_operand = df[(df["operand_order"] == 4)]
    gte_g_operand = df_gte_operand.replace(">=", ">").to_dict('records')
    gte_e_operand = df_gte_operand.replace(">=", "=").to_dict('records')

    df_lte_operand = df[(df["operand_order"] == 2)]
    lte_l_operand = df_lte_operand.replace("<=", "<").to_dict('records')
    lte_e_operand = df_lte_operand.replace("<=", "=").to_dict('records')

    all_operand = single_operand + gte_g_operand + gte_e_operand + \
                  lte_l_operand + lte_e_operand

    df_all_operand = sort_rule(all_operand)

    return df_all_operand


def check_unique_pair(pair):
    assert len(pair) <= 1, "Duplicated decision at '" + str(pair["value"]) + " " + pair["op"] + "'"
    return True


def check_conflict(decision_lt, decision_gt):
    if len(decision_lt) == 0:
        label_lt = None
    else:
        label_lt = decision_lt["label"].values[0]
    if len(decision_gt) == 0:
        label_gt = None
    else:
        label_gt = decision_gt["label"].values[0]

    if label_lt == None:
        return label_gt
    if label_gt == None:
        return label_lt
    # Check conflict
    if not label_lt == label_gt:
        raise ValueError("Rules raise a conflict at x " + decision_lt.iloc[0]["op"] + " " +
                         str(decision_lt.iloc[0]["value"]) + " is " + decision_lt.iloc[0]["label"]
                         + ", but x " + decision_gt.iloc[0]["op"] + " " +
                         str(decision_gt.iloc[0]["value"]) + " is " + decision_gt.iloc[0]["label"])
    return label_gt


def get_decision(df, boundaries, idx):
    start_value = boundaries[idx]
    end_value = boundaries[idx + 1]
    decision_lt = \
        df[(df["value"] == end_value) &
           (df["op"] == "<")]
    check_unique_pair(decision_lt)
    decision_gt = \
        df[(df["value"] == start_value) &
           (df["op"] == ">")]
    check_unique_pair(decision_gt)

    decision = check_conflict(decision_lt, decision_gt)
    while (decision == None and idx <= (len(df) - 1)):
        decision = get_decision(df, boundaries, idx + 1)
    return decision


def get_inteveral_label_list(df, boundaries):
    inteveral_label_list = np.array([None] * (len(boundaries) + 1))

    assert df["op"].iloc[0] == "<", \
        "The rule is missing a decision from -inf to " + str(df["value"].iloc[0])
    inteveral_label_list[0] = df.iloc[0]["label"]
    for idx in range(len(boundaries) - 1):
        decision = get_decision(df, boundaries, idx)
        inteveral_label_list[idx + 1] = decision
    assert df["op"].iloc[-1] == ">", \
        "The rule is missing a decision from " + str(df["value"].iloc[-1]) + " to inf"
    inteveral_label_list[-1] = df.iloc[-1]["label"]
    return inteveral_label_list


def get_value_label_list(df, boundaries, interval_label_list):
    value_label_list = np.array([None] * (len(boundaries)))
    for idx in range(len(boundaries)):
        decision = df[(df["value"] == boundaries[idx]) &
                      (df["op"] == "=")]
        check_unique_pair(decision)
        if len(decision) == 0:
            value_label_list[idx] = interval_label_list[idx + 1]
        else:
            value_label_list[idx] = decision.iloc[0]["label"]
    return value_label_list


def cut_segment(df,milestone):
    """
    Spit Dataframe into segments, base on the pair of start and end indices.

    Parameters
    ----------
    df :
        Signal dataframe .
    milestone :
        Indices dataframe

    Returns
    -------
    The list of split segments.
    """
    assert isinstance(milestone, pd.DataFrame), \
        "Please convert the milestone as dataframe " \
        "with 'start' and 'end' columns. " \
        ">>> from vital_sqi.common.utils import format_milestone" \
        ">>> milestones = format_milestone(start_milestone,end_milestone)"
    start_milestone = np.array(milestone.iloc[:,0])
    end_milestone = np.array(milestone.iloc[:, 1])
    processed_df = []
    for start, end in zip(start_milestone, end_milestone):
        processed_df.append(df.iloc[int(start):int(end)])
    return processed_df


def format_milestone(start_milestone, end_milestone):
    """

    Parameters
    ----------
    start_milestone :
        array-like represent the start indices of segment.
    end_milestone :
        array-like represent the end indices of segment.

    Returns
    -------
    a dataframe of two columns.
    The first column indicates the start indices, the second indicates the end indices
    """
    assert len(start_milestone) == end_milestone, "The list of  start indices and end indices must equal size"
    df_milestones = pd.DataFrame()
    df_milestones['start'] = start_milestone
    df_milestones['end'] = end_milestone
    return df_milestones


def check_signal_format(s):
    assert isinstance(s, pd.DataFrame), 'Expected a pd.DataFrame.'
    assert len(s.columns) == 2, 'Expect a datafram of only two columns.'
    assert isinstance(s.iloc[0, 0], pd.Timestamp), \
        'Expected type of the first column to be pd.Timestamp.'
    assert is_number(s.iloc[0, 1]), \
        'Expected type of the second column to be float'
    return True

def create_rule_def(sqi_name, upper_bound=0, lower_bound=1):
    json_rule_dict = {}
    json_rule_dict[sqi_name] = {
        "name": sqi_name,
        "def": [
            {"op": ">", "value": str(lower_bound), "label": "accept"},
            {"op": "<=", "value": str(lower_bound), "label": "reject"},
            {"op": ">=", "value": str(upper_bound), "label": "reject"},
            {"op": "<", "value": str(upper_bound), "label": "accept"},
        ],
        "desc": "",
        "ref": ""
    }
    return json_rule_dict
