import warnings

import numpy as np
import datetime as dt
import os
import json

import pandas as pd
from datetimerange import DateTimeRange
import dateparser

OPERAND_MAPPING_DICT = {
    ">":5,
    ">=":4,
    "=":3,
    "<=":2,
    "<":1
}

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
            pass
    steps = np.diff(timestamps_second)
    sampling_rate = round(1 / np.min(steps[steps != 0]))
    return sampling_rate


def generate_timestamp(start_datetime, sampling_rate, signal_length):
    """

    Parameters
    ----------
    start_datetime :
        
    sampling_rate : float
        
    signal_length : int
        

    Returns
    -------
    list : list of timestamps with length equal to signal_length.
    """
    number_of_seconds = (signal_length - 1)/sampling_rate
    if start_datetime is None:
        start_datetime = dt.datetime.now()
    end_datetime = start_datetime + dt.timedelta(seconds=number_of_seconds)
    time_range = DateTimeRange(start_datetime, end_datetime)
    timestamps = []
    for value in time_range.range(dt.timedelta(seconds=1/sampling_rate)):
        timestamps.append(value)
    if len(timestamps) != signal_length:
        raise Exception("Timestamp series generated is not valid, please "
                        "check sampling rate.")
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
        formats =  datime_formats
    for f in formats:
        try:
            return datetime.strptime(string, f)
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
            raise KeyError("SQI {0} not found".format(name))
        # sort the rule in ascending order -> value -> operand "<","=",">"
        df = sort_rule(list(np.copy(sqi['def'])))
        df = decompose_operand(df.to_dict('records'))
        boundaries = np.sort(df["value"].unique())
        inteveral_label_list = get_inteveral_label_list(df, boundaries)
        value_label_list = get_value_label_list(df, boundaries, inteveral_label_list)

        label_list = []
        for i in range(len(value_label_list)):
            label_list.append(inteveral_label_list[i])
            label_list.append(value_label_list[i])
        label_list.append(inteveral_label_list[-1])

    return sqi['def'], \
           boundaries, \
           label_list

def write_rule(name, rule_def, file_path):
    rule_dict = {}
    rule_dict['name'] = name
    rule_dict['def'] = rule_def
    with open(file_path, "w") as write_file:
        json.dump(rule_dict, write_file)
    return

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
    assert len(pair) <= 1, "Duplicated decision at '"\
                           +str(pair["value"])+" "+pair["op"]+"'"
    return True

def check_conflict(decision_lt,decision_gt):
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
        raise ValueError("Rules raise a conflict at x "+decision_lt.iloc[0]["op"]+" "+
                         str(decision_lt.iloc[0]["value"])+" is "+decision_lt.iloc[0]["label"]
                         +", but x "+decision_gt.iloc[0]["op"]+" "+
                         str(decision_gt.iloc[0]["value"])+" is "+decision_gt.iloc[0]["label"])
    return label_gt

def get_decision(df,boundaries,idx):
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
    while(decision==None and idx<=(len(df)-1)):
        decision = get_decision(df,boundaries,idx+1)
    return decision

def get_inteveral_label_list(df,boundaries):
    inteveral_label_list = np.array([None] * (len(boundaries) + 1))

    assert df["op"].iloc[0] == "<", \
        "The rule is missing a decision from -inf to " + str(df["value"].iloc[0])
    inteveral_label_list[0] = df.iloc[0]["label"]
    for idx in range(len(boundaries)-1):
        decision = get_decision(df,boundaries,idx)
        inteveral_label_list[idx+1] = decision
    assert df["op"].iloc[-1] == ">",\
        "The rule is missing a decision from "+str(df["value"].iloc[-1])+" to inf"
    inteveral_label_list[-1] = df.iloc[-1]["label"]
    return inteveral_label_list

def get_value_label_list(df,boundaries,inteveral_label_list):
    value_label_list = np.array([None] * (len(boundaries)))
    for idx in range(len(boundaries)-1):
        decision = df[(df["value"] == boundaries[idx]) &
                               (df["op"] == "==")]
        check_unique_pair(decision)
        if len(decision)==0:
            value_label_list[idx] = inteveral_label_list[idx+1]
        else:
            value_label_list[idx] = decision.iloc[0]["label"]
    value_label_list[-1] = inteveral_label_list[-2]
    return value_label_list