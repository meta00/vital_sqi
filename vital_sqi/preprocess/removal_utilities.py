"""
Trimming raw signals using: invalid values, noise at start/end of
recordings etc.

- output processed signal
- input: signal (s) in pandas df, check by type checking function (utils,
io - find it).
- bo option as_dataframe
- them option_index
"""
import numpy as np
from vital_sqi.common.utils import cut_segment, format_milestone, \
    check_signal_format
import warnings
import pmdarima as pm


def get_start_end_points(start_cut_pivot, end_cut_pivot, length_df):
    """

    Parameters
    ----------
    start_cut_pivot :
        array of starting points of the removed segment
    end_cut_pivot :
        array of relevant ending points of removed segment
    length_df :
        the length of the origin signal

    Returns
    -------

    
    """
    if 0 not in np.array(start_cut_pivot):
        start_milestone = np.hstack((0, np.array(end_cut_pivot) + 1))
        if length_df - 1 not in np.array(end_cut_pivot):
            end_milestone = np.hstack((np.array(start_cut_pivot) - 1,
                                       length_df - 1))
        else:
            end_milestone = (np.array(start_cut_pivot) - 1)
    else:
        start_milestone = np.array(end_cut_pivot) + 1
        end_milestone = np.hstack((np.array(start_cut_pivot)[1:] - 1,
                                   length_df - 1))
    return start_milestone, end_milestone


def remove_unchanged(s, duration=10, sampling_rate=100,
                     output_signal=True):
    """Unchanged signal samples, i.e., flat segments, of over an user-defined
    duration are considered noise and to be removed. This is observed in PPG
    waveform, probably due to loose sensor.

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as float
    duration : float
        (Default value = 10)
        Number of seconds considered to be noise to remove
    sampling_rate : float
        (Default value = 100)
    output_signal : bool
        (Default value = True)
        Option to output processed signal. If False, only milestones is
        returned.


    Returns
    -------
    processed_s : pandas DataFrame
        Processed signal, i.e. signal with the unchanged segments removed
    milestones: pandas DataFrame
        DataFrame of two columns containing start and end indexes for the
        retained segments.
    """

    assert check_signal_format(s) is True
    number_removed_instances = sampling_rate*duration
    signal_array = np.array(s.iloc[:, 1])
    diff = np.diff(signal_array)  # 123 35 4 0 0 0 0 0 0 123 34 3 1 5 0 0 23 45
    unchanged_idx = np.where(diff == 0)[0]  # 3 4 5 6 7 8 14 15
    if len(unchanged_idx) < 1:
        start_milestone = [0]
        end_milestone = [len(s)]
    else:
        continuous_dict = {}  # index of continuous value and the len
        continuous_len = 0
        key = -1
        for i in range(len(diff)):
            if diff[i] == 0:
                if key == -1:
                    key = i
                continuous_len = continuous_len + 1
            else:
                if continuous_len > 0:
                    continuous_dict[key] = continuous_len
                key = i+1
                continuous_len = 0

        start_cut_pivot = []
        end_cut_pivot = []
        for key in continuous_dict.keys():
            if continuous_dict[key] >= number_removed_instances:
                start_cut_pivot.append(key)
                end_cut_pivot.append(key+continuous_dict[key])

        start_milestone, end_milestone = get_start_end_points(start_cut_pivot,
                                                              end_cut_pivot,
                                                              len(s))
    milestones = format_milestone(start_milestone, end_milestone)
    if output_signal:
        processed_s = cut_segment(s, milestones)
        return processed_s, milestones
    return milestones


def remove_invalid_smartcare(s, info, output_signal=True):
    """Filtering invalid signal sample in PPG waveform recorded by the Smartcare
    oximeter based on other values generated from the oximeter such as SpO2,
    Pulse, Perfusion. This function expects additional SmartCare PPG fields
    in the input and could be adapted for other

    Invalid samples are one with either:
    - signal value = 0
    - SpO2 < 80
    - pulse > 255
    - perfusion_array < 0.1

    Parameters
    ----------
    s : pandas DataFrame
        Signal, with first column as pandas Timestamp and second column as float
    info : pandas DataFrame
        Info generated from Smartcare containing "SPO2_PCT", "PERFUSION_INDEX",
        "PULSE_BPM" columns.
    output_signal : bool
        (Default value = True)
        Option to output processed signal. If False, only milestones is
        returned.

    Returns
    -------
    processed_s : pandas DataFrame
        Processed signal, i.e. signal with the invalid samples removed
    milestones: pandas DataFrame
        DataFrame of two columns containing start and end indexes for the
        retained segments.
    
    """
    assert check_signal_format(s) is True
    info.columns = str.capitalize(info.columns)

    if {"SPO2_PCT", "PERFUSION_INDEX", "PULSE_BPM"}.issubset(set(info.columns)):
        spo2_array = np.array(info["SPO2_PCT"])
        perfusion_array = np.array(info["PERFUSION_INDEX"])
        pulse_array = np.array(info["PULSE_BPM"])
        indices_start_end = np.where((s != 0)
                                     & (spo2_array >= 80)
                                     & (pulse_array <= 255)
                                     & (perfusion_array >= 0.1))[0]
    else:
        s_channel = s.iloc[:, 1]
        indices_start_end = np.where(s_channel != 0)[0]
    diff_res = indices_start_end[1:] - indices_start_end[:-1]
    diff_loc = np.where(diff_res > 1)[0]
    start_milestone = [indices_start_end[0]]
    end_milestone = []
    for loc in diff_loc:
        end_milestone.append(indices_start_end[loc]+1)
        start_milestone.append(indices_start_end[loc+1])
    end_milestone.append(indices_start_end[-1]+1)
    milestones = format_milestone(start_milestone, end_milestone)

    if output_signal:
        processed_s = cut_segment(s, milestones)
        return processed_s, milestones
    return milestones


def trim_signal(s, duration_left=300, duration_right=300, sampling_rate=100):
    """ Trimming

    Parameters
    ----------
    s :
        
    sampling_rate :
        return: (Default value = 100)
    duration_left :
        second
        (Default value = 300)
    duration_right :
        (Default value = 300)

    Returns
    -------

    
    """
    # check if the input trimming length exceed the data length
    if int((duration_left+duration_right)*sampling_rate*2) > len(s):
        warnings.warn("Input trimming length exceed the data length. Return "
                      "the same array")
        return s

    assert check_signal_format(s) is True
    s = s.iloc[int(duration_left * sampling_rate):-int(duration_right *
                                                       sampling_rate)]
    return s


def remove_invalid_peak(nn_intervals):
    """

    Parameters
    ----------
    nn_intervals :
        

    Returns
    -------

    
    """
    #TODO
    return


def interpolate_signal(s, missing_index, missing_len, method='arima',
                       lag_ratio=10):
    """

    Parameters
    ----------
    s :
        array-like signal
    missing_index :
        array of list of starting indices missing data
    missing_len :
        array of number of missing instances,
        matching with the index list
    method :
        return:
        Example:
        > missing_index = np.where(np.diff(df.TIMESTAMP_MS) > 10)[0]
        > missing_len = [int((df.TIMESTAMP_MS.iloc[i+1] - df.TIMESTAMP_MS.iloc[i])/10-1)
        for i in missing]
        > filled_s = fill_missing_value(np.array(df1.PLETH),missing,missing_len) (Default value = 'arima')
    lag_ratio :
        (Default value = 10)

    Returns
    -------

    
    """
    assert check_signal_format(s) is True
    s_channel = s.iloc[:1]
    filled_s = []
    for pos, number_of_missing_instances in zip(missing_index, missing_len):
        seg_len = number_of_missing_instances * lag_ratio
        start_seg = max(0, int(pos - seg_len))
        ts = s[start_seg:int(pos)]
        if method is 'arima':
            model = pm.auto_arima(ts, X=None, start_p=2, d=None,
                                  start_q=2, max_p=3, max_d=3, max_q=3,
                                  start_P=1, D=None, start_Q=1, max_P=3,
                                  max_D=4, max_Q=4, max_order=5,
                                  m=int(len(ts) / 65), seasonal=True,
                                  stationary=False,
                                  information_criterion='aic', alpha=0.005,
                                  test='kpss', seasonal_test='ocsb',
                              stepwise=True, n_jobs=4, start_params=None,
                              trend=None, method='lbfgs', maxiter=50,
                                  offset_test_args=None, seasonal_test_args=None,
                              suppress_warnings=True, error_action='trace',
                                  trace=False, random=False,
                                  random_state=None, n_fits=10,
                                  return_valid_fits=False,
                                  out_of_sample_size=0, scoring='mse',
                                  scoring_args=None, with_intercept='auto')

        fc, confint = model.predict(n_periods=number_of_missing_instances,
                                    return_conf_int=True)
        filled_s = filled_s + list(ts) + list(fc)
    filled_s = filled_s + list(s_channel[int(pos):])
    s.iloc[:, 1] = filled_s
    return s