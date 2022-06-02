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
from scipy import signal
import pandas as pd
import warnings
import pmdarima as pm


def remove_unchanged(s, unchanged_seconds=10, sampling_rate=100,
                              output_index=False):
    """
    Remove flat signal waveform. Unchanged signals are considered noise. This
    is observed in PPG waveform.

    Parameters
    ----------
    s :
        array-like signal
    unchanged_seconds :
         (Default value = 10)
    sampling_rate :
         (Default value = 100)
    as_dataframe :
         (Default value = True)

    Returns
    -------
        array
         Start and end indexes of flat segments
    """
    number_removed_instances = sampling_rate*unchanged_seconds
    if as_dataframe:
        pleth_array = np.array(s["PLETH"])
    else:
        pleth_array = np.array(s)
    diff = np.diff(pleth_array)         # 123 35 4 0 0 0 0 0 0 123 34 3 1 5 0 0 23 45
    unchanged_idx = np.where(diff == 0)[0]  # 3 4 5 6 7 8 14 15
    if len(unchanged_idx) < 1:
        return [0], [len(s)]
    continuous_dict = {} #index of continuous value and the len
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
    return start_milestone, end_milestone # chuyen thanh 1 array


def remove_invalid_smartcare(s, info, as_dataframe=True):
    """
    Filter invalid signal sample in PPG recorded by Smartcare oximeter.
    Invalid samples are one with default value:
    - signal value = 0
    - SpO2 < 80)
    - pulse > 255
    - perfusion_array < 0.1

    Parameters
    ----------
    s :
        array-like signal
    as_dataframe :
         (Default value = True)

    Returns
    -------
        array-like filtered signal
    """

    if as_dataframe:
        pleth_array = np.array(s["PLETH"])
        spo2_array = np.array(s["SPO2_PCT"])
        perfusion_array = np.array(s["PERFUSION_INDEX"])
        pulse_array = np.array(s["PULSE_BPM"])
        indices_start_end = np.where((pleth_array != 0) & (spo2_array >= 80)
                                     & (pulse_array <= 255) & (
                                             perfusion_array >= 0.1))[0]
    else:
        indices_start_end = np.where(s != 0)[0]
    diff_res = indices_start_end[1:] - indices_start_end[:-1]
    diff_loc = np.where(diff_res > 1)[0]
    start_milestone = [indices_start_end[0]]
    end_milestone = []
    for loc in diff_loc:
        end_milestone.append(indices_start_end[loc]+1)
        start_milestone.append(indices_start_end[loc+1])
    end_milestone.append(indices_start_end[-1]+1)

    return start_milestone, end_milestone


def trim_signal(data, minute_remove=1, sampling_rate=100):
    """Expose

    Parameters
    ----------
    data :
        param minute_remove:
    sampling_rate :
        return: (Default value = 100)
    minute_remove :
         (Default value = 1)

    Returns
    -------

    """
    # check if the input trimming length exceed the data length
    if minute_remove*sampling_rate*2 > len(data):
        warnings.warn("Input trimming length exceed the data length. Return "
                      "the same array")
        return data
    if type(data) == type(pd.DataFrame()):
        data = data.iloc[minute_remove * 60 *
                         sampling_rate:-(minute_remove * 60 * sampling_rate)]
    else:
        data = data[minute_remove * 60 *
                    sampling_rate:-(minute_remove * 60 * sampling_rate)]
    return data


def get_start_end_points(start_cut_pivot, end_cut_pivot, length_df):
    """handy

    Parameters
    ----------
    start_cut_pivot :
        array of starting points of the removal segment
    end_cut_pivot :
        array of relevant ending points of removal segment
    length_df :
        the length of the origin signal

    Returns
    -------

    """
    if 0 not in np.array(start_cut_pivot):
        start_milestone = np.hstack((0, np.array(end_cut_pivot) + 1))
        if length_df - 1 not in np.array(end_cut_pivot):
            end_milestone = np.hstack((np.array(start_cut_pivot) - 1, length_df - 1))
        else:
            end_milestone = (np.array(start_cut_pivot) - 1)
    else:
        start_milestone = np.array(end_cut_pivot) + 1
        end_milestone = np.hstack((np.array(start_cut_pivot)[1:] - 1, length_df - 1))
    return start_milestone, end_milestone


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
    type
        Example:
        > missing_index = np.where(np.diff(df.TIMESTAMP_MS) > 10)[0]
        > missing_len = [int((df.TIMESTAMP_MS.iloc[i+1] - df.TIMESTAMP_MS.iloc[i])/10-1)
        for i in missing]
        > filled_s = fill_missing_value(np.array(df1.PLETH),missing,missing_len)

    """
    filled_s = []
    for pos, number_of_missing_instances in zip(missing_index, missing_len):
        seg_len = number_of_missing_instances * lag_ratio
        start_seg = max(0, int(pos - seg_len))
        ts = s[start_seg:int(pos)]

        model = pm.auto_arima(ts, X=None, start_p=2, d=None,
                              start_q=2, max_p=3, max_d=3,
                              max_q=3, start_P=1, D=None,
                              start_Q=1, max_P=3, max_D=4, max_Q=4, max_order=5,
                              m=int(len(ts) / 65), seasonal=True, stationary=False,
                              information_criterion='aic', alpha=0.005,
                              test='kpss', seasonal_test='ocsb',
                              stepwise=True, n_jobs=4, start_params=None,
                              trend=None, method='lbfgs', maxiter=50,
                              offset_test_args=None, seasonal_test_args=None,
                              suppress_warnings=True, error_action='trace',trace=False,
                              random=False, random_state=None, n_fits=10,
                              return_valid_fits=False, out_of_sample_size=0,
                              scoring='mse', scoring_args=None, with_intercept='auto')

        fc, confint = model.predict(n_periods=number_of_missing_instances, return_conf_int=True)
        filled_s = filled_s + list(ts) + list(fc)
    filled_s = filled_s + list(s[int(pos):])
    return filled_s
