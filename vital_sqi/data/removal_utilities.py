"""Trimming raw signals using: invalid values, noise at start/end of
recordings etc."""
import numpy as np
from scipy import signal
import pandas as pd
import warnings

def remove_invalid(df,as_dataframe=True):
    """
    Exposed
    Remove  the list of invalid data signal
    :param df:
    :param as_dataframe:
    :return:
    """

    #TODO Cover the case of different input instead of SMARTCARE device
    if as_dataframe:
        pleth_array = np.array(df["PLETH"])
        spo2_array = np.array(df["SPO2_PCT"])
        perfusion_array = np.array(df["PERFUSION_INDEX"])
        pulse_array = np.array(df["PULSE_BPM"])
        indices_start_end = np.where((pleth_array != 0) & (spo2_array >= 80)
                                     & (pulse_array <= 255) & (perfusion_array>=0.1))[0]
    else:
        indices_start_end = np.where(df!=0)[0]
    diff_res = indices_start_end[1:] - indices_start_end[:-1]
    diff_loc = np.where(diff_res>1)[0]
    start_milestone = [indices_start_end[0]]
    end_milestone = []
    for loc in diff_loc:
        end_milestone.append(indices_start_end[loc]+1)
        start_milestone.append(indices_start_end[loc+1])
    end_milestone.append(indices_start_end[-1]+1)

    return start_milestone,end_milestone

def trim_data(data,minute_remove=1,sampling_rate=100):
    """
    Expose
    :param data:
    :param minute_remove:
    :param sampling_rate:
    :return:
    """
    # check if the input trimming length exceed the data length
    if minute_remove*sampling_rate*2 > len(data):
        warnings.warn("Input trimming length exceed the data length. Return the same array")
        return data
    if type(data) == type(pd.DataFrame()):
        data = data.iloc[minute_remove * 60 * sampling_rate:-(minute_remove * 60 * sampling_rate)]
    else:
        data = data[minute_remove * 60 * sampling_rate:-(minute_remove * 60 * sampling_rate)]
    return data

def get_start_end_points(start_cut_pivot,end_cut_pivot,length_df):
    """
    handy
    :param start_cut_pivot: array of starting points of the removal segment
    :param end_cut_pivot: array of relevant ending points of removal segment
    :param length_df: the length of the origin signal
    :return:
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
    return start_milestone,end_milestone

def concate_removed_index(start_list,end_list,remove_sliding_window = 0):
    """
    handy
    :param start_list:
    :param end_list:
    :param remove_sliding_window:
    :return:
    """
    start_list = np.array(start_list)
    end_list = np.array(end_list)
    diff_list = start_list[1:]-end_list[:-1]
    end_list_rm_indices = np.where(diff_list<=remove_sliding_window)[0]
    start_list_rm_indices = np.where(diff_list <= remove_sliding_window)[0]+1
    start_out_list = np.delete(start_list,start_list_rm_indices)
    end_out_list = np.delete(end_list, end_list_rm_indices)
    return start_out_list,end_out_list

def cut_invalid_rr_peak(df):
    """
    expose
    :param df:
    :return:
    """
    #TODO
    return

def cut_by_frequency_partition(df_examine,
                                window_size=None,peak_threshold_ratio=None,
                                lower_bound_threshold=None,
                                remove_sliding_window=None,
                                overlap_rate =None):
    """
    Expose

    :param df_examine:
    :param window_size:
    :param peak_threshold_ratio:
    :param lower_bound_threshold:
    :param remove_sliding_window:
    :param overlap_rate:
    :return:
    """
    if window_size == None:
        window_size = 500
    if window_size > len(df_examine):
        window_size  = len(df_examine)
    if peak_threshold_ratio == None:
        peak_threshold_ratio = 1.8
    if lower_bound_threshold == None:
        lower_bound_threshold = 1
    if remove_sliding_window == None:
        remove_sliding_window = 0
    if overlap_rate == None:
        overlap_rate = 1

    window = signal.get_window("boxcar", window_size)
    welch_full = signal.welch(df_examine, window=window)
    peaks_full = signal.find_peaks(welch_full[1], threshold=np.mean(welch_full[1]))
    if len(peaks_full[0]) < 2:
        num_peaks_full = 2
    else:
        num_peaks_full = len(peaks_full[0])

    remove_start_indices = []
    remove_end_indices = []

    pter = 0
    while pter < len(df_examine):
        end_pointer = pter + (window_size)
        if end_pointer >= len(df_examine):
            break
        small_partition = df_examine[pter:end_pointer]
        welch_small_partition = signal.welch(small_partition, window=window)
        peaks_small_partition = signal.find_peaks(welch_small_partition[1],
                                                  threshold=np.mean(welch_small_partition[1]))
        if (len(peaks_small_partition[0]) > num_peaks_full * peak_threshold_ratio) or  \
                (len(peaks_small_partition[0]) < num_peaks_full * lower_bound_threshold):
            remove_start_indices.append(pter)
            remove_end_indices.append(end_pointer)

        pter = pter + int(window_size * overlap_rate)

    start_trim_by_freq, end_trim_by_freq = concate_removed_index(remove_start_indices, remove_end_indices,
                                                                remove_sliding_window)
    start_milestone_by_freq,end_milestone_by_freq = \
        get_start_end_points(start_trim_by_freq, end_trim_by_freq,len(df_examine))

    return start_milestone_by_freq,end_milestone_by_freq
