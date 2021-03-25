import pandas as pd
import base64
import io
import dash_html_components as html
import numpy as np
from scipy import signal

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), header=0)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

def concate_remove_index(start_list,end_list,remove_sliding_window = 0):
    start_list = np.array(start_list)
    end_list = np.array(end_list)
    diff_list = start_list[1:]-end_list[:-1]
    end_list_rm_indices = np.where(diff_list<=remove_sliding_window)[0]
    start_list_rm_indices = np.where(diff_list <= remove_sliding_window)[0]+1
    start_out_list = np.delete(start_list,start_list_rm_indices)
    end_out_list = np.delete(end_list, end_list_rm_indices)
    return start_out_list,end_out_list

def remove_short_length(start_milestone,end_milestone,min_length=500):
    remove_idx = []
    for idx in range(len(end_milestone)):
        try:
            if (end_milestone[idx] - start_milestone[idx]) < min_length:
                remove_idx.append(idx)
        except Exception as error:
            print(error)
    start_milestone = np.delete(start_milestone, remove_idx)
    end_milestone = np.delete(end_milestone, remove_idx)
    return start_milestone,end_milestone

def trim_invalid_signal(df,as_dataframe=True):
    if as_dataframe:
        pleth_array = np.array(df["PLETH"])
        spo2_array = np.array(df["SPO2_PCT"])
        perfusion_array = np.array(df["PERFUSION_INDEX"])
        pulse_array = np.array(df["PULSE_BPM"])
        indices_start_end = np.where((pleth_array != 0) & (spo2_array >= 80)
                                     & (pulse_array <= 255) & (perfusion_array >= 0.1))[0]
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

def get_invalid_SpO2(df):

    pleth_array = np.array(df["SPO2_PCT"])
    indices_start_end = np.where((pleth_array<=80) & (pleth_array >= 70))[0]

    diff_res = indices_start_end[1:] - indices_start_end[:-1]
    diff_loc = np.where(diff_res>1)[0]
    start_milestone = [indices_start_end[0]]
    end_milestone = []
    for loc in diff_loc:
        end_milestone.append(indices_start_end[loc]+1)
        start_milestone.append(indices_start_end[loc+1])
    end_milestone.append(indices_start_end[-1]+1)

    return start_milestone,end_milestone

def get_invalid_perfusion(df):

    pleth_array = np.array(df["PERFUSION_INDEX"])
    indices_start_end = np.where((pleth_array<0.2))[0]
        # indices_start_end = np.array(df[(df["PLETH"]) != 0].index)

    diff_res = indices_start_end[1:] - indices_start_end[:-1]
    diff_loc = np.where(diff_res>1)[0]

    start_milestone = [indices_start_end[0]]
    end_milestone = []
    for loc in diff_loc:
        end_milestone.append(indices_start_end[loc]+1)
        start_milestone.append(indices_start_end[loc+1])
    end_milestone.append(indices_start_end[-1]+1)

    return start_milestone,end_milestone

def get_invalid_BPM(df):

    pleth_array = np.array(df["PULSE_BPM"])
    indices_start_end = np.where((pleth_array>255))[0]

    diff_res = indices_start_end[1:] - indices_start_end[:-1]
    diff_loc = np.where(diff_res>1)[0]
    start_milestone = [indices_start_end[0]]
    end_milestone = []
    for loc in diff_loc:
        end_milestone.append(indices_start_end[loc]+1)
        start_milestone.append(indices_start_end[loc+1])
    end_milestone.append(indices_start_end[-1]+1)

    return start_milestone,end_milestone

def cut_milestone_to_keep_milestone(start_cut_pivot,end_cut_pivot,length_df):
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

def trim_by_frequency_partition(
        df_examine,window_size=500,
        peak_threshold_ratio=None,
        lower_bound_threshold=None,
        remove_sliding_window=None,
        overlap_rate =None
):
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

    start_trim_by_freq, end_trim_by_freq = concate_remove_index(remove_start_indices, remove_end_indices,
                                                                remove_sliding_window)
    start_milestone_by_freq,end_milestone_by_freq = \
        cut_milestone_to_keep_milestone(start_trim_by_freq, end_trim_by_freq,len(df_examine))

    return start_milestone_by_freq,end_milestone_by_freq
