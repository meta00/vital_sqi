import pandas as pd
import numpy as np
import heartpy as hp
import os
import plotly.graph_objs as go
from heartpy import analysis, peakdetection
from heartpy.datautils import rolling_mean, _sliding_window
import hrvanalysis as hrva
# DATA_PATH = os.path.join(os.getcwd(),"medical_signal","Work","data")
DATA_PATH = os.path.join(os.getcwd(),"..","data") #filter_SpO2_20191230T111948.658+0000.csv
import sys
import plotly.express as px
import plotly.io as pio
from scipy import signal
pio.renderers.default = "browser"

MIN_CYCLE = 5
SAMPLE_RATE = 100

df = pd.read_csv(os.path.join(DATA_PATH,"24EI-008-PPG.csv"))
indices_0 = np.array(df[df["PLETH"] == 0].index)
indices_start_end = np.hstack((0, indices_0, len(df) - 1))
diff_res = indices_start_end[1:] - indices_start_end[:-1]
sequence_diff_threshold = np.median(list(set(diff_res)))
start_cut_pivot = []
end_cut_pivot = []


def is_start(x, x_pre, threshold):
    if np.abs(x) < threshold and np.abs(x_pre) >= threshold:
        return True
    return False

def is_end(x, x_after, threshold):
    if np.abs(x) < threshold and np.abs(x_after) >= threshold:
        return True
    return False

for idx in range(1, len(diff_res) - 1):
            if is_start(diff_res[idx], diff_res[idx - 1], sequence_diff_threshold):
                start_cut_pivot.append(indices_0[idx])
            if is_end(diff_res[idx], diff_res[idx + 1], sequence_diff_threshold):
                end_cut_pivot.append(indices_0[idx])
start_milestone = np.hstack((0, np.array(end_cut_pivot) + 1))
end_milestone = np.hstack((np.array(start_cut_pivot) - 1, len(df)-1))

        # REMOVE SHORT LENGTH
remove_idx = []
for idx in range(len(end_milestone)-1):
            if (end_milestone[idx]-start_milestone[idx]) < MIN_CYCLE*SAMPLE_RATE:
                remove_idx.append(idx)
start_milestone = np.delete(start_milestone,remove_idx)
end_milestone = np.delete(end_milestone, remove_idx)

#=================================================================
#    TRIM BY FREQUENCY START FROM HERE
#=================================================================

df_examine = df["PLETH"].iloc[start_milestone[1]:end_milestone[1]]

window_size = MIN_CYCLE*SAMPLE_RATE
taper_windows = signal.hanning(window_size)

window = signal.get_window("boxcar",window_size)
welch_full = signal.welch(df_examine,window=window)
peaks_full = signal.find_peaks(welch_full[1],threshold=np.mean(welch_full[1]))

remove_start_indices = []
remove_end_indices = []

pointer = 0
peak_threshold_ratio = 1.8
overlap_rate = 1
remove_sliding_window = 0
while pointer < len(df_examine):
    end_pointer = pointer+(window_size)
    if end_pointer >= len(df_examine):
        break
    small_partition = df_examine[pointer:end_pointer]
    # small_partition = df_examine[pointer:end_pointer]*taper_windows
    welch_small_partition = signal.welch(small_partition, window=window)
    peaks_small_partition = signal.find_peaks(welch_small_partition[1],
                                              threshold=np.mean(welch_small_partition[1]))
    if len(peaks_small_partition[0])> len(peaks_full[0])*peak_threshold_ratio:
        remove_start_indices.append(pointer)
        remove_end_indices.append(end_pointer)

    pointer = pointer+int(window_size*overlap_rate)

def concate_remove_index(start_list,end_list,remove_sliding_window = 0):
    start_list = np.array(start_list)
    end_list = np.array(end_list)
    diff_list = start_list[1:]-end_list[:-1]
    end_list_rm_indices = np.where(diff_list<=remove_sliding_window)[0]
    start_list_rm_indices = np.where(diff_list <= remove_sliding_window)[0]+1
    start_out_list = np.delete(start_list,start_list_rm_indices)
    end_out_list = np.delete(end_list, end_list_rm_indices)
    return start_out_list,end_out_list

start_trim_by_freq, end_trim_by_freq = concate_remove_index(remove_start_indices,remove_end_indices,remove_sliding_window)
if 0 not in np.array(start_trim_by_freq):
    start_milestone_by_freq = np.hstack((0, np.array(end_trim_by_freq) + 1))
    end_milestone_by_freq = np.hstack((np.array(start_trim_by_freq) - 1, len(df) - 1))
else:
    start_milestone_by_freq = np.array(end_trim_by_freq) + 1
    end_milestone_by_freq = np.hstack((np.array(start_trim_by_freq)[1:] - 1, len(df) - 1))


fig = go.Figure()
for start, end in zip(start_milestone_by_freq, end_milestone_by_freq):
    fig.add_traces(
                    go.Scatter(
                        x=df_examine[int(start):int(end)].index,
                        y=df_examine[int(start):int(end)],
                        mode="lines"
                    ))

# fig = px.line(small_partition)
fig.show()