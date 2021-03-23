import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))

try:
    from ..utilities.peak_approaches import waveform_template
except Exception as e:
    from utilities.peak_approaches import waveform_template


sys.path.append(os.path.join(os.getcwd(),".."))

try:
    from ..utilities.filtering import butter_lowpass_filter,butter_highpass_filter,\
        scale_pattern
    from ..utilities.trim_utilities import trim_invalid
    from ..utilities.generate_template import \
        ppg_nonlinear_dynamic_system_template,ppg_absolute_dual_skewness_template,ppg_dual_doublde_frequency_template
except:
    from utilities.filtering import butter_lowpass_filter,butter_highpass_filter,\
        scale_pattern
    from utilities.trim_utilities import trim_invalid
    from utilities.generate_template import \
        ppg_nonlinear_dynamic_system_template,ppg_absolute_dual_skewness_template,ppg_dual_doublde_frequency_template

import plotly.io as pio
pio.renderers.default = "browser"

DATA_PATH = os.path.join(os.getcwd(), "..", "data", "18")  # 24EI-011-PPG-day1-4.csv
filename = "24EI-018-PPG-day1"  # 24EI-011-PPG-day1
ROOT_SAVED_FOLDER = os.path.join(os.getcwd(), "..", "data", "label_PPG_segment")
SAVED_FOLDER = os.path.join(ROOT_SAVED_FOLDER, filename)
SAVED_FILE_FOLDER = os.path.join(SAVED_FOLDER, "ppg")
SAVED_IMG_FOLDER = os.path.join(SAVED_FOLDER, "img")
if not os.path.exists(ROOT_SAVED_FOLDER):
    os.makedirs(ROOT_SAVED_FOLDER)
if not os.path.exists(SAVED_FOLDER):
    os.makedirs(SAVED_FOLDER)
if not os.path.exists(SAVED_FILE_FOLDER):
    os.makedirs(SAVED_FILE_FOLDER)
if not os.path.exists(SAVED_IMG_FOLDER):
    os.makedirs(SAVED_IMG_FOLDER)

def save_each_segment(filename,segment_list,display_trough_peak=False):
    """
    Save each n-second segment into csv and the relevant image
    :param filename: str, the origin file name
    :param segment_list: list, the list all split 30-second segments
    :param display_trough_peak: bool, default = False, display to trough and peak in the saved images
    :return:
    """
    extension_len = len(str(len(segment_list)))
    i = 1

    for segment in tqdm(segment_list):
        to_save = True
        zero_adding = "".join(["0"] * (extension_len-len(str(i))))
        saved_filename = filename+"-"+zero_adding+str(i)

        try:
            if to_save or i == len(segment_list):
                fig = go.Figure()
                fig.add_traces(go.Scatter(x=np.arange(1, len(segment)),
                                          y=segment, mode="lines"))
                if display_trough_peak:
                    wave = waveform_template()
                    systolic_peaks_idx, trough_idx = wave.detect_peak_trough_count_orig(segment)
                    fig.add_traces(go.Scatter(x=systolic_peaks_idx,
                                              y=segment[systolic_peaks_idx], mode="markers"))
                    fig.add_traces(go.Scatter(x=trough_idx,
                                              y=segment[trough_idx], mode="markers"))

                fig.update_layout(
                    autosize=True,
                )

                fig.write_image(os.path.join(SAVED_IMG_FOLDER, saved_filename + '.png'))
                np.savetxt(os.path.join(SAVED_FILE_FOLDER, saved_filename + '.csv'), segment, delimiter=',')  # as an array

        except Exception as e:
            print(e)
        i=i+1

def split_and_save(filename,sampling_rate=100.0,segment_length=30.0,minute_remove=5.0):
    """
    Split the data after applying bandpass filter and removing the first and last n-minutes
    (High pass filter with cutoff at 1Hz)
    The signal is split according to time domain - default is 30s
    :param filename: str, path to load file
    :param sampling_rate:float, default = 100.0. The sampling rate of the wearable device
    :param segment_length:float, default = 30.0. The length of the segment (in seconds)
    :param minute_remove: float, default = 5.0. The first and last of n-minutes to be removed
    :return:
    """
    df_origin = pd.read_csv(os.path.join(DATA_PATH, filename + ".csv"))
    df_origin = df_origin.iloc[minute_remove * 60 * sampling_rate:-(minute_remove * 60 * sampling_rate)]
    df_pleth = np.array(df_origin["PLETH"])

    start_milestone, end_milestone = trim_invalid(df_pleth, False)

    print(start_milestone)
    print(end_milestone)
    segments = []
    for start, end in zip(start_milestone, end_milestone):
        segment_seconds = segment_length * sampling_rate
        chunk = df_origin.iloc[int(start):int(end)]

        signal_bp = np.array(chunk["PLETH"])
        signal_bp = butter_highpass_filter(signal_bp, cutoff=1, fs=sampling_rate, order=1)

        segments = segments + [signal_bp[segment_seconds * i:segment_seconds * (i + 1)]
                               for i in range(0, int(np.ceil(len(signal_bp) / segment_seconds)))]

    save_each_segment(filename, np.array(segments), display_trough_peak=True)

if __name__ == "__main__":
    """
    Test the split segment script
    """
    split_and_save(filename)



