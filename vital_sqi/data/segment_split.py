""" Splitting long recordings into segments"""

import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np
import warnings
import os
from vital_sqi.data.removal_utilities import remove_invalid,trim_data
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.preprocess.band_filter import BandpassFilter
from vital_sqi.preprocess.preprocess_signal import scale_pattern

def save_segment_image(segment,saved_filename,save_img_folder,display_trough_peak):
    """
    handy
    :param segment:
    :param saved_filename:
    :param save_img_folder:
    :param display_trough_peak:
    :return:
    """
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=np.arange(1, len(segment)),
                              y=segment, mode="lines"))
    if display_trough_peak:
        wave = PeakDetector()
        systolic_peaks_idx, trough_idx = wave.detect_peak_trough_count_orig(segment)
        fig.add_traces(go.Scatter(x=systolic_peaks_idx,
                                  y=segment[systolic_peaks_idx], mode="markers"))
        fig.add_traces(go.Scatter(x=trough_idx,
                                  y=segment[trough_idx], mode="markers"))
    fig.update_layout(
        autosize=True,
    )
    fig.write_image(os.path.join(save_img_folder, saved_filename + '.png'))

def save_each_segment(filename,segment_list,save_file_folder,
                      save_image,save_img_folder,display_trough_peak):
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
        zero_adding = "".join(["0"] * (extension_len-len(str(i))))

        try:
            saved_filename = filename+"-"+zero_adding+str(i)
            if save_image:
                save_segment_image(segment, saved_filename, save_img_folder, display_trough_peak)

            np.savetxt(os.path.join(save_file_folder, saved_filename + '.csv'), segment, delimiter=',')  # as an array
        except Exception as e:
            warnings.warn(e)
        i=i+1

def split_by_time(signal,filename,sampling_rate=100.0,segment_length_second=30.0,minute_remove=5.0,
                  is_trim=False,save_file_folder=None,save_image=False,save_img_folder=None,display_trough_peak=True):
    """
    Expose
    Split the data after applying bandpass filter and removing the first and last n-minutes
    (High pass filter with cutoff at 1Hz)
    The signal is split according to time domain - default is 30s
    :param filename: str, path to load file
    :param sampling_rate:float, default = 100.0. The sampling rate of the wearable device
    :param segment_length:float, default = 30.0. The length of the segment (in seconds)
    :param minute_remove: float, default = 5.0. The first and last of n-minutes to be removed
    :return:
    """
    if save_file_folder == None:
        save_file_folder = '.'
    save_file_folder = os.path.join(save_file_folder, "ppg")
    if not os.path.exists(save_file_folder):
        os.makedirs(save_file_folder)

    if  save_image == True:
        if save_img_folder == None:
            save_img_folder = '.'
        save_img_folder = os.path.join(save_img_folder, "img")
        if not os.path.exists(save_img_folder):
            os.makedirs(save_img_folder)

    # df_origin = pd.read_csv(os.path.join(folder_path, filename + ".csv"))
    # df_origin = df_origin.iloc[minute_remove * 60 * sampling_rate:-(minute_remove * 60 * sampling_rate)]
    if is_trim:
        signal = trim_data(signal,minute_remove,sampling_rate)
    # df_pleth = np.array(df_origin["PLETH"])

    start_milestone, end_milestone = remove_invalid(signal, False)

    print(start_milestone)
    print(end_milestone)
    segments = []
    filter = BandpassFilter()
    for start, end in zip(start_milestone, end_milestone):
        segment_seconds = segment_length_second * sampling_rate
        # chunk = signal.iloc[int(start):int(end)]
        # signal_bp = filter.signal_highpass_filter(signal, cutoff=1, fs=sampling_rate, order=1)

        segments = segments + [signal[segment_seconds * i:segment_seconds * (i + 1)]
                               for i in range(0, int(np.ceil(len(signal) / segment_seconds)))]

    save_each_segment(filename, np.array(segments),save_file_folder,
                      save_image,save_img_folder,display_trough_peak)

# if __name__ == "__main__":
#     """
#     Test the split segment script
#     """
#     split_by_time(folder_path,filename)



