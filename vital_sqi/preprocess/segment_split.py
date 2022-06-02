""" Splitting long recording into segments
- By duration with option for overlapping
- By beat

To be revised: gom 4 functions cuoi vao split_segment, bo save_segment_image.

- save_segment
- split_segment

"""

import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np
import warnings
import os
from vital_sqi.preprocess.removal_utilities import remove_invalid_smartcare, \
    trim_signal
from vital_sqi.common.rpeak_detection import PeakDetector


def save_segment_image(segment, saved_filename, save_img_folder,
                       display_trough_peak):
    """

    Parameters
    ----------
    segment :
        param saved_filename:
    save_img_folder :
        param display_trough_peak:
    saved_filename :
        
    display_trough_peak :
        

    Returns
    -------

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


def save_segment(filename, segment_list, save_file_folder,
                      save_image=None, save_img_folder, display_trough_peak):
    """
    Save segment waveform and plot (optional) to csv and image file.
    Input is a segment with timestamps.

    Parameters
    ----------
    filename :
        str, the origin file name
    segment_list :
        list, the list all split 30-second segments
    display_trough_peak :
        bool, default = False, display to trough and peak in the saved images
    save_file_folder :
        
    save_image :
        
    save_img_folder :
        

    Returns
    -------

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
        i = i+1


def split_segment(s, by=[duration-0, beat-1], duration=None, overlapping=None,
                   peak_detector=None):
    return

def split_to_segments(signal_data, filename=None, sampling_rate=100.0,
                    segment_length_second=30.0, minute_remove=5.0,
                    wave_type="ecg", split_type="time", is_trim=False,
                    save_file_folder=None, save_image=False,
                    save_img_folder=None, display_trough_peak=True):
    """Expose - to be removed
    Split the data after applying bandpass filter and removing the first and last n-minutes
    (High pass filter with cutoff at 1Hz)
    The signal is split according to time domain - default is 30s

    Parameters
    ----------
    filename :
        str, path to load file (Default value = None)
    sampling_rate :
        float, default = 100.0. The sampling rate of the wearable device
    segment_length :
        float, default = 30.0. The length of the segment (in seconds)
    minute_remove :
        float, default = 5.0. The first and last of n-minutes to be removed
    signal_data :
        
    segment_length_second :
         (Default value = 30.0)
    wave_type :
         (Default value = "ecg")
    split_type :
         (Default value = "time")
    is_trim :
         (Default value = False)
    save_file_folder :
         (Default value = None)
    save_image :
         (Default value = False)
    save_img_folder :
         (Default value = None)
    display_trough_peak :
         (Default value = True)

    Returns
    -------

    """
    if filename is None:
        filename = 'segment'
    if save_file_folder is None:
        save_file_folder = '../data'
    save_file_folder = os.path.join(save_file_folder, wave_type)
    if not os.path.exists(save_file_folder):
        os.makedirs(save_file_folder)

    if save_image is True:
        if save_img_folder is None:
            save_img_folder = '.'
        save_img_folder = os.path.join(save_img_folder, "img")
        if not os.path.exists(save_img_folder):
            os.makedirs(save_img_folder)

    if is_trim:
        signal_data = trim_signal(signal_data, minute_remove, sampling_rate)

    start_milestone, end_milestone = remove_invalid_smartcare(signal_data,
                                                              False)

    segments = []
    for start, end in zip(start_milestone, end_milestone):
        segment_seconds = segment_length_second * sampling_rate
        sub_signal_data = signal_data[int(start):int(end)]
        if split_type == 'peak_interval':
            chunk_indices = get_split_rr_index(segment_seconds,sub_signal_data)
        else:
            chunk_indices = get_split_time_index(segment_seconds, sub_signal_data)
        segments = segments + [sub_signal_data[chunk_indices[i]:chunk_indices[i+1]]
                               for i in range(len(chunk_indices)-1)]

    save_segment(filename, np.array(segments), save_file_folder, save_image,
                 save_img_folder, display_trough_peak)


def generate_segment_idx(segment_length, sampling_rate, signal_array):
    """ handy

    Parameters
    ----------
    segment_length :
        
    sampling_rate :
        
    signal_array :
        

    Returns
    -------

    """
    segments = []
    for channel in signal_array:
        segments.append(np.arange(0,len(channel),segment_length*sampling_rate))
    return np.asarray(segments)


def get_split_time_index(segment_seconds, sequence):
    """handy
    Return the index of splitting points

    Parameters
    ----------
    segment_seconds :
        the length of each cut split (in seconds)
    sequence :
        return:

    Returns
    -------

    """
    indices = [int(segment_seconds * i)
               for i in range(0, int(np.ceil(len(sequence) / segment_seconds)))]
    return indices


def get_split_rr_index(segment_seconds, sequence):
    """handy
    Return the index of the splitting points

    Parameters
    ----------
    segment_seconds :
        the length of each cut split (in seconds)
    sequence :
        return:

    Returns
    -------

    """
    detector = PeakDetector()
    indices = [0]
    for i in range(0, int(np.ceil(len(sequence) / segment_seconds))):
        chunk = sequence[int(segment_seconds * i):
                         int(segment_seconds * (i + 1) + 60)]
        peak_list, trough_list = detector.ppg_detector(chunk)
        if len(trough_list)>0:
            indices.append(int(trough_list[-1]+segment_seconds * i))
        else:
            indices.append(int(segment_seconds * (i+1)))
    return indices