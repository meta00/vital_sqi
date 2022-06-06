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
from vital_sqi.common.utils import cut_segment, check_signal_format
from vital_sqi.common.rpeak_detection import PeakDetector


def save_segment(filename, segment_list, save_file_folder,
                      save_image=None, save_img_folder=None):
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

                fig = go.Figure()
                fig.add_traces(go.Scatter(x=np.arange(1, len(segment)),
                                          y=segment, mode="lines"))
                fig.update_layout(autosize=True)
                fig.write_image(os.path.join(save_img_folder, saved_filename + '.png'))

            np.savetxt(os.path.join(save_file_folder, saved_filename + '.csv'), segment, delimiter=',')  # as an array
        except Exception as e:
            warnings.warn(e)
        i = i+1


def split_segment(s, split_type=0, duration=30.0,
                      overlaping=1,sampling_rate=100,
                   peak_detector=7, wave_type='ppg'):
    """
    Save segment waveform and plot (optional) to csv and image file.
    Input is a segment with timestamps.

    Parameters
    ----------
    s :
        array-like represent the signal.
    split_type :
        0: split by time
        1: split by beat
    duration :
        the duration of each segment if split by time in seconds, default = 30 (second)
        the number of complex/beat in each segement if split by beat, default = 30(beat/segment)

    sampling_rate :
        device sampling rate

    peak_detector :
        The type of peak detection if split the segment by beat.

    wave_type:
        Type of signal. Either 'ppg' or 'ecg'

    Returns
    -------
    >>>from vital_sqi.common.utils import generate_timestamp
    >>>s = np.arange(100000)
    >>>timestamps = generate_timestamp(None,100,len(s))
    >>>df = pd.DataFrame(np.hstack((np.array(timestamps).reshape(-1,1),
                                 np.array(s).reshape(-1,1))))
    >>>split_segment(df,overlaping=3)
    """
    assert check_signal_format(s) is True
    if split_type is 0:
        chunk_size = int(duration * sampling_rate)
        chunk_step = int(overlaping * sampling_rate)
        chunk_indices = [
                [int(i), int(i + chunk_size)] for i in
                range(0, len(s), chunk_size - chunk_step)
            ]
    else:
        if wave_type == 'ppg':
            detector = PeakDetector(wave_type='ppg')
            peak_list, trough_list = detector.ppg_detector(s,detector_type=peak_detector)
        else:
            detector = PeakDetector(wave_type='ecg')
            peak_list, trough_list = detector.ecg_detector(s, detector_type=peak_detector)
        chunk_indices = [
                [peak_list[i], peak_list[i+duration]] for i in range(0,
                                                                     len(peak_list),int(duration-overlaping))
        ]
        chunk_indices[0] = 0
    milestones = pd.DataFrame(chunk_indices)
    segments = cut_segment(s, milestones)
    return segments, milestones

