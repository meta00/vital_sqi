import pandas as pd
import numpy as np
import heartpy as hp
import os
from scipy import fft
from tqdm import tqdm
from heartpy import analysis, peakdetection
from heartpy.datautils import rolling_mean, _sliding_window
import hrvanalysis as hrva
import datetime
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from openpyxl import Workbook,load_workbook
from scipy.ndimage import gaussian_filter1d
from heartpy.filtering import smooth_signal
import heartpy as hp
from scipy.interpolate import CubicSpline
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import sys
import os
import shutil
from dtw import dtw
from scipy.signal import argrelextrema
sys.path.append(os.path.join(os.getcwd(),".."))
try:
    from ..utilities.trim_utilities import *
except Exception as e:
    from utilities.trim_utilities import *

if "sqi" not in os.getcwd():
    os.chdir(os.path.join(os.getcwd(),"medical_signal","Work","sqi"))
sys.path.append(os.path.join(os.getcwd(),".."))
print(sys.path)
try:
    from ..utilities.filtering import butter_lowpass_filter,butter_highpass_filter,\
        smooth,scale_pattern,smooth_window,get_clipping_pivots
    from ..utilities.generate_template import custom_window
except:
    from utilities.filtering import butter_lowpass_filter,butter_highpass_filter,\
        smooth,scale_pattern,smooth_window,get_clipping_pivots
    from utilities.generate_template import custom_window

try:
    from ..visualization.plotting import plot_clipping
except:
    from visualization.plotting import plot_clipping

try:
    from ..sqi.SQI import kurtosis_sqi,skewness_sqi,zero_crossings_rate_sqi,entropy_sqi,signal_to_noise_sqi
except:
    from sqi.SQI import kurtosis_sqi, skewness_sqi, zero_crossings_rate_sqi, entropy_sqi, signal_to_noise_sqi


import plotly.io as pio

pio.renderers.default = "browser"

def get_cycles():
    cycle_list = []
    window_size = 500
    window = signal.get_window("boxcar", window_size)
    welch_full = signal.welch(np.array(df_origin["PLETH"]), window=window)
    peaks_full = signal.find_peaks(welch_full[1], threshold=np.mean(welch_full[1]))

def clipping_df(s):
    min_s = np.min(s)
    s = s+np.abs(min_s)
    max_s = np.max(s)
    s = s - max_s / 2

    S = fft.fft(s)
    absS = abs(S)
    return s

def tapering(signal_data,scan_window = 100):
    # window = custom_window(int(len(signal_data)))
    local_minima = argrelextrema(signal_data, np.less)
    if len(local_minima[0])>=2:
        signal_data = signal_data[local_minima[0][0]:local_minima[0][-1]+1]
    window = signal.windows.tukey(len(signal_data),0.9)
    signal_data_tapered = np.array(window) * (signal_data) #-min(signal_data)
    return np.array(signal_data_tapered)

def get_clipping(s,pivot_list,start=0,end=10,samples = -1,use_tol=False,scale_data=False,random=True):
    template_list = []
    width = np.median(np.diff(pivot_list))
    if samples == -1:
        list_peaks = np.arange(len(pivot_list)-1)
    else:
        list_peaks = range(start, end)
        if random:
            list_peaks = np.random.randint(1, len(pivot_list)-1, samples)

    tol_median = np.median(np.diff(pivot_list))
    for i in list_peaks:
        signal_data = s[pivot_list[i]:pivot_list[i+1]]
        # signal_data = butter_highpass_filter(signal_data, cutoff=1, fs=100, order=1)
            # print(len(signal_data),width)
        if scale_data:
            signal_data = scale_pattern(signal_data,width)
        else:
            width = len(signal_data)
        # signal_data = s[pivot_list[i]:pivot_list[i] + int(width)]
        # Taper data
        # window = custom_window(int(width))
        # signal_data_tapered = np.array(window) * (signal_data-min(signal_data))
        signal_data_tapered = tapering(signal_data)

        if (((pivot_list[i + 1] - pivot_list[i]) <= tol_median * 1.25)
                and ((pivot_list[i + 1] - pivot_list[i]) >= tol_median * 0.75)):
            template_list.append(signal_data_tapered.tolist())
        elif not use_tol:
            template_list.append(signal_data_tapered.tolist())
    if scale_data:
        template_mean = (np.mean(np.array(template_list), axis=0))
    else:
        #TODO
        template_mean = []

    check_len=0
    for template in template_list:
        check_len += len(template)
    return template_list,template_mean

def compare_differences(last_row,current_row,tol=0.5):
    if len(last_row) == 0:
        return True
    total_diff = np.sum(np.abs(np.array(last_row)-np.array(current_row)))
    if total_diff < tol:
        return False
    return True

def reorganize_file(filename,SAVED_FOLDER,SAVED_IMG_FOLDER):
    df_template = pd.read_csv(os.path.join(SAVED_FOLDER, filename + "_sqi.csv"), header=0)
    for i in range(0,4):
        if not os.path.exists(os.path.join(SAVED_IMG_FOLDER,str(i))):
            os.makedirs(os.path.join(SAVED_IMG_FOLDER,str(i)))
    for i in range(0,4):
        file_list = np.array(df_template[df_template["quality"]==i]["name"])
        for file in file_list:
            shutil.move(os.path.join(SAVED_IMG_FOLDER, file+ ".png"),
                        os.path.join(SAVED_IMG_FOLDER, str(i), file + ".png"))

def save_each_template(filename,template_list,template_mean=None,
                       detrend=False,origin=False,enhanced=False,
                       resampling=False,save_all=True,display_trough_peak=False):
    extension_len = len(str(len(template_list)))

    df_template = pd.DataFrame(columns=["name", "kurtosis_sqi", "skewness_sqi",
                                        "entropy_sqi", "signal_to_noise_sqi",
                                        "zero_crossings_rate_sqi", "quality"])
    i = 1
    if template_mean != None:
        template_list = np.vstack((template_list, template_mean))
    latest_row_arr = []
    for template in tqdm(template_list):
        to_save = True
        zero_adding = "".join(["0"] * (extension_len-len(str(i))))
        # template = butter_highpass_filter(template, cutoff=1, fs=100, order=1)
        if i == len(template_list):
            saved_filename = filename+"-mean_template"
        else:
            saved_filename = filename+"-"+zero_adding+str(i)
        if origin:
            saved_filename = saved_filename+"_origin"
        if enhanced:
            saved_filename = saved_filename+"_enhanced"
        if detrend:
            template = signal.detrend(template)
            saved_filename = saved_filename + "_detrend"
        if resampling:
            template = signal.resample(template,int(np.ceil(len(template)/5)))
            saved_filename = saved_filename + "_resampling"
        # save ppg data to ppg file
        try:
            # save figure to image file
            row_content = [saved_filename]
            for sqi_method in sqi_methods:
                row_content.append(sqi_method(np.array(template)))
            row_content.append("")
            row_append = pd.Series(row_content, index=df_template.columns)
            if not save_all:
                to_save = compare_differences(latest_row_arr[1:-1],row_content[1:-1])
            if to_save or i == len(template_list):
                df_template = df_template.append(row_append,ignore_index=True)
                fig = go.Figure()
                fig.add_traces(go.Scatter(x=np.arange(1, len(template)),
                                          y=template, mode="lines"))
                if display_trough_peak:
                    systolic_peaks_idx, trough_idx = custom_peaks_detection(template)
                    fig.add_traces(go.Scatter(x=systolic_peaks_idx,
                                              y=template[systolic_peaks_idx], mode="markers"))
                    fig.add_traces(go.Scatter(x=trough_idx,
                                              y=template[trough_idx], mode="markers"))

                fig.update_layout(
                    autosize=True,
                    # width=750,
                    # height=1000,
                    # margin=dict(l=50,r=50,b=100,t=100,pad=4),
                    # paper_bgcolor="LightSteelBlue",
                )

                fig.write_image(os.path.join(SAVED_IMG_FOLDER, saved_filename + '.png'))
                np.savetxt(os.path.join(SAVED_FILE_FOLDER, saved_filename + '.csv'), template, delimiter=',')  # as an array

        except Exception as e:
            print(e)
        i=i+1
        if to_save:
            latest_row_arr = row_content
    if not origin:
        df_template.to_csv(os.path.join(SAVED_FOLDER, filename + "_sqi.csv"), index=False)

def join_templates(template_list):
    template_concat = []
    for template in template_list:
        template_concat = template_concat + template
    return template_concat

def compute_feature(s, local_extrema):
    amplitude = s[local_extrema]
    diff = np.diff(amplitude)
    diff = np.hstack((diff[0],diff,diff[-1]))
    mean_diff = np.mean(np.vstack((diff[1:],diff[:-1])),axis=0)
    amplitude = amplitude.reshape(-1,1)
    mean_diff = mean_diff.reshape(-1, 1)
    return np.hstack((amplitude,mean_diff))

def custom_peaks_detection(s, peak_threshold_ratio=None, trough_threshold_ratio=None):
    local_maxima = signal.argrelmax(s)[0]
    local_minima = signal.argrelmin(s)[0]

    clusterer = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,
                       tol=0.0001, precompute_distances='deprecated', verbose=0,
                       random_state=None, copy_x=True, n_jobs='deprecated', algorithm='auto')
    convert_maxima = compute_feature(s, local_maxima)
    clusterer.fit(convert_maxima)
    systolic_group = clusterer.predict(convert_maxima[np.argmax(s[local_maxima])].reshape(1,-1))
    labels = clusterer.predict(convert_maxima)

    systolic_peaks_idx = local_maxima[np.where(labels == systolic_group)]

    convert_minima = compute_feature(s, local_minima)
    clusterer.fit(convert_minima)
    trough_group = clusterer.predict(convert_minima[np.argmin(s[local_minima])].reshape(1, -1))
    labels = clusterer.predict(convert_minima)

    trough_idx = local_minima[np.where(labels == trough_group)]

    return systolic_peaks_idx,trough_idx

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.getcwd(), "..", "data", "11")  # 24EI-011-PPG-day1-4.csv
    filename = "24EI-011-PPG-day1"  # 24EI-011-PPG-day1
    ROOT_SAVED_FOLDER = os.path.join(os.getcwd(),"..","data","label_PPG_segment")
    SAVED_FOLDER = os.path.join(ROOT_SAVED_FOLDER,filename)
    SAVED_FILE_FOLDER = os.path.join(SAVED_FOLDER,"ppg")
    SAVED_IMG_FOLDER = os.path.join(SAVED_FOLDER, "img")
    if not os.path.exists(ROOT_SAVED_FOLDER):
        os.makedirs(ROOT_SAVED_FOLDER)
    if not os.path.exists(SAVED_FOLDER):
        os.makedirs(SAVED_FOLDER)
    if not os.path.exists(SAVED_FILE_FOLDER):
        os.makedirs(SAVED_FILE_FOLDER)
    if not os.path.exists(SAVED_IMG_FOLDER):
        os.makedirs(SAVED_IMG_FOLDER)

    # file_cut = "24EI-013-PPG-day1-5"
    df_origin = pd.read_csv(os.path.join(DATA_PATH, filename + ".csv"))
    # df_small = pd.read_csv(os.path.join(DATA_PATH, file_cut + ".csv"), index_col=0)

    # signal_data = np.array(df_origin["PLETH"])
    # signal_hp = butter_highpass_filter(signal_data, cutoff=1, fs=100, order=1)
    # signal_hp = butter_highpass_filter(signal_hp, cutoff=1, fs=100, order=1)

    # USE THIS FOR FULL DATA
    # pivots = get_clipping_pivots(signal_hp)
    # template_list, template_mean = get_clipping(signal_hp, pivots, samples=-1)
    SAMPLE_RATE = 100
    MIN_CYCLE = 5
    # df = signal_hp[5 * 60 * SAMPLE_RATE:-(5 * 60 * SAMPLE_RATE)]
    df_origin = df_origin.iloc[5 * 60 * SAMPLE_RATE:-(5 * 60 * SAMPLE_RATE)]
    df_pleth = np.array(df_origin["PLETH"])
    # start_milestone, end_milestone = trim_missing(df_origin,True)
    start_milestone, end_milestone = trim_missing(df_pleth, False)
    # start_milestone, end_milestone = remove_short_length(start_milestone, end_milestone, MIN_CYCLE * SAMPLE_RATE)

    print(start_milestone)
    print(end_milestone)
    segments = []
    for start, end in zip(start_milestone, end_milestone):
        segment_seconds = 3000 #aka 5 mins
        chunk = df_origin.iloc[int(start):int(end)]

        # signal_data = np.array(chunk["PLETH"])
        signal_hp = np.array(chunk["PLETH"])
        signal_hp = butter_highpass_filter(signal_hp, cutoff=1, fs=100, order=1)
        signal_hp = butter_highpass_filter(signal_hp, cutoff=1, fs=100, order=1)
        # signal_hp = butter_lowpass_filter(signal_hp, cutoff=1, fs=100, order=1)
        # signal_hp = butter_highpass_filter(signal_hp, cutoff=1, fs=100, order=1)

        segments = segments+[signal_hp[segment_seconds*i:segment_seconds*(i+1)]
                    for i in range(0,int(np.ceil(len(signal_hp)/segment_seconds)))]
        # fig.add_traces(
        #     go.Scatter(
        #         x=df["PLETH"][int(start):int(end)].index,
        #         y=df["PLETH"][int(start):int(end)],
        #         mode="lines"
        #     ))

    # #TODO cut into 20s segments aka 300 instances
    # segment_seconds = 1000
    # segments = [signal_hp[segment_seconds*i:segment_seconds*(i+1)]
    #             for i in range(0,int(np.ceil(len(signal_hp)/segment_seconds)))]
    #
    # origin_segments = [signal_data[segment_seconds*i:segment_seconds*(i+1)]
    #             for i in range(0,int(np.ceil(len(signal_data)/segment_seconds)))]
    #
    # # segment_list = []
    # # for segment in segments:
    # #     pivots = get_clipping_pivots(segment)
    # #     template_list, template_mean = get_clipping(segment, pivots, samples=-1)
    # #     template_concat = join_templates(template_list)
    # #     segment_list.append(template_concat)
    #
    # # save_each_template(filename, segment_list[:100])
    # # save_each_template(filename, origin_segments[:100], origin=True)
    # # save_each_template(filename, segments[1000:1100], enhanced=True)
    # # save_each_template(filename, segments[1000:1100], enhanced=True)
    #
    save_each_template(filename, np.array(segments), display_trough_peak=True)
    # save_each_template(filename, np.array(origin_segments)[[1665]], display_trough_peak=True,origin=True)

