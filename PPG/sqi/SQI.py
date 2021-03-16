#====================================================
#======= Apply ARIMA to calibrate the signal ========
#====================================================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from dtw import dtw
# import dtw as dtw_root
from scipy.stats import kurtosis,skew,entropy
import os

from statsmodels.tsa.stattools import adfuller
import sys
sys.path.append("..")
try:
    from ..utilities.generate_template import ppg_absolute_dual_skewness_template, \
        ppg_dual_doublde_frequency_template,ppg_nonlinear_dynamic_system_template
except:
    from utilities.generate_template import ppg_absolute_dual_skewness_template, \
        ppg_dual_doublde_frequency_template,ppg_nonlinear_dynamic_system_template
# from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
# from pandas.plotting import register_matplotlib_converters
from scipy import signal

#=====================================================================
#-----   LOAD DATA ---------------------------------------------------

# NORMAL_FILE = "D:\Workspace\Python\oucru\ECG\Work\data/20191230T111948.658+0000_clean.csv" #20191230T111948.658+0000_clean
# ABNORMAL_FILE_0 = "D:\Workspace\Python\oucru\ECG\Work/744469_abnormal.csv"
# ABNORMAL_FILE_1 = "D:\Workspace\Python\oucru\ECG\Work/742830_abnormal.csv"
# # NORMAL_FILE = "744620_normal.csv"
# # ABNORMAL_FILE_0 = "744469_abnormal.csv"
# # ABNORMAL_FILE_1 = "742830_abnormal.csv"
#
#
# # df_normal = (pd.read_csv(NORMAL_FILE,header=0)).iloc[:,0]
# df_normal = (pd.read_csv(NORMAL_FILE,header=0))["PLETH"].iloc[:]
# df_abnormal_0 = (pd.read_csv(ABNORMAL_FILE_0,header=0)).iloc[:,0]
# df_abnormal_1 = (pd.read_csv(ABNORMAL_FILE_1,header=0)).iloc[:,0]


def get_stationarity(timeseries):
    """
    Check the score of stationary condition
    :param timeseries:
    :return:
    """
    # rolling statistics
    rolling_mean = pd.DataFrame(timeseries).rolling(window=1000).mean()
    rolling_std = pd.DataFrame(timeseries).rolling(window=1000).std()

    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Dickey–Fuller test:
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# get_stationarity(df_normal)
#============================================
# Convert data into stationary series
#============================================
def log_stationary(df_input):
    """
    Transform the series to the log series to achieve stationary condition
    :param df_input:
    :return:
    """
    df_log = np.log(df_input)
    # decomposition = seasonal_decompose(df_log)
    df_normal_min = min(df_input)
    df_normal_shifting = 2*np.abs(df_normal_min) + df_input
    df_log = np.log(df_normal_shifting)
    return df_log
#
# # Compute DTW
# small_sample_1 = np.array(df_normal).reshape(-1,1)[:250]
# small_sample_2 = np.array(df_normal).reshape(-1,1)[250:500]
# # dtw_res = dtw(small_sample_1, small_sample_2,open_end=False)
# dtw_res = dtw(small_sample_1, small_sample_1, keep_internals=True,
#     step_pattern=dtw.rabinerJuangStepPattern(6, "c"))
# #dtw_res.plot(type="twoway",offset=-2)
# cost_matrix = dtw_res.costMatrix
# mean_diff = np.mean(cost_matrix[~np.isnan(cost_matrix)])
# # np.where(np.isnan(cost_matrix), np.ma.array(cost_matrix, mask=np.isnan(cost_matrix)).mean(axis=0), cost_matrix)
# cost_matrix = np.where(np.isnan(cost_matrix), 0, cost_matrix)
# # dtwPlotThreeWay

number_samples = 20
freq = 250 # Refers to the device  frequency

def get_template_by_chunk(input_data,number_samples=10,freq=250):
    """
    #=====================================================
    # CREATE TEMPLATE:
    # MEAN OF the fist 10 samples
    # SAMPLE SIZE ADJUST WITHT THE DEVICE FREQUENCY 250Hz
    #=====================================================
    :param input_data:
    :param number_samples:
    :param freq:
    :return:
    """
    template_list = []
    # template_dict = {}
    # template_gradient_list = []
    for i in range(number_samples):
        template_list.append(input_data[i*freq:(i+1)*freq])
        # template_dict[i] = (input_data[i * freq:(i + 1) * freq])
        # template_gradient = np.array(template_list[i][1:-1]) - np.array(template_list[i][0:-2])
        # template_gradient_list.append(template_gradient)
    return template_list

# # Return the template mean of the first 10 samples
# template_list = get_template_by_chunk(df_normal)
# template_mean = np.mean(template_list,axis=0)


def get_template_by_centroid(input_data,number_samples=10,freq=250):
    """
    #=====================================================
    # CREATE TEMPLATE:
    # 1) Find the peak from the given frequency 250Hz
    # 2) get the left span element using the min from first 10 samples
    # 3) get the right span element by using the median
    #=====================================================
    :param input_data:
    :param number_samples:
    :param freq:
    :return:
    """
    template_list = get_template_by_chunk(input_data,number_samples,freq)
    peak_idx = np.array(template_list).argmax(axis=1)
    template_list_adjust = []
    width = min(min(peak_idx),min(freq-peak_idx))
    width_left = np.min(peak_idx)
    width_right = int(np.median(freq-peak_idx))
    for i in range(number_samples):
        peak_idx_ = peak_idx[i]+i*freq
        template_list_adjust.append(input_data[peak_idx_-width_left:peak_idx_+width_right])
    return template_list_adjust,width_left,width_right

# template_list_adjust,width_left,width_right = get_template_by_centroid(df_normal,number_samples)
#
# template_mean_adj = np.mean(template_list_adjust,axis=0)
# #=================================================
# # TESTING THE PERFORMANCE OF DTW
# #=================================================
# test_sample_center = np.argmax(df_normal[3000:3250])
# test_sample = df_normal[test_sample_center-width_left:test_sample_center+width_right]
# for template_adj in template_list_adjust:
#     dtw_res = dtw(test_sample, template_adj, keep_internals=True,
#                   step_pattern=dtw_root.rabinerJuangStepPattern(6, "c"))
#     dtw_res.plot(type="twoway")
#     plt.show()

# dtw_res = dtw(test_sample,template_mean, keep_internals=True,
#     step_pattern=rabinerJuangStepPattern(6, "c"))
def perfusion_sqi(x,y,filter=True):
    """
    The perfusion index is the ratio of the pulsatile blood flow to the nonpulsatile
    or static blood in peripheral tissue.
    In other words, it is the difference of the amount of light absorbed through the pulse of
    when light is transmitted through the finger, which can be defined as follows:
    PSQI=[(ymax−ymin)/x¯|]×100
    where PSQI is the perfusion index, x¯ is the statistical mean of the x signal (raw PPG signal),
    and y is the filtered PPG signal
    :param x: mean of the raw signal
    :param y: array of filter signnal
    :return:
    """
    if filter:
        return (np.max(y)-np.min(y))/np.abs(x)*100

def kurtosis_sqi(x,axis=0, fisher=True, bias=True, nan_policy='propagate'):
    """
    Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution.
    That is, data sets with high kurtosis tend to have heavy tails, or outliers.
    Data sets with low kurtosis tend to have light tails, or lack of outliers.
    A uniform distribution would be the extreme case.

    Kurtosis is a statistical measure used to describe the distribution of observed data around the mean.
    It represents a heavy tail and peakedness or a light tail and flatness of a distribution
    relative to the normal distribution, which is defined as:

    :param x:
    :return:
    """
    # mean_x = np.mean(x)
    # std_x = np.std(x)
    # return np.mean(((x-mean_x)/std_x)**4)
    return kurtosis(x,axis, fisher, bias, nan_policy)

def skewness_sqi(x,axis=0, bias=True, nan_policy='propagate'):
    """
    Skewness is a measure of symmetry, or more precisely, the lack of symmetry.
    A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.

    Skewness is a measure of the symmetry (or the lack of it) of a probability distribution, which is defined as
    SSQI=1/N∑i=1N[xi−μˆx/σ]3
    where μˆx and σ are the empirical estimate of the mean and standard deviation of xi,
    respectively; and N is the number of samples in the PPG signal.
    :param x:
    :return:
    """
    # mean_x = np.mean(x)
    # std_x = np.std(x)
    # return np.mean(((x-mean_x)/std_x)**3)
    return skew(x,axis, bias, nan_policy)

def entropy_sqi(x,qk=None, base=None, axis=0):
    x_ = x - min(x)
    return entropy(x_,qk,base,axis)

def signal_to_noise_sqi(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def systolic_wave_matching_sqi(x):
    #TODO
    return

def relative_power_sqi(x):
    #TODO
    return

def zero_crossing_rate(x):
    zero_crosses = np.nonzero(np.diff(x > 0))[0]
    return zero_crosses

def mean_crossing_rate(x):
    zero_crosses = np.nonzero(np.diff(x > 0))[0]
    #TODO
    return zero_crosses

from librosa  import feature
# feature.zero_crossing_rate()

def zero_crossings_rate_sqi(y, threshold=1e-10, ref_magnitude=None, pad=True, zero_pos=True, axis=-1):

    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        y = y.copy()
        y[np.abs(y) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        y_sign = np.signbit(y)
    else:
        y_sign = np.sign(y)

    # Find the change-points by slicing
    slice_pre = [slice(None)] * y.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [slice(None)] * y.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * y.ndim
    padding[axis] = (1, 0)

    crossings =  np.pad(
        (y_sign[tuple(slice_post)] != y_sign[tuple(slice_pre)]),
        padding,
        mode="constant",
        constant_values=pad,
    )

    # return crossings, np.mean(crossings, axis=0, keepdims=True)
    return np.mean(crossings, axis=0, keepdims=True)[0]

def dtw_sqi(x,template_type=0):
    if template_type == 0:
        reference = ppg_nonlinear_dynamic_system_template(len(x)).reshape(-1)
    elif template_type == 1:
        reference = ppg_dual_doublde_frequency_template(len(x))
    elif template_type == 2:
        reference = ppg_absolute_dual_skewness_template(len(x))
    alignmentOBE = dtw(x, reference,keep_internals=True,step_pattern='asymmetric',
        open_end=True, open_begin=True)

    match_distance = []
    for i in range(len(alignmentOBE.index2)):
        match_distance.append(alignmentOBE.costMatrix[i][alignmentOBE.index2[i]])
    trace_ratio = np.sum(match_distance)/alignmentOBE.costMatrix.trace()

    return trace_ratio