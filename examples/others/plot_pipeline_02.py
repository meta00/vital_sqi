"""
Pipeline - PPG (Smartcare)
==========================

This example shows the whole structure to compute the
signal quality indexes (SQI) given a .csv file with
the PPG/ECG data.

It works with full patient .csv file. The dtw function
is very slow and therefore it has not been included in
the example (improve).

.. warning:: The user needs to configure the path to
             load the Smartcare.csv file (see example
             below). This is because the data has not
             been included in the repository.

"""

#################################################################
# Libraries
# ----------

# Generic
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scipy
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy

# Heartpy
import heartpy as hp

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# vitalSQI
from vital_sqi.data.signal_io import ECG_reader
from vital_sqi.data.signal_io import PPG_reader
from vital_sqi.dataset import load_ppg, load_ecg


# ----------------------------
# Constant
# ----------------------------
# Constant to define whether we are using the terminal in
# order to plot the dataframes. Keep to false by default
# specially when building the documentation.
TERMINAL = True


#################################################################
# Load data (smartcare)
# ---------------------
#
# First, lets load the data from the smartcare csv file.
#
# Questions:
#    - Why COUNTER starts in 9?
#

# ----------------------------
# Load data
# ----------------------------
# Filepath
filepath = './data/'
filename = '01NVa-003-2001 Smartcare.csv'

# Load patient data
# .. note: To speed up the run for trial/error yuu
#          can include the number of rows to load
#          from the file as a parameter to the
#          read_csv function (e.g. nrows=100000)
signals = pd.read_csv(os.path.join(filepath, filename),
    nrows=200000)

# Calculate sampling rate
fs = 1 / (signals.TIMESTAMP_MS.diff().median() * 0.001)

# Datetime start
start_datetime = '2021-01-01 17:00:00'

# Show
print("\nLoaded signals:")
signals

if TERMINAL:
    print(signals)


#################################################################
# Format the data
# ---------------
#
# In this step, we format the signal so that the index is represented
# by timedelta (increment in time between consecutive samples) and in
# addition to the default channels (e.g. 0, 1) we include a column
# with an incremental index (idx). This index points to the original
# .csv file.

# ----------------------------
# Pandas
# ----------------------------
# Display (shows timedelta aligned)
pd.Timedelta.__str__ = lambda x: x._repr_base('all')

# ----------------------
# Format data
# ----------------------
# Include column with index
signals = signals.reset_index()

# .. note: We are assuming that the data signals index has been
#          recorded every fs no matter whether the patient moved,
#          the device was disconnected and connected again, ...
#
# .. note: In this particular example in which we have the
#          timestamps, the timedelta could be also created
#          directly from the time stamp (hence no need to
#          use signal.index / fs
# Create timedelta
#signals['timedelta'] = \
#    pd.to_timedelta(signals.index / fs, unit='s')

signals['timedelta'] = \
    pd.to_timedelta(signals.TIMESTAMP_MS, unit='ms')

# Create datetimes (if needed)
signals['date'] = pd.to_datetime(start_datetime)
signals['date']+= pd.to_timedelta(signals.timedelta)

# Set the timedelta index (keep numeric index too)
signals = signals.set_index('timedelta')

# Rename column index to avoid confusion
signals = signals.rename(columns={'index': 'idx'})

###############################################
# Lets see the dataframe

# Show
print("\nSignals:")
signals

if TERMINAL:
    print(signals)

###############################################
# Lets plot one signal. Note that you can use the slider
# (below the main series) to select a smaller section of the
# series to improve the visualization.

# Plot (matplotlib)
#fig, axes = plt.subplots(nrows=2, ncols=1)
#axes = axes.flatten()
#signals.set_index('date').PLETH.plot(ax=axes[0])
#signals.set_index('date').IR_ADC.plot(ax=axes[1])

# Plot (plotly)
# .. note: displaying all the data in an interactive
#          HTML might be a bit expensive. If running
#          locally you might want to use just
#          matplotlib.

# Create figure
fig = go.Figure(go.Scatter(
    x=signals.date,
    y=signals.IR_ADC,
    name='IR_ADC'))
fig.update_xaxes(rangeslider_visible=True)
# fig.show() # Uncomment if running locally

#################################################################
# Basic preprocessing
# -------------------
# This is more about the general preprocessing of the signal,
# if there are specific preprocessing steps to generate
# specific vital signals (e.g. hr) they will be implemented on
# their own in the next step.

####################################
# Lets trim the first/last 5 minutes

# -------------------------
# Trim first/last 5 minutes
# -------------------------
# Offset
offset = pd.Timedelta(minutes=5)

# Indexes
idxs = (signals.index > offset) & \
       (signals.index < signals.index[-1] - offset)

# Filter
signals = signals[idxs]

####################################
# Lets resample the data

# Implement!

####################################
# Lets imput missing data

# Implement!

####################################
# Lets do tappering??

# Implement!

####################################
# Lets show the preprocessed signals

# Show
print("\nPreprocessing:")
signals

if TERMINAL:
    print(signals)



#################################################################
# Let's compute other signals (vitals) from raw data
#
# For instance, some quality indexes might be derived from signals
# such as the hear rate (hr) computed by detecting the peaks on
# ecg signals. Thus, their own methods can be specified to generate
# the signal from the raw data.
def bbpf_rep(s):
    """Butter Band Pass Filter Scipy

    This is equivalent to the BandPassFilter() class implemented
    in the vital_sqi package. Need to understand the difference
    between filtfilt and lfilter.

    There is no need to compute the nyquist frequency. We can
    just pass the frequency sample and scipy will adjust the
    cutoff frequencies according to the nyquist theorem:

                f = f / 0.5*fs

    Note that scipy has also a bandpass filter (btype=band) but
    it is not possible to choose different orders for the high
    pass and low pass filters.
    """
    from scipy import signal
    # Configure high/low filters
    bh, ah = signal.butter(1, 1, fs=fs, btype='high', analog=False)
    bl, al = signal.butter(4, 20, fs=fs, btype='low', analog=False)
    # Apply filters
    aux = signal.filtfilt(bh, ah, s)
    aux = signal.lfilter(bl, al, aux)
    # Return
    return aux


# Add vital signals
signals['PLETH_BPF'] = bbpf_rep(signals.PLETH)

# Show
print("\nSignals (all):")
signals

if TERMINAL:
    print(signals)



#################################################################
# Compute SQIs
# ------------
#
# .. warning:: The method dtw is too slow.

##########################################
# Lets define our own SQI functions.
#
# .. note: This should be the real focus and strength of this
#          package, to have a series of sqi techniques very
#          easy to compute. Also it would be great if they can
#          be made compatible with pandas.
#
# .. note: I am creating a function so that the final column
#          has the name of the function instead of the long
#          name of the sqi function.
#
# .. note: The following sqi indexes were giving errors:
#
#     - ValueError: Can only compare identically-labeled Series object.
#         sq.standard_sqi.zero_crossings_rate_sqi(x)
#         sq.standard_sqi.mean_crossing_rate_sqi(x)

# Library
import vital_sqi.sqi as sq


def snr(x, axis=0, ddof=0):
    """Signal to noise ratio"""
    a = np.asanyarray(x)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def zcr(x):
    """Zero crossing rate"""
    return 0.5 * np.mean(np.abs(np.diff(np.sign(x))))

def mcr(x):
    """Mean crossing rate"""
    return zcr(x - np.mean(x))

def perfusion(x, y):
    """Perfusion

    Parameters
    ----------
    x: raw signal
    y: filtered signal
    """
    return (np.max(y) - np.min(y)) / np.abs(np.mean(x)) * 100

def correlogram(x):
    """Correlogram"""
    return sq.rpeaks_sqi.correlogram_sqi(x)

def msq(x):
    """Mean signal quality (is this the acronym?)

    Feedback for the package:
      The msq_sqi should also allow to receive two peak detector
      ids as input parameters. At the moment only allows a list
      of peaks and a peak detector id.
    """
    # Library
    from vital_sqi.common.rpeak_detection import PeakDetector
    # Detection of peaks
    detector = PeakDetector()
    peak_list, trough_list = detector.ppg_detector(x, 7)
    # Return
    return sq.standard_sqi.msq_sqi(x, peaks_1=peak_list, peak_detect2=6)

def dtw(x):
    """Dynamic time warping

    .. note: It is very slow!!

    Returns
    -------
    [mean, std]
    """
    # Library
    from vital_sqi.common.rpeak_detection import PeakDetector
    # Detection of peaks
    detector = PeakDetector()
    peak_list, trough_list = detector.ppg_detector(x, 7)
    # Per beats
    dtw_list = sq.standard_sqi.per_beat_sqi(\
        sqi_func=sq.dtw_sqi, troughs=trough_list,
        signal=x, taper=True, template_type=1
    )
    # Return mean
    return [np.mean(dtw_list),
            np.std(dtw_list)]


def sqi_all(x):
    """Compute all SQIs.

    .. note: If some variables are required for different
             SQIS (e.g. detecting the peaks), it might be
             possible to save some time by computing them
             first instead of having each method to compute
             them within the method. There is a trade off
             between the efficiency gained (depends on the
             implementation of the PeakDetectgor) and the
             the simplicity of the code and usage.
    """
    # Information
    dinfo = {
        'first': x.idx.iloc[0],
        'last': x.idx.iloc[-1],
        'skew': skew(x['PLETH']),
        'kurtosis': kurtosis(x['PLETH']),
        'snr': snr(x['PLETH']),
        'mcr': mcr(x['PLETH']),
        'zcr': zcr(x['PLETH_BPF']),
        'msq': msq(x['PLETH_BPF']),
        'perfusion': perfusion(x['PLETH'], x['PLETH_BPF']),
        'correlogram': correlogram(x['PLETH_BPF']),
        #'dtw': dtw(x['0_bbpf'])
    }

    # Return
    return pd.Series(dinfo)


##########################################
# Lets compute the SQIs
#
# In this section we will demonstrate two possible approaches
# using pandas to compute the signal quality indexes (SQIs)
# for each window. Each of the approaches have their own
# benefits and drawbacks.
#
# - The first approach uses the ``.agg`` function.
# - The second approach uses the ``.apply`` function.
#
# .. note:: It is important to highlight that not all the SQIs
#           are applied to the same columns. Some can be applied
#           to the raw signals whereas others have to be applied
#           to the filtered data.

#########################################
# First lets use the method agg

# ---------------------
# Compute SQIs
# ---------------------
# Group by 30s windows/aggregate
sqis = signals \
    .groupby(pd.Grouper(freq='30s')) \
    .agg({'idx': ['first', 'last'],
          'PLETH': [skew, kurtosis, snr, mcr],
          'IR_ADC': [skew, kurtosis, snr, mcr],
          'PLETH_BPF': [zcr, msq, correlogram] #dtw]
          })

# .. note: We are assuming that the whole signal has been
#          read in one chunk. This will not work if using
#          batches, will window ids be necessary?
# Add window id (if needed)
sqis['w'] = np.arange(sqis.shape[0])

# Show
print("\nSQIs (agg):")
sqis

if TERMINAL:
    print(sqis)


#########################################
# Now lets use apply to compare

# Group by 30s windows/apply
sqis2 = signals \
    .groupby(pd.Grouper(freq='30s')) \
    .apply(sqi_all)

# Show
print("\nSQIs (apply):")
sqis2

if TERMINAL:
    print(sqis2)




##########################################
# Select HQ windows
# -----------------------------
#
# Lets apply some signal quality rules. The
# aim of this part is to select those windows
# that are of good quality for further analysis.

# ---------------------
# Apply SQI Rules
# ---------------------

# Apply random rule
sqis['keep'] = np.random.choice(a=[False, True], size=(sqis.shape[0],))

# Create basic rule
criteria = list(zip(*[
    (sqis['PLETH']['skew'].between(-2, -1), True),
    (sqis['PLETH']['skew'].between(0, 1), True)
]))

# Apply rule (default False)
sqis['keep'] = np.select(criteria[0], criteria[1], False)

# Keep all
#sqis['keep'] = True

# Keep only valid
sqis = sqis[sqis.keep]

# Show
print("\nSQIs (valid):")
sqis

if TERMINAL:
    print(sqis)





#################################################################
# Lets go back to raw data
# ------------------------
#

#########################################################
# Lets extract the original signal only for those windows
# that are of good quality.

# -------------------------------------
# Extract windows from original signals
# -------------------------------------
# .. note: This might be fragile as it is not really using
#          the index but the position. Anyways, the index
#          column is just incremental isn't? Or am I missing
#          special conditions when this might not happen?
#
# .. note: Could it be done more efficiently?
#
# .. note: We could include the window ids if needed. This could
#          help linking the quality indexes stored in sqis.csv
#          and the valid sections of the signal stored in the
#          signals.csv file.
#
# Keep slices and concatenate
slices = [signals[signals.idx.between(start, stop)]
    for start, stop in zip(sqis['idx']['first'],
                           sqis['idx']['last'])]

# Concatenate only valid sections
result = pd.concat(slices)

# Show
print("\nSignals (for valid sqis)")
result

if TERMINAL:
    print(result)



#########################################################
# Lets plot the result
#

# Plot (matplotlib)
#fig, axes = plt.subplots(nrows=2, ncols=1)
#axes = axes.flatten()
#signals.set_index('date').PLETH.plot(ax=axes[0])
#signals.set_index('date').IR_ADC.plot(ax=axes[1])
#plt.show()

# Plot (plotly)
# Create figure
fig = go.Figure(go.Scatter(
    x=result.date,
    y=result.IR_ADC,
    name='IR_ADC'))
fig.update_xaxes(rangeslider_visible=True)
#fig.show() # Uncomment if running locally
