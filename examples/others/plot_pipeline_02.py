"""
Exploiting pandas!!
=====================

This example shows the whole structure to compute the
signal qality indexes (SQI) given a .csv file with
the PPG/ECG data.

See more notes at the end.
"""

#################################################################
# Load data
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

# vitalSQI
from vital_sqi.data.signal_io import ECG_reader
from vital_sqi.dataset import load_ppg, load_ecg


# ----------------------------
# Constant
# ----------------------------
# Constant to define whether we are using the terminal in
# order to plot the dataframes. Keep to false by default
# specially when building the documentation.
TERMINAL = True


# ----------------------------
# Load data
# ----------------------------
# Filepath
# filepath = '../../tests/test_data'
# filename = 'example.edf'

# Load patiet data
# data = ECG_reader(os.path.join(filepath, filename), 'edf')

# Load sample dataset
data = load_ecg()

# .. note: Loading the data using load_ppg() does not work.
#          It might be because the methods are returning
#          the data in different formats.
# data = load_ppg()

# The attributes!
print(data)
print(data.signals)
print(data.sampling_rate)
print(data.start_datetime)
print(data.wave_type)
print(data.sqis)
print(data.info)

# Set sample frequency
fs = data.sampling_rate

#################################################################
# Formatting
# ----------
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
# Load DataFrame
signals = pd.DataFrame(data.signals)

signals = pd.concat([signals]*5, ignore_index=True)

# Include column with index
signals = signals.reset_index()

# .. note: We are assuming that the data signals index has been
#          recorded every fs no matter whether the patient moved,
#          the device was disconnected and connected again, ...
# Create timedelta
signals['timedelta'] = \
    pd.to_timedelta(signals.index / fs, unit='s')

# Create datetimes (if needed)
#signals['date'] = pd.to_datetime(data.start_datetime)
#signals['date']+= pd.to_timedelta(signals.timedelta)

# Set the timedelta index (keep numeric index too)
signals = signals.set_index('timedelta')

# Rename column index to avoid confusion
signals = signals.rename(columns={'index': 'idx'})

###############################################
# Lets see the display and plot the raw signals

# Show
print("\nSignals:")
signals

if TERMINAL:
    print(signals)

# Plot
fig, axes = plt.subplots(nrows=2, ncols=1)
axes = axes.flatten()
signals[0].plot(ax=axes[0])
signals[1].plot(ax=axes[1])




#################################################################
# Preprocessing
# -------------
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

def hr(s):
    """Heart rate signal (random int)"""
    return np.random.randint(low=40, high=150, size=s.shape)

def rr(s):
    """Respiratory rate signal (random int)"""
    return np.random.randint(low=15, high=25, size=s.shape)

def bbpf(s):
    """Butter Band Pass Filter from vital_sqi"""
    from vital_sqi.preprocess.band_filter import BandpassFilter
    f = BandpassFilter(band_type='butter', fs=fs)
    aux = f.signal_highpass_filter(s, cutoff=1, order=1)
    aux = f.signal_lowpass_filter(aux, cutoff=20, order=4)
    return aux

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
signals['0_hr'] = hr(signals[0])
signals['0_rr'] = rr(signals[0])
signals['0_bbpf'] = bbpf(signals[0])
signals['0_bbpf_rep'] = bbpf_rep(signals[0])

# Show
print("\nSignals (all):")
signals

if TERMINAL:
    print(signals)



#################################################################
# Compute SQIs
# ------------

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

def own(x):
    """Own defined SQI (randint)"""
    return np.random.randint(100)

def own2(x):
    """Own defined SQI (fixed array)"""
    return [1,2]

def snr(x, axis=0, ddof=0):
    """Signal to noise ratio"""
    a = np.asanyarray(x)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
    #return np.mean(sq.standard_sqi.signal_to_noise_sqi(x))

def zcr(x):
    """Zero crossing rate"""
    return 0.5 * np.mean(np.abs(np.diff(np.sign(x))))
    #return sq.standard_sqi.zero_crossings_rate_sqi(x)

def mcr(x):
    """Mean crossing rate"""
    return zcr(x - np.mean(x))
    #return sq.standard_sqi.mean_crossing_rate_sqi(x)

def perfusion(x, y):
    """Perfusion

    Parameters
    ----------
    x: raw signal
    y: filtered signal
    """
    return (np.max(y) - np.min(y)) / np.abs(np.mean(x)) * 100
    #sq.standard_sqi.perfusion_sqi(y=x['0_bbpf'], x=x[0])

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


def all(x):
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
        'skew': skew(x[0]),
        'kurtosis': kurtosis(x[0]),
        'snr': snr(x[0]),
        'mcr': mcr(x[0]),
        'own': own(x[0]),
        'zcr': zcr(x['0_bbpf']),
        'msq': msq(x['0_bbpf']),
        'perfusion': perfusion(x[0], x['0_bbpf']),
        'correlogram': correlogram(x['0_bbpf']),
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

# ---------------------
# Compute SQIs
# ---------------------
# Group by 30s windows/aggregate
sqis = signals \
    .groupby(pd.Grouper(freq='30s')) \
    .agg({'idx': ['first', 'last'],
          0: [skew, kurtosis, snr, mcr, own, own2],
          1: [skew, kurtosis, snr, mcr],
          '0_bbpf': [zcr, msq, correlogram] #dtw]
          })

# Group by 30s windows/apply
sqis2 = signals \
    .groupby(pd.Grouper(freq='30s')) \
    .apply(all)

# Flatten correlogram vector to columns
sqis2[['corr_peak_%s'%i for i in range(6)]] = \
    sqis2.correlogram.tolist()

# Flatten dtw vector to columns
#sqis2[['dtw_mean', 'dtw_std']] = \
#    sqis2.dtw.tolist()

# Remove columns
# .. note: There is no need to remove the columns with
#          the vectors now that we now how easy it is
#          to 'expand' them.
#sqis2 = sqis2.drop(columns=['correlogram', 'dtw'])

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

# Show
print("\nSQIs (apply):")
sqis2

if TERMINAL:
    print(sqis2)




##########################################
# Lets apply some signal quality rules

# ---------------------
# Apply SQI Rules
# ---------------------

# Apply random rule
sqis['keep'] = np.random.choice(a=[False, True], size=(sqis.shape[0],))

# Create basic rule
criteria = list(zip(*[
    (sqis[0]['skew'].between(-2.9, -2.6), True),
    (sqis[0]['skew'].between(4, 5), True)
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
# Lets extract the valid windows from the original signal

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
slices = [signals.iloc[start:stop, :] for start, stop
    in zip(sqis['idx']['first'],  sqis['idx']['last'])]

# Concatenate only valid sections
result = pd.concat(slices)

# Show
print("\nSignals (for valid sqis)")
result

if TERMINAL:
    print(sqis)


#########################################################
# Lets plot the result

# Create figure
fig, axes = plt.subplots(nrows=2, ncols=1)
axes = axes.flatten()

# Plot
result[0].plot(ax=axes[0])
result[1].plot(ax=axes[1])

# Adjust layout
plt.tight_layout()




#################################################################
# Further analysis (other tutorial)
# ---------------------------------
#
# Now that we have selected those sections in which the signal
# quality is appropriate. We can do further analysis, we can
# find the peaks to identify the heart rate, we can describe
# the windows statistically, ....
#
#
#

# Show
plt.show()



#
#.. warning:: Should we use TimeInterval indexes for windows?
#
#.. warning:: Generalising rules:
#
#             https://stackoverflow.com/questions/50098025/mapping-ranges-of-values-in-pandas-dataframe
#
#.. warning:: This is a very basic example and might fail when using
#             the reading in batches function from pandas. In such
#             scenario, consider using a map reduce approach, which
#             should not require many changes anyways.#
#
#             https://pythonspeed.com/articles/chunking-pandas/
#
#.. warning:: Useful to filter periods in which value is constant,
#             maybe due to lost of connection or something similar.#
#
#             https://stackoverflow.com/questions/55271735/pandas-finding-start-end-values-of-consecutive-indexes-in-a-pandas-dataframe
#             https://stackoverflow.com/questions/62361446/python-dataframe-get-index-start-and-end-of-successive-values
#