"""
Exploiting pandas!!
=====================

This example....

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
filepath = '../../tests/test_data'
filename = 'example.edf'

# Load
data = ECG_reader(os.path.join(filepath, filename), 'edf')

# The attributes!
print(data)
print(data.signals)
print(data.sampling_rate)
print(data.start_datetime)
print(data.wave_type)
print(data.sqi_indexes)
print(data.info)

fs = data.sampling_rate

#################################################################
# Formatting
# ----------


# ----------------------------
# Pandas
# ----------------------------
# Questions:
# Could we exploit pandas?
# Will it have any limitation?

# Display (shows timedelta aligned)
pd.Timedelta.__str__ = lambda x: x._repr_base('all')

# ----------------------
# Format data
# ----------------------
# Load DataFrame
signals = pd.DataFrame(data.signals)

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

# Rename column to avoid confusion
signals = signals.rename(columns={'index': 'idx'})

# Show
print("\nSignals:")
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
print(signals)



#################################################################
# Let's compute other signals (vitals) from raw data
#
# For instance, some quality index might be derived from signals
# such as the hear rate (hr) based on ecg signals by detection
# of peaks. Thus, their own methods can be specified to generate
# the signal from the raw data.

def hr(s):
    return np.random.randint(low=40, high=150, size=s.shape)

def rr(s):
    return np.random.randint(low=15, high=25, size=s.shape)

def bbpf(s):
    """Butter Band Pass Filter"""
    from vital_sqi.preprocess.band_filter import BandpassFilter
    f = BandpassFilter(band_type='butter', fs=fs)
    aux = f.signal_highpass_filter(s, cutoff=1, order=1)
    aux = f.signal_lowpass_filter(s, cutoff=20, order=4)
    return aux

# Add vital signals
signals['0_hr'] = hr(signals[0])
signals['0_rr'] = rr(signals[0])
signals['0_bbpf'] = bbpf(signals[0])




#################################################################
# Compute SQIs
# ------------

##########################################
# Lets first see how the windows look like

# Implement!


##########################################
# Lets define our own SQI function.
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
#
#     - NameError: name PeakDetector is not defined.
#         sq.standard_sqi.msq_sqi(x)
#
#     - AttributeError: module 'vital_sqi.sqi' has no attribute 'rpeaks_sqi'
#         sq.rpeaks_sqi.correlogram_sqi(x)
#
#        .. note: Implementation issue
#         manually handled exception: name 'isscalar' is not defined warning.
#
#    - AttributeError: module 'vital_sqi.sqi.standard_sqi' has no attribute 'per_beat_sqi'

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

def msq(x):
    """Mean SQ"""
    return sq.standard_sqi.msq_sqi(x)

def corr(x):
    """Correlogram"""
    a = sq.rpeaks_sqi.correlogram_sqi(x)

def dtw(x):
    """

    .. note: Implementation issue
        manually handled exception: name 'isscalar' is not defined warning.

    .. note: Implementation issue
        AttributeError: module 'vital_sqi.sqi.standard_sqi' has no attribute 'per_beat_sqi'
    """
    from vital_sqi.common.rpeak_detection import PeakDetector
    detector = PeakDetector()
    peak_list, trough_list = \
        detector.ppg_detector(x, 7)
    dtw_list = sq.standard_sqi.per_beat_sqi(\
        sqi_func=sq.dtw_si, troughs=trough_list,
        signal=x, taper=True, template_type=1
    )
    return np.mean(dtw_list)


# .. note: What if it is a complex SQI that requires first
#          to compute the peaks and then apply some numpy
#          functions?

# from vital_sqi.sqi.standard_sqi import msq_sqi

# The msq_sqi uses a PeakDetector (although at the moment
# it is missing the library so it breaks). When included,
# it raises a weird warning but returns a value.

##########################################
# Lets compute the SQIs

# ---------------------
# Compute SQIs
# ---------------------
# Group by 30s segments
sqis = signals \
    .groupby(pd.Grouper(freq='30s')) \
    .agg({'idx': ['first', 'last'],
          0: [skew, kurtosis, snr, mcr],
          1: [skew, kurtosis, snr, mcr, own, own2],
          '0_bbpf': [zcr, msq], #dtw, corr]
          })


# .. note: We are assuming that the whole signal has been
#          read in one chunk. This will not work if using
#          batches, will window ids be necessary?
# Add window id (if needed)
sqis['w'] = np.arange(sqis.shape[0])

# Show
print("\nSQIs (all):")
sqis

if TERMINAL:
    print(sqis)

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