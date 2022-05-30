"""
PPG pipeline
====================

.. warning:: Include code to show how to create a pipeline to
             clean an ecg signal. It is probably in one of the
             jupyter notebooks (maybe ecg_qc.ipynb).

"""


###################################
# Load the data
# -------------
#

# First, lets load the data (ppg sample data)

# Libraries
import pandas as pd

# Load data
#data = pd.read_csv(path)


###################################
# Preprocessing
# -------------
#
# Wearable devices need some time to pick up stable signals. For this reason,
# it is a common practice to trim the data. In the following example, the f
# first and last 5 minutes of each recording are trimmed to exclude unstable
# signals.

# Trim data
#data = trim(data, start=5, end=5)

###################################
# Now, lets remove the following noise:
#
#   - ``PLETH`` is 0 or unchanged values for xxx time
#   - ``SpO2`` < 80
#   - ``Pulse`` > 200 bpm or ``Pulse`` < 40 bpm
#   - ``Perfusion`` < 0.2
#   - ``Lost connection``: sampling rate reduced due to (possible) Bluetooth connection
#     lost. Timestamp column shows missing timepoints. If the missing duration is
#     larger than 1 cycle (xxx ms), recording is split. If not, missing timepoints
#     are interpolated.

# Remove invalid PLETH
#idxs_1 = data.PLETH == 0
#idxs_2 = unchanged(period=xxxx)
#data = data[~(idxs_1 | idxs_2)]

# Remove invalid ranges
#data = data[data.SpO2>=80]
#data = data[data.Pulse.between(40, 200)]
#data = data[data.Perfusion>=0.2]

# Remove lost connection
#data = data[lost_connection(min_fs, max_fs) ??

# The recording is then split into files.

#######################################
# Lets filter the data with a band pass filter; high pass filter (cut off at 1Hz)

#######################################
# Lets detrend the signal

# Lets split the data
#4.1. Cut data by time domain. Split data into sub segments of 30 seconds
#4.2. Apply the peak and trough detection methods in peak_approaches.py to get single PPG cycles in each segment
#4.3. Shift baseline above 0 and tapering each single PPG cycle to compute the mean template
#Notes: the described process is implemented in split_to_segments.py

#######################################
# SQI scores
# ----------
#
# Lets compute the SQI scores

#######################################
# Visualization
# -------------