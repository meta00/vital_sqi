[![Build Status](https://travis-ci.com/meta00/vital_sqi.svg?token=CDjcmJqzLe7opuWagsPJ&branch=main)](https://travis-ci.com/meta00/vital_sqi)
[![codecov](https://codecov.io/gh/meta00/vital_sqi/branch/main/graph/badge.svg?token=6RV5BUK340)](https://codecov.io/gh/meta00/vital_sqi)
# vital_sqi: A python package for signal quality control of physiological signals

# Description
- SQI indexes.
- QC pipeline for ECG and PPG.
## PPG
1. Data format
   
2. Preprocessing

2.1. Trimming: First and last 5 minutes of each recording are trimmed as they usually contain unstable signals. Wearables need some time to pick up signals.

2.2. Noise removal: The following is considered noise, thus removed. The recording is then split into files.
   - PLETH column: 0 or unchanged values for xxx time
   - Invalid values: SpO2 < 80 and Pulse > 200 or Pulse < 40
   - Perfusion < 0.2
   - Lost connection: sampling rate reduced due to (possible) Bluetooth connection lost. Timestamp column shows missing timepoints. If the missing duration is larger than 1 cycle (xxx ms), recording is split. If not, missing timepoints are interpolated.

3. Filtering

3.1. Bandpass filter: High pass filter (cut off at 1Hz)

3.2. Detrend

4. Split Data 

4.1. Cut data by time domain. Split data into sub segments of 30 seconds

4.2. Apply the peak and trough detection methods in peak_approaches.py 
to get single PPG cycles in each segment 

4.3. Shift baseline above 0 and 
tapering each single PPG cycle to compute the mean template

Notes: the described process is implemented in split_to_segments.py

5. SQI scores

Most of the statistical SQI is adopted from the paper
<i>Optimal Signal Quality Index for Photoplethysmogram Signals, Mohamed Elgendi</i>

The indices are classified into statistical domain (skewness, kurtosis,...) 
and signal processing domain (entropy, zero crossing rate, mean crossing rate). 
Besides these two domains, we used the dynamic time warping with different templates 
as another domain. 

The PPG templates are generated in <i>generate_template.py</i>

The mean template is compared with each generated templated. According to DTW techniques,
a list of key mapping is projected. The DTW score compute the ratio from the mean distance 
of the mapping points and the trace of the difference matrix.

### utilities package
1) Filtering

1.1. Bandpass filtering: includes high-pass and low-pass filtering. Usage: remove any additive signal frequency below or above the cutoff threshold 

1.2. Tapering: Pin the leftmost and rightmost signal to the zero baseline
    and amplify the remainder according to the window shape (ex: hann, cosine,...)

1.3.  Scale pattern: spanning or squeeze the selected sequence to the chosen window size. 
Use to equalized the single complex/cycle of PPG

1.4. Smooth: use a convolution windows to smooth the scale or squeeze signal

2) Generate template

Generate a synthetic data of PPG and ECG

2.1.  ppg_dual_doublde_frequency_template

Generate a PPG template by using 2 sine waveforms.
    The first waveform double the second waveform frequency
    
2.2.  ppg_absolute_dual_skewness_template

Generate a PPG template by using 2 skewness distribution. Return a 1-D numpy array of PPG waveform
    having diastolic peak at the high position
    
2.3.  ppg_nonlinear_dynamic_system_template

Generate a PPG template from the paper 
<i>An Advanced Bio-Inspired PhotoPlethysmoGraphy (PPG) and ECG Pattern Recognition System for Medical Assessment</i>

2.4. ecg_dynamic_template

Generate a ECG synthetic template from the paper
<i>A dynamical model for generating synthetic electrocardiogram signals</i>

3) peak_approaches

Contains a list of PPG peak detection methods as described in 
<i>Systolic Peak Detection in Acceleration Photoplethysmograms Measured
    from Emergency Responders in Tropical Conditions</i>

3.1. detect_peak_trough_kmean

Using clustering technique to separate the list of systolic and diastolic peak

3.2. detect_peak_trough_count_orig

using local extreme technique with threshold

3.3. detect_peak_trough_slope_sum

analyze the slope sum to get local extreme

3.4. detect_peak_trough_moving_average_threshold

examine second derivative and the average line to determine threshold

### sqi packages

SQI.py includes list of all available SQI scores as described in the paper
<i>Elgendi, Mohamed, Optimal signal quality index for photoplethysmogram signals, Bioengineering,</i>

There are 3 types of SQI: statistical domain, signal processing domain & DTW domain
- statistical domain: kurtosis_sqi, skewness_sqi, entropy_sqi
- signal processing domain: zero_crossings_rate_sqi, signal_to_noise_sqi, mean_crossing_rate_sqi
- DTW domain: dtw_sqi

## ECG

1. Data format

2. Preprocessing

2.1. Trimming: First and last 5 minutes of each recording are trimmed as they usually contain unstable signals. Wearables need some time to pick up signals.

2.2. Noise removal: The following is considered noise, thus removed. The recording is then split into files.

2.3.

# Installation
```
pip install vital_sqi
```
# Contributing
# License

