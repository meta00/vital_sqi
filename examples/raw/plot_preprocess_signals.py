"""
Preprocessing signal
====================

"""

#%%

import vital_sqi
from vital_sqi.data.signal_io import ECG_reader, PPG_reader
import os

#%% md

### Read an EDF file using ECG_reader
#The function returns an Signal SQI object and a Dictionary contains the information of the file settings**

#%%

# file_name = "example_edf.edf"
# file_name = "../../tests/test_data/example.edf"
# ecg_data = ECG_reader(file_name,'edf')
from vital_sqi.dataset import load_ecg,load_ppg
ecg_data = load_ecg()
#%%

ecg_data.info

#%% md

### List all of the attributes in Signal_SQI object
#1. signals: a numpy array contains the raw amplitude values of the devices
#2. sampling_rate: the sampling rate derives from the signal data
#3. wave_type: the types of signal. Only 2 types are accepted: either 'ecg' or 'ppg'
#4. sqi_indexes: a list of sqi_indexes

#%%

ecg_data.__dict__

#%%

all_channels = ecg_data.signals

#%%

channel_1 = all_channels[:,0]
ecg_sample_idx = int(len(all_channels)/2)

#%%

ecg_sample_complex_for_tapering = \
    channel_1[ecg_sample_idx-45:ecg_sample_idx+210]
ecg_sample_complex = channel_1[ecg_sample_idx+80:ecg_sample_idx+225]

#%%

import plotly.graph_objects as go
import numpy as np

#%% md

### We will focus on 1 QRS-complex to examine the function

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_complex))
#fig.show()
fig

#%%

# file_name = "ppg_smartcare.csv"
# ppg_data = PPG_reader(os.path.join("../../tests/test_data",file_name),
#                       signal_idx=['PLETH'],
#                       timestamp_idx= ['TIMESTAMP_MS'],
#                       info_idx=['SPO2_PCT','PULSE_BPM','PERFUSION_INDEX'])
ppg_data = load_ppg()

#%%

ppg_data.__dict__

#%%

ppg_sample_idx = int(len(ppg_data.signals)/2)

#%%

ppg_sample_complex_for_tapering = \
    ppg_data.signals[0][ppg_sample_idx+185:ppg_sample_idx+225]
ppg_sample_complex = ppg_data.signals[0][ppg_sample_idx+195:ppg_sample_idx+267]

#%% md

### Subsequently, we also focus on 1 PPG waveform to examine the function

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex))
#fig.show()
fig

#%% md

## Examples on preprocessing function
#import the preprocess package

#%%

from vital_sqi import preprocess

#%% md

### Taper data into the zerobaseline to remove the edge effect

#%%

ecg_sample_tapering_zerobaseline = preprocess.tapering(ecg_sample_complex,
                                                       shift_min_to_zero=True)
ppg_sample_tapering_zerobaseline = preprocess.tapering(ppg_sample_complex,
                                                       shift_min_to_zero=True)

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_complex,
                         name='original_signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_tapering_zerobaseline,
                         name='tapered signal'))
#fig.show(title='ecg tapering')
fig
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex,
                         name='original_signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_tapering_zerobaseline)),
                         y= ppg_sample_tapering_zerobaseline,
                         name='tapered signal'))
#fig.show()
fig

#%% md

### The tapering data will pin the first and last part at the zero pivot. The remaining will be scale according to the windows format

#The default tapering method shifts the segment by the value equal to the minimum value to the zero baseline set shift_min_to_zero=False**

#%%

ecg_sample_tapering_zerobaseline = \
    preprocess.tapering(ecg_sample_complex,shift_min_to_zero=False)
ppg_sample_tapering_zerobaseline = \
    preprocess.tapering(ppg_sample_complex,shift_min_to_zero=False)

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_complex,
                         name='original_signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_tapering_zerobaseline,
                         name='tapered signal'))
#fig.show(title='ECG')
fig
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex,
                         name='original_signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_tapering_zerobaseline,
                         name='tapered signal'))
#fig.show(itle='PPG')
fig
#%% md

### Different windows format can be used to perform tapering process
#window is imported from the scipy package (scipy.signal.window). Default is using Tukey window**


#%%

import scipy.signal.windows as wd

#%% md

#Initialize a hann windows and cast it as a list-type.**

#%%

window_ecg = list(wd.hann(len(ecg_sample_complex)))
window_ppg = list(wd.hann(len(ppg_sample_complex)))

#%%

ecg_sample_tapering_hann = preprocess.tapering(ecg_sample_complex,
                                           window=window_ecg,
                                           shift_min_to_zero=False)
ppg_sample_tapering_hann = preprocess.tapering(ppg_sample_complex,
                                           window=window_ppg,
                                           shift_min_to_zero=False)

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_complex,
                         name='original signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_tapering_zerobaseline,
                         name='tukey window tapering'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_tapering_hann,
                         name='hann window tapering'))

fig.update_layout(
    title='ECG'
)
fig

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex,
                         name='original signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_tapering_zerobaseline,
                         name='tukey window tapering'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_tapering_hann,
                         name='hann window tapering'))
fig.update_layout(
    title='PPG'
)

#%% md

### Example of smoothing function.
#Apply a convolutional window to smooth the signal (the default windows is flat and can be assigned with different distribution)**

#%%

ecg_sample_smoothing_5 = preprocess.smooth(ecg_sample_complex)
ecg_sample_smoothing_9 = preprocess.smooth(ecg_sample_complex,window_len=9)

ppg_sample_smoothing_5 = preprocess.smooth(ppg_sample_complex)
ppg_sample_smoothing_9 = preprocess.smooth(ppg_sample_complex,window_len=9)

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_complex,
                         name='original_signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_smoothing_5,
                         name='smoothing - sliding window length = 5'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_smoothing_9,
                         name = 'smoothing - sliding window length = 9'))
#fig.show()
fig

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex,
                         name='original_signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_smoothing_5,
                         name='smoothing - sliding window length = 5'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_smoothing_9,
                         name = 'smoothing - sliding window length = 9'))
#fig.show()
fig

#%% md

### Example of squeezing function
#We will use the default resampling function from scipy package**

#%%

from scipy import signal

#%%

ecg_sample_squeezing = signal.resample(ecg_sample_complex,
                                       int(len(ecg_sample_complex)/2))
ppg_sample_squeezing = signal.resample(ppg_sample_complex,
                                       int(len(ppg_sample_complex)/2))

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_complex))
#fig.show(title='original data')
fig

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_squeezing)),
                         y= ecg_sample_squeezing))
#fig.show(title='squeezed data')
fig

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex,))
#fig.show(title='original data')
fig
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_squeezing)),
                         y= ppg_sample_squeezing))
#fig.show(title='squeezed data')
fig

#%% md

### Example of expanding function

#%%

ecg_sample_expanding = signal.resample(ecg_sample_squeezing,
                                       int(len(ecg_sample_squeezing)*2))
ppg_sample_expanding = signal.resample(ppg_sample_squeezing,
                                       int(len(ppg_sample_squeezing)*2))

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex,
                         name="original data"))
#fig.show(title='original data')
fig
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_expanding)),
                         y= ppg_sample_expanding,
                         name="the expanded data from the squeezed data"))
#fig.show()
fig

#%% md

# Example with bandpass filter

#%%

from vital_sqi.preprocess.band_filter import BandpassFilter

#%%

butter_bandpass = BandpassFilter("butter",fs=256)
cheby_bandpass = BandpassFilter("cheby1",fs=256)
ellip_bandpass = BandpassFilter("ellip",fs=256)

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_complex,
                         name='original signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= butter_bandpass.signal_highpass_filter(
    ecg_sample_complex,cutoff=1,order=5),
                        name='highpass filtered - cutoff 1Hz'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= butter_bandpass.signal_highpass_filter(
    ecg_sample_complex,cutoff=0.8,order=5),
                        name='highpass filtered - cutoff 0.8Hz'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= butter_bandpass.signal_highpass_filter(
    ecg_sample_complex,cutoff=0.6,order=5),
                        name='highpass filtered - cutoff 0.6Hz'))
#fig.show()
fig

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex,
                         name='original signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= butter_bandpass.signal_highpass_filter(
    ppg_sample_complex,cutoff=1,order=5),
                        name='highpass filtered - cutoff 1Hz'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= butter_bandpass.signal_highpass_filter(
    ppg_sample_complex,cutoff=0.8,order=5),
                        name='highpass filtered - cutoff 0.8Hz'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= butter_bandpass.signal_highpass_filter(
    ppg_sample_complex,cutoff=0.6,order=5),
                        name='highpass filtered - cutoff 0.6Hz'))
#fig.show()
fig

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ecg_sample_complex,name='original signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= butter_bandpass.signal_highpass_filter(
    ecg_sample_complex,cutoff=1,order=5)
                         ,name='butterworth highpass'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= cheby_bandpass.signal_highpass_filter(
    ecg_sample_complex,cutoff=1,order=5)
                        ,name='chebyshev highpass'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ellip_bandpass.signal_highpass_filter(
    ecg_sample_complex,cutoff=1,order=5)
                        ,name='elliptic highpass'))
#fig.show()
fig

#%%

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= ppg_sample_complex,name='original signal'))
fig.add_trace(go.Scatter(x=np.arange(len(ppg_sample_complex)),
                         y= butter_bandpass.signal_highpass_filter(
    ppg_sample_complex,cutoff=1,order=5)
                         ,name='butterworth highpass'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= cheby_bandpass.signal_highpass_filter(
    ppg_sample_complex,cutoff=1,order=5)
                        ,name='chebyshev highpass'))
fig.add_trace(go.Scatter(x=np.arange(len(ecg_sample_complex)),
                         y= ellip_bandpass.signal_highpass_filter(
    ppg_sample_complex,cutoff=1,order=5)
                        ,name='elliptic highpass'))
#fig.show()
fig