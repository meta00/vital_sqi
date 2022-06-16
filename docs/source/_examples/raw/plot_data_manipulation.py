"""
Data manipulation
====================

.. note:: This is a copy of the jupyter notebook with the
          following name: Data_manipulation_ECG_PPG.ipynb.
          The other option is to use the sphinx extension
          sphinx-nbexamples.

"""

#%%

from vital_sqi.data.signal_io import ECG_reader,PPG_reader
import os
file_name = os.path.abspath('../../tests/test_data/example.edf')
ecg_data = ECG_reader(file_name, 'edf')
# ecg_data = load_ecg()
file_name = os.path.abspath('../../tests/test_data/ppg_smartcare.csv')
ppg_data = PPG_reader(file_name,
                    signal_idx=['PLETH'],
                    timestamp_idx=['TIMESTAMP_MS'],
                    info_idx=['PULSE_BPM', 'SPO2_PCT', 'PERFUSION_INDEX'],
                    sampling_rate=100,
                    start_datetime='2020/12/30 10:00:00')
# ppg_data = load_ppg()

#%%

all_channels = ecg_data.signals
channel_1 = all_channels.iloc[:, 1]

#%% md

### Example of splitting the whole data into subsegment using time domain for ECG.

#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

print(len(channel_1))

#%% md

#The whole channel length will be splitted into each 30-second segment

#%%

from vital_sqi.preprocess.segment_split import split_segment

#%%

save_file_name = "example_file"
save_file_folder = "subsegments_time"
split_segment(channel_1,
                  filename=None,
                  sampling_rate=256,
                  segment_length_second=10.0,
                  wave_type=ecg_data.wave_type,
                  split_type="time",
                  save_file_folder=save_file_folder)

#%% md

#The function requires the sampling rate and the defined length (in seconds) of the split segment to calculate the cutting points. User also defined a location to save the output of cut files**

#The split_to_subsegments output the saved segment at the defined save folder. Save files takes the format of "[file_name]-[segment_number].csv"

#%%

print(os.listdir("subsegments_time/ecg/"))

#%%

segment_51 = np.loadtxt("subsegments_time/ecg/segment-051.csv")
segment_52 = np.loadtxt("subsegments_time/ecg/segment-052.csv")

#%%

#Uncomment the plotly code to use interactive plot

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(segment_51)),
#                          y= segment_51,
#                          name='segment 51'))
# fig.add_trace(go.Scatter(x=np.arange(len(segment_51),
#                                      len(segment_51)+len(segment_52)),
#                          y= segment_52,
#                          name='segment 52'))
# fig.show()


fig = plt.Figure()
plt.plot(np.arange(len(segment_51)),
         segment_51)
plt.plot(np.arange(len(segment_51),len(segment_51)+len(segment_52)),
         segment_52)
plt.show()

#%% md

### Example of splitting the whole data into subsegment using time domain for PPG.

#%%

save_file_name = "example_file"
save_file_folder = "subsegments_time"
if not os.path.exists(save_file_folder):
    os.makedirs(save_file_folder)
split_to_segments(ppg_data.signals.iloc[:, 1],
                  filename=None,
                  sampling_rate=100,
                  segment_length_second=10.0,
                  wave_type=ppg_data.wave_type,
                  split_type="time",
                  save_file_folder=save_file_folder)

#%%
ppg_folder = os.path.join(save_file_folder, "ppg")
file_list = os.listdir(ppg_folder)
print(file_list)

#%%
segment_1 = np.loadtxt(os.path.join(ppg_folder, file_list[0]))
segment_2 = np.loadtxt(os.path.join(ppg_folder, file_list[0]))

#%%

# Uncomment the plotly code to use interactive plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(segment_1)),
#                          y= segment_1,
#                          name='segment 1'))
# fig.add_trace(go.Scatter(x=np.arange(len(segment_1),
#                                      len(segment_1)+len(segment_2)),
#                          y= segment_2,
#                          name='segment 2'))
# fig.show()

fig = plt.Figure()
plt.plot(np.arange(len(segment_1)),
         segment_1)
plt.plot(np.arange(len(segment_1),len(segment_1)+len(segment_2)),
         segment_2)
plt.show()

#%% md

### Example of splitting the whole data into subsegment using frequency domian for ECG.

### Notes on the difference of splitting point as comparing with time domain splitting. Uncomment the plotly code - interactive plot - for better observation

#%%

save_file_name = "example_file"
save_file_folder = "subsegments_frequency"
split_to_segments(channel_1,
                  filename=None,
                  sampling_rate=256,
                  segment_length_second=10.0,
                  split_type="peak_interval",
                  save_file_folder=save_file_folder)

#%%

segment_51 = np.loadtxt("subsegments_frequency/ecg/segment-051.csv")
segment_52 = np.loadtxt("subsegments_frequency/ecg/segment-052.csv")

#%%

# Uncomment the plotly code to use interactive plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(segment_51)),
#                          y= segment_51,
#                          name='segment 51'))
# fig.add_trace(go.Scatter(x=np.arange(len(segment_51),
#                                      len(segment_51)+len(segment_52)),
#                          y= segment_52,
#                          name='segment 52'))
# fig.show()

fig = plt.Figure()
plt.plot(np.arange(len(segment_51)),
         segment_51)
plt.plot(np.arange(len(segment_51),len(segment_51)+len(segment_52)),
         segment_52)
plt.show()

#%% md

### Example of splitting the whole data into subsegment using frequency domian for PPG.

#%%

save_file_name = "example_file"
save_file_folder = "subsegments_frequency"
split_to_segments(ppg_data.signals.iloc[:, 0],
                  filename=None,
                  sampling_rate=256,
                  segment_length_second=10.0,
                  wave_type=ppg_data.wave_type,
                  split_type="peak_interval",
                  save_file_folder=save_file_folder)

#%%

segment_1 = np.loadtxt("subsegments_frequency/ppg/segment-01.csv")
segment_2 = np.loadtxt("subsegments_frequency/ppg/segment-02.csv")

#%%

# Uncomment to use interactive plot
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(segment_1)),
#                          y= segment_1,
#                          name='segment 1'))
# fig.add_trace(go.Scatter(x=np.arange(len(segment_1),
#                                      len(segment_1)+len(segment_2)),
#                          y= segment_2,
#                          name='segment 2'))
# fig.show()

fig = plt.Figure()
plt.plot(np.arange(len(segment_1)),
         segment_1)
plt.plot(np.arange(len(segment_1),len(segment_1)+len(segment_2)),
         segment_2)
plt.show()

#%% md

### Example of trimming the first and the last n-minute data.

#%%

from vital_sqi.data import trim_data

#%%

trimmed_data = trim_data(channel_1,minute_remove=10)

#%%

trimmed_data

#%%

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(channel_1)),
#                          y= channel_1,
#                          name='full data'))
# fig.show()

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(trimmed_data)),
#                          y= trimmed_data,
#                          name='trimmed data'))
# fig.show()

fig = plt.Figure()
plt.plot(np.arange(len(channel_1)),
         channel_1)
plt.show()
plt.plot(np.arange(len(trimmed_data)),
         trimmed_data, color=u'#ff7f0e')
plt.show()

#%% md

#The before and after trimming 5 minutes segment**

#%%

trimmed_data_ppg = trim_data(ppg_data.signals.iloc[:, 1], minute_remove=1)

#%%

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(ppg_data.signals)),
#                          y= ppg_data.signals,
#                          name='full data'))
# fig.show()

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(trimmed_data_ppg)),
#                          y= trimmed_data_ppg,
#                          name='trimmed data'))
# fig.show()

fig = plt.Figure()
plt.plot(np.arange(len(ppg_data.signals.iloc[:, 1])),
         ppg_data.signals.iloc[:, 1])
plt.show()
plt.plot(np.arange(len(trimmed_data_ppg)),
         trimmed_data_ppg,
         color=u'#ff7f0e')
plt.show()

#%% md

### Example of before and after removing the unchanged value of the n-continuous second.

#%%

from vital_sqi.data.removal_utilities import remove_unchanged_squences

#%%

# create a series of unchanged value in the trimmed_data list
idx = np.random.randint(int(len(trimmed_data)/2))
sampling_rate = 256
unchanged_data = trimmed_data.copy()
unchanged_data[idx:idx+sampling_rate*20] = max(trimmed_data)

#%%

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(len(unchanged_data)),
#                          y= unchanged_data,
#                          name='trimmed data'))
# fig.show()

fig = plt.Figure()
plt.plot(np.arange(len(unchanged_data)),
         unchanged_data)
plt.show()

#%%

start_list, end_list = \
    remove_unchanged_squences(unchanged_data,
                              unchanged_seconds=10,
                              sampling_rate=256,
                              as_dataframe=False)

#%%

# fig = go.Figure()
# for start,end in zip(start_list,end_list):
#     fig.add_trace(go.Scatter(x=np.arange(start,end),
#                              y= unchanged_data[start:end],
#                              name='trimmed data'))
# fig.show()

fig = plt.Figure()
for start, end in zip(start_list, end_list):
    plt.plot(np.arange(start, end),
         unchanged_data[start:end])
plt.show()

#%% md

### Example of removing invalid signal data (signal = 0 and other vital signs exceed the normal range)

#%%

from vital_sqi.data import remove_invalid

#%%

trimmed_data = trim_data(channel_1, minute_remove=10)
error_data = trimmed_data.copy()

#%%

# create a series of unchanged value in the trimmed_data list
idx = np.random.randint(int(len(error_data)/2))
sampling_rate = 256
error_data[idx:idx+sampling_rate*20] = 0

#%%

start_list, end_list = remove_invalid(error_data,as_dataframe=False)

#%%

# fig = go.Figure()
# for start,end in zip(start_list,end_list):
#     fig.add_trace(go.Scatter(x= np.arange(start,end),
#                              y= trimmed_data[start:end],
#                              name='trimmed data'))
# fig.show()
fig = plt.Figure()
for start,end in zip(start_list,end_list):
    plt.plot(np.arange(start,end),
         trimmed_data[start:end])
plt.show()

#%% md

### One example of removing invalid signal data using the frequency domain

#%%

from vital_sqi.data import cut_by_frequency_partition

#%%

start_list, end_list = \
    cut_by_frequency_partition(trimmed_data,
                              window_size=30000,
                              peak_threshold_ratio=4,
                              lower_bound_threshold=2)

#%% md

#Welch method is applied for the whole data to obtain the common frequency component.**

#*After that, a scanning window with the size of 3000 samples is computed for each subsegment to analyse its frequency component.**
#Any windows having its component exceeds the peak_threshold_ratio (the ratio between the number of subsegment's components and the number of whole data components) will be removed.**

#%%

# fig = go.Figure()
# for start,end in zip(start_list,end_list):
#     fig.add_trace(go.Scatter(x= np.arange(start,end),
#                              y= trimmed_data[start:end],
#                              name='trimmed data'))
# fig.show()

fig = plt.Figure()
plt.plot(np.arange(len(trimmed_data)),
         trimmed_data)
plt.show()


fig = plt.Figure()
for start,end in zip(start_list,end_list):
    plt.plot(np.arange(start,end),
         trimmed_data[start:end])
plt.show()

#%%

out = PPG_reader(os.path.join(os.getcwd(),'../../', 'tests/test_data/ppg_smartcare.csv'),
                 timestamp_idx = ['TIMESTAMP_MS'], signal_idx = ['PLETH'],
                 info_idx = ['PULSE_BPM','SPO2_PCT','PERFUSION_INDEX'])

#%%

start_list, end_list = \
    cut_by_frequency_partition(ppg_data.signals.iloc[:, 1],
                              window_size=30000,
                              peak_threshold_ratio=2,
                              lower_bound_threshold=2)

#%%

# fig = go.Figure()
# for start,end in zip(start_list,end_list):
#     fig.add_trace(go.Scatter(x= np.arange(start,end),
#                              y= ppg_data.signals[start:end],
#                              name='trimmed data'))
# fig.show()
fig = plt.Figure()
plt.plot(np.arange(len(ppg_data.signals.iloc[:, 1])),
         ppg_data.signals.iloc[:, 1])
plt.show()

# .. note gives an error
#fig = plt.Figure()
#for start,end in zip(start_list,end_list):
#    plt.plot(np.arange(start,end),
#         ppg_data.signals[start:end])
#plt.show()
