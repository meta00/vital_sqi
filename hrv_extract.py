import os
import pandas as pd
import vital_sqi as vs


files = [os.path.join('/Users/haihb/Documents/innovation/01NVa/Dengue/data/set_9_shock/',
					  _) for _ in
		 os.listdir(
	'/Users/haihb/Documents/innovation/01NVa/Dengue/data/set_9_shock/') if _.endswith('asc')]
print(files)
dat = []
segments = []
for i in range(0, 1): # len(files)):
	print(files[i])
	dat.append(pd.read_csv(files[i], skiprows=1, delimiter='\t', usecols=[
		'Time', 'ECG1'], nrows=90000))
	# print(len(dat[i]))
	# timestamps = []
	# for k in range(0, len(dat[i])):
	# 	timestamps.append(pd.Timestamp(dat[i].iloc[:, 0][k], unit='s'))
	# del dat[i]['Time']
	# dat[i].insert(0, 'timestamps', timestamps)
	# if isinstance(dat[i]['timestamps'][0], pd.Timestamp) is True:
	# 	print('aaa')

	segments, milestones = vs.preprocess.split_segment(dat[i], sampling_rate=300,
										split_type=0, duration=300,
										wave_type='ecg',
										overlapping=0)

	feat = []
	for m in range(0, len(segments)):
		time_domain_features, frequency_domain_features, \
		geometrical_features, csi_cvi_features = vs.sqi.get_all_features_hrva(
			segments[m], sample_rate=300, rpeak_method=7, wave_type='ecg')
		feat.append([time_domain_features, frequency_domain_features,
					 geometrical_features, csi_cvi_features])

print(len(segments))
