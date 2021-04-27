import pytest


class TestECGReader(object):
    def test_on_something(self):
        pass


class TestECGWriter(object):
    def test_on_something(self):
        pass


class TestPPGReader(object):
    def test_on_something(self):
        pass


class TestPPGWriter(object):
    def test_on_something(self):
        pass
# out = ECG_reader('/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests'
#             '/test_data/ecg_test_w.csv', 'csv', sampling_rate = 100)
# ECG_writer(out, '/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests'
#             '/test_data/ecg_test_w.csv', 'csv')

# out = PPG_reader('/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests'
#             '/test_data/ppg_smartcare.csv', timestamp_idx = [
#     'TIMESTAMP_MS'], signal_idx = ['PLETH'], info_idx = ['PULSE_BPM',
#                                                          'SPO2_PCT','PERFUSION_INDEX'],
#                  start_datetime = '2020-04-12 10:00:00')
# PPG_writer(out, '/Users/haihb/Documents/Work/Oucru/innovation/vital_sqi/tests'
#             '/test_data/ppg_smartcare_w.csv')