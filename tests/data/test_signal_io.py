import tempfile
import pytest

from vital_sqi.data.signal_io import *


class TestECGReader(object):

    def test_on_filename(self):
        with pytest.raises(AssertionError) as exc_info:
            file_type = 'sth'
            ECG_reader('path_to_file', file_type=file_type)
            assert exc_info.match('File not found')

    def test_on_valid_edf(self):
        file_name = os.path.abspath('tests/test_data/example.edf')
        assert isinstance(ECG_reader(file_name, 'edf'), SignalSQI) is True

    def test_on_valid_mit(self):
        file_name = os.path.abspath('tests/test_data/a103l')
        assert isinstance(ECG_reader(file_name, 'mit'), SignalSQI) is True

    def test_on_float_sampling_rate(self):
        file_name = os.path.abspath('tests/test_data/example.edf')
        out = ECG_reader(file_name, 'edf', sampling_rate = 120.8)
        assert out.sampling_rate == 121

    def test_on_start_datetime(self):
        file_name = os.path.abspath('tests/test_data/example.edf')
        out = ECG_reader(file_name, 'edf', sampling_rate = 120.8,
                         start_datetime = '2020/12/12 10:10:00')
        assert isinstance(out.start_datetime, dt.datetime) is True

    def test_on_KeyError(self):
        # lines 70-74, 78-83, 101-105
        pass

    def test_on_date_mit(self):
        # lines 111 - 127 find another mit file with date
        pass

    def test_on_valid_csv(self):
        file_name = os.path.abspath('tests/test_data/ecg_test1.csv')
        assert isinstance(ECG_reader(file_name, 'csv',
                                     channel_name = ['Time', '1']), SignalSQI)
        assert isinstance(ECG_reader(file_name, 'csv',
                                     channel_num = [0, 1]), SignalSQI)

    def test_on_csv_infer_sampling_rate(self):
        file_name = os.path.abspath('tests/test_data/ecg_test_w.csv')
        out = ECG_reader(file_name, 'csv', channel_name = ['Time', '1'])
        assert out.sampling_rate == 256
        file_name = os.path.abspath('tests/test_data/ecg_test2.csv')
        with pytest.raises(Exception) as exc_info:
            out = ECG_reader(file_name, 'csv', channel_name = ['Time', '1'])
        assert exc_info.match("Sampling rate not found nor inferred")


class TestECGWriter(object):

    def test_on_valid_edf(self):
        file_in = os.path.abspath('tests/test_data/example.edf')
        out = ECG_reader(file_in, 'edf')
        file_out = tempfile.gettempdir() + '/out.edf'
        assert ECG_writer(out, file_out, file_type='edf', info=out.info) is \
               True
        assert ECG_writer(out, file_out, file_type='edf', info=None)\
               is True

    def test_on_valid_mit(self):
        file_in = os.path.abspath('tests/test_data/a103l')
        out = ECG_reader(file_in, 'mit')
        file_out = tempfile.gettempdir() + '/out_mit'
        assert ECG_writer(out, file_out, file_type='mit', info=out.info) is \
               not None
        with pytest.raises(Exception) as exc_info:
            ECG_writer(out, file_out, file_type='mit', info=None)
        assert exc_info.match("Header dict needed")

    def test_on_valid_csv(self):
        file_in = os.path.abspath('tests/test_data/ecg_test1.csv')
        out = ECG_reader(file_in, 'csv', channel_name = ['Time', '1'])
        file_out = tempfile.gettempdir() + '/ecg_test_write.csv'
        assert ECG_writer(out, file_out, 'csv') is True


class TestPPGReader(object):
    file_name = os.path.abspath('tests/test_data/ppg_smartcare.csv')

    def test_on_valid_ppg(self):
        file_name = os.path.abspath('tests/test_data/ppg_smartcare.csv')
        assert isinstance(PPG_reader(file_name, signal_idx=['PLETH'],
                   timestamp_idx=['TIMESTAMP_MS'],
                   info_idx=['PULSE_BPM', 'SPO2_PCT', 'PERFUSION_INDEX'],
                          sampling_rate=100,
                          start_datetime='2020/12/30 10:00:00'), SignalSQI) is\
               True
        assert isinstance(PPG_reader(file_name, signal_idx = ['PLETH'],
                                     timestamp_idx = ['TIMESTAMP_MS'],
                                     info_idx = ['PULSE_BPM', 'SPO2_PCT',
                                                 'PERFUSION_INDEX']),
                          SignalSQI) is \
               True

    def test_on_timeunit_error(self):
        file_name = os.path.abspath('tests/test_data/ppg_smartcare.csv')
        with pytest.raises(Exception) as exc_info:
            PPG_reader(file_name,
                        signal_idx=['PLETH'],
                        timestamp_idx=['TIMESTAMP_MS'],
                        info_idx=['PULSE_BPM', 'SPO2_PCT', 'PERFUSION_INDEX'],
                        timestamp_unit=None)
        assert exc_info.match("Missing sampling_rate, not able to infer "
                              "sampling_rate without timestamp_unit")
        with pytest.raises(Exception) as exc_info:
            PPG_reader(file_name,
                       signal_idx=['PLETH'],
                       timestamp_idx=['TIMESTAMP_MS'],
                       info_idx=['PULSE_BPM', 'SPO2_PCT', 'PERFUSION_INDEX'],
                       timestamp_unit='abc')
        assert exc_info.match("Timestamp unit must be either second")


class TestPPGWriter(object):

    def test_on_valid_ppg(self):
        file_in = os.path.abspath('tests/test_data/ppg_smartcare.csv')
        out = PPG_reader(file_in, signal_idx = ['PLETH'],
                                     timestamp_idx = ['TIMESTAMP_MS'],
                                     info_idx = ['PULSE_BPM', 'SPO2_PCT',
                                                 'PERFUSION_INDEX'],
                                     sampling_rate = 100,
                                     start_datetime = '2020/12/30 10:00:00')
        file_out = tempfile.gettempdir() + '/ppg_test_write.csv'
        assert PPG_writer(out, file_out, 'csv') is True
        file_out = tempfile.gettempdir() + '/ppg_test_write.xlsx'
        assert PPG_writer(out, file_out, 'xlsx') is True
