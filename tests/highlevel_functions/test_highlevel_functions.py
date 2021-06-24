import pytest
import os
import numpy as np
from vital_sqi.data.signal_io import PPG_reader
from vital_sqi.highlevel_functions.highlevel import \
    compute_all_SQI,compute_multiple_SQIs,get_clean_signals,\
    get_cutpoints,basic_ecg_pipeline,basic_ppg_pipeline,signal_preprocess

class TestSignalPreProcess(object):
    def test_on_signal_preprocess(self):
        file_name = os.path.abspath('tests/test_data/ppg_smartcare.csv')
        mock_data = PPG_reader(file_name, signal_idx=['PLETH'],
                         timestamp_idx=['TIMESTAMP_MS'],
                         info_idx=['PULSE_BPM', 'SPO2_PCT',
                                   'PERFUSION_INDEX'],
                         sampling_rate=100,
                         start_datetime='2020/12/30 10:00:00')
        mock_signal = mock_data.signals
        with pytest.raises(Exception) as exc_info:
            signal_preprocess()
        assert exc_info.match("No signal provided")
        out_data = signal_preprocess(mock_signal)
        assert len(mock_signal)==len(out_data)

class TestComputeAllSQI(object):
    def test_on_compute_all_SQI(self):
        file_name = os.path.abspath('tests/test_data/ppg_smartcare.csv')
        mock_data = PPG_reader(file_name, signal_idx=['PLETH'],
                               timestamp_idx=['TIMESTAMP_MS'],
                               info_idx=['PULSE_BPM', 'SPO2_PCT',
                                         'PERFUSION_INDEX'],
                               sampling_rate=100,
                               start_datetime='2020/12/30 10:00:00')
        mock_signal = mock_data.signals[0]
        with pytest.raises(Exception) as exc_info:
            compute_all_SQI(signal=None,segments=[],raw_signal=[])
        assert exc_info.match("Signal or segments not provided, can't extract all SQI's")
        with pytest.raises(Exception) as exc_info:
            compute_all_SQI(signal=[],segments=None,raw_signal=[])
        assert exc_info.match("Signal or segments not provided, can't extract all SQI's")
        with pytest.raises(Exception) as exc_info:
            compute_all_SQI(signal=[],segments=[],raw_signal=None)
        assert exc_info.match("Signal or segments not provided, can't extract all SQI's")
        mock_segment_length = 1000
        mock_segment = np.arange(np.floor(len(mock_signal)/mock_segment_length))\
                       *mock_segment_length
        mock_segment = [int(idx) for idx in mock_segment]
        mock_sqi_list = compute_all_SQI(signal=mock_signal,segments=mock_segment,
                                        raw_signal=mock_signal)
        assert len(mock_sqi_list) > 0
        mock_sqi_dict = mock_sqi_list[0]
        assert type(mock_sqi_dict) is dict
        assert len(mock_sqi_dict.keys()) > 0
        for key in mock_sqi_dict.keys():
            assert mock_sqi_dict[key] is not None

class TestComputeMultipleSQIs(object):
    def test_on_compute_multiple_SQIs(self):
        pass

class TestMakeRuleSet(object):
    def test_on_make_rule_set(self):
        pass

class TestGetCutPoints(object):
    def test_on_get_cutpoints(self):
        pass

class TestGetCleanSignals(object):
    def test_on_get_clean_signals(self):
        pass

class TestBasicPpgPipeline(object):
    def test_on_basic_ppg_pipeline(self):
        pass

class TestBasicEcgPipeline(object):
    def test_on_basic_ecg_pipeline(self):
        pass
