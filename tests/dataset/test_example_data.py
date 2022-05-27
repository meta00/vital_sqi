import pytest
from vital_sqi.dataset import load_ppg, load_ecg
from vital_sqi.data.signal_io import SignalSQI


class TestLoadECG(object):
    def test_on_load_ecg(self):
        mock_data = load_ecg()
        assert type(mock_data) is SignalSQI


class TestLoadPPG(object):
    def test_on_load_ppg(self):
        mock_data = load_ppg()
        assert type(mock_data) is SignalSQI
