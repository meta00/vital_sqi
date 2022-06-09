import pytest
import os
from vital_sqi.data.signal_io import PPG_reader
from vital_sqi.highlevel_functions.highlevel_functions import \
    compute_all_SQI, compute_multiple_SQIs,get_clean_signals, get_cutpoints,\
    basic_ecg_pipeline, basic_ppg_pipeline

class TestGetAllFeaturesHeartpy(object):
    def test_on_get_all_features_heartpy(self):
        pass


class TestCalculateSQI(object):
    def test_on_calculate_SQI(self):
        pass


class TestPerBeatSQI(object):
    def test_on_per_beat_sqi(self):
        pass


class TestGetSqiDict(object):
    def test_on_get_sqi_dict(self):
        pass


class TestGetSqi(object):
    def test_on_get_sqi(self):
        pass


class TestSegmentPPGSQIExtraction(object):
    def test_on_segment_PPG_SQI_extraction(self):
        pass


class TestComputeSQI(object):
    def test_on_compute_sqi(self):
        pass


class TestGetDecision(object):
    def test_on_get_decision(self):
        pass