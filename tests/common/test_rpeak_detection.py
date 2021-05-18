import pytest
import numpy as np
from sklearn.cluster import KMeans
from scipy import signal

from vital_sqi.preprocess.band_filter import BandpassFilter
from vital_sqi.common.generate_template import ecg_dynamic_template
from vital_sqi.common.rpeak_detection import PeakDetector
import warnings
from ecgdetectors import Detectors,panPeakDetect

class TestPeakDetector(object):
    def test_on_init(self):
        detector = PeakDetector()
        pass

class TestECGDetector(object):
    def test_ecg_detector(self):
        detector = PeakDetector()
        pass

class TestPPGDetector(object):
    def test_on_ppg_detector(self):
        detector = PeakDetector()
        pass

    def test_on_matched_filter_detector(self):
        detector = PeakDetector()
        pass

    def test_on_compute_feature(self):
        detector = PeakDetector()
        pass

    def test_on_detect_peak_trough_clusterer(self):
        detector = PeakDetector()
        pass

    def test_on_detect_peak_trough_count_orig(self):
        detector = PeakDetector()
        pass

    def test_on_detect_peak_trough_slope_sum(self):
        detector = PeakDetector()
        pass

    def test_on_search_for_onset(self):
        detector = PeakDetector()
        pass

    def test_on_get_moving_average(self):
        detector = PeakDetector()
        pass

    def test_on_detect_peak_trough_billauer(self):
        detector = PeakDetector()
        pass