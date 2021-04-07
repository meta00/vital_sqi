import pytest
import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy import signal
from vital_sqi.preprocess.band_filter import BandpassFilter

class TestBandpassFilter:
    def test_on_init(self):
        band_filter = BandpassFilter()
        pass
class TestSignalBypass(object):
    def test_on_signal_bypass(self):
        band_filter = BandpassFilter()
        pass
class TestSignalLowpassFilter(object):
    def test_on_signal_lowpass_filter(self):
        band_filter = BandpassFilter()
        pass
class TestSignalHighpassFilter(object):
    def test_on_signal_highpass_filter(self):
        band_filter = BandpassFilter()
        pass

