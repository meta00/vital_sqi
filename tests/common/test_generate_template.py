import pytest
import numpy as np

from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import scipy
from scipy.signal import argrelextrema
import plotly.io as pio
from scipy.integrate import solve_ivp
from vital_sqi.preprocess.preprocess_signal import squeeze_template

class TestPPGDualDoubleFrequencyTemplate(object):
    def test_on_ppg_dual_double_frequency_template(self):
        pass
class TestSkewFunc(object):
    def test_on_skew_func(self):
        pass
class TestPPGAbsoluteDualSkewnessTemplate(object):
    def test_on_ppg_absolute_dual_skewness_template(self):
        pass
class TestPpgNonlinearDynamicSystemTemplate(object):
    def test_on_ppg_nonlinear_dynamic_system_template(self):
        pass
class TestInterp(object):
    def test_on_interp(self):
        pass

class TestECGDynamicTemplate(object):
    def test_on_ecg_dynamic_template(self):
        pass
class TestOrdinaryDifferentialEquation(object):
    def test_on_ordinary_differential_equation(self):
        pass
class TestRRProcess(object):
    def test_on_rr_process(self):
        pass