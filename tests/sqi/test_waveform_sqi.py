import pytest
from vital_sqi.sqi.waveform_sqi import *
from vital_sqi.data.signal_io import ECG_reader
import os


class TestBandEnergySqi(object):
	file_name = os.path.abspath('tests/test_data/example.edf')
	out = ECG_reader(file_name, 'edf')

	def test_on_valid(self):
		out = band_energy_sqi(signal=self.out.signals.iloc[:, 1],
							sampling_rate=self.out.sampling_rate,
							band=[0, 0.5])
		assert isinstance(out, float)
		out = band_energy_sqi(signal=self.out.signals.iloc[:, 1],
							sampling_rate=self.out.sampling_rate,
							band=[0, 0.5])
		assert isinstance(out, float)

	def test_on_band(self):
		with pytest.raises(AssertionError) as exc_info:
			out = band_energy_sqi(signal=self.out.signals.iloc[:, 1],
								sampling_rate=self.out.sampling_rate,
								band='[0, 0.5]')
		assert exc_info.match("Invalid band values")
		with pytest.raises(AssertionError) as exc_info:
			out = band_energy_sqi(signal=self.out.signals.iloc[:, 1],
								sampling_rate=self.out.sampling_rate,
								band=[0.5, 0])
		assert exc_info.match("Invalid band values")

	def test_on_sampling_rate(self):
		with pytest.raises(AssertionError) as exc_info:
			out = band_energy_sqi(signal=self.out.signals.iloc[:, 1],
								sampling_rate='',
								band=[0.5, 0])
		assert exc_info.match("Invalid sampling rate value")

