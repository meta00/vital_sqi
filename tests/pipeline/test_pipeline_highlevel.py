import tempfile
import pytest

from vital_sqi.pipeline.pipeline_highlevel import get_ppg_sqis, get_ecg_sqis,\
											get_qualified_ppg, get_qualified_ecg
from vital_sqi.data.signal_sqi_class import SignalSQI
import os


class TestGetPPGSQIs(object):
	def test_on_get_ppg_sqis(self):
		file_in = os.path.abspath('tests/data/ppg_smartcare.csv')
		sqi_dict = os.path.abspath('tests/data/sqi_dict.json')
		segments, signal_sqi_obj = get_ppg_sqis(file_in,
												timestamp_idx=['TIMESTAMP_MS'],
												signal_idx=['PLETH'],
												sqi_dict_filename=sqi_dict)
		assert isinstance(segments, list) is True
		assert isinstance(signal_sqi_obj, SignalSQI) is True
		assert signal_sqi_obj.sqis is not None


class TestGetQualifiedPPG(object):
	def test_on_get_qualified_ppg(self):
		file_in = os.path.abspath('tests/data/ppg_smartcare.csv')
		sqi_dict = os.path.abspath('tests/data/sqi_dict.json')
		rule_dict_filename = os.path.abspath(
			'tests/data/rule_dict_test.json')
		ruleset_order = {3: 'skewness_sqi',
						2: 'kurtosis_sqi',
						1: 'perfusion_sqi'}
		timestamp_idx = ['TIMESTAMP_MS']
		signal_idx = ['PLETH']
		output_dir = tempfile.gettempdir()
		signal_obj = get_qualified_ppg(file_in, sqi_dict_filename=sqi_dict,
									signal_idx=signal_idx,
									timestamp_idx=timestamp_idx,
									rule_dict_filename=rule_dict_filename,
									ruleset_order=ruleset_order,
									output_dir=output_dir,
									save_image=True)
		assert isinstance(signal_obj, SignalSQI) is True
		assert os.path.isdir(os.path.join(output_dir, 'accept', 'img')) is True
		assert os.path.isdir(os.path.join(output_dir, 'reject', 'img')) is True


class TestGetECGSQIs(object):
	def test_on_get_ecg_sqis(self):
		file_in = os.path.abspath('tests/data/example.edf')
		sqi_dict = os.path.abspath('tests/data/sqi_dict.json')
		segments, signal_sqi_obj = get_ecg_sqis(file_in, sqi_dict, 'edf')
		assert isinstance(segments, list) is True
		assert isinstance(signal_sqi_obj, SignalSQI) is True
		assert signal_sqi_obj.sqis is not None


class TestGetQualifiedSQIs(object):
	def test_on_get_qualified_ecg(self):
		file_in = os.path.abspath('tests/data/example.edf')
		sqi_dict = os.path.abspath('tests/data/sqi_dict.json')
		rule_dict_filename = os.path.abspath(
			'tests/data/rule_dict_test.json')
		ruleset_order = {3: 'skewness_sqi', 2: 'kurtosis_sqi',
						 1: 'perfusion_sqi'}
		output_dir = tempfile.gettempdir()
		signal_obj = get_qualified_ecg(file_name=file_in,
									sqi_dict_filename=sqi_dict,
									file_type='edf', duration=30,
									rule_dict_filename=rule_dict_filename,
									ruleset_order=ruleset_order,
									output_dir=output_dir)
		assert isinstance(signal_obj, SignalSQI) is True
		assert os.path.isdir(os.path.join(output_dir, 'accept', 'img')) is True
		assert os.path.isdir(os.path.join(output_dir, 'reject', 'img')) is True

# file_name = "../../tests/data/ppg_smartcare.csv"
		# json_rule_file_name = "../resource/rule_dict.json"
		# with open(json_rule_file_name) as rule_file:
		#     json_rule_dict = json.loads(rule_file.read())
		# rule_set_order={
		#     2:'sdsd_sqi',
		#     1:'sdnn_sqi'
		# }

		# get_qualified_ppg(file_name,timestamp_idx = ['TIMESTAMP_MS'],
		#                                     signal_idx = ['PLETH'],
		#                                     sqi_dict=None,
		#                                     rule_dict=json_rule_dict,
		#                                     ruleset_order = rule_set_order
		#                                     )
		#                                     # info_idx = ['PULSE_BPM','SPO2_PCT','PERFUSION_INDEX'])


# file_in = os.path.abspath('../../tests/data/example.edf')
# sqi_dict = os.path.abspath('../../tests/data/sqi_dict.json')
# segments, signal_sqi_obj = get_ecg_sqis(file_in, sqi_dict, 'edf')

# import tempfile
# file_in = os.path.abspath('../../tests/data/example.edf')
# sqi_dict = os.path.abspath('../../tests/data/sqi_dict.json')
# rule_dict_filename = os.path.abspath(
#     '../../tests/data/rule_dict_test.json')
# ruleset_order = {3: 'skewness_sqi',
#                   2: 'kurtosis_sqi',
#                   1: 'perfusion_sqi'}
# timestamp_idx = ['TIMESTAMP_MS']
# signal_idx = ['PLETH']
# output_dir = tempfile.gettempdir()
# signal_obj = get_qualified_ecg(file_name=file_in,
# 									sqi_dict_filename=sqi_dict,
# 									file_type='edf', duration=30,
# 									rule_dict_filename=rule_dict_filename,
# 									ruleset_order=ruleset_order,
# 									output_dir=output_dir)
#
# print(signal_obj)
