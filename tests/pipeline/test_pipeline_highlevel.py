import pytest
from vital_sqi.pipeline.pipeline_highlevel import get_ppg_sqis
from vital_sqi.data.signal_sqi_class import SignalSQI
import os


class TestGetPPGSQIs(object):
	def test_on_get_ppg_sqis(self):
		file_in = os.path.abspath('tests/test_data/ppg_smartcare.csv')
		# file_name = "../../tests/test_data/ppg_smartcare.csv"
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

		sqi_dict = os.path.abspath('tests/test_data/sqi_dict.json')
		segments, signal_sqi_obj = get_ppg_sqis(file_in,
												timestamp_idx=['TIMESTAMP_MS'],
												signal_idx=['PLETH'],
												sqi_dict_filename=sqi_dict)
		assert isinstance(segments, list) is True
		assert isinstance(signal_sqi_obj, SignalSQI) is True
		assert signal_sqi_obj.sqis is not None
