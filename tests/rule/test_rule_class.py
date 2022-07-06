import pytest
from vital_sqi.rule.rule_class import Rule
import os
import tempfile


class TestRuleClass(object):

    def test_on_init(self):
        assert isinstance(Rule('test_sqi'), Rule) is True

    def test_on_set(self):
        out = Rule('test_sqi')
        with pytest.raises(AttributeError) as exc_info:
            out.name = 'a/b'
        assert exc_info.match("Name of SQI rule must be a string")
        with pytest.raises(AttributeError) as exc_info:
            out.rule = []
        assert exc_info.match("Rule definition must be a dict or None")

    def test_on_update(self):
        out = Rule('test_sqi')
        out.name = 'new_test_sqi'
        assert out.name == 'new_test_sqi'
        with pytest.raises(AttributeError) as exc_info:
            out.name = 12
        assert exc_info.match('Name of SQI rule must be a string')
        with pytest.raises(AttributeError) as exc_info:
            out.name = 'a.b'
        assert exc_info.match('containing only letter, number')

    def test_on_load(self):
        out = Rule('perfusion_sqi')
        source = os.path.abspath('tests/data/rule_dict_test.json')
        out.load_def(source)
        assert isinstance(out.rule['def'], list) is True
        with pytest.raises(Exception) as exc_info:
            out.name = 'random_sqi'
            out.load_def(source)
        assert exc_info.match('not found')
        with pytest.raises(Exception) as exc_info:
            source = os.path.abspath('tests/data/file_not_exist.json')
            out.load_def(source)
        assert exc_info.match('Source file not found')

    def test_on_update(self):
        out = Rule('test_sqi')
        out.update_def(op_list=["<=", ">"], value_list=[5, 5], label_list=[
            "accept", "reject"])
        assert out.rule['labels'][0] == 'accept'
        out.update_def(op_list=["<=", ">", '<', ">="],
                       value_list=[3, 3, 10, 10],
                       label_list=["reject", "accept", "accept", "reject"])
        assert out.rule['labels'][0] == 'reject'
        with pytest.raises(ValueError) as exc_info:
            out.update_def(op_list = ["<=", ">", '>', "<="],
                           value_list = [10, 10, 3, 3],
                           label_list = ["reject", "accept", "accept",
                                         "reject"])
        assert exc_info.match('conflict')
        with pytest.raises(ValueError) as exc_info:
            out.update_def(op_list = ["abc", ">", '>', "<="],
                           value_list = [10, 10, 3, 3],
                           label_list = ["reject", "accept", "accept",
                                         "reject"])
        assert exc_info.match('Invalid operand')
        with pytest.raises(ValueError) as exc_info:
            out.update_def(op_list = ["<=", ">", '<', ">="],
                           value_list = ['10', 3, 10, 10],
                           label_list = ["reject", "accept", "accept",
                                         "reject"])
        assert exc_info.match('Invalid threshold')

    def test_on_apply_rule(self):
        out = Rule('test_sqi')
        out.update_def(op_list=["<=", ">", '<', ">="],
                       value_list=[3, 3, 10, 10],
                       label_list=["reject", "accept", "accept", "reject"])
        assert out.apply_rule(11) == 'reject'
        assert out.apply_rule(10) == 'reject'
        assert out.apply_rule(3) == 'reject'

    def test_on_save(self):
        rule_obj = Rule('perfusion_sqi')
        source = os.path.abspath('tests/data/rule_dict_test.json')
        rule_obj.load_def(source)
        file_out = tempfile.gettempdir() + '/rule_dict.json'
        rule_obj.save_def(file_out)
        assert os.path.isfile(file_out)
        rule_obj.update_def(op_list=["<=", ">", '<', ">="],
                            value_list=[3, 3, 10, 10],
                            label_list=["reject", "accept", "accept", "reject"])
        rule_obj.save_def(file_out, overwrite=True)
        assert os.path.isfile(file_out)
        rule_obj = Rule('new_test_sqi')
        rule_obj.update_def(op_list=["<=", ">", '<', ">="],
                            value_list=[3, 3, 10, 10],
                            label_list=["reject", "accept", "accept", "reject"])
        rule_obj.save_def(file_out, overwrite=True)
