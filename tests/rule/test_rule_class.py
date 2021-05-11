import pytest
from vital_sqi.rule.rule_class import Rule
import os


class TestRuleClass(object):

    def test_on_init(self):
        assert isinstance(Rule('test_sqi'), Rule) is True

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
        out = Rule('test_sqi')
        source = os.path.abspath('tests/test_data/rule_dict_test.json')
        out.load_def(source)
        assert isinstance(out.rule_def, list) is True
        with pytest.raises(Exception) as exc_info:
            out.name = 'random_sqi'
            out.load(source)
        assert exc_info.match('not found')
        with pytest.raises(Exception) as exc_info:
            source = os.path.abspath('tests/test_data/file_not_exist.json')
            out.load_def(source)
        assert exc_info.match('Source file not found')

    def test_on_save(self):
        pass

    def test_on_apply_rule(self):
        pass
