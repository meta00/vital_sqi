import pytest
from vital_sqi.rule import *
import os


class TestRuleSet(object):
    r1 = Rule("sqi1")
    r2 = Rule("sqi2")
    r3 = Rule("sqi3")
    source = os.path.abspath('tests/test_data/rule_dict_test.json')
    r1.load_def(source)
    r2.load_def(source)
    r3.load_def(source)
    r = {3: r1, 2: r2, 1: r3}
    s = RuleSet(r)

    def test_on_init(self):
        assert isinstance(self.s, RuleSet)

    def test_on_set(self):
        with pytest.raises(AttributeError) as exc_info:
            self.s.rules = []
        assert exc_info.match('Rule set must be of dict type')
        with pytest.raises(ValueError) as exc_info:
            r = {'3': self.r1, 2: self.r2, 1: self.r3}
            self.s.rules = r
        assert exc_info.match('Order must be of type int')
        with pytest.raises(ValueError) as exc_info:
            r = {3: 1, 2: self.r2, 1: self.r3}
            self.s.rules = r
        assert exc_info.match('Rules must be of class Rule')
        with pytest.raises(ValueError) as exc_info:
            r = {4: self.r1, 2: self.r2, 1: self.r3}
            self.s.rules = r
        assert exc_info.match('Order must contain consecutive numbers')
        with pytest.raises(ValueError) as exc_info:
            r = {4: self.r1, 2: self.r2, 3: self.r3}
            self.s.rules = r
        assert exc_info.match('Order must start with 1')

    def test_on_export(self):
        assert self.s.export_rules() == True

    def test_on_execute(self):
        dat = pd.DataFrame([[6, 100, 0]], columns = ['sqi1', 'sqi2', 'sqi3'])
        assert self.s.execute(dat) == 'accept'
        dat = pd.DataFrame([[10, 100, 0]], columns = ['sqi1', 'sqi2', 'sqi3'])
        assert self.s.execute(dat) == 'reject'
        with pytest.raises(AssertionError) as exc_info:
            self.s.execute([])
        assert exc_info.match('Expected data frame')
        with pytest.raises(AssertionError) as exc_info:
            dat = [[6, 1, 1], [1, 100, 3], [0, 0, 0]]
            dat = pd.DataFrame(dat, columns = ['sqi1', 'sqi2', 'sqi3'])
            self.s.execute(dat)
        assert exc_info.match('Expected data frame of 1 row')
        with pytest.raises(KeyError) as exc_info:
            dat = pd.DataFrame([[6, 100, 0]], columns = ['sqi1', 'sqi2',
                                                         'sqi4'])
            self.s.execute(dat)
        assert exc_info.match('not found in input data frame')



