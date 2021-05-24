"""
Class RuleSet contains rules of SQIs and their splitting orders, used to
build a decision flow/tree.
"""
from vital_sqi.rule.rule_class import Rule
from pyflowchart import *
import pandas as pd


class RuleSet:
    """ """

    def __init__(self, rules):
        self.rules = rules

    def __setattr__(self, name, value):
        if name == 'rules':
            if not isinstance(value, dict):
                raise AttributeError('Rule set must be of dict type.')
            order = list(value.keys())
            for i in order:
                if not isinstance(i, int):
                    raise ValueError('Order must be of type int')
            if sorted(order) != list(range(min(order), max(order) + 1)):
                raise ValueError('Order must contain consecutive numbers')
            if sorted(order)[0] != 1:
                raise ValueError('Order must start with 1.')
            rs = list(value.values())
            for i in rs:
                if not isinstance(i, Rule):
                    raise ValueError('Rules must be of class Rule')
        super().__setattr__(name, value)

    def export_rules(self):
        """ """
        rules = self.rules
        st = StartNode('')
        e = EndNode('')
        ops = []
        conds = []
        for key, value in rules.items():
            ops.append(OperationNode(value.name))
            conds.append(ConditionNode(value.write_rule()))
        # define the direction the connection will leave the node from
        st.connect(ops[0])
        for i in range(len(ops)):
            ops[i].connect(conds[i])
            conds[i].connect_no(e)
            if i < len(ops) - 1:
                conds[i].connect_yes(ops[i+1])
        conds[-1].connect_yes(e)
        fc = Flowchart(st)
        return fc.flowchart()

    def execute(self, value_df):
        """

        Parameters
        ----------
        value_df :
            

        Returns
        -------

        """
        assert isinstance(value_df, pd.DataFrame), \
            'Expected data frame, found {0}'.format(type(value_df))
        assert len(value_df) == 1, 'Expected data frame of 1 row but got {0}' \
                                   ' instead'.format(len(value_df))
        rules = self.rules
        rules = rules.items()
        rules = sorted(rules)
        decision = 'accept'
        for r in rules:
            rule = r[1]
            try:
                value = value_df.iloc[0][rule.name]
            except:
                raise KeyError('SQI {0} not found in input data frame'.format(
                        rule.name))
            decision = rule.apply_rule(value)
            if decision == 'reject':
                break
        return decision

# r1 = Rule("sqi1")
# r2 = Rule("sqi2")
# r3 = Rule("sqi3")
# import os
# source = os.path.abspath('/Users/haihb/Documents/Work/Oucru/innovation'
#                            '/vital_sqi/tests/test_data/rule_dict_test.json')
# r1.load_def(source)
# r2.load_def(source)
# r3.load_def(source)
# r = {'3' : r1, 2 : r2, 1 : r3}
# s = RuleSet(r)
# s.export_rules()
# dat = [6, 100, 0]
# dat = pd.DataFrame([dat], columns = ['sqi1', 'sqi2', 'sqi3'])
# s.execute(dat)