"""
Class Rule contains thresholds and its corresponding labels of an SQI.
"""
import warnings

import numpy as np
import pandas as pd
from vital_sqi.common.utils import parse_rule,write_rule,\
    reorganize_rule


class Rule:
    """ """

    def __init__(self, name, rule_def=None):
        self.name = name
        self.rule_def = rule_def

    def __setattr__(self, name, value):
        if name == 'name':
            if not isinstance(value, str):
                raise AttributeError('Name of SQI rule must be a string '
                                     'containing only letter, number, '
                                     'and hyphens')
        if name == 'rule_def':
            if not (isinstance(value, list) or value is None):
                raise AttributeError('Rule definition must be a list or None')
        super().__setattr__(name, value)

    def load_def(self, source=None):
        """

        Parameters
        ----------
        source :
             (Default value = None)

        Returns
        -------

        """
        self.rule_def = parse_rule(self.name, source)
        return self

    def update_def(self,op_list,value_list,label_list):
        """

        Parameters
        ----------
        op
        value
        label

        Returns
        -------

        """

        for op in op_list:
            if op not in ["<","<=",">",">=","="]:
                raise ValueError("Invalid operand: Expect string operands, "
                                 "instead found {0}".op+" type {1}".format(op))
        for value in value_list:
            if np.isnumeric(value):
                raise ValueError("Invalid threshold: Expect numeric type threshold, "
                                 "instead found {0}".value+" type {1}".format(value))
        for label in label_list:
            assert label is str, "Label must be 'accept' or 'reject' string"
            if label != "reject" or label != "accept":
                label = None

        thresholder_list = []
        for idx in range(len(label_list)):
            thresholder = {}
            thresholder["op"] = op
            thresholder["value"] = value
            thresholder["label"] = label
            thresholder_list.append(thresholder)

        if self.rule_def is None:
            self.rule_def = []

        try:
            self.reorganize_rule(self.rule_def, thresholder)
        except Exception as err:
            warnings.warn(err)
        return

    def save_def(self,file_type="json"):
        """ """
        write_rule(self.name,self.rule_def)
        return

    def apply_rule(self, x):
        """

        Parameters
        ----------
        x :

        Returns
        -------

        """
        pass

if __name__ == "__main__":
    rule = Rule("test_sqi")
    rule.load_def("../resource/rule_dict.json")
    def_str = rule.save_def()
    # conflicted_rule(rule.rule_def)
    thresholder = {}
    thresholder["op"] = "<="
    thresholder["value"] = 5
    thresholder["label"] = "accept"
    thresholder_1 = {}

    thresholder_1["op"] = ">"
    thresholder_1["value"] = 5
    thresholder_1["label"] = "reject"

    reorganize_rule(rule.rule_def)

    reorganize_rule(rule.rule_def, [thresholder,thresholder_1])

    reorganize_rule([thresholder,thresholder_1])