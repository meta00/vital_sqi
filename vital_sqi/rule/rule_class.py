"""
Class Rule contains thresholds and its corresponding labels of an SQI.
"""
import warnings
import pandas as pd
from vital_sqi.common.utils import parse_rule,write_rule,\
    converted_rule
import bisect
import re
import numpy as np


class Rule:
    """ """

    def __init__(self, name, rule_def=None):
        self.name = name
        self.rule_def = rule_def

    def __setattr__(self, name, value):
        if name == 'name':
            if not isinstance(value, str) or not bool(re.match("^[A-Za-z0-9_-]*$", value)):
                raise AttributeError('Name of SQI rule must be a string '
                                     'containing only letter, number, '
                                     'hyphens and underscores')
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
            if not np.isscalar(value):
                raise ValueError("Invalid threshold: Expect numeric type threshold, "
                                 "instead found {0}".value+" type {1}".format(value))
        for label in label_list:
            assert (type(label) is str) or (label is None), \
                "Label must be 'accept' or 'reject' string"
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
        converted_rule(self.rule_def,thresholder_list)
        update_rule(self)
        return

    def save_def(self,file_path,file_type="json"):
        """ """
        write_rule(self.name,self.rule_def,file_path)
        return

    def apply_rule(self, x):
        """

        Parameters
        ----------
        x :

        Returns
        -------

        """
        converted_rule_dict = converted_rule(self.rule_def)
        boundaries = converted_rule_dict['boundaries']
        labels = converted_rule_dict['label_list']
        if np.any(boundaries == x):
            return labels[(np.where(boundaries == x)[0][0])*2+1]
        else:
            new_labels = []
            for i in range(len(labels)):
                if i % 2 == 0:
                    new_labels.append(labels[i])
            return new_labels[bisect.bisect_left(boundaries, x)]
