"""
Class Rule contains thresholds and its corresponding labels of an SQI. Labels
are either 'accept' or 'reject'
"""
import json
from vital_sqi.common.utils import parse_rule, update_rule
import bisect
import re
import os
import numpy as np


class Rule:
    """ """

    def __init__(self, name, rule=None):
        self.name = name
        self.rule = rule

    def __setattr__(self, name, value):
        if name == 'name':
            if not isinstance(value, str) or \
                    not bool(re.match("^[A-Za-z0-9_-]*$", value)):
                raise AttributeError('Name of SQI rule must be a string '
                                     'containing only letter, number, '
                                     'hyphens and underscores')
        if name == 'rule':
            if not (isinstance(value, dict) or value is None):
                raise AttributeError('Rule definition must be a dict or None')
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
        rule_def, boundaries, labels = parse_rule(self.name, source)
        self.rule = {'def': rule_def,
                     'boundaries': boundaries,
                     'labels': labels}
        return

    def update_def(self, op_list, value_list, label_list):
        """

        Parameters
        ----------
        op_list :
            
        value_list :
            
        label_list :
            

        Returns
        -------

        Examples
        --------
        >>> rule = Rule("test_sqi")
        >>> rule.load_def("../resource/rule_dict.json")
        >>> rule.update_def(op_list=["<=", ">"],
                        value_list=[5, 5],
                        label_list=["accept", "reject"])
        >>> print(rule.rule['def'])
        [{'op': '>', 'value': '10', 'label': 'reject'},
        {'op': '>=', 'value': '3', 'label': 'accept'},
        {'op': '<', 'value': '3', 'label': 'reject'},
        {'op': '<=', 'value': 5, 'label': 'accept'},
        {'op': '>', 'value': 5, 'label': 'reject'}]
        """
        for op in op_list:
            if op not in ["<", "<=", ">", ">=", "="]:
                raise ValueError("Invalid operand: Expect string operands, "
                                 "instead found {0}" + op
                                 + ", type {1}".format(op, type(op)))
        for value in value_list:
            if not np.isreal(value):
                raise ValueError("Invalid threshold: Expect numeric type "
                                 "threshold, instead found {0}" + str(value)
                                 + ", type {1}".format(value, type(value)))
        for label in label_list:
            assert isinstance(label, str) or label is None, \
                "Label must be 'accept' or 'reject' string"
            if label != "reject" or label != "accept":
                label = None

        threshold_list = []
        for idx in range(len(label_list)):
            threshold = {"op": op_list[idx], "value": value_list[idx],
                         "label": label_list[idx]}
            threshold_list.append(threshold)

        if self.rule is None:
            self.rule = {'def': None, 'boundaries': None, 'labels': None}
        self.rule['def'], self.rule['boundaries'], self.rule[
            'labels'] = update_rule(self.rule['def'], threshold_list)
        return

    def save_def(self, file_path, file_type="json", overwrite=False):
        """

        Parameters
        ----------
        file_path :
            
        file_type :
            (Default value = "json")
        overwrite :
             (Default value = False)

        Returns
        -------

        
        """
        assert isinstance(file_path, str) and len(file_path) > 0, \
            "Invalid output file path"
        if overwrite:
            assert os.path.isfile(file_path), "Overwritten file doesn't exist"
            with open(file_path) as file_in:
                all_rules = json.load(file_in)
                assert isinstance(all_rules, dict), "Invalid file format."
            if np.any(np.array(list(all_rules.keys())) == self.name):
                all_rules[self.name]['def'] = self.rule['def']
            else:
                all_rules[self.name] = {'name': self.name,
                                        'def': self.rule['def']}
            with open(file_path, 'w') as file_out:
                json.dump(all_rules, file_out)
        else:
            with open(file_path, "w") as file_out:
                name = self.name
                rule = self.rule['def']
                rule_def = {name: {'name': name, 'def': rule}}
                json.dump(rule_def, file_out)
        return

    def apply_rule(self, x):
        """

        Parameters
        ----------
        x :
            

        Returns
        -------

        
        """
        boundaries = self.rule['boundaries']
        labels = self.rule['labels']
        if np.any(boundaries == x):
            return labels[(np.where(boundaries == x)[0][0])*2+1]
        else:
            new_labels = []
            for i in range(len(labels)):
                if i % 2 == 0:
                    new_labels.append(labels[i])
            return new_labels[bisect.bisect_left(boundaries, x)]

    def write_rule(self):
        rules = self.rule['def']
        r_strs = []
        for r in rules:
            r_str = "x" + " " + r['op'] + str(r['value']) + ": " + r['label']
            r_strs.append(r_str)
        return "\n".join(r_strs)