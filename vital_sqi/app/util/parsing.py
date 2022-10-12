import base64
import pandas as pd
import io
import dash_html_components as html
import pathlib
from vital_sqi.rule.rule_class import Rule
from vital_sqi.rule.ruleset_class import RuleSet
from vital_sqi.common.utils import update_rule
import json
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../temp").resolve()

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), header=0)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
        elif 'json' in filename:
            return json.load(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df.to_dict()

def generate_rule(rule_name,rule_def):
    rule_def, boundaries, label_list = update_rule(rule_def, is_update=False)
    rule_detail = {'def': rule_def,
                     'boundaries': boundaries,
                     'labels': label_list}
    rule = Rule(rule_name,rule_detail)
    return rule

def generate_rule_set(rule_set_dict):
    rule_set = {}
    for i,rule_dict in enumerate(rule_set_dict):
        rule_name = rule_dict["name"]
        rule_def = rule_dict["def"]
        rule = generate_rule(rule_name,rule_def)
        rule_order = rule_dict["order"]
        rule_set[rule_order] = rule
    return RuleSet(rule_set)

def parse_rule_list(rule_def):
    rule_dict_list = []
    for i,rule in enumerate(rule_def):
        rule_dict = {
            "op":rule['op'],
            "value":rule['value'],
            "label":rule['label']
        }
        rule_dict_list.append(rule_dict)
    return rule_dict_list

def generate_boundaries(boundaries):
    bound_list = []
    for idx,boundary in enumerate(boundaries):
        if idx == 0:
            bound_list.append("[-inf, "+str(boundaries[idx])+"]")
            bound_list.append(str(boundaries[idx]))
        else:
            bound_list.append("[" + str(boundaries[idx-1])+", "+ str(boundaries[idx]) + "]")
            bound_list.append(str(boundaries[idx]))
    bound_list.append("[" + str(boundaries[idx-1])+", inf]")
    return bound_list