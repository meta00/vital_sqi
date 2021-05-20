import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from vital_sqi.app.app import app
from dash.exceptions import PreventUpdate
from vital_sqi.app.util.parsing import generate_rule_set,generate_boundaries
import pathlib

layout = html.Div([
    # html.Div([
    #     dcc.Input(
    #         id='editing-columns-name',
    #         placeholder='Enter a column name...',
    #         value='',
    #         style={'padding': 10}
    #     ),
    #     html.Button('Add Column', id='editing-columns-button', n_clicks=0)
    # ], style={'height': 50}),

    html.Div(id='confirmed-rule-table')
])

@app.callback(Output('confirmed-rule-table', 'children'),
              Input('rule-dataframe', 'data'))
def on_data_set_table(data):
    if data is None:
        raise PreventUpdate

    rule_set = generate_rule_set(data)

    tables = []
    for rule_order in rule_set.rules:
        rule_name = rule_set.rules[rule_order].name
        rule_content = rule_set.rules[rule_order].rule
        boundaries = generate_boundaries(rule_content['boundaries'])
        labels = (rule_content['labels'])
        print(boundaries)

        table_header = [
            html.Thead(html.Tr([
                html.Th(bound) for bound in boundaries
            ]))
        ]

        label_row = html.Tr([
            html.Td(label_detail) for label_detail in labels
        ])

        table_body = [html.Tbody([label_row])]

        tables.append(
            dbc.Table(table_header + table_body,
                              bordered=True,
                              dark=True,
                              hover=True,
                              responsive=True,
                              striped=True
                              )
        )
    children = tables
    return children
