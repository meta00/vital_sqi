import os

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

    html.Div(id='confirmed-rule-table'),
    html.Div(id='export-message'),
    dbc.Button(
            "Export",
            # color="link",
            color="primary",
            id='export-button',
            style={
                'display':'inline-block'
            }
        ),
    dcc.Download(id='download-content'),
    dcc.Download(id='download-decision-dataframe'),
    # dbc.Button(
    #         'Save',
    #         id='save-decision-button',
    #         color="primary"
    #     ),
    dbc.Button(
            "Apply",
            # color="link",
            color="success",
            id='apply-button',
            style={
                'display': 'inline-block'
            }
        ),
    html.Div(id="applied-rule-table"),
])

@app.callback(
    Output('download-content', 'data'),
    Output('export-message','children'),
    Input('export-button', 'n_clicks'),
    Input('rule-dataframe', 'data'),
    prevent_initial_call=True
)
def export_rule_set(n_clicks,rule_set_dict):
    ctx = dash.callback_context
    change_id = [p['prop_id'] for p in ctx.triggered][0]
    if rule_set_dict is None:
        raise PreventUpdate
    if 'export-button' in change_id:
        rule_set = generate_rule_set(rule_set_dict)
        file_content = rule_set.export_rules()
        # filename = f"{uuid.uuid1()}.txt"
        filename = "exported_rule.txt"
        msg="Please copy the exported content and " \
            "paste it in the following website to visualize " \
            "https://flowchart.js.org/"
        return [dict(content=file_content,filename=filename),msg]

    return [None,None]

def save_file(filename, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = (content.encode("utf8").split(b";base64,"))[0]
    path = os.path.join(os.getcwd(),filename)
    with open(path, "wb") as fp:
        fp.write(data)
    return path

@app.callback(
    Output('applied-rule-table','children'),
    Output('download-decision-dataframe','data'),
    Input('apply-button','n_clicks'),
    Input('rule-dataframe', 'data'),
    Input('dataframe', 'data')
)
def apply_rule_set(n_clicks,rule_set_dict, sqi_table):
    ctx = dash.callback_context
    change_id = [p['prop_id'] for p in ctx.triggered][0]
    if rule_set_dict is None:
        raise PreventUpdate
    if 'apply-button' in change_id:
        rule_set = generate_rule_set(rule_set_dict)
        sqi_columns = [rule['name'] for rule in rule_set_dict]
        df = pd.DataFrame(sqi_table)
        output_label = []
        for idx in range(len(df)):
            row_data = pd.DataFrame(dict(df[sqi_columns].iloc[idx]), index=[0])
            output_label.append(rule_set.execute(row_data))
        # dat =
        df['output_decision'] = output_label
        output_columns = ['file_name']+sqi_columns+['output_decision']
        decision_table = dash_table.DataTable(
            id='decision-table',
            columns=[{"name": i, "id": i, 'deletable': True} for i in output_columns],
            data=df[output_columns].to_dict('records'),
            style_table={'overflowX': 'auto'},
            editable=True,
            style_cell={'textAlign': 'left'},
            filter_action="native",
            sort_action="native",
            sort_mode="single",
            page_size=15,
        )
        children = html.Div([
            decision_table,
        ])
        fname = "decision_table.csv"
        decision_content = dict(
            content=df.to_csv(index=False),
            filename=fname
        )
        return [children,decision_content]
    return [None,None]

# @app.callback(
#     Output('download-decision-dataframe','data'),
#     Input('save-decision-button','n_clicks'),
#     Input('dataframe', 'data')
# )
# def save_decision_table(n_clicks,sqi_table):
#     ctx = dash.callback_context
#     change_id = [p['prop_id'] for p in ctx.triggered][0]
#     if 'save-decision-button' in change_id:
#         df = pd.DataFrame(sqi_table)
#         fname = "decision_table.csv"
#         return dict(content=df.to_csv(index=False),filename=fname)
#     return None

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
            html.Thead(html.Tr([html.Th('Range')]+[
                html.Th(bound) for bound in boundaries
            ]))
        ]

        label_row = html.Tr([html.Td('Decision')]+[
            html.Td(label_detail) for label_detail in labels
        ])

        table_body = [html.Tbody([label_row])]

        tables.append(
            dbc.Table(table_header + table_body,
                              bordered=True,
                              dark=True,
                              hover=True,
                              responsive=True,
                              striped=True,
                              )
        )
    children = tables

    return children
