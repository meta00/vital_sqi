import dash
import dash_table
import numpy as np
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from plotly import graph_objects as go
import pandas as pd
from vital_sqi.app.util.parsing import parse_rule_list,generate_rule,generate_rule_set
from vital_sqi.app.app import app
from vital_sqi.common.utils import update_rule

def generate_detail(idx,column_name,data=[]):
    sqi_detail = html.Div([
        dbc.CardHeader(
            [
                html.Div(
                    [
                    dbc.Checklist(
                        options=[
                            {"label": column_name, "value": 1},
                        ],
                        value=[],
                        id={
                            'type': 'switch-selection',
                            'index': idx
                        },
                        inline=True,
                        switch=True,
                        style={'display': 'inline-block'}
                    ),"- Order = ",
                    dcc.Input(
                        id={
                            'type': 'input-order',
                            'index': idx
                            # "input_{}".format(_),
                        },
                        type="number",
                        min=1,
                        placeholder="",
                        style={'display': 'inline-block',
                               'width':'80px'}
                    )
                ],
                ),
                dbc.Button(
                    "Expand",
                    # color="link",
                    color="primary",
                    id={
                        'type': 'group-toggle',
                        'index': idx
                    },
                )
            ]
        ),
        # dbc.Button(
        #     column_name,
        #     id={
        #         'type': 'group-toggle',
        #         'index': idx
        #     },
        #     # id=f"group-{idx}-toggle",
        #     className="mb-3",
        #     color="primary",
        # ),
        dbc.Collapse(
            dbc.CardBody([
                dash_table.DataTable(
                    id={
                        'type': 'rules-table',
                        'index': idx
                    },
                    columns=[
                        {
                            'name': 'Operand',
                            'id': 'op',
                            'presentation': 'dropdown'
                        },
                        {
                            'name': 'Value',
                            'id': 'value',
                            'type': 'numeric',
                        },
                        {
                            'name': 'Label',
                            'id': 'label',
                            'presentation': 'dropdown'
                        },
                    ],
                    data=data,
                    css=[
                        {"selector": ".Select-menu-outer", "rule": "display: block !important"}
                    ],
                    dropdown={
                        'op': {
                            'options': [{'label': i, 'value': i}
                                        for i in ['>', '>=', '=', '<=', '<']]
                        },
                        'label': {
                            'options': [{'label': i, 'value': i} for i in ['accept', 'reject']]
                        }
                    },
                    editable=True,
                    row_deletable=True
                ),
                dbc.Button('Add Row', id={'type': 'editing-rows-button',
                                          'index': idx}, n_clicks=0),
                dbc.Button('Visualize Rule', id={'type':'visualize-rule-button',
                                                 'index':idx}, n_clicks=0),
                # dcc.Graph(id={'type':'rules-graph','index':idx})
            ]),
            id={
                'type':'collapse',
                'index':idx
            }
        ),
    ]
    )
    return sqi_detail

layout = html.Div([
    dbc.Checklist(
        options=[
            {"label": "Select All", "value": 1},
            ],
        value=[],
        id='select-all',
        inline=True,
        switch=True,
    ),
    dbc.Button(
        "Confirm",
        id="confirm-rule-button",
        color="success"
    ),
    html.Div(
        id="sqi-list",
        className="accordion",
    )
])

@app.callback(
    Output('rule-dataframe','data'),
    Input('confirm-rule-button','n_clicks'),
    Input({"type":"switch-selection","index":ALL}, "value"),
    Input({'type':'rules-table','index':ALL}, "data"),
    Input({'type':'input-order','index':ALL}, "value"),
    State({"type":"switch-selection","index":ALL}, "options")
)
def send_to_rule_set(confirm_click, rule_selection_list,
                     table_components,order_list,column_list):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'confirm-rule-button' in changed_id:
        rule_set = []
        for (i, value) in enumerate(table_components):
            single_rule = rule_selection_list[i]
            rule_order = order_list[i]
            if len(single_rule) > 0 or rule_order is not None:
                rule_name = column_list[i][0]['label']
                rule_def = table_components[i]
                if rule_order is None:
                    rule_order = 1
                rule_dict ={
                    'name': rule_name,
                    'order': rule_order,
                    'def': rule_def
                }
                rule_set.append(rule_dict)
                # Check if generated rule is correct
                # rule = generate_rule(rule_name,rule_def)
                # rule_set_dict[rule_order] = rules
        if len(rule_set) > 0:
            verified_rule_set = generate_rule_set(rule_set)
        return rule_set
    return None

@app.callback(
    Output({"type":"switch-selection","index":ALL}, "value"),
    Input('select-all','value'),
    Input('sqi-list', 'children')
)
def toggle_select_all(checked,checked_data):
    if len(checked)>0:
        return [[1]]*len(checked_data)
    return [[]]*len(checked_data)

@app.callback(
    Output({'type':'rules-table','index':MATCH}, 'data'),
    Input({'type':'editing-rows-button','index':MATCH}, 'n_clicks'),
    Input('rule-set-store', 'data'),
    State({'type':'rules-table','index':MATCH}, 'data'),
    State({'type':'switch-selection','index':MATCH}, 'options'))
def add_row(n_clicks, rule_set, rows, columns):
    ctx = dash.callback_context
    # load data file
    change_id = [p['prop_id'] for p in ctx.triggered][0]
    if rows == None:
        rows = []
    if rule_set is None:
        # Modified here
        rows = []
    if 'editing-rows-button' in change_id:

        if n_clicks > 0:
            rows.append({
                'op': '>',
                'value': 0,
                'label': 'accept'
            })
    return rows

@app.callback(
    Output('rules-graph', 'figure'),
    Input('visualize-rule-button', 'n_clicks'),
    Input('rules-table', 'data'),
    Input('rules-table', 'columns')
)
def display_output(n_clicks,rows,columns):
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    if n_clicks<1:
        return fig
    if len(rows) > 0:
        # all_rule, boundaries, interval = [[],[],[]]
        all_rule, boundaries, interval = update_rule([],rows)
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)'
    )
    x_accept = np.arange(100)[np.r_[:2,5:40,60:68,[85,87,88]]]
    fig.add_traces(go.Scatter(x=x_accept,y=np.zeros(len(x_accept)),line_color='#ffe476'))

    x_reject = np.setdiff1d(np.arange(100),x_accept)
    fig.add_traces(go.Scatter(x=x_reject, y=np.zeros(len(x_reject)), line=dict(color="#0000ff")))

    return fig

@app.callback(
    Output('sqi-list', 'children'),
    Input('dataframe', 'data'),
    Input('rule-set-store', 'data')
)
def on_data_set_table(data,data_rule):
    if data is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    columns = list(df.columns)
    sqi_list = []
    # df_rule = pd.DataFrame(data_rule)
    for idx in range(len(columns)):
        column_name = columns[idx]
        data = []
        if column_name in data_rule.keys():
            data = parse_rule_list(data_rule[column_name]['def'])
            # data = [{"Operand":"","Value":"","Label":""}]
            sqi_detail = generate_detail(idx,columns[idx],data)
            sqi_list.append(sqi_detail)
    return sqi_list

@app.callback(
    Output({"type":"collapse","index":MATCH}, "is_open"),
    Input({"type":"group-toggle","index":MATCH}, "n_clicks"),
    State({"type":"collapse","index":MATCH}, "is_open")
)
def toggle_collapse(n_clicks, is_open):
    is_triggered = False
    try:
        if n_clicks > 0:
            is_triggered = True
        if is_triggered:
            return not is_open
    except:
        return is_open
    return is_open