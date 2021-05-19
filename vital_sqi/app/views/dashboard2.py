import dash
import dash_table
import numpy as np
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from plotly import graph_objects as go
import pandas as pd
from vital_sqi.app.app import app
from vital_sqi.common.utils import update_rule

def generate_detail(idx,column_name):
    sqi_detail = html.Div([
        dbc.CardHeader(
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
                    data=[
                    ],
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
    Output('rule_dataframe','data'),
    Input('confirm-rule-button','n_clicks'),
    Input({"type":"switch-selection","index":ALL}, "value"),
    Input({'type':'rules-table','index':ALL}, "data"),
    State({"type":"switch-selection","index":ALL}, "options")
)
def send_to_rule_set(confirm_click, rule_list,table_components,column_list):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'confirm-rule-button' in changed_id:
        for (i, value) in enumerate(table_components):
            rule = rule_list[i]
            if len(rule) > 0:
                sqi_name = column_list[i][0]['label']
                rule_def = table_components[i]
                print(rule)
                #TODO parse children to rule set
        return table_components
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
    State({'type':'rules-table','index':MATCH}, 'data'),
    State({'type':'rules-table','index':MATCH}, 'columns'))
def add_row(n_clicks, rows, columns):
    if rows == None:
        rows = []
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

@app.callback(Output('sqi-list', 'children'),
              Input('dataframe', 'data'))
def on_data_set_table(data):
    if data is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    columns = list(df.columns)
    sqi_list = []
    for idx in range(len(columns)):
        sqi_detail = generate_detail(idx,columns[idx])
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