import os
import dash
import dash_table
import numpy as np
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory, send_file
from plotly import graph_objects as go
from app import app
# from vital_sqi.common.utils import update_rule

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# server = app.server

layout = html.Div([
    dash_table.DataTable(
        id='adding-rows-ta'
           'ble',
        columns=[
            {
            'name': 'Operand',
            'id': 'op',
            'presentation':'dropdown'
            },
            {
            'name': 'Value',
            'id': 'value',
            'type': 'numeric',
            },
            {
            'name': 'Label',
            'id': 'label',
            'presentation':'dropdown'
            },
        ],
        data=[
        ],
        dropdown={
            'op':{
                'options':[{'label': i, 'value': i}
                           for i in ['>', '>=', '=','<=','<']]
            },
            'label':{
                'options':[{'label': i, 'value': i} for i in ['accept','reject']]
            }
        },
        editable=True,
        row_deletable=True
    ),

    html.Button('Add Row', id='editing-rows-button', n_clicks=0),
    html.Button('Visualize Rule', id='visualize-rule-button', n_clicks=0),
    dcc.Graph(id='adding-rows-graph')
    # dcc.Slider(
    #         min=0,
    #         max=100,
    #         value=65,
    #         ranges={
    #
    #         },
    #         marks={
    #             0: {'label': '0 °C', 'style': {'color': '#77b0b1'}},
    #             26: {'label': '26 °C'},
    #             37: {'label': '37 °C'},
    #             100: {'label': '100 °C', 'style': {'color': '#f50'}}
    #         }
    #     ),
    # html.Div(id='adding-rows-graph',children=[])
])


@app.callback(
    Output('adding-rows-table', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    State('adding-rows-table', 'data'),
    State('adding-rows-table', 'columns'))
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
    Output('adding-rows-graph', 'figure'),
    Input('visualize-rule-button', 'n_clicks'),
    Input('adding-rows-table', 'data'),
    Input('adding-rows-table', 'columns')
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
        all_rule, boundaries, interval = [[],[],[]]
        # all_rule, boundaries, interval = update_rule([],rows)
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

# @app.callback(
#     Output('adding-rows-graph', 'figure'),
#     Input('adding-rows-table', 'data'),
#     Input('adding-rows-table', 'columns'))
# def display_output(rows, columns):
#     return {
#         'data': [{
#             'type': 'heatmap',
#             'z': [[row.get(c['id'], None) for c in columns] for row in rows],
#             'x': [c['name'] for c in columns]
#         }]
#     }

# if __name__ == '__main__':
#     app.run_server(debug=True)