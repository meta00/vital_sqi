import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from vital_sqi.app.app import app
import pathlib

layout = html.Div([
    html.Div([
        dcc.Input(
            id='editing-columns-name',
            placeholder='Enter a column name...',
            value='',
            style={'padding': 10}
        ),
        html.Button('Add Column', id='editing-columns-button', n_clicks=0)
    ], style={'height': 50}),

    html.Div(id='data-table'),
    # html.Div(id='summary-table')
])

# @app.callback(
#     Output('summary-table', 'children'),
#     Input('dataframe', 'data')
# )
# def on_summary_table():
#     return

@app.callback(Output('data-table', 'children'),
              Input('dataframe', 'data'))
def on_data_set_table(data):
    if data is None:
        raise PreventUpdate
    df = pd.DataFrame(data)
    df_2 = pd.DataFrame(data)
    for e in df.columns[2:]:
        df_2 = df_2.append({e: df[e].mean()}, ignore_index=True)

    dash_table.DataTable(
        id='summary table',
        columns=[{"name": j, "id": j, 'deletable': True} for j in df_2.columns],
        data=df_2.to_dict('records'),
    ),

    children = dash_table.DataTable(
        id='editing-columns',
        columns=[{"name": i, "id": i, 'deletable': True} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        editable=True,
        style_cell={
            # all three widths are needed
            'textAlign': 'left',
            'minWidth': '110px', 'width': '110px', 'maxWidth': '110px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in df.to_dict('records')
        ],
        tooltip_duration=None,
        filter_action="native",
        sort_action="native",
        sort_mode="single",
        page_action="native",
        page_size=30,
    )
    return children
