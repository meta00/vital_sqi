import pandas as pd
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
df = pd.read_csv('./data-analysis.csv')
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        dcc.Input(
            id='editing-columns-name',
            placeholder='Enter a column name...',
            value='',
            style={'padding': 10}
        ),
        html.Button('Add Column', id='editing-columns-button', n_clicks=0)
    ], style={'height': 50}),

    dash_table.DataTable(
        id='editing-columns',
        columns=[{"name": i, "id": i, 'deletable': True} for i in df.columns],
        data=df.to_dict('records'),
        editable=True,
        style_cell={'textAlign': 'left'},
        filter_action="native",
        sort_action="native",
        sort_mode="single",
    ),
])
if __name__ == '__main__':
    app.run_server(debug=True);