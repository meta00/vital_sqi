import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from vital_sqi.app.util.parsing import parse_data
from vital_sqi.app.app import app
from vital_sqi.app.views import dashboard1,dashboard2, dashboard3
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Menu", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Dashboard 1",id='dashboard_1_link', disabled=True,
                            href="/views/dashboard1", active="exact"),
                dbc.NavLink("Dashboard 2", id='dashboard_2_link',disabled=True,
                            href="/views/dashboard2", active="exact"),
                dbc.NavLink("Dashboard 3", id='dashboard_3_link',disabled=True,
                            href="/views/dashboard3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
    # Store dataframe
    dcc.Store(id='dataframe', storage_type='local'),
    dcc.Store(id='rule-set-store', storage_type='local'),
    dcc.Store(id='rule-dataframe', storage_type='local'),
    dcc.Location(id='url', refresh=False),
    sidebar,
    content
])

home_content = html.Div([
    html.H2("SQIs Table"),
    dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '50px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '2px',
                'textAlign': 'center',
                'margin': '2px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
    html.H2("Rule Table (Optional)"),
    dcc.Upload(
            id='upload-rule',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '50px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '2px',
                'textAlign': 'center',
                'margin': '2px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
    # dbc.Progress(id='upload-progress',striped= True,animated= True)
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/views/dashboard1':
         return dashboard1.layout
    elif pathname == '/views/dashboard2':
         return dashboard2.layout
    elif pathname == '/views/dashboard3':
        return dashboard3.layout
    else:
        return home_content

@app.callback(
              Output('dataframe','data'),
              Output('dashboard_1_link','disabled'),
              Output('dashboard_2_link','disabled'),
              Output('dashboard_3_link','disabled'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('dataframe', 'data')
)
def update_output(content, filename, last_modified,state_data):
    if content is not None:
        df = parse_data(content,filename)
        return [df,False,False,False]
    elif state_data is not None:
        return [state_data,False,False,False]
    return [None,True,True,True]

#Load rule set
@app.callback(
              Output('rule-set-store','data'),
              Input('upload-rule', 'contents'),
              State('upload-rule', 'filename'),
              State('upload-rule', 'last_modified'),
              State('rule-set-store', 'data')
)
def upload_rule(content, filename, last_modified,state_data):
    if content is not None:
        df = parse_data(content,filename)
        return df
    elif state_data is not None:
        return state_data
    return None


if __name__ == '__main__':
    app.run_server(debug=True)