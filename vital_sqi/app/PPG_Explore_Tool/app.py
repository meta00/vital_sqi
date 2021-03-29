import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../PPG/tools"))
import pandas as pd

try:
    from .trim_utilities import *
except Exception as e:
    from trim_utilities import *

MIN_CYCLE = 5
MIN_MINUTES = 5
SAMPLE_RATE = 100
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

global df,df_cut
df = pd.DataFrame()
df_cut = []

global start_milestone,end_milestone
start_milestone = []
end_milestone = []

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

app.layout = html.Div([
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
    dcc.Graph(id='Mygraph'),
    # dcc.Graph(id='Trimgraph'),
    html.Div([
            html.Button('Remove Invalid Signal', id='trim_button', n_clicks=0)
            ]),
    html.Details([
        html.Div(dcc.Input(id='input-window-size', type='text')),
        html.P('window-size (Default= Min cycle * samle rate (5 * 100)'),

        html.Div(dcc.Input(id='input-peak-threshold', type='text')),
        html.P('peak-ratio-threshold (Default = 1.8 - Increase if the window-size increases)'),

        html.Div(dcc.Input(id='input-remove-sliding-window', type='text')),
        html.P('sliding window to concatenate the removal milestone (Default = 0)')
       
    ]),
    html.Div([
        html.Button('Trim By Frequency', id='trim_freq_button', n_clicks=0),
        html.Div(id='save_message')
        ]),
    html.Div(id='output-data-upload')
])

@app.callback(
    Output('Mygraph', 'figure'),
    [
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('trim_button', 'n_clicks'),
        Input('trim_freq_button', 'n_clicks'),
        dash.dependencies.State('input-window-size', 'value'),
        dash.dependencies.State('input-peak-threshold', 'value'),
        dash.dependencies.State('input-remove-sliding-window', 'value'),
    ])
def update_graph(contents, filename, btn_trim, btn_trim_freq,
                 input_window_size,input_peak_threshold,input_remove_sliding_window):
    global df,start_milestone,end_milestone,df_cut
    fig = {
        'layout': go.Layout(
            plot_bgcolor=colors["graphBackground"],
            paper_bgcolor=colors["graphBackground"])
    }

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'trim_button' in changed_id:
        print("TRIM BUTTON CLICK")
        start_milestone, end_milestone = trim_invalid_signal(df)
        start_milestone,end_milestone = remove_short_length(start_milestone,end_milestone,MIN_CYCLE*SAMPLE_RATE)
         # UPDATE FIGURE
        fig = go.Figure()
        for start, end in zip(start_milestone, end_milestone):
                fig.add_traces(
                    go.Scatter(
                        x=df["PLETH"][int(start):int(end)].index,
                        y=df["PLETH"][int(start):int(end)],
                        mode="lines"
                    ))
        return fig
    elif 'trim_freq_button' in changed_id:
        print("TRIM BUTTON FREQ CLICK")
        fig = go.Figure()
        if len(start_milestone) == 0 or len(end_milestone)==0:
            return  fig
        df_cut = []

        try:
            window_size = int(input_window_size)
        except Exception as error:
            window_size = SAMPLE_RATE * MIN_CYCLE
        try:
            peak_threshold = float(input_peak_threshold)
        except Exception as error:
            peak_threshold = None
        try:
            remove_sliding_window = int(input_remove_sliding_window)
        except Exception as error:
            remove_sliding_window = None


        for start_idx,end_idx in zip(start_milestone,end_milestone):
            df_examine = df.iloc[int(start_idx):int(end_idx)]
            start_milestone_by_freq, end_milestone_by_freq = \
                trim_by_frequency_partition(df_examine["PLETH"],
                                            start_milestone,
                                            end_milestone,
                                            window_size=window_size,
                                            peak_threshold_ratio = peak_threshold,
                                            remove_sliding_window = remove_sliding_window)

            start_milestone_by_freq, end_milestone_by_freq = \
                remove_short_length(start_milestone_by_freq, end_milestone_by_freq,
                                                                 MIN_MINUTES * 60 * SAMPLE_RATE)

            # UPDATE FIGURE
            for start, end in zip(start_milestone_by_freq, end_milestone_by_freq):
                fig.add_traces(
                    go.Scatter(
                        x=df_examine["PLETH"].iloc[int(start):int(end)].index,
                        y=df_examine["PLETH"].iloc[int(start):int(end)],
                        mode="lines"
                    ))
                df_cut.append(df_examine.iloc[int(start):int(end)])
            
        return fig

    elif 'upload-data' in changed_id:
        print("GET CONTENT")
        df = parse_data(contents, filename)
        fig = go.Figure()
        fig.add_traces(go.Scatter(x=df["PLETH"].index, y=df["PLETH"], mode="lines"))

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
