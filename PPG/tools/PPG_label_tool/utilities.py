
import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
import io
import numpy as np


try:
    from ..utilities.peak_approaches import waveform_template
except:
    from utilities.peak_approaches import waveform_template

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
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

def parse_img(image_name):
    # test_png = 'test.png'
    image_base64 = base64.b64encode(open(image_name, 'rb').read()).decode('ascii')

    src = 'data:image/png;base64,{}'.format(image_base64)
    return src

def parse_contents(contents,fname,idx=0):
    explanations = [
        "systolic.png",
        "diastolic.png",
        "amplitude.png",
        "AuC.png",
        "dicrotic.png",
        "flat.png",
        "peak_detection.png",
        "decision.png"
    ]
    data = parse_data(contents,'csv')
    return html.Div([
        html.Hr(),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Div(fname),
        html.Table(
            #Header,
            [html.Tr([html.Th(
                    # col
                html.P([
                html.Span(col,id=col),
                 dbc.Tooltip(
                     # Explanation image
                     [
                        html.Img(src=parse_img('web_img/'+ques), )#style={'height': '300px', 'width': '1000px'})
                     ]
                     , placement='top'
                     ,target=col)
                ])
                )
                for col,ques in zip(["Systolic-Peak","Diastolic-Peak","Amplitude",
                                               "Area-Under-Curve","Dicrotic-Appearance","Flatten",
                                               "Peak-Detection",
                                               "Decision"
                                               ],explanations)])] +
            [html.Tr([
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-systolic-peak' + str(idx),
                            options=[
                                {'label': 'Yes', 'value': '1'},
                                {'label': 'No', 'value': '-1'},
                                {'label': 'TBD', 'value': '0'}
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-diastolic-peak' + str(idx),
                            options=[
                                {'label': 'Yes', 'value': '1'},
                                {'label': 'No', 'value': '-1'},
                                {'label': 'TBD', 'value': '0'}
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-amplitude' + str(idx),
                            options=[
                                {'label': 'Significant Increase/Decrease', 'value': '-1'},
                                {'label': 'Stable', 'value': '1'},
                                {'label': 'TBD', 'value': '0'},
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-auc' + str(idx),
                            options=[
                                {'label': 'Narrower/Wider', 'value': '-1'},
                                {'label': 'Stable', 'value': '1'},
                                {'label': 'TBD', 'value': '0'},
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-dicrotic' + str(idx),
                            options=[
                                {'label': 'High position', 'value': '2'},
                                {'label': 'Mid position', 'value': '1'},
                                {'label': 'Low position', 'value': '-1'},
                                {'label': 'None', 'value': '-2'},
                                {'label': 'TBD', 'value': '0'}
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-flatten' + str(idx),
                            options=[
                                {'label': 'Yes', 'value': '-1'},
                                {'label': 'No', 'value': '1'},
                                {'label': 'TBD', 'value': '0'},
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-peak-detection' + str(idx),
                            options=[
                                {'label': 'Perfect', 'value': '1'},
                                {'label': 'Missed detection', 'value': '0'},
                                {'label': 'Wrong detection','value':'-1'},
                                {'label': 'Missed and  Wrong detection', 'value': '-2'},
                                # {'label': 'TBD', 'value': '0'}
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-decision' + str(idx),
                            options=[
                                {'label': 'Good', 'value': '1'},
                                {'label': 'Bad', 'value': '-1'},
                                {'label': 'TBD', 'value':'0'}
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                ])
            ]
        ),

        create_chart(data,idx),
        html.Hr()
    ])


def create_chart(data,i):
    data = np.array(data.astype("float")).reshape(1,-1)
    wave = waveform_template()
    peak_list_1, trough_list_1 = wave.detect_peak_trough_count_orig(data.reshape(-1))
    peak_list_2, trough_list_1 = wave.detect_peak_trough_kmean(data.reshape(-1))
    peak_list_3, trough_list_1 = wave.detect_peak_trough_moving_average_threshold(data.reshape(-1))
    set_peak_list_1 = set(peak_list_1)
    set_peak_list_2 = set(peak_list_2)
    set_peak_list_3 = set(peak_list_3)
    peak_list = list(set_peak_list_1.union(set_peak_list_2).union(set_peak_list_3))
    zoom_layout = go.Layout(
        yaxis=dict(
            range=[np.min(data)-3000, np.max(data)+3000],
            fixedrange= True
        ),
        xaxis=dict(
            range=[0, 500]
        )
    )
    fig = go.Figure(layout=zoom_layout)
    fig.add_traces(go.Scatter(x=np.arange(1, len(data[0])),
                              y=data[0], mode="lines"))
    fig.add_traces(go.Scatter(x=(np.array(peak_list)+1),
                              y=data.reshape(-1)[peak_list], mode="markers"))
    return dcc.Graph(id='graph_'+str(i),figure=fig,style={'height':'400px', 'width':'1200px'})