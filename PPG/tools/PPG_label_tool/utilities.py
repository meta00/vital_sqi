import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
import pandas as pd
import io
import numpy as np

def create_chart(data,i):
    data = np.array(data.astype("float")).reshape(1,-1)
    zoom_layout = go.Layout(
        yaxis=dict(
            range=[-30000, 30000],
            fixedrange= True
        ),
        xaxis=dict(
            range=[0, 500]
        )
    )
    fig = go.Figure(layout = zoom_layout)
    fig.add_traces(go.Scatter(x=np.arange(1, len(data[0])),
                              y=data[0], mode="lines"))
    return dcc.Graph(id='graph_'+str(i),figure=fig,style={'height':'400px', 'width':'1200px'})

def parse_file_label(div_list):
    file_list = []
    amplitude_list = []
    trend_list = []
    width_list = []
    auc_list = []
    dicrotic = []
    flatten = []
    label_list = []
    for i in range(len(div_list)):
        component_dict = div_list[i]
        content_component = component_dict['props']['children']
        file_list.append(content_component[1]['props']['children'])
        label_radioitem_list = (content_component[2]['props']['children'][1]['props']['children'])
        amplitude_list.append(label_radioitem_list[0]['props']['children']['props']['value'])
        trend_list.append(label_radioitem_list[1]['props']['children']['props']['value'])
        width_list.append(label_radioitem_list[2]['props']['children']['props']['value'])
        auc_list.append(label_radioitem_list[3]['props']['children']['props']['value'])
        dicrotic.append(label_radioitem_list[4]['props']['children']['props']['value'])
        flatten.append(label_radioitem_list[5]['props']['children']['props']['value'])
        label_list.append(label_radioitem_list[6]['props']['children']['props']['value'])

    return file_list,amplitude_list,trend_list,width_list,auc_list,dicrotic,flatten,label_list



def parse_img(image_name):
    # test_png = 'test.png'
    image_base64 = base64.b64encode(open(image_name, 'rb').read()).decode('ascii')

    src = 'data:image/png;base64,{}'.format(image_base64)
    return src

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), header=None)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
    except Exception as e:
        print(e)
    return df

def parse_contents(contents,fname,idx=0):
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
                     html.Img(src=parse_img('my-image.png'), style={'height': '200px', 'width': '200px'})
                     # app.get_asset_url('my-image.png')
                     , placement='top'
                     ,target=col)
                ])
            )
                      for col in ["Amplitude","Baseline-Trend","Baseline-Width",
                                               "Area-Under-Curve","Dicrotic-Appearance","Flatten",
                                               "Decision"
                                               ]])] +
            [html.Tr([
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-amplitude' + str(idx),
                            options=[
                                {'label': 'Increase', 'value': '1'},
                                {'label': 'Decrease', 'value': '-1'},
                                {'label': 'Stable', 'value': '0'},
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-trend' + str(idx),
                            options=[
                                {'label': 'Increase', 'value': '1'},
                                {'label': 'Decrease', 'value': '-1'},
                                {'label': 'Both', 'value':2},
                                {'label': 'Stable', 'value': '0'},
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-width' + str(idx),
                            options=[
                                {'label': 'Increase', 'value': '1'},
                                {'label': 'Decrease', 'value': '-1'},
                                {'label': 'Both', 'value': 2},
                                {'label': 'Stable', 'value': '0'},
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-auc' + str(idx),
                            options=[
                                {'label': 'Narrower', 'value': '1'},
                                {'label': 'Wider', 'value': '-1'},
                                {'label': 'Both', 'value': 2},
                                {'label': 'Stable', 'value': '0'},
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-dicrotic' + str(idx),
                            options=[
                                {'label': '1 Peak', 'value': '1'},
                                {'label': 'More than 2 peaks', 'value': '3'},
                                {'label': 'Both systolic and acrolic', 'value': '2'},
                            ],
                            value='2',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-flatten' + str(idx),
                            options=[
                                {'label': 'Having flat valley', 'value': '1'},
                                {'label': 'No flat valley', 'value': '0'},
                            ],
                            value='0',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    html.Td(
                        dcc.RadioItems(
                            id='radio-item-decision' + str(idx),
                            options=[
                                {'label': 'Acceptable', 'value': '1'},
                                {'label': 'Non Acceptable', 'value': '0'},
                                {'label': 'TBD', 'value':'-1'}
                            ],
                            value='-1',
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                ])
            ]
        ),

        create_chart(data,idx),
        html.Hr()
    ])
