import os
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory, send_file
import dash_auth
import boto3
try:
    from .utilities import *
except Exception as e:
    from utilities import *

UPLOAD_DIRECTORY = "uploads"
BUCKET = os.environ['S3_BUCKET_NAME']

# Keep this out of source code repository - save in a file or a database
USERNAME = os.environ['USER_NAME']
PASSWORD = os.environ['PASSWORD']
VALID_USERNAME_PASSWORD_PAIRS = {
    USERNAME: PASSWORD
}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded                                                                                                                                                  
        multiple=True
    ),
    html.Div([
        html.Button('Confirm', id='confirm-button', n_clicks=0),
        html.H2("Exported File"),
        html.Ul(id="file-to-download"),
    ]),
    html.Div(id='save_message'),
    html.Div(id='output-image-upload'),
])

def upload_file(file_name, bucket):
    """
    Function to upload a file to an S3 bucket
    """
    object_name = file_name
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(file_name, bucket, object_name)

    return response

def make_s3_connection():
    s3 = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return s3

def download_file(file_name, bucket):
    """
    Function to download a given file from an S3 bucket
    """
    s3 = boto3.resource('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    output = f"/downloads/{file_name}"
    # s3.Bucket(bucket).download_file(file_name, output)
    s3.meta.client.download_file(bucket,file_name,file_name)
    print("download successful")
    return output

def list_files(bucket):
    """
    Function to list files in a given S3 bucket
    """
    s3 = boto3.client('s3')
    contents = []
    for item in s3.list_objects(Bucket=bucket)['Contents']:
        contents.append(item)

    return contents

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents'),
               State('upload-image', 'filename')
              ])
def update_output(images,list_of_names):
    if not images:
        return
    children = [parse_contents(images[i],fname,i) for i,fname in zip(range(len(images)),list_of_names)]
    return children

@server.route("/downloads/<path:file_path>",methods=['GET'])
def download(file_path):
    """Serve a file from the upload directory."""
    output = download_file(file_path,BUCKET)
    url = 'https://%s.s3.amazonaws.com/%s' % (BUCKET, file_path)
    print('================================')
    print(url)
    return send_from_directory(".", file_path,cache_timeout=0, as_attachment=True)

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/downloads/{}".format(urlquote(filename))
    return html.A(filename, href=location)

# @server.route('/')
def upload_to_s3(file_name,file_content):

  s3 = make_s3_connection()
  s3.upload_fileobj(io.BytesIO(bytearray(file_content,'utf-8')), BUCKET, file_name)

def parse_file_label(div_list):
    file_list = []
    systolic_list = []
    diastolic_list = []
    amplitude_list = []
    trend_list = []
    width_list = []
    auc_list = []
    dicrotic = []
    flatten = []
    peak_detection = []
    label_list = []
    for i in range(len(div_list)):
        component_dict = div_list[i]
        content_component = component_dict['props']['children']
        file_list.append(content_component[1]['props']['children'])
        label_radioitem_list = (content_component[2]['props']['children'][1]['props']['children'])

        systolic_list.append(label_radioitem_list[0]['props']['children']['props']['value'])
        diastolic_list.append(label_radioitem_list[1]['props']['children']['props']['value'])
        amplitude_list.append(label_radioitem_list[2]['props']['children']['props']['value'])
        auc_list.append(label_radioitem_list[3]['props']['children']['props']['value'])
        dicrotic.append(label_radioitem_list[4]['props']['children']['props']['value'])
        flatten.append(label_radioitem_list[5]['props']['children']['props']['value'])
        peak_detection.append(label_radioitem_list[6]['props']['children']['props']['value'])
        label_list.append(label_radioitem_list[7]['props']['children']['props']['value'])
        # label_list.append(content_component[2]['props']['value'])

    return file_list,systolic_list,diastolic_list,amplitude_list,\
           trend_list,width_list,auc_list,dicrotic,flatten,peak_detection,label_list


@app.callback(Output('file-to-download', 'children'),
            [Input('confirm-button', 'n_clicks')]+
            [Input(component_id="output-image-upload",component_property='children')])
def confirm_selection(btn_confirm,children):
    if btn_confirm > 0:
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'confirm-button' in changed_id:
            df = pd.DataFrame(columns=["file_name"])
            file_list, amplitude_list, trend_list, width_list, auc_list, dicrotic, flatten, label_list = parse_file_label(
                children)
            df["file_name"] = file_list
            df["amplitude"] = amplitude_list
            df["trend"] = trend_list
            df["width"] = width_list
            df["auc"] = auc_list
            df["dicrotic"] = dicrotic
            df["flatten"] = flatten
            df["decision"] = label_list
            name = "Download_file.csv"
            file_content = df.to_csv(index=False)

            upload_to_s3(name, file_content)
            files = [name]
            return [html.Li(file_download_link(filename)) for filename in files]
    else:
        return [html.Li("Click confirm to export csv file")]


if __name__ == '__main__':
    app.run_server(debug=True)