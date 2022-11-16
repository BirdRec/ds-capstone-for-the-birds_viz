import os
import pandas as pd
import numpy as np
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_reusable_components as drc
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import tensorflow as tf

external_stylesheets = [dbc.themes.PULSE]

################################################################################
# APP INITIALIZATION
################################################################################
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# this is needed by gunicorn command in procfile
server = app.server

################################################################################
# IMPORT DATA
################################################################################
# Get absolute path of repository
path_repo = os.path.dirname(os.path.abspath('app'))

# Load models
path_model_150 = os.path.join(path_repo, 'models', 'MobileNetV3L_europe-40_p150_e75')
path_model_300 = os.path.join(path_repo, 'models', 'MobileNetV3L_europe-40_p300_e10')
model_150 = tf.keras.models.load_model(path_model_150)
model_300 = tf.keras.models.load_model(path_model_300)

# Load labels
df = pd.read_csv(os.path.join(path_repo, 'labelmap_europe-40.txt'), header=None, names=['species'])
labelmap = df.species.tolist()

# Preallocate click-counter
cc = 0

################################################################################
# LAYOUT
################################################################################
app.layout = dbc.Container([
    # First Row
    dbc.Row([
        dbc.Col([html.Img(id='logo',
                          src=app.get_asset_url('bird_branch.svg'))
                 ], width=3),
        dbc.Col([html.H1(id='title',
                         children='BirdRec Interactive Dash Plotly Dashboard',
                         style={'textAlign': 'center'})
                 ], width=6),
        dbc.Col([])
    ]),

    dbc.Row([
        # dbc.Col([html.Div(id='output-image-upload')], width=8),
        dbc.Col([html.Hr(),
                 html.Div(id='output-image-upload')], width=8),

        dbc.Col([html.Hr(),
                 html.Div('1) Upload a picture:', style={'margin-top': '20px',
                                                         'margin-bottom': '20px'}),
                 html.Div([
                     dcc.Upload(id='upload-image', children=dbc.Button('Upload Image', color='primary', size='lg',
                                                                       style={"width": "100%",
                                                                              "height": "50px", }
                                                                       )),
                     #dcc.Upload(id='upload-image',
                     #           children=["Drag and Drop or ",
                     #                     html.A(children="Select an Image"),
                     #                     ],
                     #           style={"width": "100%",
                     #                  "height": "50px",
                     #                  "lineHeight": "50px",
                     #                  "borderWidth": "1px",
                     #                  "borderStyle": "dashed",
                     #                  "borderRadius": "5px",
                     #                  "borderColor": "darkgray",
                     #                  "textAlign": "center",
                     #                  "margin": "10px", }
                     #           )
                 ],),
                 #
                 html.Div(style={'margin-top': '20px', 'margin-bottom': '50px'}),
                 html.Div('2) Choose a model:'),
                 html.Div([
                     dcc.Dropdown(id='my-dropdown',
                                  options={
                                      'model_150': '150x150 MobileNetV3L',
                                      'model_300': '300x300 MobileNetV3L',
                                  },
                                  value='150x150 MobileNetV3L',
                                  style={"width": "100%",
                                         "height": "50px",
                                         # "lineHeight": "50px",
                                         # "borderWidth": "1px",
                                         # "borderStyle": "dashed",
                                         "borderRadius": "5px",
                                         "borderColor": "darkgray",
                                         # "textAlign": "center",
                                         "margin": "10px", }
                                  )
                 ]),
                 html.Div(style={'margin-top': '50px', 'margin-bottom': '50px'}),
                 html.Div('3) Predict the photo:'),
                 html.Div(style={'margin-top': '20px', 'margin-bottom': '20px'}),
                 html.Div(children=dbc.Button('Classify!', id='my_button', n_clicks=0, color='primary', size='lg',
                                              style={"width": "100%",
                                                     "height": "50px",}
                                              ),
                          #style={"width": "100%",
                          #       "height": "50px", }
                          ),
                 #html.Div(style={'margin-top': '20px', 'margin-bottom': '20px'}),
                 #html.Div(dcc.Graph(id='prediction_chart', figure={})),
                 #
                 ], width=4)

    ]),

    dbc.Row([
        dbc.Col([html.Div(dcc.Graph(id='prediction_chart', figure={})),
        ], width=8),
    ]),

])


################################################################################
# INTERACTION CALLBACKS
################################################################################
# https://dash.plotly.com/basic-callbacks

# https://dash.plotly.com/dash-core-components/upload
@app.callback([Output(component_id='output-image-upload', component_property='children'),
               Output(component_id='prediction_chart', component_property='figure')],
              [Input(component_id='upload-image', component_property='contents'),
              State(component_id='upload-image', component_property='filename'),
               Input(component_id='my-dropdown', component_property='value'),
              Input(component_id='my_button', component_property='n_clicks')],
              prevent_initial_call=True
              )
# Main Function of Callback
def parse_contents(contents, filename, value, n_clicks):
    # print properties of imported image
    print('contents: ', contents)
    print('type: ', type(contents))
    print('filename: ', filename)

    # We don't have access to path of uploaded image due to some inbuilt browser security reasons preventing access
    # the filesystem directly - all we get is a cryptic string of type base 64. With the help of
    # dash_reusable_components.py we can transform it to a numpy array.

    # split contents string object into its components
    content_type, content_string = contents.split(',')
    print('type: ', content_type)
    print('string: ', content_string)
    # convert b64 object to numpy
    img = drc.b64_to_numpy(content_string, to_scalar=False)
    print('Image properties:')
    print(type(img))
    print(img.shape)

    # check, which model was chosen
    print('model:')
    print('value: ', value)
    if value == 'model_150':
        model = model_150
    elif value == 'model_300':
        model = model_300

    # predict
    pred_vals, pred_label = predict(model, value, labelmap, img)

    # plot
    fig = bar_plot(pred_vals, pred_label)

    # get access to click-counter
    global cc

    # return, if button has been clicked
    if n_clicks > cc:
        # update click-counter
        cc = n_clicks

        return html.Div([
            html.Img(src=contents, style={  # 'height':'10%',
                'width': '80%',
            }),
            html.H5(filename),
            html.Hr(),
        ], style={'textAlign': 'center'}
        ), fig


# Nested Helper Functions for Main Function:
# - predict
# - bar_plot
def predict(model, value, label, image):
    '''
    image: image as numpy array
    label: labelmap (as list)
    model: the model
    value: str from dropdown menu
    '''
    if value == 'model_150':
        IMG_SHAPE = 150
    elif value == 'model_300':
        IMG_SHAPE = 300

    # preprocessing
    # convert numpy array to tf tensor object
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, size=[IMG_SHAPE, IMG_SHAPE])  # resize the image

    # prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # get 3 indices with the highest values and sort descending
    ind = np.argpartition(pred.flatten(), -3)[-3:]
    top3 = pred.flatten()[ind]

    # create list of tuples with indices and values
    top = []
    for idx, val in zip(ind, top3):
        top.append((val, idx))

    # per default, it will be sorted by first entry of tuples, but ascending
    top.sort(reverse=True)

    #
    values = [top[i][0] * 100 for i in range(3)]
    names = [label[top[i][1]] for i in range(3)]

    return values, names


def bar_plot(values, names):
    z = [12, 24, 48]  # colors of the viridis colormap
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker=dict(color=z,
                    colorscale='viridis')
    )
    )

    # Adding labels
    annotations = []
    y_s = np.round(values, decimals=2)

    for yd, xd in zip(y_s, names):
        # labeling the bar
        annotations.append(dict(
            y=xd, x=yd + 5,
            text=str(yd) + '%',
            font=dict(family='Arial', size=16,
                      # color='rgb(50, 171, 96)'
                      ),
            showarrow=False
        ))
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(visible=False, showticklabels=False)
    # fig.show()

    return fig


# Add the server clause:
if __name__ == "__main__":
    app.run_server()
