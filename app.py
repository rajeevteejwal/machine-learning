from flask import Flask, render_template, request, Response, jsonify
import pickle
import numpy as np
import base64
from PIL import Image
#from keras.models import load_model
from tensorflow import keras
import re
import io
#from io import BytesIO
#from skimage.io import imsave, imread#, imresize
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import cv2
from camera import VideoCamera

#import os
#import sys

app = Flask(__name__)
video_stream = VideoCamera()
global cnn_model, graph, model

tf_config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session(config=tf_config)

model = pickle.load(open("model.pkl", 'rb'))
set_session(sess)
cnn_model = keras.models.load_model('model.h5')
graph = tf.compat.v1.get_default_graph()


def convertImage(imgData):
    img_str = re.search(r'base64,(.*)', imgData).group(1)
    return img_str


'''def convertImage(imgData):
    im = Image.open(BytesIO(base64.b64decode(imgData)))
    im.save('output.png', 'PNG')
'''


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/digit_recognize')
def digit_recognition():
    return render_template('digit_recognizer.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    result = "5"
    img_data = request.get_data()
    img_data = img_data.decode('utf-8')
    img_str = convertImage(img_data)
    image_bytes = io.BytesIO(base64.b64decode(img_str))
    im = Image.open(image_bytes).convert('L')
    #arr = np.array(im)[:, :, 0]
    #x= imread('out.png',mode='L')
    #x=np.invert(im)
    #x = imresize(x,28,28)
    #x = x.reshape(1,28,28,1)
    img = im.resize((28, 28), Image.ANTIALIAS)
    pixels = np.asarray(img, dtype='uint8')
    pixels = np.invert(pixels)
    pixels = np.resize(pixels, (28, 28))
    x = pixels.reshape((28, 28, 1))
    x = np.expand_dims(x, axis=0)
    with graph.as_default():
        set_session(sess)
        out = cnn_model.predict(x)
        result = np.argmax(out, axis=1)[0]
        return str(result)


@app.route('/calculate', methods=['POST'])
def calculate_interest_rate():
    if request.method == 'POST':
        result = request.form
        features = {}
        for key, value in result.items():
            if key == 'home-type':
                print(result.get(key))
                if int(result.get(key)) == 1:
                    features["MORT"] = 0
                    features["OWN"] = 0
                    features["RENT"] = 1
                if int(result.get(key)) == 2:
                    features["MORT"] = 1
                    features["OWN"] = 0
                    features["RENT"] = 0
                if int(result.get(key)) == 3:
                    features["MORT"] = 0
                    features["OWN"] = 1
                    features["RENT"] = 0
            else:
                features[key] = float(result.get(key))

        input_feat = [np.array(list(features.values()))]
        prediction = model.predict(input_feat)
        output = None
        if prediction:
            output = round(prediction[0][0], 2)
        else:
            output = "Some Error Occured"
        return render_template("index.html", prediction="You should get loan @{}%".format(output))


@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                mimetype='multipart/x-mixed-replace; boundary=frame')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':
    app.run()