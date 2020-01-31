from flask import Flask, render_template, request
import pickle
import numpy as np
import base64
from PIL import Image
from io import BytesIO
#from skimage.io import imsave, imread, imresize
from keras.models import load_model
#import tensorflow as tf
#import re
#import os
#import sys

app = Flask(__name__)
#global cnn_model, graph, model

model = pickle.load(open("model.pkl",'rb'))

cnn_model = load_model('model.h5')
#graph = tf.get_default_graph()

'''def convertImage(imgData):
    imgstr  = re.search(r'base64,(.*'.imgData1).group(1)
    with open('output.png','wb') as output:
        output.wite(imgstr.decode('base64'))'''

def convertImage(imgData):
    im = Image.open(BytesIO(base64.b64decode(imgData)))
    im.save('output.png', 'PNG')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/digit_recognize')
def digit_recognition():
    return render_template('digit_recognizer.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    result = "5"
    imgData = request.get_data()
    convertImage(imgData)
    '''x= imread('out.png',mode='L')
    x=np.invert(x)
    x = imresize(x,28,28)
    x = x.reshape(1,28,28,1)
    with graph.as_default():
        out = cnn_model.predict(x)
        result = np.array_str(np.argmax(out))'''
    return result

@app.route('/calculate', methods = ['POST'])
def calculate_interest_rate():
    if request.method == 'POST':
        result = request.form
        features = {}
        for key,value in result.items():
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
            output = round(prediction[0][0],2)
        else:
            output = "Some Error Occured"
        return render_template("index.html" , prediction = "You should get loan @{}%".format(output))


if __name__ == '__main__':
    app.run()
