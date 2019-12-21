from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl",'rb'))

@app.route('/')
def index():
    return render_template('index.html')


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
