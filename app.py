from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods = ['POST'])
def calculate_interest_rate():
    if request.method == 'POST':
        result = request.form
        print(result)
        return render_template("show_interest_rate.html" , result=result)


if __name__ == '__main__':
    app.run()
