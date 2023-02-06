from flask import Flask, render_template, request, flash, url_for
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# loading the csv data to a Pandas DataFrame
dataset_data = pd.read_csv('MODIFIED.csv')
X = dataset_data[['DAY', 'MONTH', 'YEAR']]
Y = dataset_data['Power']
X_train, X_test, y_train, y_test = train_test_split(X, Y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)
print(X_test)


df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})
print(df)
graph = df.head(25)

app = Flask(__name__)


@app.route('/')
def intro1():
    return render_template('intr.html')


@app.route('/index')
def index():
    return render_template('index1.html')


@app.route('/ack')
def ack():
    return render_template('aboutus1.html')


@app.route('/res')
def res():
    return render_template('research1.html')


@app.route('/data')
def data():
    return render_template('data1.html')


@app.route('/result', methods=['POST'])
def result():
    input_data = [int(x) for x in request.form.values()]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    y_pred = regressor.predict(input_data_reshaped)
    y_pred = round((y_pred[0]), 2)
    print(y_pred)
    return render_template("predict1.html", result=(y_pred))


if __name__ == "__main__":
    app.run(debug=True)
