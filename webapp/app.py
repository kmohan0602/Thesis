import numpy as np
# import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='templates')
# model = pickle.load(open('../outputs/azure_devops_test.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    ## For rendering results on HTML GUI

    print('in predict fn')

if __name__ == '__main__':
	app.run()