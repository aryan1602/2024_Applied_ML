#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:29:49 2024

@author: aryan
"""

from flask import Flask, request, jsonify
from score import *
import pickle
import os
import signal

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def get_score():
    data = request.get_json()
    print(data)
   
    prediction, propensity = predict(data['text'])
    if prediction == 1 :
        prediction = "spam"
    else:
        prediction = "Not spam"
    response = {'prediction': prediction, 'propensity': propensity}
    return jsonify(response)

@app.route('/', methods=['GET'])
def text():
    
    return "Hello World"

@app.route('/shutdown', methods=['POST'])
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return 'Server shutting down...'

def predict(text):
    model = pickle.load(open('model.pkl','rb'))
    return score(text, model, 0.5)

if __name__ == '__main__':
    app.run(debug=True)
