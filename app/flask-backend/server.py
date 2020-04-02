from math import expm1

import pandas as pd
from flask import Flask, jsonify, request, render_template
from tensorflow import keras
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input
import json

import pandas as pd



app = Flask(__name__)

def echo():
    return "hello from test"

def getTopPredictions(preds):
    predsDict = {
        'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0,
        'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 0, 'N': 0,
        'O': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0,
        'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0, 'Å': 0,
        'Ä': 0, 'Ö': 0,
    }
    # map preds index with probability to correct letter from dictonary
    for i, key in enumerate(predsDict, start=0):
        predsDict[key] = preds[i]

    # sort by dictonary value and returns as list
    all_preds = sorted(predsDict.items(), reverse=True, key=lambda x: x[1])
    top_preds = [all_preds[0], all_preds[1], all_preds[2]]
    return top_preds, all_preds

def getModel():
    model = keras.models.load_model("./model96.hdf5")
    return model

def predictSingleImage(filepath, model):
    inputImage = Image.open(filepath)
    inputImage = inputImage.resize((224, 224))

    imageDataArray = image.img_to_array(inputImage)
    imageDataArray = np.expand_dims(imageDataArray, axis=0)
    imageDataArray = preprocess_input(imageDataArray)

    predictions = model.predict(imageDataArray)
    return predictions

def serialisePreds(predictions):
    topPreds = []
    for i, pred in enumerate(predictions[0], start=0):
        letter, accuracy = predictions[0][i]
        topPreds.append((letter, round(accuracy*100, 2)))
    return topPreds

# curl -d 'hello' -H "Content-Type: application/json" -X POST 192.168.86.189:5000
#
@app.route("/", methods=["POST"])
def index():
    return 'Welcome to our KEX'

@app.route("/predict", methods=["POST"])
def predict():
    loaded_model = getModel()
    predictions = predictSingleImage("./G1.jpg", loaded_model)
    predictions = getTopPredictions(predictions[0])
    predictions = serialisePreds(predictions)
    response = {'predictions': predictions, 'letterSent': 'G'}
    return jsonify(response)
