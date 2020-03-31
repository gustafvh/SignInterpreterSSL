from math import expm1

import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras


app = Flask(__name__)

def getModel():
    model = keras.models.load_model("../../output/model-weights-best-96.hdf5")

    return model.summary()

# curl -d '{first: 566}' -H "Content-Type: application/json" -X POST http://localhost:5000
@app.route("/", methods=["POST"])
def index():
    modelInfo = getModel()
    return jsonify(modelInfo)

@app.route("/predict", methods=["POST"])
def predict():
    return jsonify("predict")
