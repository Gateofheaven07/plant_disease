#!/usr/bin/env python
# coding: utf-8

# In[2]:


# install sekali jika belum
# !pip install flask flask-cors

from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import Predictor

app = Flask(__name__)
CORS(app)

predictor = Predictor()

@app.get("/")
def root():
    return jsonify(ok=True)

@app.post("/predict")
def predict():
    if 'file' not in request.files:
        return jsonify(error="file is required"), 400
    data = request.files['file'].read()
    return jsonify(predictor.predict_bytes(data))

# untuk menjalankan langsung dari notebook:
# from werkzeug.serving import run_simple
# run_simple("0.0.0.0", 8000, app, use_reloader=False, use_debugger=True)

