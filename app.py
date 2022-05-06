#!/usr/bin/env python3.9
# *-* coding: utf-8*-*

from pathlib import Path
from flask import Flask, request
from pipeline.training import classifier as cl


app = Flask(__name__)
root = Path(__file__).parent
name = root / "model/model.pkl"
model = cl.load_model(name=str(name))


@app.route('/predictions', methods=['POST'])
def predict():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        response = cl.predict(json, model)
        return response
    else:
        return 'Content-Type not supported!'


if __name__ == "__main__":
    app.run()
