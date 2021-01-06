from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import torch
from PIL import Image
import albumentations as aug
from efficientnet_pytorch import EfficientNet

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = torch.load("best.pth")
model.eval()
def model_predict(file, model):
    image = Image.open(file).convert('RGB')
    image = np.array(image)
    transforms = aug.Compose([
            aug.Resize(224,224),
            aug.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225),max_pixel_value=255.0,always_apply=True),
            ])
    image = transforms(image=image)["image"]
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor([image], dtype=torch.float)
    preds = model(image)
    probs  = preds.detach().numpy()[0]
    probs = np.exp(probs)/np.sum(np.exp(probs))
    return probs


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    # Get the file from post request
    f = request.files['file']
    labs=['Cat','Dog']

    # Make prediction
    probs = model_predict(f, model)
    # result = labs[preds]
    probs = ["%.8f" % x for x in probs]
    outs = {}
    for i in range(len(labs)):
        outs[labs[i]]=probs[i]
    return outs


if __name__ == '__main__':
    app.run(debug=True)
