from flask import Flask, request, render_template, Response
from PIL import Image
from io import BytesIO
import pickle
import mysql.connector
import pandas as pd
import requests
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
from main import cosine_similarity, load_vector_db, query


app = Flask(__name__)

@app.route('/queryimg', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(BytesIO(file.read()))
            return query(img,feature_tensors)
    return render_template('upload.html')

if __name__ == '__main__':
    vector_db_path = "vector_db.pkl"
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    modules = list(resnet18.children())[:-1]
    resnet18 = torch.nn.Sequential(*modules)
    resnet18.eval()
    feature_tensors = load_vector_db()
    app.run(debug=True)
