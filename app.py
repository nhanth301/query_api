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


def cosine_similarity(vector1, vector2):
    """
    Tính cosine similarity giữa hai vector.

    Tham số:
    vector1 (numpy array): Vector thứ nhất.
    vector2 (numpy array): Vector thứ hai.

    Trả về:
    float: Giá trị cosine similarity giữa hai vector.
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity
def load_vector_db():
    with open(vector_db_path, 'rb') as file:
        feature_tensors = pickle.load(file)
    return feature_tensors
def query(image, feature_tensors, n=5):
    transform = transforms.Compose([
        transforms.Resize((86, 128)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image.convert('RGB'))
    feature = resnet18(img_tensor.unsqueeze(0)).squeeze().detach().numpy()
    sims = [(int(id), cosine_similarity(feature, f)) for (id, f) in feature_tensors]
    sorted_sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return [idx for (idx, sim) in sorted_sims[:n]]

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
