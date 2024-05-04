from flask import Flask, request, render_template, Response
from PIL import Image
from io import BytesIO
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from main import cosine_similarity, load_vector_db, query
from main2 import load_titles, queryText


app = Flask(__name__)

@app.route('/queryimg', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        try:
            n = int(request.form.get('n'))
        except:
            n = 10

        if file:
            img = Image.open(BytesIO(file.read()))
            return query(img,feature_tensors,n)
    return render_template('upload.html')

@app.route('/querytext', methods=['GET', 'POST'])
def upload_text():
    if request.method == 'POST':
        text = request.form.get('text')
        try:
            n = int(request.form.get('n'))
        except:
            n = 10
        if text:
            ids = queryText(text,titles,n)
            # return [text for (idx,text) in titles if (idx in ids)]
            return ids
    return render_template('upload2.html')

if __name__ == '__main__':
    vector_db_path = "vector_db.pkl"
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    modules = list(resnet18.children())[:-1]
    resnet18 = torch.nn.Sequential(*modules)
    resnet18.eval()
    feature_tensors = load_vector_db()
    titles = load_titles()
    app.run(debug=True)
