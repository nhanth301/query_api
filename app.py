from flask import Flask, request, render_template, Response, jsonify
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
import json
from main3 import load_vector_db, query, img_embedding
from main2 import queryText, load_embedd, load_titles, text_embedding
from transformers import ViTImageProcessor, ViTModel
from main import print_res
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize


app = Flask(__name__)

@app.route('/queryimg', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        try:
            n = int(request.form.get('n'))
        except:
            n = 9
        if file:
            img = Image.open(BytesIO(file.read())).convert("RGB")
            input = processor(images=img, return_tensors="pt")
            output = model(**input)
            # print_res(query(output.pooler_output.detach().numpy().reshape(-1),feature_tensors,n))
            return jsonify({"ids":query(output.pooler_output.detach().numpy().reshape(-1),feature_tensors,n)})
    return render_template('upload.html')

@app.route('/querytext', methods=['GET', 'POST'])
def upload_text():
    if request.method == 'POST':
        text = request.form.get('text')
        try:
            n = int(request.form.get('n'))
        except:
            n = 9
        if text:
            model = SentenceTransformer('dangvantuan/vietnamese-embedding')
            embeded_text = model.encode(tokenize(text))
            ids = queryText(embeded_text,embeddings,n)
            return jsonify({"ids":ids})
    return render_template('upload2.html')

if __name__ == '__main__':
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    feature_tensors = load_vector_db()
    embeddings = load_embedd()
    app.run(debug=True)
