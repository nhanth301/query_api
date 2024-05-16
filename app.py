from flask import Flask, request, render_template, Response, jsonify
from PIL import Image
from io import BytesIO
import base64
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
from main2 import queryText, load_embedd, load_titles, text_embedding, queryTextJC
from transformers import ViTImageProcessor, ViTModel
from main import print_res
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize


app = Flask(__name__)

@app.route('/queryimg', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        base64_img = request.json["img"]
        img_data = base64.b64decode(base64_img)
        try:
            n = int(request.json["n"])
        except:
            n = 9
        img = img = Image.open(BytesIO(img_data)).convert("RGB")
        input = processor(images=img, return_tensors="pt")
        output = img_model(**input)
        return jsonify({"ids":query(output.pooler_output.detach().numpy().reshape(-1),feature_tensors,n)})
    return render_template('upload.html')

@app.route('/querytext', methods=['GET', 'POST'])
def upload_text():
    if request.method == 'POST':
        text = request.json["text"]
        try:
            n = int(request.json["n"])
        except:
            n = 9
        if text:
            embeded_text = text_model.encode(tokenize(text))
            ids = queryText(embeded_text,embeddings,n)
            return jsonify({"ids":ids})
    return render_template('upload2.html')

@app.route('/querytext2', methods=['POST'])
def upload_text2():
    if request.method == 'POST':
        text = request.json["text"]
        try:
            n = int(request.json["n"])
        except:
            n = 9
        if text:
            ids = queryTextJC(text, titles, n)
            return jsonify({"ids":ids})


if __name__ == '__main__':
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    img_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    text_model = SentenceTransformer('dangvantuan/vietnamese-embedding')
    titles = load_titles()
    feature_tensors = load_vector_db()
    embeddings = load_embedd()
    app.run(debug=True)
