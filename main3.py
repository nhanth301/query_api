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
from transformers import ViTImageProcessor, ViTModel

host = 'localhost'
database = 'ltdd'
user = 'root'
password = '30113011'
vector_db_path = "vector_db.pkl"

def connect_db():
    connection = mysql.connector.connect(
    host=host,
    database=database,
    user=user,
    password=password
    )

    if connection.is_connected():
        print('Kết nối thành công!')
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM book")
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=cursor.column_names)
        # Đóng kết nối
        cursor.close()
        connection.close()
        print('Đã đóng kết nối.')
    else:
        df = None
        print('Kết nối không thành công.')
    return df

def img_embedding(processor, model):
    df = connect_db()
    embeddings = []
    for index, row in df.iterrows():
        image =Image.open(requests.get(row['img_link'],stream=True).raw).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        output = outputs.pooler_output
        embeddings.append((row['id'],output.detach().numpy().reshape(-1)))
        print("OK")
    with open(vector_db_path, 'wb') as file:
        pickle.dump(embeddings, file)

def load_vector_db():
    with open(vector_db_path, 'rb') as file:
        feature_tensors = pickle.load(file)
    return feature_tensors

def query(feature, feature_tensors, n=5):
    sims = [(int(id), cosine_similarity(feature, f)) for (id, f) in feature_tensors]
    sorted_sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return [idx for (idx, sim) in sorted_sims[:n]]

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
