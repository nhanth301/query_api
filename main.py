import mysql.connector
import pandas as pd
import requests
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
from torch.nn.functional import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Thay đổi các thông số sau thành thông tin của cơ sở dữ liệu MySQL của bạn
host = 'localhost'
database = 'ltdd'
user = 'root'
password = '30113011'
vector_db_path = "vector_db.pkl"
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
modules = list(resnet18.children())[:-1]
resnet18 = torch.nn.Sequential(*modules)
resnet18.eval()

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
def download_img():
    folder_path = "images"

    # Tạo thư mục nếu nó không tồn tại
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Duyệt qua từng dòng trong DataFrame
    for index, row in df.iterrows():
        img_link = row['img_link']
        img_id = row['id']
        try:
            # Tải hình ảnh từ URL
            response = requests.get(img_link)

            # Kiểm tra nếu yêu cầu thành công (status code = 200)
            if response.status_code == 200:
                # Lấy nội dung của hình ảnh
                img_data = response.content

                # Đường dẫn lưu trữ hình ảnh mới với tên là ID
                img_path = os.path.join(folder_path, f"{img_id}.jpg")

                # Lưu hình ảnh vào thư mục và đổi tên thành ID.jpg
                with open(img_path, 'wb') as img_file:
                    img_file.write(img_data)

                print(f"Đã tải và lưu trữ hình ảnh {img_id}.jpg thành công.")
            else:
                print(f"Lỗi khi tải hình ảnh từ URL {img_link}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Lỗi xảy ra khi xử lý URL {img_link}: {e}")

def load_img():
    transform = transforms.Compose([
    transforms.Resize((86, 128)),
    transforms.ToTensor(),
    ])
    def load(path, transform):
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        return img_tensor
    images = []
    for filename in os.listdir('images'):
        id = filename.split('.')[0]
        img_tensor = load(f'images/{id}.jpg', transform)
        images.append((id,img_tensor))
    return images
def extract_feature():
    def extract(tensor, model):
        features = model(tensor.unsqueeze(0))
        features = features.view(features.size(0),-1)
        return features
    img_tensor = load_img()
    feature_list = []
    for idx, tensor in img_tensor:
        features = extract(tensor,resnet18)
        feature_list.append((idx,features.squeeze().detach().numpy()))
    with open(vector_db_path, 'wb') as file:
        pickle.dump(feature_list, file)


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

def load_vector_db():
    with open(vector_db_path, 'rb') as file:
        feature_tensors = pickle.load(file)
    return feature_tensors

def print_res(ids):
    for idx in ids:
        img = Image.open(f'images/{idx}.jpg').convert('RGB')
        plt.imshow(img)
        plt.axis('off')  # Tắt trục
        plt.show()

