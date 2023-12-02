import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN

import numpy as np
import pandas as pd
from PIL import Image
import sys

class BasicTrainDataset(Dataset):
    def __init__(self, df, transform=None, img_path='/DATA/train/images/'):
        self.df = df.copy()
        self.df['risk'] = self.df['risk'].apply(lambda x: 1 if x == 'high' else 0)
        self.transform = transform
        self.img_path = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, label = self.df.iloc[idx]
        img_fname = f'{self.img_path}{img_name}'
        img = Image.open(img_fname)

        if self.transform:
            img = self.transform(img)

        return img, label

class CombinedTrainDataset(Dataset):
    def __init__(self, gray_dataset, cluster_dataset, edge_dataset):
        self.gray_dataset = gray_dataset
        self.cluster_dataset = cluster_dataset
        self.edge_dataset = edge_dataset

    def __len__(self):
        return len(self.gray_dataset)

    def __getitem__(self, idx):
        # 이미지 및 레이블
        gray_img, label = self.gray_dataset[idx]
        clust_img, _ = self.cluster_dataset[idx]
        edge_img, _ = self.edge_dataset[idx]

        # 이미지를 쌓아 3채널 생성
        combined_img = torch.stack([gray_img, clust_img, edge_img])

        return combined_img, label

class CombinedTrainDataset2(Dataset):
    def __init__(self, gray_dataset, cluster_dataset):
        self.gray_dataset = gray_dataset
        self.cluster_dataset = cluster_dataset

    def __len__(self):
        return len(self.gray_dataset)

    def __getitem__(self, idx):
        # 이미지 및 레이블
        gray_img, label = self.gray_dataset[idx]
        clust_img, _ = self.cluster_dataset[idx]

        # 이미지를 쌓아 2채널 생성
        combined_img = torch.stack([gray_img, clust_img])

        return combined_img, label

class BasicPredictDataset(Dataset):
    def __init__(self, df, transform=None, img_path='/DATA/test/images/'):
        self.data = df['filename'].tolist()
        self.transform = transform
        self.img_path = img_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        img_fname = f'{self.img_path}{img_name}'
        img = Image.open(img_fname)

        if self.transform:
            img = self.transform(img)

        return img

class CombinedPredictDataset(Dataset):
    def __init__(self, gray_dataset, cluster_dataset, edge_dataset):
        self.gray_dataset = gray_dataset
        self.cluster_dataset = cluster_dataset
        self.edge_dataset = edge_dataset

    def __len__(self):
        return len(self.gray_dataset)

    def __getitem__(self, idx):
        # 이미지
        gray_img = self.gray_dataset[idx]
        clust_img = self.cluster_dataset[idx]
        edge_img = self.edge_dataset[idx]

        # 이미지를 쌓아 3채널 생성
        combined_img = torch.stack([gray_img, clust_img, edge_img])

        return combined_img

class CombinedPredictDataset2(Dataset):
    def __init__(self, gray_dataset, cluster_dataset):
        self.gray_dataset = gray_dataset
        self.cluster_dataset = cluster_dataset

    def __len__(self):
        return len(self.gray_dataset)

    def __getitem__(self, idx):
        # 이미지 및 레이블
        gray_img = self.gray_dataset[idx]
        clust_img = self.cluster_dataset[idx]

        # 이미지를 쌓아 2채널 생성
        combined_img = torch.stack([gray_img, clust_img])

        return combined_img
    
    
class OpenTransform:
    def __call__(self, gray_image_tensor):
        # 모폴로지 Open 적용
        tensor = self.open(gray_image_tensor)
        return tensor

    def open(self, gray_image_tensor):
        # 이미지 텐서를 NumPy 배열로 변환
        image_array = gray_image_tensor.numpy()
        image_array = (image_array[0] * 255).astype('uint8')
        
        # Open 연산
        kernel = np.ones((3, 3), np.uint8)
        open_array = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)

        # NumPy 배열을 이미지 텐서로 변환
        open_tensor = torch.from_numpy(open_array).float() / 255.0
        
        return open_tensor

class HistTransform:
    def __call__(self, gray_image_tensor):
        # Histogram Equalization 적용
        equalized_tensor = self.histogram_equalization(gray_image_tensor)
        return equalized_tensor

    def histogram_equalization(self, gray_image_tensor):
        # 이미지 텐서를 NumPy 배열로 변환
        image_array = gray_image_tensor.numpy()

        # 히스토그램 평활화 적용
        equalized_array = cv2.equalizeHist((image_array * 255).astype('uint8'))

        # NumPy 배열을 이미지 텐서로 변환
        equalized_tensor = torch.from_numpy(equalized_array).float() / 255.0

        return equalized_tensor

class CannyTransform:
    def __call__(self, gray_image_tensor):
        # Canny Edge Detection 적용
        edges_tensor = self.canny_edge_detection(gray_image_tensor)
        return edges_tensor

    def canny_edge_detection(self, gray_image_tensor, low_threshold=50, high_threshold=150):
        # 이미지 텐서를 NumPy 배열로 변환
        image_array = gray_image_tensor.numpy()

        # Canny Edge Detection 적용
        edges_array = cv2.Canny((image_array * 255).astype('uint8'), low_threshold, high_threshold)

        # NumPy 배열을 이미지 텐서로 변환
        edges_tensor = torch.from_numpy(edges_array).float() / 255.0

        return edges_tensor

class DBSCANTransform:
    def __call__(self, gray_image_tensor):
        # DBSCAN 클러스터링을 적용하고 군집에 따라 색상 지정
        clustered_tensor = self.DBSCAN_clustering(gray_image_tensor)
        return clustered_tensor

    def DBSCAN_clustering(self, gray_image_tensor, eps=0.5, min_samples=5):
        # 이미지 텐서를 NumPy 배열로 변환
        image_array = gray_image_tensor.numpy()
        image_array = (image_array * 255).astype('uint8')

        # 2D 이미지를 1D 배열로 펼치기
        flattened_array = image_array.ravel()

        # DBSCAN 클러스터링 적용
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(flattened_array.reshape(-1, 1))

        # 군집 레이블을 이미지로 변환
        clustered_array = labels.reshape(image_array.shape)

        # NumPy 배열을 이미지 텐서로 변환
        clustered_tensor = torch.from_numpy(clustered_array).float() / max(labels)

        return clustered_tensor

class HDBSCANTransform():
    def __init__(self, CLUSTER):
        self.CLUSTER = CLUSTER
    
    def __call__(self, gray_image_tensor):
        # DBSCAN 클러스터링을 적용하고 군집에 따라 색상 지정
        clustered_tensor = self.HDBSCAN_clustering(gray_image_tensor)
        return clustered_tensor

    def HDBSCAN_clustering(self, gray_image_tensor):
        # 이미지 텐서를 NumPy 배열로 변환
        image_array = gray_image_tensor.numpy()
        image_array = (image_array * 255).astype('uint8')

        # 2D 이미지를 1D 배열로 펼치기
        flattened_array = image_array.ravel()

        # HDBSCAN 클러스터링 적용
        labels = self.CLUSTER.fit_predict(flattened_array.reshape(-1, 1))

        # 군집 레이블을 이미지로 변환
        clustered_array = labels.reshape(image_array.shape)

        # NumPy 배열을 이미지 텐서로 변환
        clustered_tensor = torch.from_numpy(clustered_array).float() / max(labels)

        return clustered_tensor

def getTransformations(CLUSTER=HDBSCAN()):
    gray_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Grayscale(),
        OpenTransform()
    ])

    gray_Hist_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Grayscale(),
        OpenTransform(),
        HistTransform()
    ])

    gray_cluster_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Grayscale(),
        OpenTransform(),
        HistTransform(),
        HDBSCANTransform(CLUSTER)
    ])

    gray_edge_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Grayscale(),
        OpenTransform(),
        HistTransform(),
        CannyTransform()
    ])

    return gray_transform, gray_Hist_transform, gray_cluster_transform, gray_edge_transform


### 사용법
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# gray_transform, gray_cluster_transform, gray_edge_transform = getTransformations()

# # Load the data
# df = pd.read_csv(f'/DATA/sample/sample.csv')

# # train / validation split with stratified sampling
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=20231101)

# for train_idx, val_idx in skf.split(df, df['risk']):
#     train_df = df.iloc[train_idx]
#     val_df = df.iloc[val_idx]
#     break

# # dataset 예시
# img_path = '/DATA/sample/images/'
# gray_dataset = BasicDataset(train_df, transform=gray_transform, img_path=img_path)
# cluster_dataset = BasicDataset(train_df, transform=gray_cluster_transform, img_path=img_path)
# edge_dataset = BasicDataset(train_df, transform=gray_edge_transform, img_path=img_path)

# train_dataset = CombinedDataset(gray_dataset, cluster_dataset, edge_dataset)