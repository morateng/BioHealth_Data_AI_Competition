import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

import numpy as np
import pandas as pd
from PIL import Image
import preprocessing as pre

class AugmentationDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        
        if not isinstance(sample, Image.Image):
            sample = transforms.ToPILImage()(sample)
            
        if self.transform:
            transformed_sample = self.transform(sample)
        return transformed_sample, label
    
def get_Aug_Transformations(mode='default'):
    if mode == 'default':
        default_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(), #좌우로 flip
            transforms.ToTensor(),
            ])
        return default_transform
    
    elif mode == 'crop_only':
        crop_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            ])
        return crop_transform
    
    elif mode == 'rotation_only':
        rotation_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            ])
        return rotation_transform
        
    elif mode == 'flip_only':
        flip_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(), #좌우로 flip
            transforms.ToTensor(),
            ])
        return flip_transform
    
## 사용법
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# gray_transform, gray_cluster_transform, gray_edge_transform = pre.getTransformations()

# crop_only_transform = get_Aug_Transformations('crop_only')
# rotation_transform = get_Aug_Transformations('rotation')
# flip_transform = get_Aug_Transformations('flip')

# df = pd.read_csv(f'/DATA/sample/sample.csv')

# # train / validation split with stratified sampling
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=20231101)

# for train_idx, val_idx in skf.split(df, df['risk']):
#     train_df = df.iloc[train_idx]
#     val_df = df.iloc[val_idx]
#     break

# print("augmentation 전 데이터 개수: ", len(train_idx))

# # augmentation dataset 예시
# img_path = '/DATA/sample/images/'

# original_dataset = pre.BasicTrainDataset(train_df, transform=None, img_path=img_path)
# crop_only_augmentation_dataset = pre.BasicDataset(train_df, transform=crop_only_transform, img_path=img_path)
# rotation_augmentation_dataset = pre.BasicDataset(train_df, transform=rotation_transform, img_path=img_path)
# flip_augmentation_dataset = pre.BasicDataset(train_df, transform=flip_transform, img_path=img_path)

# augmentation_dataset = torch.utils.data.ConcatDataset([original_dataset, crop_only_augmentation_dataset, rotation_augmentation_dataset, flip_augmentation_dataset])

# gray_dataset = AugmentationDataset(augmentation_dataset, transform=gray_transform)
# cluster_dataset = AugmentationDataset(augmentation_dataset, transform=gray_cluster_transform)
# edge_dataset = AugmentationDataset(augmentation_dataset, transform=gray_edge_transform)

# train_dataset = pre.CombinedTrainDataset(gray_dataset, cluster_dataset, edge_dataset)

# print("augmentation 후 데이터개수")
# print(len(augmentation_dataset))
# print(len(gray_dataset))
# print(len(cluster_dataset))
# print(len(edge_dataset))
# print(len(train_dataset))

# for images, label in gray_dataset:
#     print("gray\n", images.shape)
#     #print(images[0])
    
# for images, label in train_dataset:
#     print("combined\n", images.shape)