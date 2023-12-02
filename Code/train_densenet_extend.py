import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
from lion_pytorch import Lion

import sys, os
sys.path.append('/USER/RESULT/utils')
import augmentation as aug
import preprocessing as pre

torch.cuda.empty_cache()
from torch.cuda.amp import autocast
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 사용하려는 GPU 인덱스로 설정
torch.cuda.set_per_process_memory_fraction(0.9)

# 사용할 모델 클래스(DenseNet121)
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.model = torchvision.models.densenet121(pretrained = True)
        n_features = self.model.classifier.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.model.classifier = self.fc

    def forward(self, x):
        x = self.model(x)
        return x
    
def train(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    losses = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
    return losses

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 10
num_batches = 170

# transformations
base_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.52, 0.52, 0.52], std=[0.11, 0.11, 0.11])
])
flip_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(), #좌우로 flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.52, 0.52, 0.52], std=[0.11, 0.11, 0.11])
])
random_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.52, 0.52, 0.52], std=[0.11, 0.11, 0.11])
])

df = pd.read_csv(f'/DATA/train/train.csv')

# prepare dataset
img_path='/DATA/train/images/'
## data augmentation
original = pre.BasicTrainDataset(df, transform=base_transform, img_path = img_path)
flip = pre.BasicTrainDataset(df, transform=flip_transform, img_path = img_path)
random = pre.BasicTrainDataset(df, transform=random_transform,  img_path = img_path)
augmentation = torch.utils.data.ConcatDataset([original, flip, random])

train_loader = DataLoader(augmentation, batch_size=num_batches, shuffle=True, num_workers=4, pin_memory=True)


model = DenseNet().to(device)
model.load_state_dict(torch.load('./save/DenseNet/train_best.pth'))


print("Model : ", model)

criterion = nn.BCEWithLogitsLoss()
optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch: 0.99 ** epoch,last_epoch=-1,verbose=False)

best_train_loss = float('inf')
for epoch in range(num_epochs):
    train_losses = train(model, train_loader, criterion, optimizer, scheduler, device)
    print('Epoch {}, Train Loss: {:.4f}'.format(epoch+1, np.mean(train_losses)))
    if np.mean(train_losses) < best_train_loss:
        best_train_loss = np.mean(train_losses)
        
        torch.save(model.state_dict(), f'./save/DenseNet/train_best_extend.pth')