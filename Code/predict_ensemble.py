import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.model = torchvision.models.resnet50(pretrained=True)
        n_features = self.model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.model.fc = self.fc

    def forward(self, x):
        x = self.model(x)
        return x

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()

        self.model = torchvision.models.efficientnet_v2_s(pretrained = True)
        n_features = self.model.classifier[1].in_features
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


# Dataset
class BaselineTestDataset(Dataset):
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


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.52, 0.52, 0.52], std=[0.11, 0.11, 0.11])
])

#models
resnet = ResNet().to(device)
resnet.load_state_dict(torch.load('./save/ResNet/train_best_extend.pth'))
densenet = DenseNet().to(device)
densenet.load_state_dict(torch.load('./save/DenseNet/train_best_extend.pth'))
efficientnet = EfficientNet().to(device)
efficientnet.load_state_dict(torch.load('./save/EfficientNet/train_best_extend.pth'))


resnet.eval()
densenet.eval()
efficientnet.eval()

# hyperparameters
num_batches = 170

#test 
df = pd.read_csv(f'/DATA/test/test.csv')
img_path='/DATA/test/images/'
test_dataset = BaselineTestDataset(df, transform=transform, img_path=img_path)
test_loader = DataLoader(test_dataset, batch_size=num_batches, shuffle=False, num_workers=3, pin_memory=True)

preds_list = []
with torch.no_grad():
    for image in tqdm(test_loader):
        image = image.to(device)
        outputs_res = resnet(image).view(-1)
        outputs_den = densenet(image).view(-1)
        outputs_effi = efficientnet(image).view(-1)
        
        outputs_endemble = outputs_res*0.6 + outputs_effi*0.2 + outputs_den*0.2
        preds = torch.sigmoid(outputs_endemble).round()
        preds_list += preds.cpu().numpy().tolist()
        
df['risk'] = preds_list
df['risk'] = df['risk'].apply(lambda x: 'high' if x == 1 else 'low')
df.to_csv('./submission.csv', index=False)

