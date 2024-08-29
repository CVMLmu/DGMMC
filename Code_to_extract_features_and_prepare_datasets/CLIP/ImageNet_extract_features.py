import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from PIL.Image import BICUBIC
from torchvision import datasets, transforms
from torchvision.datasets import ImageNet

class MyDataset(Dataset):
    def __init__(self, split = 'train', transform= None):
        self.imagenet = ImageNet(IMAGENET_PATH, split=split, transform = transform)
        
    def __getitem__(self, index):
        image, target = self.imagenet[index]
        
        return image, target, index

    def __len__(self):
        return len(self.imagenet)
    
IMAGENET_PATH = os.path.join('/home/jeremy/Documents/Datasets/ImageNet')

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)

preprocess = transforms.Compose([
    transforms.Resize(224, BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
train_set = MyDataset('train', preprocess)
val_set = MyDataset('val', preprocess)
#test_set = MyDataset('test', preprocess)

features_dir= os.path.join('datasets', 'imagenet_clip_features')

# train
train_dir = os.path.join(features_dir, 'train')
train_loader = DataLoader(train_set, batch_size=64, shuffle=False)

with torch.no_grad():
    for data in tqdm(train_loader):

        images, labels, index = data[0], data[1], data[2]

        images = images.to(device)

        features = model.encode_image(images)

        features = features.cpu()

        for j in range(len(features)):
            feature_path = os.path.join(train_dir, '{}.pt'.format(index[j]))
            torch.save(features[j], feature_path)

# validation
val_dir = os.path.join(features_dir, 'val')
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

with torch.no_grad():
    for data in tqdm(val_loader):

        images, labels, index = data[0], data[1], data[2]

        images = images.to(device)

        features = model.encode_image(images)

        features = features.cpu()

        for j in range(len(features)):
            feature_path = os.path.join(val_dir, '{}.pt'.format(index[j]))
            torch.save(features[j], feature_path)