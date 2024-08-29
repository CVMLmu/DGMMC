import pandas as pd
import torch
import os
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageNet, MNIST

class ESC50Dataset(Dataset):
    def __init__(self, dir_path, file_path):
        self.path = dir_path

        df = pd.read_csv(file_path)

        labels = df['Label'].values
        filenames = df['Filename'].values

        self.filenames = filenames
        self.labels = labels
        
    def __getitem__(self, index):
        feature_path = os.path.join(self.path, '{}.pt'.format(self.filenames[index]))
        feature = torch.load(feature_path)
        label = self.labels[index]

        return feature, label, index

    def __len__(self):
        return len(self.filenames)

class CIFAR10Dataset(Dataset):
    def __init__(self, features_path, dataset_path, train):
        self.cifar10 = torchvision.datasets.CIFAR10(root=dataset_path, train=train, download=True, transform=None)
        self.features_path = features_path
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]

        features_path = os.path.join(self.features_path, '{}.pt'.format(index))
        features = torch.load(features_path)
        
        return features, target, index

    def __len__(self):
        return len(self.cifar10)
    
class CIFAR100Dataset(Dataset):
    def __init__(self, features_path, train):
        self.cifar100 = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=None)
        self.features_path = features_path
        
    def __getitem__(self, index):
        data, target = self.cifar100[index]

        features_path = os.path.join(self.features_path, '{}.pt'.format(index))
        features = torch.load(features_path)
        
        return features, target, index

    def __len__(self):
        return len(self.cifar100)
    
class ImageNetFeaturesDataset(Dataset):
    def __init__(self, dataset_dir, features_dir, split = 'train', transform= None):
        self.imagenet = ImageNet(dataset_dir, split=split, transform = transform)
        self.features_path = features_dir
        
    def __getitem__(self, index):
        image, target = self.imagenet[index]

        features_path = os.path.join(self.features_path, '{}.pt'.format(index))
        features = torch.load(features_path)
        
        return features, target, index

    def __len__(self):
        return len(self.imagenet)
    
class MNISTFeaturesDataset(Dataset):
    def __init__(self, dataset_dir, features_dir, is_train_set, transform):
        """Pytorch dataset in order to load the images from MNIST with the corresponding classes and also the features provided by pretrained model (e.g. CLIP, IMAGEBIND).

        Args:
            dataset_dir (string): Path to the directory containing the downloaded dataset on disk.
            features_dir (string): Path to the directory containing the features provided by pretrained models.
            is_train_set (bool): Boolean defining if the traing or test images are considered.
            transform (pytorch transforms) : Transform to apply on the images
        """
        self.mnist = MNIST(root=dataset_dir, train=is_train_set, download=True, transform=transform)
        self.features_path = features_dir
        
    def __getitem__(self, index):
        data, target = self.mnist[index]

        features_path = os.path.join(self.features_path, '{}.pt'.format(index))
        features = torch.load(features_path)
        
        return features, target, index

    def __len__(self):
        return len(self.mnist)