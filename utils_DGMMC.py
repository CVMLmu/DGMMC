import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src_DGMMC.DGMMC import DGMMC_spherical
import math

class DGMMClassifier(torch.nn.Module):
    def __init__(self, out_features, nb_classes, nb_components = 3, init_means=None):
        super().__init__()

        self.DGMMC = DGMMC_spherical(features_dim=out_features, nb_classes=nb_classes, nb_components=nb_components, init_means=init_means)

    def forward(self, x):

        x = self.DGMMC(x)

        return x

class CrossEntropy(torch.nn.Module):
    reduction: str

    def __init__(self, label_smoothing = 0):
        super().__init__()

        assert label_smoothing>=0 or label_smoothing <=1

        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            target = target * (1-self.label_smoothing) + self.label_smoothing / target.shape[1]
        mul = target * input
        sum = -torch.sum(mul, 1)
        avg =  torch.mean(sum)
        return avg
    
def get_means_bandwidth_from_features(nb_classes, loader, pca = None, d = None):
    
    dict_features = {}

    for i in range(0, nb_classes):
        dict_features[i] = []

    with torch.no_grad():
        with tqdm(range(math.ceil(len(loader))), desc="Processing : ", unit='batchs') as pbar:
            for i, data in enumerate(loader,0):
                features, labels = data[0], data[1]
                features  =torch.flatten(features, start_dim=1)

                if pca is not None:
                    features = features.numpy()
                    features = pca.transform(features)
                    features = torch.from_numpy(features).float()

                if d is not None:
                    features = features[:, 0:d]

                for j in range(0, len(labels)):
                    dict_features[labels[j].item()].append(features[j])

                pbar.update()
            
            pbar.close()

    means = []
    stds = []
    for i in range(0, nb_classes):
        dict_features[i] = torch.stack(dict_features[i], dim=0)
        dict_features[i] = torch.mean(dict_features[i], dim=0)
        means.append(dict_features[i])
        stds.append(torch.std(dict_features[i]))

    means = torch.stack(means, dim=0)
    
    stds = torch.stack(stds, dim=0)

    return means, stds

def train_from_features_PCA(n_classes, device, model, loader, criterion, optimizer, pca = None, d = None):
    index = 0
    total_loss = 0
    total_accuracy = 0

    with tqdm(range(len(loader))) as pbar:
        for (inputs, labels, _) in loader:

            inputs = inputs.float()

            if pca is not None:
                inputs = inputs.numpy()
                inputs = pca.transform(inputs)
                if d is not None:
                    inputs = inputs[:, 0:d]
                inputs = torch.from_numpy(inputs).float()
            
            #inputs = inputs[:, 0:d]

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            target = torch.nn.functional.one_hot(labels, n_classes)

            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            total_loss += loss.item()
            total_accuracy += (predicted == labels).sum().item()
            index +=labels.size(0)

            pbar.update()

    pbar.close()

    return model, total_loss/len(loader), total_accuracy/index

def test_from_features_PCA(n_classes, device, model, loader, criterion, pca = None, d=None):
    index = 0
    total_loss = 0
    total_accuracy = 0

    with tqdm(range(len(loader))) as pbar:
        for (inputs, labels, _) in loader:

            inputs = inputs.float()

            if pca is not None:
                inputs = inputs.numpy()
                inputs = pca.transform(inputs)
                if d is not None:
                    inputs = inputs[:, 0:d]
                inputs = torch.from_numpy(inputs).float()
            
            #inputs = inputs[:, 0:d]


            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            target = torch.nn.functional.one_hot(labels, n_classes)

            loss = criterion(outputs, target)

            _, predicted = torch.max(outputs.data, 1)

            total_loss += loss.item()
            total_accuracy += (predicted == labels).sum().item()
            index +=labels.size(0)

            pbar.update()

    pbar.close()

    return total_loss/len(loader), total_accuracy/index