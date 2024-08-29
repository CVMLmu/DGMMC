import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src_SDGM.SDGM import SDGM


class SDGMClassifier(torch.nn.Module):
    def __init__(self, out_features, nb_classes, nb_components = 3,covtype = 'diag'):
        super().__init__()

        self.SDGM = SDGM(out_features, nb_classes, nb_components, covtype)

    def forward(self, x):

        x = self.SDGM(x)

        return x
    
def get_kl_weight(epoch, max_epoch): 
    return min(1, 1e-9 * epoch / max_epoch)

def train_from_features_PCA(epoch, n_epochs, device, model, loader, criterion, optimizer, pca = None, d=None):
    index = 0
    total_loss = 0
    total_accuracy = 0

    kl_weight = get_kl_weight(epoch, n_epochs)

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

            loss = criterion(outputs, labels, 1, kl_weight)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            total_loss += loss.item()
            total_accuracy += (predicted == labels).sum().item()
            index +=labels.size(0)

            pbar.update()

    pbar.close()

    return model, total_loss/len(loader), total_accuracy/index

def test_from_features_PCA(epoch, n_epochs, device, model, loader, criterion, pca = None, d=None):
    index = 0
    total_loss = 0
    total_accuracy = 0

    kl_weight = get_kl_weight(epoch, n_epochs)

    with tqdm(range(len(loader))) as pbar:
        for (inputs, labels,_) in loader:

            inputs = inputs.float()

            if pca is not None:
                inputs = inputs.numpy()
                inputs = pca.transform(inputs)
                if d is not None:
                    inputs = inputs[:, 0:d]
                inputs = torch.from_numpy(inputs).float()
            
            #inputs = inputs[:, 0:d]


            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels, 1, kl_weight)

            _, predicted = torch.max(outputs.data, 1)

            total_loss += loss.item()
            total_accuracy += (predicted == labels).sum().item()
            index +=labels.size(0)

            pbar.update()

    pbar.close()

    return total_loss/len(loader), total_accuracy/index