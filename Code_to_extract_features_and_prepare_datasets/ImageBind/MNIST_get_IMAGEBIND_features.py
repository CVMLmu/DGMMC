import os
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from PIL.Image import BICUBIC
from torchvision import datasets, transforms

class ImageBind(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = imagebind_model.imagebind_huge(pretrained=True)
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):

        x = self.encoder(x)[ModalityType.VISION]

        return x
    
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ImageBind()
model.to(device)

preprocess = transforms.Compose([
    transforms.Resize(224, BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Load the dataset
train = MNIST('./datasets', download=True, train=True, transform=preprocess)
test = MNIST('./datasets', download=True, train=False, transform=preprocess)



def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(DataLoader(dataset, batch_size=100)):

            inputs = torch.dstack([inputs] * 3)
            inputs = inputs.reshape(-1, 3, 224, 224)

            inputs = inputs.to(device)

            inputs = {
                ModalityType.VISION: inputs
            }

            features = model(inputs).cpu()

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)

train_path = os.path.join('features', 'IMAGEBIND', 'train')
for j in range(0, len(train_features)):

    feature_path = os.path.join(train_path, '{}.pt'.format(j))
    torch.save(train_features[j], feature_path)


test_features, test_labels = get_features(test)

test_path = os.path.join('features', 'IMAGEBIND', 'test')
for j in range(0, len(test_features)):

    feature_path = os.path.join(test_path, '{}.pt'.format(j))
    torch.save(test_features[j], feature_path)