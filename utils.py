import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

def get_trained_PCA(train_loader, nb_components, whiten = False):

    x = []
    with torch.no_grad():
        for images, labels, _ in tqdm(train_loader):
            images = torch.flatten(images, start_dim=1)
            images = images.numpy()
            x.append(images)
                
    x = np.concatenate(x, axis=0)

    pca = PCA(n_components=nb_components,
            whiten=whiten,
            svd_solver='auto')

    pca.fit(x)

    return pca