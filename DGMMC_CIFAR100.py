import os
import numpy as np
import pandas as pd
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from src_SDGM.SDGM import SDGM
from src_SDGM.torch_ard import ELBOLoss
from src.Datasets import CIFAR100Dataset
from utils_DGMMC import DGMMClassifier, train_from_features_PCA, test_from_features_PCA, get_means_bandwidth_from_features, CrossEntropy
from utils import get_trained_PCA
from torch.utils.data import Dataset, DataLoader


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('Code running on :', device)

    P = range(5, 105, 5)
    G = [1]
    runs = [0,1,2]
    embeddings = ['CLIP', 'IMAGEBIND']

    classes = 100
    batch_size = 64
    nb_epochs = 30

    EXPERIMENT_PATH = os.path.join('experiments', 'CIFAR100')
    FEATURES_ABOSLUTE_PATH = os.path.join('/home/jeremy/Documents/Datasets/CIFAR100', 'Features')

    for embedding in embeddings:

        embeding_folder = os.path.join(EXPERIMENT_PATH, embedding)
        if os.path.isdir(embeding_folder) is False:
            os.mkdir(embeding_folder)

        SDGM_folder_path = os.path.join(embeding_folder, 'DGMMC')
        if os.path.isdir(SDGM_folder_path) is False:
            os.mkdir(SDGM_folder_path)

        results_path = os.path.join(SDGM_folder_path, 'results')
        if os.path.isdir(results_path) is False:
            os.mkdir(results_path)

        models_path = os.path.join(SDGM_folder_path, 'models')
        if os.path.isdir(models_path) is False:
            os.mkdir(models_path)

        trainset = CIFAR100Dataset(os.path.join(FEATURES_ABOSLUTE_PATH, embedding, 'train'), train=True)
        train_ds, val_ds = random_split(trainset, [math.floor(0.90*len(trainset)), len(trainset) - math.floor(0.90*len(trainset))])

        trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers = 8, pin_memory = False)
        valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,num_workers = 8, pin_memory = False)

        testset = CIFAR100Dataset(os.path.join(FEATURES_ABOSLUTE_PATH, embedding, 'test'), train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,num_workers = 8, pin_memory = False)

        if embedding == 'CLIP':
            pca = get_trained_PCA(trainloader, 768)
        else:
            pca = get_trained_PCA(trainloader, 1024)


        temp_dir = os.path.join('temp_dataset', 'train')
        for (features, labels, index) in DataLoader(trainset, batch_size=2*batch_size, num_workers=8):
            features = features.numpy()
            features = pca.transform(features)
            features = torch.from_numpy(features).float()

            for i in range(0, len(features)):
                torch.save(features[i], os.path.join(temp_dir, '{}.pt'.format(index[i])))
        
        temp_dir = os.path.join('temp_dataset', 'test')
        for (features, labels, index) in DataLoader(testset, batch_size=2*batch_size, num_workers=8):
            features = features.numpy()
            features = pca.transform(features)
            features = torch.from_numpy(features).float()

            for i in range(0, len(features)):
                torch.save(features[i], os.path.join(temp_dir, '{}.pt'.format(index[i])))

        train_ds.dataset.features_path = os.path.join('temp_dataset', 'train')
        val_ds.dataset.features_path = os.path.join('temp_dataset', 'train')
        testset.features_path = os.path.join('temp_dataset', 'test')

        for p in P:
            for g in G:
                for run in runs:

                    cumsum = np.cumsum(pca.explained_variance_ratio_)

                    if p != 100:
                        d = np.argmax(cumsum >= p/100) + 1
                    else:
                        if embedding == 'CLIP':
                            d = 768
                        else:
                            d = 1024

                    print(d)

                    init_means, init_stds = get_means_bandwidth_from_features(classes, trainloader, pca, d)

                    model = DGMMClassifier(d,classes,g, init_means)
                    model.to(device)

                    criterion = CrossEntropy()

                    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs, eta_min=1e-4)
                    best_loss = math.inf
                    
                    model_path = os.path.join(models_path, 'model_P_{}_G_{}_run_{}.pt'.format(p, g, run))

                    tr = []
                    val = []
                    for epoch in range(nb_epochs):

                        model, train_loss, train_acc = train_from_features_PCA(classes, device, model, trainloader, criterion, optimizer, pca, d)
                        tr.append(np.hstack((train_loss, train_acc)))

                        val_loss, val_acc = test_from_features_PCA(classes, device, model, valloader, criterion, pca, d)
                        val.append(np.hstack((val_loss, val_acc)))

                        print("[Epoch {}/{}] tr_loss: {:.4f} -- tr_acc: {:.3f} -- val_loss: {:.4f} -- val_acc: {:.3f}".format(epoch, nb_epochs, train_loss, train_acc, val_loss, val_acc))

                        if val_loss < best_loss:
                            torch.save(model, model_path)
                            best_loss = val_loss

                        scheduler.step()

                    best_model = torch.load(model_path)
                    best_model.eval()
                    best_model.to(device)

                    test_loss, test_acc = test_from_features_PCA(classes, device, best_model, testloader, criterion, pca, d)
                    print("Test: test_loss: {:.5f} -- test_acc: {:.3f}".format(test_loss, test_acc))

                    # Save results
                    tr = np.stack(tr, axis=0)
                    df_tr = pd.DataFrame(tr, columns=['loss', 'acc'])
                    fpath = os.path.join(results_path, 'train_P_{}_G_{}_run_{}.csv'.format(p, g, run))
                    df_tr.to_csv(fpath, sep=';')
                    
                    val = np.stack(val, axis=0)
                    df_val = pd.DataFrame(val, columns=['loss', 'acc'])
                    fpath = os.path.join(results_path, 'val_P_{}_G_{}_run_{}.csv'.format(p, g, run))
                    df_val.to_csv(fpath, sep=';')

                    te = np.vstack((test_loss, test_acc)).transpose()
                    df_test = pd.DataFrame(te, columns=['loss', 'acc'])
                    fpath = os.path.join(results_path, 'test_P_{}_G_{}_run_{}.csv'.format(p, g, run))
                    df_test.to_csv(fpath, sep=';')

                    feat_infos = np.vstack((p/100, d)).transpose()
                    df_feat_info = pd.DataFrame(feat_infos, columns=['P', 'Features_kept'])
                    fpath = os.path.join(results_path, 'Features_P_{}_G_{}_run_{}.csv'.format(p, g, run))
                    df_feat_info.to_csv(fpath, sep=';')