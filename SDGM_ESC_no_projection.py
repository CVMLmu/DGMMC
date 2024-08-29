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
from src.Datasets import ESC50Dataset
from utils_SDGM import SDGMClassifier, train_from_features_PCA, test_from_features_PCA
from utils import get_trained_PCA

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('Code running on :', device)

    G = [2]
    folders = [1,2,3,4,5]
    cov_types = ['diag', 'full']
    d = 1024
    classes = 50
    batch_size = 64
    nb_epochs = 30

    EXPERIMENT_PATH = os.path.join('experiments_no_projection', 'ESC')
    FEATURES_ABOSLUTE_PATH = os.path.join('/home/jeremy/Documents/Datasets/ESC50', 'features')

    for i in folders:

        folder_path = os.path.join(EXPERIMENT_PATH, 'fold_{}'.format(i))

        SDGM_folder_path = os.path.join(folder_path, 'SDGM')
        if os.path.isdir(SDGM_folder_path) is False:
            os.mkdir(SDGM_folder_path)

        results_path = os.path.join(SDGM_folder_path, 'results')
        if os.path.isdir(results_path) is False:
            os.mkdir(results_path)

        models_path = os.path.join(SDGM_folder_path, 'models')
        if os.path.isdir(models_path) is False:
            os.mkdir(models_path)

        trainset = ESC50Dataset(FEATURES_ABOSLUTE_PATH, os.path.join(folder_path, 'train_data.csv'))
        train_ds, val_ds = random_split(trainset, [math.floor(0.90*len(trainset)), len(trainset) - math.floor(0.90*len(trainset))])

        trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory = True)
        valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory = True)

        testset = ESC50Dataset(FEATURES_ABOSLUTE_PATH, os.path.join(folder_path, 'test_data.csv'))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory = True)


        for cov_type in cov_types:
                for g in G:

                    model = SDGMClassifier(d, classes, g, cov_type)
                    model.to(device)

                    criterion = ELBOLoss(model, F.cross_entropy).to("cuda")

                    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs, eta_min=1e-4)
                    best_loss = math.inf
                    
                    model_path = os.path.join(models_path, 'model_D_{}_G_{}_cov_{}.pt'.format(d, g, cov_type))

                    tr = []
                    val = []
                    for epoch in range(nb_epochs):

                        model, train_loss, train_acc = train_from_features_PCA(epoch, nb_epochs, device, model, trainloader, criterion, optimizer)
                        tr.append(np.hstack((train_loss, train_acc)))

                        val_loss, val_acc = test_from_features_PCA(epoch, nb_epochs, device, model, valloader, criterion)
                        val.append(np.hstack((val_loss, val_acc)))

                        print("[Epoch {}/{}] tr_loss: {:.4f} -- tr_acc: {:.3f} -- val_loss: {:.4f} -- val_acc: {:.3f}".format(epoch, nb_epochs, train_loss, train_acc, val_loss, val_acc))

                        if val_loss < best_loss:
                            torch.save(model, model_path)
                            best_loss = val_loss

                        scheduler.step()

                    best_model = torch.load(model_path)
                    best_model.eval()
                    best_model.to(device)

                    test_loss, test_acc = test_from_features_PCA(epoch, nb_epochs, device, best_model, testloader, criterion)
                    print("Test: test_loss: {:.5f} -- test_acc: {:.3f}".format(test_loss, test_acc))

                    # Save results
                    tr = np.stack(tr, axis=0)
                    df_tr = pd.DataFrame(tr, columns=['loss', 'acc'])
                    fpath = os.path.join(results_path, 'train_D_{}_G_{}_cov_{}.csv'.format(d, g, cov_type))
                    df_tr.to_csv(fpath, sep=';')
                    
                    val = np.stack(val, axis=0)
                    df_val = pd.DataFrame(val, columns=['loss', 'acc'])
                    fpath = os.path.join(results_path, 'val_D_{}_G_{}_cov_{}.csv'.format(d, g, cov_type))
                    df_val.to_csv(fpath, sep=';')

                    te = np.vstack((test_loss, test_acc)).transpose()
                    df_test = pd.DataFrame(te, columns=['loss', 'acc'])
                    fpath = os.path.join(results_path, 'test_D_{}_G_{}_cov_{}.csv'.format(d, g, cov_type))
                    df_test.to_csv(fpath, sep=';')