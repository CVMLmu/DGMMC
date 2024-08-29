import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def create_results_array(values, channels):

    mean = np.mean(values, axis=channels, keepdims=True)
    std = np.std(values, axis=channels, keepdims=True)

    x,y = mean.shape

    results = np.zeros((x, 2*y))

    for i in range(0, y):
        results[:, 2*i] = mean[:,i]
        results[:, 2*i+1] = std[:,i]

    return results

P = range(5, 105, 5)
G = [1]
runs = [0,1,2]
embeddings = ['CLIP', 'IMAGEBIND']

EXPERIMENT_PATH = os.path.join('experiments', 'CIFAR100')

colors = {
    'diag' : 'red',
    'full' : 'blue'
}

labels = {
    'diag' : 'SDGM-D',
    'full' : 'SDGM-F'
}

idx = 0

plt.figure(figsize=(8,8))

#########
#SDGM
########

cov_types = ['full', 'diag']

for embedding in embeddings:
    for cov_type in cov_types:
        general_results = []
        for g in G:
            for p in P:
                complete_results = []
                for run in runs:

                    df_path = os.path.join(EXPERIMENT_PATH, embedding, 'SDGM','results', 'test_P_{}_G_{}_cov_{}run_{}.csv'.format(p,g, cov_type, run))
                    df = pd.read_csv(df_path, sep=';')

                    values = df.values
                    accuracy =  values[0][2]

                    complete_results.append(accuracy)

                complete_results = np.stack(complete_results, axis=0)

                general_results.append(complete_results)

        general_results = np.stack(general_results, axis=0)
        general_results = general_results.reshape((len(G), len(P), len(runs)), order = 'F')

        means = np.mean(general_results, axis=2)
        std = np.std(general_results, axis=2)
        test_array = np.ravel([means,std],'F').reshape((len(G), 2*len(P)))

        index = pd.MultiIndex.from_product([P,['avg','std']])
        df = pd.DataFrame(test_array, index = G, columns = index)
        print(df)
        df.to_csv(os.path.join('results', 'CIFAR100_SDGM_cov_{}.csv'.format(cov_type)), sep=';')

        if embedding == 'CLIP':
            plt.plot(P, means.squeeze(), color=colors[cov_type], marker='o',linewidth=2, markersize=8, label = '(C.) {}'.format(labels[cov_type]))
        else:
            plt.plot(P, means.squeeze(), color=colors[cov_type], linestyle='dashed', marker='o',linewidth=2, markersize=8, label = '(I.) {}'.format(labels[cov_type]))


for embedding in embeddings:
    general_results = []
    for g in G:
        for p in P:
            complete_results = []
            for run in runs:

                df_path = os.path.join(EXPERIMENT_PATH, embedding, 'DGMMC','results', 'test_P_{}_G_{}_run_{}.csv'.format(p,g, run))
                df = pd.read_csv(df_path, sep=';')

                values = df.values
                accuracy =  values[0][2]

                complete_results.append(accuracy)

            complete_results = np.stack(complete_results, axis=0)

            general_results.append(complete_results)

    general_results = np.stack(general_results, axis=0)
    general_results = general_results.reshape((len(G), len(P), len(runs)), order = 'F')

    means = np.mean(general_results, axis=2)
    std = np.std(general_results, axis=2)
    test_array = np.ravel([means,std],'F').reshape((len(G), 2*len(P)))

    index = pd.MultiIndex.from_product([P,['avg','std']])
    df = pd.DataFrame(test_array, index = G, columns = index)
    print(df)
    df.to_csv(os.path.join('results', 'CIFAR100_DGMMC.csv'), sep=';')

    if embedding == 'CLIP':
        plt.plot(P, means.squeeze(), color='green', marker='o',linewidth=2, markersize=8, label = '(C.) DGMMC-S')
    else:
        plt.plot(P, means.squeeze(), color='green', linestyle='dashed', marker='o',linewidth=2, markersize=8, label = '(I.) DGMMC-S')


plt.xlabel('Relevant information kept (in %)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()