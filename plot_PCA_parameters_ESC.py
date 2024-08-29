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
folders = [1,2,3,4,5]

EXPERIMENT_PATH = os.path.join('experiments', 'ESC')

colors = ['black', 'red', 'green']
idx = 0
plt.figure(figsize=(8,8))

#########
#SDGM
########

values = []

'''cov_types = ['full', 'diag']

for cov_type in cov_types:
    general_results = []
    for g in G:
        for p in P:
            complete_results = []
            for folder in folders:

                df_path = os.path.join(EXPERIMENT_PATH, 'fold_{}'.format(folder), 'SDGM','results', 'Features_P_{}_G_{}_cov_{}.csv'.format(p,g, cov_type))
                df = pd.read_csv(df_path, sep=';')

                values = df.values
                d =  int(values[0][2])

                complete_results.append(d)

            complete_results = np.stack(complete_results, axis=0)

            general_results.append(complete_results)

    general_results = np.stack(general_results, axis=0)
    general_results = general_results.reshape((len(G), len(P), len(folders)), order = 'F')

    means = np.mean(general_results, axis=2)
    std = np.std(general_results, axis=2)
    test_array = np.ravel([means,std],'F').reshape((len(G), 2*len(P)))

    index = pd.MultiIndex.from_product([P,['avg','std']])
    df = pd.DataFrame(test_array, index = G, columns = index)
    print(df)
    #df.to_csv(os.path.join('results', 'ESC_SDGM_cov_{}.csv'.format(cov_type)), sep=';')


    plt.plot(P, means.squeeze(), color=colors[idx], marker='o',linewidth=2, markersize=8, label = labels[idx])

    idx+=1'''


general_results = []
for g in G:
    for p in P:
        complete_results = []
        for folder in folders:

            df_path = os.path.join(EXPERIMENT_PATH, 'fold_{}'.format(folder), 'DGMMC','results', 'Features_P_{}_G_{}.csv'.format(p,g))
            df = pd.read_csv(df_path, sep=';')

            values = df.values
            d =  values[0][2]

            complete_results.append(d)

        complete_results = np.stack(complete_results, axis=0)

        general_results.append(complete_results)

general_results = np.stack(general_results, axis=0)
general_results = general_results.reshape((len(G), len(P), len(folders)), order = 'F')

means = np.mean(general_results, axis=2)
std = np.std(general_results, axis=2)
test_array = np.ravel([means,std],'F').reshape((len(G), 2*len(P)))

index = pd.MultiIndex.from_product([P,['avg','std']])
df = pd.DataFrame(test_array, index = G, columns = index)
print(df)
#df.to_csv(os.path.join('results', 'ESC_DGMMC.csv'), sep=';')

plt.plot(P, means.squeeze(), color=colors[idx], marker='o',linewidth=2, markersize=8, label = 'ImageBind')

idx+=1

plt.xlabel('Cumulative variance ratio (in %)')
plt.ylabel('Eigenvectors conserved (d)')
plt.legend()
plt.show()