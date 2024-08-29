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

D = [1024]
G = [1]
folders = [1,2,3,4,5]

EXPERIMENT_PATH = os.path.join('experiments_no_projection', 'ESC')

#########
#SDGM
########

values = []

cov_types = ['full', 'diag']

for cov_type in cov_types:
    general_results = []
    for g in G:
        for d in D:
            complete_results = []
            for folder in folders:

                df_path = os.path.join(EXPERIMENT_PATH, 'fold_{}'.format(folder), 'SDGM','results', 'test_D_{}_G_{}_cov_{}.csv'.format(d,g, cov_type))
                df = pd.read_csv(df_path, sep=';')

                values = df.values
                accuracy =  values[0][2]

                complete_results.append(accuracy)

            complete_results = np.stack(complete_results, axis=0)

            general_results.append(complete_results)

    general_results = np.stack(general_results, axis=0)
    general_results = general_results.reshape((len(G), len(D), len(folders)), order = 'F')

    means = np.mean(general_results, axis=2)
    std = np.std(general_results, axis=2)
    test_array = np.ravel([means,std],'F').reshape((len(G), 2*len(D)))

    index = pd.MultiIndex.from_product([D,['avg','std']])
    df = pd.DataFrame(test_array, index = G, columns = index)
    print(df)
    if d == 768:
        df.to_csv(os.path.join('results_no_projection', 'ESC_CLIP_SDGM_cov_{}.csv'.format(cov_type)), sep=';')
    else:
        df.to_csv(os.path.join('results_no_projection', 'ESC_IMAGEBIND_SDGM_cov_{}.csv'.format(cov_type)), sep=';')


general_results = []
for g in G:
    for d in D:
        complete_results = []
        for folder in folders:

            df_path = os.path.join(EXPERIMENT_PATH, 'fold_{}'.format(folder), 'DGMMC','results', 'test_D_{}_G_{}.csv'.format(d,g))
            df = pd.read_csv(df_path, sep=';')

            values = df.values
            accuracy =  values[0][2]

            complete_results.append(accuracy)

        complete_results = np.stack(complete_results, axis=0)

        general_results.append(complete_results)

general_results = np.stack(general_results, axis=0)
general_results = general_results.reshape((len(G), len(D), len(folders)), order = 'F')

means = np.mean(general_results, axis=2)
std = np.std(general_results, axis=2)
test_array = np.ravel([means,std],'F').reshape((len(G), 2*len(D)))

index = pd.MultiIndex.from_product([D,['avg','std']])
df = pd.DataFrame(test_array, index = G, columns = index)
print(df)
if d == 768:
    df.to_csv(os.path.join('results_no_projection', 'ESC_CLIP_DGMMC.csv'), sep=';')
else:
    df.to_csv(os.path.join('results_no_projection', 'ESC_IMAGEBIND_DGMMC.csv'), sep=';')