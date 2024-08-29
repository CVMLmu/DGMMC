import os
import numpy as np
import pandas as pd

meta_data_path = os.path.join('ESC', 'meta', 'esc50.csv')
experiments_path = os.path.join('experiments_SDGM_APCA', 'ESC')

df = pd.read_csv(meta_data_path)
print(df)

folds = df['fold']
filenames = df['filename']
labels = df['target']

for fold in np.unique(folds):

    fold_path = os.path.join(experiments_path, 'fold_{}'.format(fold))
    if os.path.isdir(fold_path) is False:
        os.mkdir(fold_path)
    
    train_files = []
    test_files = []

    for i in range(len(filenames)):
        
        if folds[i] == fold:
            test_files.append((filenames[i], labels[i]))
        else:
            train_files.append((filenames[i], labels[i]))
    
    train_files = pd.DataFrame(train_files, columns=['Filename', 'Label'])
    train_files_path = os.path.join(fold_path, 'train_data.csv')
    train_files.to_csv(train_files_path)

    test_files = pd.DataFrame(test_files, columns=['Filename', 'Label'])
    test_files_path = os.path.join(fold_path, 'test_data.csv')
    test_files.to_csv(test_files_path)
