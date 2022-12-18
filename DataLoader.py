import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def ten_fold_cross(data, label):
    cross_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=88)
    folds = []
    for train, test in cross_fold.split(data, label):
        x_train, y_train = data[train], label[train]
        x_test, y_test = data[test], label[test]
        
        folds.append((x_train, y_train, x_test, y_test))
    return folds
    
    
def load_dataset():
    file_location_1 = 'Data/project3_dataset1.txt'
    file_location_2 = 'Data/project3_dataset2.txt'
    
#     dataset_1 = open(file_location_1, 'r')
#     dataset_1.read

    dataset_1 = pd.read_csv(file_location_1, header=None, sep='\t').to_numpy()
    dataset_2 = pd.read_csv(file_location_2, header=None, sep='\t')
    
    dataset_2[4] = dataset_2[4].replace(['Present','Absent'], [0,1]) 
    

    
    dataset_2 = dataset_2.to_numpy()
    
    
    data_1 = dataset_1[:,:-1]
    data_2 = dataset_2[:,:-1]
    
    
    label_1 = np.array(dataset_1[:,-1], dtype='int')
    label_2 = np.array(dataset_2[:,-1], dtype='int')
    return ten_fold_cross(data_1, label_1), ten_fold_cross(data_2, label_2)
    
    
    
    
    
    
#     test_size_1 = int(train_test_split * len(dataset_1))
#     test_size_2 = int(train_test_split * len(dataset_2))
    
    
    
    
#     test_X_1 = dataset_1[:test_size_1,:-1]
#     test_Y_1 = dataset_1[:test_size_1,-1]
#     train_X_1 = dataset_1[test_size_1:,:-1]
#     train_Y_1 = dataset_1[test_size_1:,-1]
    
    
#     test_X_2 = dataset_2[:test_size_2,:-1]
#     test_Y_2 = dataset_2[:test_size_2,-1]
#     train_X_2 = dataset_2[test_size_2:,:-1]
#     train_Y_2 = dataset_2[test_size_2:,-1]
    
#     test_Y_1 = np.array(test_Y_1, dtype='int')
#     test_Y_2 = np.array(test_Y_2, dtype='int')
#     train_Y_1 = np.array(train_Y_1, dtype='int')
#     train_Y_2 = np.array(train_Y_2, dtype='int')
    
    
    
    
    
#     return (train_X_1, train_Y_1, test_X_1, test_Y_1), (train_X_2, train_Y_2, test_X_2, test_Y_2)
