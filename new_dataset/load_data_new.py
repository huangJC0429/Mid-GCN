import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.sparse as sp


def data_split(nnodes, Y):
    np.random.seed(15)
    idx = np.arange(nnodes)
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=0.1 + 0.1,  # 0.1+0.1
                                                   test_size=0.8,
                                                   stratify=Y)

    if Y is not None:
        stratify = Y[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=0.5,
                                          test_size=0.5,
                                          stratify=stratify)
    return idx_train, idx_val, idx_test
import os
def load_new_data(dataset):
    data = np.load("../new_dataset/Processed_data/"+dataset+".npz", allow_pickle=True)
    adj, X, Y = data['adj'], data['X'], data['Y']
    print(adj.sum())
    print(adj.shape[0])
    print(X.shape)

    if os.path.exists("../new_dataset/Processed_data//split_" + dataset + ".npz"):
        split = np.load("../new_dataset/Processed_data//split_" + dataset + ".npz", allow_pickle=True)
        idx_train, idx_val, idx_test = split['idx_train'], split['idx_val'], split['idx_test']
    else:
        idx_train, idx_val, idx_test = data_split(len(Y), Y)
        huang_split = {'idx_train': idx_train,
                      'idx_val': idx_val,
                      'idx_test': idx_test}
        np.savez('./Processed_data/split_' + dataset + '.npz', **huang_split)
    return sp.csr_matrix(adj), X, Y.astype('int'), idx_train, idx_val, idx_test
# load_new_data('git2')
