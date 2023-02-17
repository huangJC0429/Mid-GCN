import torch, os
import numpy as np
import scipy.sparse as sp
import torch_geometric.datasets as geo_data
import torch_geometric.transforms as T

def feature_flip(data_name, feature_dim, x, train_mask):
    if feature_dim > 0:
        for i in range(x.shape[0]):
            at_idx = np.random.choice(x.shape[1], size=int(feature_dim), replace=False)
            idex_fea = x[i, at_idx].toarray()
            at_fea = np.where(idex_fea == 0, 1, 0)
            x[i, at_idx] = at_fea
    return x

def add_gaussion_nosie(data_name, rate, x, train_mask):
    # Feature added noise
    noise = np.random.normal(loc=0, scale=1, size=(x.shape[0], int(x.shape[1] * (rate / 100)))).astype(np.float64)
    x = np.hstack((x.A.astype(np.float64), noise))
    x = sp.csr_matrix(x)
    print(x.shape)
    return x