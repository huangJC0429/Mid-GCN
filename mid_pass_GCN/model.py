import torch
import torch.nn as nn
from mid_pass_GCN.layers import GraphConv, GraphAttConv
import numpy as np
import torch.nn.functional as F
import torch_sparse

class DeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, combine, nlayer=2, args=None):
        super(DeepGCN, self).__init__()
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i == 0 else nhid, nhid, bias=False)
            for i in range(nlayer - 1)
        ])
        self.out_layer = GraphConv(nfeat if nlayer == 1 else nhid, nclass)

        self.combine = combine
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.nrom = nn.BatchNorm1d(nhid)

        # self.relu = nn.GELU()


    def forward(self, x, data):
        adj = data.mid_adj
        # adj = data.adj# _origin
        # print(data.adj.to_dense())
        # exit()
       #  new_adj = self._preprocess_adj(adj, normalize)
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.relu(x)

        x = self.dropout(x)
        x = self.out_layer(x, adj)
        x = torch.log_softmax(x, dim=-1)
        return [x, None]


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, combine, nlayer=2, args=None):
        super(GAT, self).__init__()
        alpha_droprate = dropout
        self.gac1 = GraphAttConv(nfeat, nhid, 8, alpha_droprate)  # 8 head attention
        self.gac2 = GraphAttConv(nhid, nclass, 1, alpha_droprate)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ELU(True)

    def forward(self, x, data):
        adj = data.adj_origin
        x = self.dropout(x) # ?
        x = self.gac1(x, adj)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gac2(x, adj)
        x = torch.log_softmax(x, dim=-1)
        return [x, None]


