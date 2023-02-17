import torch, os
import numpy as np
import scipy.sparse as sp
import torch_geometric.datasets as geo_data
import torch_geometric.transforms as T

device = torch.device('cuda')
DATA_ROOT = 'data'
if not os.path.isdir(DATA_ROOT):
    os.mkdir(DATA_ROOT)


def load_data(data_name='cora', normalize_feature=True, missing_rate=0, citation_random=False, train_size=20,
              cuda=False):
    # can use other dataset, some doesn't have mask
    if data_name in ['cora', 'citeseer', 'pubmed']:
        data = geo_data.Planetoid(os.path.join(DATA_ROOT, data_name), data_name).data
        if citation_random:
            random_coauthor_amazon_splits(data, max(data.y), None, train_size)
    elif data_name in ['Photo', 'Computers']:
        data = geo_data.Amazon(os.path.join(DATA_ROOT, data_name), data_name, T.NormalizeFeatures()).data
        random_coauthor_amazon_splits(data, max(data.y) + 1, None)
    elif data_name in ['cora_ml', 'dblp']:
        data = geo_data.CitationFull(os.path.join(DATA_ROOT, data_name), data_name, T.NormalizeFeatures()).data
        random_coauthor_amazon_splits(data, max(data.y) + 1, None, train_size)
    else:
        data = geo_data.WikiCS(os.path.join(DATA_ROOT, data_name), data_name, T.NormalizeFeatures()).data
        data.train_mask = data.train_mask.type(torch.bool)[:, 0]
        data.val_mask = data.val_mask.type(torch.bool)[:, 0]
    print(max(data.y))
    # original split
    data.train_mask = data.train_mask.type(torch.bool)
    data.val_mask = data.val_mask.type(torch.bool)
    data.test_mask = data.test_mask.type(torch.bool)
    # data.test_mask = data.test_mask.type(torch.bool)

    # expand test_mask to all rest nodes
    # data.test_mask = ~(data.train_mask + data.val_mask)

    # get adjacency matrix
    n = len(data.x)
    adj = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    data.degree_martix = to_torch_sparse(sp.diags(np.array(adj.sum(axis=1)).flatten()))

    data.noeye_adj = to_torch_sparse(normalize_adj(adj - sp.eye(adj.shape[0], adj.shape[1]))).to(device)
    # data.adj_origin = to_torch_sparse(adj)  # Adjacency matrix
    data.adj_origin = to_torch_sparse(adj)  # Adjacency matrix

    # middle-normalize,
    mid_adj = middle_normalize_adj(adj)
    data.mid_adj = to_torch_sparse(mid_adj)
    # exit()

    adj = normalize_adj(adj)  # symmetric normalization works bad, but why? Test more.
    data.adj = to_torch_sparse(adj)

    # normalize feature
    if normalize_feature:
        data.x = row_l1_normalize(data.x)

    # generate missing feature setting
    indices_dir = os.path.join(DATA_ROOT, data_name, 'indices')
    if not os.path.isdir(indices_dir):
        os.mkdir(indices_dir)
    missing_indices_file = os.path.join(indices_dir, "indices_missing_rate={}.npy".format(missing_rate))
    if not os.path.exists(missing_indices_file):
        erasing_pool = torch.arange(n)[~data.train_mask]  # keep training set always full feature
        size = int(len(erasing_pool) * (missing_rate / 100))
        idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
        np.save(missing_indices_file, idx_erased)
    else:
        idx_erased = np.load(missing_indices_file)
    # erasing feature for random missing
    if missing_rate > 0:
        data.x[idx_erased] = 0

    if cuda:
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.adj = data.adj.to(device)
        data.mid_adj = data.mid_adj.to(device)
        data.adj_origin = data.adj_origin.to(device)
        data.degree_martix = data.degree_martix.to(device)

    return data


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # add self-loop and normalization also affects performance a lot
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def process(adj, features, labels, idx_train, idx_val, idx_test, alpha):

    data = Data()

    mid_adj = middle_normalize_adj(adj, alpha)  # + sp.eye(adj.shape[0])
    norm_adj = normalize_adj(adj)
    mid_adj = to_torch_sparse(mid_adj).to(device)
    norm_adj = to_torch_sparse(norm_adj).to(device)
    features = torch.tensor(features.A).float().to(device)
    # if fea == 0:
    features = row_l1_normalize(features)
    labels = torch.tensor(labels).long().to(device)

    data.mid_adj = mid_adj
    data.adj = norm_adj
    data.adj_origin = to_torch_sparse(adj).to(device)
    data.x = features
    data.y = labels

    data.train_mask = idx_train
    data.val_mask = idx_val
    data.test_mask = idx_test

    return data




def middle_normalize_adj(adj, alpha):
    """Middle normalize adjacency matrix."""
    # add self-loop and normalization also affects performance a lot
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    DAD = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return (alpha * sp.eye(adj.shape[0], adj.shape[1]) - DAD).dot(sp.eye(adj.shape[0], adj.shape[1]) + DAD)


def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx


def to_torch_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_l1_normalize(X):
    norm = 1e-6 + X.sum(dim=1, keepdim=True)
    return X / norm


def dense_tensor2sparse_tensor(adj):
    indices = torch.nonzero(adj != 0)
    indices = indices.t()
    re_adj = torch.reshape(adj, (-1, 1))
    nonZeroRows = torch.abs(re_adj).sum(dim=1) != 0
    re_adj = re_adj[nonZeroRows]
    value = re_adj.t().squeeze()
    shape = torch.Size(adj.shape)
    new_adj = torch.sparse_coo_tensor(indices, value, shape)
    return new_adj


def random_coauthor_amazon_splits(data, num_classes, lcc_mask, train_size=20):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)



    train_index = torch.cat([i[:train_size] for i in indices], dim=0)
    val_index = torch.cat([i[train_size:train_size + 30] for i in indices], dim=0)

    rest_index = torch.cat([i[train_size + 30:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

class Data():
    def __init__(self):
        self.A = None

if __name__ == "__main__":
    import sys

    print(sys.version)
    # test goes here
    data = load_data(cuda=True)
    print(data.train_mask[:150])