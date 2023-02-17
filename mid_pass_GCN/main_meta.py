import torch
import argparse
import logging
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCNSVD
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, PrePtbDataset
import datetime
import sys
sys.path.append("..")
from mid_pass_GCN.model import DeepGCN
import sys
sys.path.append("..")
from mid_pass_GCN.utils import process
from sklearn.preprocessing import MinMaxScaler, scale
from feature_filp.feature_flip_utils import feature_flip
from new_dataset.load_data_new import load_new_data

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'git2', 'film'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.0, help='pertubation rate')
parser.add_argument('--attack', type=str, default='meta', choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--nlayer', type=int, default=2, help='Number of layers, works for Deep model.')
parser.add_argument('--combine', type=str, default='mul', help='{add, mul}}')
parser.add_argument('--hid', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.') # 0.005
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.') # 300
parser.add_argument('--alpha', type=float, default=0.5, help='hyperparameters.')
parser.add_argument('--feature_flip', type=int, default=0, help='Random flip features.')
args = parser.parse_args()
result_list = []
OUT_PATH = "./results/"

def train(net, optimizer, data):
    net.train()
    optimizer.zero_grad()
    output, output_list = net(data.x, data)
    loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss = loss_train
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss, acc

def val(net, data):
    net.eval()
    output, output_list = net(data.x, data)
    loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    return loss_val, acc_val

def test(net, data):
    net.eval()
    output, output_list = net(data.x, data)
    loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return loss_test, acc_test

acc_all = []
for i in range(0, 10):

    args.cuda = torch.cuda.is_available()
    print('cuda:%s' % args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make sure you use the same data splits as you generated attacks
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)
    if args.ptb_rate == 0:
        args.attack = "no"
    # load original dataset (to get clean features and labels)
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'polblogs', 'cora_ml']:
        data = Dataset(root='../data', name=args.dataset, setting='nettack')
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    elif args.dataset in ['film']:
        dataset_split = '../hetero_data/pei_data_split/splits/' + str(args.dataset) + '_split_0.6_0.2_' + str(
            0) + '.npz'
        data = load_data(args.dataset, dataset_split)

        features = data.x.cpu().detach().numpy()
        print(features)
        features = sp.csr_matrix(features)
        adj, labels, idx_train, idx_val, idx_test = data.adj_origin, data.y, data.train_mask, data.val_mask, data.test_mask
    else:
        adj, features, labels, idx_train, idx_val, idx_test = load_new_data(args.dataset)
        features = sp.csr_matrix(features)
    if args.dataset == 'pubmed':
        idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0],
                                                          val_size=0.1, test_size=0.8,
                                                          stratify=encode_onehot(labels),
                                                          seed=15)

    print(len(idx_train))
    print(len(idx_val))
    print(len(idx_test))
    """
           Load the data after the attack.
           """
    if args.attack == 'no':
        perturbed_adj = adj
    else:
        if args.dataset in ['cora', 'citeseer' 'polblogs', 'cora_ml']:
            if args.attack == 'meta':
                if args.ptb_rate == 0.1 or args.ptb_rate == 0.2:
                    perturbed_data_file = "../meta_modified_graph_data/%s_meta_adj_%.1f.npz" % (args.dataset, args.ptb_rate)
                else:
                    perturbed_data_file = "../meta_modified_graph_data/%s_meta_adj_%.2f.npz" % (args.dataset, args.ptb_rate)
            if args.attack == 'nettack':
                perturbed_data_file = "../nettack_modified_graph_data/%s_nettack_adj_%.1f.npz" % (
                    args.dataset, args.ptb_rate)

            print("perturbed data file is:", perturbed_data_file)
            perturbed_adj = sp.load_npz(perturbed_data_file)
        else:
            if args.attack == 'meta':
                if args.ptb_rate == 0.1 or args.ptb_rate == 0.2:
                    perturbed_data_file = "../meta_modified_graph_data/%s_meta_adj_%.1f.npy" % (args.dataset, args.ptb_rate)
                else:
                    perturbed_data_file = "../meta_modified_graph_data/%s_meta_adj_%.2f.npy" % (args.dataset, args.ptb_rate)
            if args.attack == 'nettack':
                perturbed_data_file = "../nettack_modified_graph_data/%s_nettack_adj_%.1f.npz" % (
                    args.dataset, args.ptb_rate)

            print("perturbed data file is:", perturbed_data_file)
            perturbed_adj = np.load(perturbed_data_file)
            perturbed_adj = sp.csr_matrix(perturbed_adj)

    nclass = max(labels) + 1
    features = feature_flip(args.dataset, args.feature_flip, features, idx_train)
    data = process(perturbed_adj, features, labels, idx_train, idx_val, idx_test, args.alpha)

    net = DeepGCN(features.shape[1], args.hid, nclass,
                                         dropout=args.dropout,
                                         combine=args.combine,
                                         nlayer=args.nlayer)
    net = net.to(device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
    # train
    best_acc = 0
    best_loss = 1e10
    import time
    s = time.time()
    for epoch in range(args.epochs):

        train_loss, train_acc = train(net, optimizer, data)
        val_loss, val_acc = val(net, data)
        test_loss, test_acc = test(net, data)

        print('Epoch %d: train loss %.3f train acc: %.3f, val loss: %.3f val acc %.3f | test acc %.3f.' %
                      (epoch, train_loss, train_acc, val_loss, val_acc, test_acc))
        # save model
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), OUT_PATH + 'checkpoint-best-acc' + str(args.nlayer) + str(args.dataset) + '.pkl')
        # if best_loss > val_loss:
        #     best_loss = val_loss
        #     torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-loss'+str(args.nlayer) + str(args.data) + '.pkl')

    e = time.time()
    print("avr epoch time:", (e-s)/args.epochs)
    # pick up the best model based on val_acc, then do test
    net.load_state_dict(torch.load(OUT_PATH + 'checkpoint-best-acc' + str(args.nlayer) + str(args.dataset) + '.pkl'))

    val_loss, val_acc = val(net, data)
    test_loss, test_acc = test(net, data)

    print("-" * 50)
    print("Vali set results: loss %.3f, acc %.3f." % (val_loss, val_acc))
    print("Test set results: loss %.3f, acc %.3f." % (test_loss, test_acc))
    print("=" * 50)
    acc_all.append(test_acc.cpu().numpy() * 100)
print('10_avr:', np.average(acc_all))
print('10_std:', np.std(acc_all))
print('10_var:', np.var(acc_all))




