from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCNJaccard
import argparse
from tqdm import tqdm
import sys
sys.path.append("..")
from mid_pass_GCN.model import DeepGCN, GAT
from deeprobust.graph.defense import GCNSVD
import sys
sys.path.append("..")
from mid_pass_GCN.utils import process
import sys
sys.path.append("..")
from new_dataset.load_data_new import load_new_data
from random import  sample

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'cora_ml', 'citeseer', 'polblogs'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.0, help='pertubation rate')
parser.add_argument('--attack', type=str, default='meta', choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--k', type=int, default=150, help='Truncated Components.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--nlayer', type=int, default=2, help='Number of layers, works for Deep model.')
parser.add_argument('--combine', type=str, default='mul', help='{add, mul}}')
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=0.2, help='hyperparameters.')
parser.add_argument('--feature_flip', type=int, default=0, help='Random flip features.')
parser.add_argument('--ptb_num', type=int, default=5,  help='pertubation number')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
OUT_PATH = "./results_nettack/"

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

if args.dataset in ['cora', 'citeseer', 'polblogs']:
    data = Dataset(root='../data', name=args.dataset, setting='prognn')
    adj_orig, features_orig, labels_orig = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
else:
    # adj, features, labels, idx_train, idx_val, idx_test = load_facebook_data(args.dataset)
    adj_orig, features_orig, labels_orig, idx_train, idx_val, idx_test = load_new_data(args.dataset)
    # std = MinMaxScaler()
    # features = std.fit_transform(features)
    # features = sp.csr_matrix(features)
    # features = scale(features)
    features_orig = sp.csr_matrix(features_orig)
if args.dataset == 'pubmed':
    idx_train, idx_val, idx_test = get_train_val_test(adj_orig.shape[0],
                                                        val_size=0.1, test_size=0.8,
                                                        stratify=encode_onehot(labels_orig),
                                                        seed=15)

idx_unlabeled = np.union1d(idx_val,idx_test)

print("ptb_num:",args.ptb_num)
print("idx_train:",idx_train.shape)
print("idx_val:",idx_val.shape)
print("idx_test:",idx_test.shape)

# Setup Surrogate model
surrogate = GCN(nfeat=features_orig.shape[1], nclass=labels_orig.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features_orig, adj_orig, labels_orig, idx_train, idx_val, patience=30)

def select_nodes_larger10():
    degrees = adj_orig.sum(0).A1
    node_list = []
    for idx in idx_test:
        if degrees[idx] > 10:
            node_list.append(idx)
    if args.dataset == 'git2':
        t = len(node_list)
        pubmed_node_list = sample(node_list,int(t*0.08))
        return pubmed_node_list
    else:
        return node_list


def multi_test_poison():
    cnt = 0
    node_list = select_nodes_larger10()
    num = len(node_list)
    # print(num)
    # exit()
    acc_all = []
    for target_node in tqdm(node_list):
        print('target_node:', target_node)
        print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
        n_perturbations = args.ptb_num
        print("n_perturbations:", n_perturbations)
        if n_perturbations == 0:
            modified_adj = adj_orig
        else:
            model = Nettack(surrogate, nnodes=adj_orig.shape[0], attack_structure=True, attack_features=False,
                            device=device)
            model = model.to(device)
            model.attack(features_orig, adj_orig, labels_orig, target_node, n_perturbations, verbose=False)
            modified_adj = model.modified_adj
            # print((modified_adj.A==adj_orig.A).all())

            # modified_features = model.modified_features

        # print(modified_adj.shape)
        print('=== [Poisoning] testing %s attacked targeted nodes respectively ===' % target_node)
        nclass = max(labels_orig) + 1
        data = process(modified_adj, features_orig, labels_orig, idx_train, idx_val, idx_test, args.alpha)

        net = DeepGCN(features_orig.shape[1], args.hid, nclass,
                      dropout=args.dropout,
                      combine=args.combine,
                      nlayer=args.nlayer)
        net = net.to(device)
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
        # train
        best_acc = 0
        best_loss = 1e10
        for epoch in range(args.epochs):

            train_loss, train_acc = train(net, optimizer, data)
            val_loss, val_acc = val(net, data)
            test_loss, test_acc = test(net, data)

            # train_loss_list.append(train_loss.cpu().detach().numpy().astype(np.float64))
            # val_loss_list.append(val_loss.cpu().detach().numpy().astype(np.float64))
            # test_loss_list.append(test_loss.cpu().detach().numpy().astype(np.float64))

            # print('Epoch %d: train loss %.3f train acc: %.3f, val loss: %.3f val acc %.3f | test acc %.3f.' %
            #       (epoch, train_loss, train_acc, val_loss, val_acc, test_acc))
            # save model
            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(net.state_dict(),
                           OUT_PATH + 'checkpoint-best-acc' +str(args.ptb_num) + str(args.nlayer) + str(args.dataset) + '.pkl')
            # if best_loss > val_loss:
            #     best_loss = val_loss
            #     torch.save(net.state_dict(), OUT_PATH+'checkpoint-best-loss'+str(args.nlayer) + str(args.data) + '.pkl')

        # pick up the best model based on val_acc, then do test
        net.load_state_dict(
            torch.load(OUT_PATH + 'checkpoint-best-acc' +str(args.ptb_num) + str(args.nlayer) + str(args.dataset) + '.pkl'))

        val_loss, val_acc = val(net, data)
        test_loss, test_acc = test(net, data)

        print("-" * 50)
        print("Vali set results: loss %.3f, acc %.3f." % (val_loss, val_acc))
        print("Test set results: loss %.3f, acc %.3f." % (test_loss, test_acc))
        print("=" * 50)
        output, output_list = net(data.x, data)
        acc_test = (output.argmax(1)[target_node] == labels_orig[target_node])
        acc = acc_test.item()
        acc_all.append(test_acc.cpu().numpy() * 100)
        print(acc)
        if acc == 1 or acc == True:
            cnt += 1
    acc_classify = cnt / num
    print('classification rate : %s' % np.average(acc_all))
    print('target_acc',acc)
    return acc_classify




if __name__ == '__main__':
    for args.ptb_num in [0, 1, 2, 3, 4, 5]:
        result_list = []
        s = 0
        for i in range(0, 5):
            acc_classfiy = multi_test_poison()
            result_list.append(acc_classfiy*100)
            s +=1
        print("result_list:", result_list)
        avg = np.mean(result_list)
        var = np.var(result_list)
        std = np.std(result_list)

        print("%d average acc:"%s, avg)
        print("%d variance:"%s, var)
        print("%dStandard Deviation:"%s, std)




