import argparse
import time
import sys
sys.path.append(r'../defense')
sys.path.append(r'../defense/hgcn')
from MD import MD
from MDGrand import CFSG
from GAT import Net
from GNNGuard.Mettack import test
from train import *

from deeprobust.graph import utils
from deeprobust.graph.defense import GCN, GCNSVD, RGCN, ProGNN, GCNJaccard
from utils import *


import copy

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
                    default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
                    choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=1e-3, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=5e-4, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
                    help='whether use symmetric matrix')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def GCN1(data, device, k):

    features = data.features
    labels = data.labels

    a = features.shape[1]
    start = time.perf_counter()
    model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(data.features, data.adj, data.labels, data.idx_train, data.idx_val)
    end = time.perf_counter()
    model.eval()
    output = model.test(data.idx_test)

    return output, end-start

def Jaccard1(data, device, threhold):

    features = data.features
    labels = data.labels
    start = time.perf_counter()
    model = GCNJaccard(nfeat=features.shape[1],
                       nhid=16,
                       nclass=labels.max().item() + 1,
                       dropout=0.5, device=device)
    model.fit(data.features, data.adj, data.labels, data.idx_train, data.idx_val, threshold=threhold, verbose=False)
    end = time.perf_counter()
    model.eval()
    output = model.test(data.idx_test)

    return output, end-start

def GCNSVD1(data, device, k):

    features = data.features
    labels = data.labels
    start = time.perf_counter()
    model = GCNSVD(nfeat=features.shape[1],
                   nhid=16,
                   nclass=labels.max().item() + 1,
                   dropout=0.5, device=device)
    model.fit(data.features, data.adj, data.labels, data.idx_train, data.idx_val, k, 200, True, False)
    end = time.perf_counter()
    model.eval()
    output = model.test(data.idx_test)

    return output, end-start

def RGCN1(data, device, k):

    features = data.features
    labels = data.labels
    start = time.perf_counter()
    model = RGCN(nnodes=features.shape[0],
                 nfeat=features.shape[1],
                 nhid=16,
                 nclass=labels.max().item() + 1)
    model.fit(data.features, data.adj, data.labels, data.idx_train, data.idx_val, 200, False)
    end = time.perf_counter()
    model.eval()
    model.test(data.idx_test)
    output5 = model.output
    output = utils.accuracy(output5[data.idx_test], data.labels[data.idx_test])

    return output, end-start

def CFS1(data, device, metric):

    features = data.features
    labels = data.labels

    start = time.perf_counter()
    modified_adj = MD(data.features, data.adj, metric, 0)
    model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(data.features, modified_adj, data.labels, data.idx_train, data.idx_val)
    end = time.perf_counter()
    model.eval()
    output = model.test(data.idx_test)

    return output, end-start


def ProGNN1(data, device, k):

    features = data.features
    labels = data.labels
    start = time.perf_counter()
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout, device=device)
    perturbed_adj1, features1, labels1 = utils.preprocess(data.adj, data.features, data.labels, preprocess_adj=False,device=device)
    prognn = ProGNN(model, args, device)
    prognn.fit(features1, perturbed_adj1, labels1, data.idx_train, data.idx_val)
    end = time.perf_counter()
    prognn.test(features1, labels1, data.idx_test)

    adj = prognn.best_graph
    if prognn.best_graph is None:
        adj = prognn.estimator.normalize()
    output6 = prognn.model(features1, adj)

    output = utils.accuracy(output6[data.idx_test], data.labels[data.idx_test])

    return output, end-start

def GAT1(data, device, attack):
    start = time.perf_counter()
    model, data = Net(data.num_features, data.num_classes).to(device), data.to(device)
    output = torch.tensor(model.fit(data, 200))
    end = time.perf_counter()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(200, model.test(data)[0], model.test(data)[1], output))

    return output, end-start


def CfsGAT1(data, device, metric):

    start = time.perf_counter()
    modified_adj = MD(data.features, data.adj, metric, 0)
    data1 = dataConver1(data, modified_adj)[0]
    model1, data1 = Net(data1.num_features, data1.num_classes).to(device), data1.to(device)
    output = torch.tensor(model1.fit(data1, 200))
    end = time.perf_counter()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(200, model1.test(data1)[0], model1.test(data1)[1], output))

    return output, end-start


def CfsRGCN1(data, device, metric):

    features = data.features
    labels = data.labels
    start = time.perf_counter()
    modified_adj = MD(data.features, data.adj, metric, 0)
    model1 = RGCN(nnodes=features.shape[0],
                  nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1)
    model1.fit(data.features, modified_adj, data.labels, data.idx_train, data.idx_val, 200, False)
    end = time.perf_counter()
    model1.eval()
    model1.test(data.idx_test)
    output5 = model1.output
    output = utils.accuracy(output5[data.idx_test], data.labels[data.idx_test])

    return output, end-start

def GNNGuard1(data, device, k):
    output, t = test(data, device)
    return output, t

def HGCN1(data, device, k):
    output, t = train(data)
    return torch.from_numpy(np.array([output['acc']])), t

def CfsHGCN1(data, device, metric):

    modified_adj = MD(data.features, data.adj, metric, 0)
    data1 = copy.copy(data)
    data1.adj = modified_adj
    output, t = train(data1)
    return torch.from_numpy(np.array([output['acc']])), t


def CFSG1(data, device, k):

    features = data.features
    labels = data.labels
    start = time.perf_counter()
    model = CFSG(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5, device=device, metric=k)
    model.fit(data.features, data.adj, data.labels, data.idx_train, data.idx_val, threshold=0)
    end = time.perf_counter()
    model.eval()
    output = model.test(data.idx_test)

    return output, end-start

