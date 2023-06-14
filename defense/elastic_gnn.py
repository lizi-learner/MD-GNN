import os
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
import torch_geometric.transforms as T

import torch
from torch import Tensor, optim
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_sparse
from torch_sparse import SparseTensor, matmul
from torch.nn import Linear
import argparse
from copy import deepcopy
from deeprobust.graph import utils

path=os.getcwd()
path,_=os.path.split(path)
print(path)
clean_data_dir=path+'/data/clean_data/'
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser(description='ElasticGNN')
    # parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--log_steps', type=int, default=200)
    # parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--conv', type=str, default='ElasticGNN')
    # parser.add_argument('--num_layers', type=int, default=2)
    # parser.add_argument('--nclass', type=int, default=64)
    # parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--weight_decay', type=float, default=0.0005)
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--runs', type=int, default=1)
    # parser.add_argument('--normalize_features', type=str2bool, default=True)
    # parser.add_argument('--random_splits', type=int, default=0, help='default: fix split')
    # parser.add_argument('--seed', type=int, default=12321312)
    parser.add_argument('--K', type=int, default=5)
    # parser.add_argument('--lambda1', type=float, default=9)
    # parser.add_argument('--lambda2', type=float, default=9)
    parser.add_argument('--L21', type=str2bool, default=True)
    # parser.add_argument('--ptb_rate', type=float, default=0)

    args = parser.parse_args()
    # args.ogb = True if 'ogb' in args.dataset.lower() else False
    return args

def get_inc(edge_index):
    # print(edge_index)
    size = edge_index.sizes()[1]
    # print(soze)
    # print(size)
    row_index = edge_index.storage.row()
    col_index = edge_index.storage.col()
    # print(row_index)
    # print(col_index)
    # 上三角矩阵
    mask = row_index >= col_index  # remove duplicate edge and self loopk)
    # print(len(mask))
    row_index = row_index[mask]
    col_index = col_index[mask]
    edge_num = row_index.numel()
    # print("edge_num:", edge_num)
    row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
    col = torch.cat([row_index, col_index])
    value = torch.cat([torch.ones(edge_num), -1 * torch.ones(edge_num)]).cuda()
    # print(row)
    # print(col)
    inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
                       sparse_sizes=(edge_num, size))
    return inc


def inc_norm(inc, edge_index):
    ## edge_index: unnormalized adjacent matrix
    ## normalize the incident matrix
    edge_index = torch_sparse.fill_diag(edge_index, 1.0)  ## add self loop to avoid 0 degree node
    deg = torch_sparse.sum(edge_index, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    inc = torch_sparse.mul(inc, deg_inv_sqrt.view(1, -1))  ## col-wise
    return inc


def check_inc(edge_index, inc):
    nnz = edge_index.nnz()
    # print('nnz:',nnz)
    deg = torch.eye(edge_index.sizes()[0]).cuda()
    # print('deg:',deg)
    adj = edge_index.to_dense()
    # print('adj',adj)
    # print(inc)
    lap = (inc.t() @ inc).to_dense()
    # print('lap:',lap)
    lap2 = deg - adj
    # print('lap2:',lap2)
    diff = torch.sum(torch.abs(lap2 - lap)) / nnz
    # print(diff.shape)
    assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'
    # return diff

def get_transform(transform,normalize_features=True):
    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform
    return transform

class EMP(MessagePassing):
    r"""The elastic message passing layer from the paper
        "Elastic Graph Neural Networks", ICML 2021
    """
    _cached_adj_t: Optional[SparseTensor]
    _cached_inc = Optional[SparseTensor]

    def __init__(self,
                 K: int,
                 lambda1: float = None,
                 lambda2: float = None,
                 L21: bool = True,
                 dropout: float = 0,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 **kwargs):

        super(EMP, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L21 = L21
        self.dropout = dropout
        self.cached = cached

        assert add_self_loops == True and normalize == True, ''
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_adj_t = None
        self._cached_inc = None  ## incident matrix

    def reset_parameters(self):
        self._cached_adj_t = None
        self._cached_inc = None

    # x = self.prop(x, adj_t, data=data)
    def forward(self, x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                data=None) -> Tensor:
        """"""
        # 参数K的作用：传播步数
        if self.K <= 0: return x

        assert isinstance(edge_index, SparseTensor), "Only support SparseTensor now"
        assert edge_weight is None, "edge_weight is not supported yet, but it can be extented to weighted case"
        # print(edge_index)
        self.unnormalized_edge_index = edge_index

        if self.normalize:
            cache = self._cached_inc
            if cache is None:
                # print("edge_index=", edge_index)
                inc_mat = get_inc(edge_index=edge_index)  ## compute incident matrix before normalizing edge_index
                # print(inc_mat)
                inc_mat = inc_norm(inc=inc_mat, edge_index=edge_index)  ## normalize incident matrix

                if self.cached:
                    self._cached_inc = inc_mat
                    self.init_z = torch.zeros((inc_mat.sizes()[0], x.size()[-1])).cuda()
            else:
                inc_mat = self._cached_inc

            cache = self._cached_adj_t
            if cache is None:
                # print("cache:", cache)
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    add_self_loops=self.add_self_loops, dtype=x.dtype)
                # print(edge_index)

                if x.size()[0] < 30000:
                    # print(edge_index)
                    # print(inc_mat)
                    check_inc(edge_index=edge_index, inc=inc_mat)  ## ensure L=B^TB

                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

        hh = x
        x = self.emp_forward(x=x, hh=hh, edge_index=edge_index, inc=inc_mat, K=self.K)
        # print(x.shape)
        return x

    def emp_forward(self, x, hh, K, edge_index, inc):
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma = 1 / (1 + lambda2)
        beta = 1 / (2 * gamma)

        if lambda1 > 0:
            z = self.init_z.detach()

        for k in range(K):

            if lambda2 > 0:
                # y = x - gamma * (x - hh + lambda2 * (x - self.propagate(edge_index, x=x, edge_weight=None, size=None)))
                ## simplied as the following if gamma = 1/(1+lambda2):
                y = gamma * hh + (1 - gamma) * self.propagate(edge_index, x=x, edge_weight=None, size=None)
            else:
                y = gamma * hh + (1 - gamma) * x  # y = x - gamma * (x - hh)

            if lambda1 > 0:
                x_bar = y - gamma * (inc.t() @ z)
                z_bar = z + beta * (inc @ x_bar)
                if self.L21:
                    z = self.L21_projection(z_bar, lambda_=lambda1)
                else:
                    z = self.L1_projection(z_bar, lambda_=lambda1)
                x = y - gamma * (inc.t() @ z)
            else:
                x = y  # z=0

            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def L1_projection(self, x: Tensor, lambda_):
        # component-wise projection onto the l∞ ball of radius λ1.
        return torch.clamp(x, min=-lambda_, max=lambda_)

    def L21_projection(self, x: Tensor, lambda_):
        # row-wise projection on the l2 ball of radius λ1.
        row_norm = torch.norm(x, p=2, dim=1)
        scale = torch.clamp(row_norm, max=lambda_)
        index = row_norm > 0
        scale[index] = scale[index] / row_norm[index]  # avoid to be devided by 0
        return scale.unsqueeze(1) * x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, lambda1={}, lambda2={}, L21={})'.format(
            self.__class__.__name__, self.K, self.lambda1, self.lambda2, self.L21)

class ElasticGNN(torch.nn.Module):

    def __init__(self, nfeat, nhid, nclass,dropout=0.5,lr=0.01,weight_decay=5e-4,
                 with_bias=True,lambda_1=0,lambda_2=0, device=None):
        super(ElasticGNN, self).__init__()
        assert device is not None,"Please specify 'device'!"
        self.device = device


        args = parse_args()
        self.weight_decay = weight_decay
        self.lin1 = Linear(nfeat, nhid,bias=False)
        self.lin2 = Linear(nhid, nclass,bias=False)
        self.dropout = dropout
        self.lr = lr
        self.with_bias=with_bias
        self.output = None
        self.prop = EMP(K=args.K,
                        lambda1=lambda_1,
                        lambda2=lambda_2,
                        L21=args.L21,
                        cached=True)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):

        x, adj_t, = data.x, data.adj_t
        # print(x.shape)
        # print(adj_t)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        # print(x.shape)
        # print(adj_t)
        x = self.prop(x, adj_t, data=data)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=500, **kwargs):
        """Train the MedianGCN conv, when idx_val is not None, pick the best conv
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        # self.device = self.conv1.weight.device
        if initialize:
            self.reset_parameters()

        self.data = pyg_data.to(self.device)
        # print("self.data:",pyg_data)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)
        # if idx_val is None:
        #     self._train_without_val(labels, idx_train, train_iters, verbose)
        # else:
        #     if patience < train_iters:
        #         self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
        #     else:
        #         self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn conv ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):

            self.train()
            optimizer.zero_grad()
            output, embeddings = self.myforward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_ssl = self.lambda_ * self.regression_loss(embeddings)
            loss_total = loss_train + loss_ssl
            loss_total.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best conv according to the performance on validation ===')
        self.load_state_dict(weights)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training ElasticGNN conv ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask
        # print("train_mask:",train_mask)
        # print("val_mask:",val_mask)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.load_state_dict(weights)

    @torch.no_grad()
    def test(self, pyg_data=None):
        self.eval()
        data = pyg_data.to(self.device) if pyg_data is not None else self.data
        test_mask = data.test_mask
        labels = data.y
        output = self.forward(data)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, pyg_data=None):
        """
        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object

        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of MedianGCN
        """

        self.eval()
        data = pyg_data.data.to(self.device) if pyg_data is not None else self.data
        return self.forward(data)

if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg
    from utils.utils_2 import Dataset_,judge_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # from deeprobust.graph.adj_changes import MedianGCN
    judge_dir(clean_data_dir)
    data = Dataset(root=clean_data_dir, name='cora')

    # data = Dataset_(dataset='cora')
    # datapy = Planetoid(root=clean_data_dir + '\pubmed\\', name='Pubmed')
    # # print(datapy)
    # print("Pubmed:",datapy[0])
    # Pubmed: Data(x=[19717, 500], edge_index=[2, 88648], y=[19717], train_mask=[19717], val_mask=[19717], test_mask=[19717])
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ElasticGNN = ElasticGNN(nfeat=features.shape[1],nhid=64,
    #                       nclass=labels.max().item() + 1,
    #                       device=device)
    # ElasticGNN = ElasticGNN.to(device)
    # # print(ElasticGNN)
    # pyg_data = Dpr2Pyg(data)
    #
    # # print("cora:",pyg_data.data)
    # transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    # data1 = transform(pyg_data.data)
    # # print(data1)
    # # Data(x=[2485, 1433], edge_index=[2, 10138], y=[2485], train_mask=[2485], val_mask=[2485], test_mask=[2485])
    # ElasticGNN.fit(data1, verbose=True)  # train with earlystopping
    # ElasticGNN.test()
    # print(ElasticGNN.predict().size())