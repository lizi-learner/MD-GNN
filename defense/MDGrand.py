import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
from numba import njit
import torch.optim as optim

class CFSG(GCN):
    """GCNJaccard first preprocesses input graph via droppining dissimilar
    edges and train a GCN based on the processed graph. See more details in
    Adversarial Examples on Graph Data: Deep Insights into Attack and Defense,
    https://arxiv.org/pdf/1903.01610.pdf.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCNJaccard.


    """
    def __init__(self, nfeat, nhid, nclass, binary_feature=True, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device='cpu', metric='similarity'):

        super(CFSG, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device
        self.binary_feature = binary_feature
        self.metric = metric

    def fit(self, features, adj, labels, idx_train, idx_val=None, threshold=0, train_iters=200, initialize=True, verbose=False, **kwargs):
        """First drop dissimilar edges with similarity smaller than given
        threshold and then train the gcn model on the processed graph.
        When idx_val is not None, pick the best model according to the
        validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        threshold : int
            similarity threshold for dropping edges. If two connected nodes with similarity smaller than threshold, the edge between them will be removed.
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.threshold = threshold
        modified_adj = self.drop_dissimilar_edges(features, adj, self.metric)
        features = self.feature_propagation(features, modified_adj, 5)
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, train_iters=train_iters, initialize=initialize, verbose=verbose)

    def feature_propagation(self, features, adj, order):

        features = normalize(features)

        adj = adj + sp.eye(adj.shape[0])
        D1_ = np.array(adj.sum(axis=1)) ** (-0.5)
        D2_ = np.array(adj.sum(axis=0)) ** (-0.5)
        D1_ = sp.diags(D1_[:, 0], format='csr')
        D2_ = sp.diags(D2_[0, :], format='csr')
        A_ = adj.dot(D1_)
        A_ = D2_.dot(A_)

        D1 = np.array(adj.sum(axis=1)) ** (-0.5)
        D2 = np.array(adj.sum(axis=0)) ** (-0.5)
        D1 = sp.diags(D1[:, 0], format='csr')
        D2 = sp.diags(D2[0, :], format='csr')

        A = adj.dot(D1)
        A = D2.dot(A)

        features = torch.FloatTensor(np.array(features.todense()))

        A = sparse_mx_to_torch_sparse_tensor(A)

        features = propagate(features, A, order)

        return sp.csr_matrix(features.numpy())

    def drop_dissimilar_edges(self, features, adj, metric='order'):
        """Drop dissimilar edges.(Faster version using numba)
        """
        print('deleting edges...')
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)

        # 取出稀疏矩阵上三角部分的元素
        adj_triu = sp.triu(adj, format='csr')

        # 选择标准
        if metric == 'Cfs': # jaccard + cn
            modified_adj, removed_cnt = dropedge_order_jaccard(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        if metric == 'Cfs1': # cosine + cn
            modified_adj, removed_cnt = dropedge_order_cosine(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)
        if metric == 'Cs': # cn
            modified_adj, removed_cnt = dropedge_order(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, threshold=self.threshold)
        if metric == 'Cs1': # cn with single nodes
            modified_adj, removed_cnt = dropedge_order1(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, threshold=self.threshold)
        if metric == 'Jaccard1': # jaccard without single nodes
            modified_adj, removed_cnt = dropedge_jaccard(adj, adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=self.threshold)

        print('removed %s edges in the original graph' % removed_cnt)
        return modified_adj



def dropedge_order(adj_triu, A, iA, jA, threshold=0):

    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    adj_triu1 = adj_triu.A
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    adj_triu1[n1][n2] = 0
                    degrees[n1] -= 1
                    degrees[n2] -= 1
                    removed_cnt += 1


    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')

    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, removed_cnt


def dropedge_order_cosine(adj_triu, A, iA, jA, features, threshold):
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    a, b = features[n1], features[n2]
                    intersection = a.multiply(b).count_nonzero()
                    C = intersection * 1.0 / np.sqrt((a.count_nonzero() * b.count_nonzero()))
                    if threshold == 0:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(C)
                        removed_cnt += 1
                    elif C < threshold:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(C)
                        removed_cnt += 1

    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):

        max_index = np.argmin(score)

        if degrees[l1[max_index]] != 1 and degrees[l2[max_index]] != 1:
            adj_triu1[l1[max_index]][l2[max_index]] = 0

            degrees[l1[max_index]] -= 1
            degrees[l2[max_index]] -= 1
            cnt += 1
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, cnt


def dropedge_order_jaccard(adj_triu, A, iA, jA, features, threshold):
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                if degrees[n1] != 1 and degrees[n2] != 1:
                    a, b = features[n1], features[n2]
                    intersection = a.multiply(b).count_nonzero()
                    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
                    if threshold == 0:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(J)
                        removed_cnt += 1
                    elif J < threshold:
                        l1.append(n1)
                        l2.append(n2)

                        score.append(J)
                        removed_cnt += 1

    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):

        max_index = np.argmin(score)

        if degrees[l1[max_index]] != 1 and degrees[l2[max_index]] != 1:
            adj_triu1[l1[max_index]][l2[max_index]] = 0

            degrees[l1[max_index]] -= 1
            degrees[l2[max_index]] -= 1
            cnt += 1
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, cnt


def dropedge_order1(adj_triu, A, iA, jA, threshold):
    # 不考虑单个点的情况
    removed_cnt = 0
    degrees = adj_triu.A.sum(0)
    S = np.dot(adj_triu.A, adj_triu.A) - np.diag(degrees)

    adj_triu1 = adj_triu.A
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):

            n1 = row
            n2 = jA[i]

            order = S[n1][n2]

            if order == 0:
                adj_triu1[n1][n2] = 0
                removed_cnt += 1

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')

    modified_adj = adj_triu1 + adj_triu1.transpose()

    return modified_adj, removed_cnt


def dropedge_jaccard(adj_triu, A, iA, jA, features, threshold = 0.03):

    removed_cnt = 0
    degrees = adj_triu.A.sum(0)

    l1 = []
    l2 = []
    score = []
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]

            a, b = features[n1], features[n2]
            intersection = a.multiply(b).count_nonzero()
            J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)

            if J < threshold and degrees[n1] != 1 and degrees[n2] != 1:
                l1.append(n1)
                l2.append(n2)
                score.append(J)
                removed_cnt += 1
    print('removed_cnt: {}'.format(removed_cnt))
    score = np.array(score)
    adj_triu1 = adj_triu.A
    cnt = 0
    for i in range(removed_cnt):
        # 若去掉边不导致独立点，去掉分数最低的边
        max_index = np.argmin(score)
        if degrees[l1[max_index]] != 1 and degrees[l2[max_index]] != 1:
            adj_triu1[l1[max_index]][l2[max_index]] = 0
            # print(np.nonzero(np.array(adj_triu1))[0].size)
            degrees[l1[max_index]] -= 1
            degrees[l2[max_index]] -= 1
            cnt += 1
        score[max_index] = 100

    adj_triu1 = sp.csr_matrix(adj_triu1)
    adj_triu1 = sp.triu(adj_triu1, format='csr')
    modified_adj = adj_triu1 + adj_triu1.transpose()
    return modified_adj, cnt


def propagate(feature, A, order):
    # feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        # print(y.add_(x))
        y.add_(x)

    return y.div_(order + 1.0).detach_()

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)