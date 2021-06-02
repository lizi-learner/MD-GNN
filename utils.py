import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deeprobust.graph.data import Dataset
import torch
from torch_geometric.data.data import Data


def dataConver(d_name, root='../data/'):
    '''
    Load data from npz and convert to data
    :param d_name: data name
    :param root: path
    :return: data
    '''
    dataset = Dataset(root=root, name=d_name, setting='nettack',seed=15, require_mask=True)
    adj, features, labels = dataset.adj, dataset.features, dataset.labels
    features = features.todense()
    edge_index ,_= matrixToedge(adj)
    data = Data(x=torch.tensor(features,dtype=torch.float32),
                edge_index=edge_index,
                y=torch.LongTensor(labels))
    data.val_mask = torch.tensor(dataset.val_mask)
    data.train_mask = torch.tensor(dataset.train_mask)
    data.test_mask = torch.tensor(dataset.test_mask)
    data.num_classes = labels.max().item() + 1
    return [data]


def dataConver1(dataset, perturbed_adj):
    '''
    Load data from npz and convert to data
    :param d_name: data name
    :param root: path
    :return: data
    '''
    # dataset = Dataset(root=root, name=d_name, setting='nettack',seed=15, require_mask=True)
    adj, features, labels = perturbed_adj, dataset.features, dataset.labels
    features = features.todense()
    edge_index ,_= matrixToedge(adj)
    data = Data(x=torch.tensor(features,dtype=torch.float32),
                edge_index=edge_index,
                y=torch.LongTensor(labels))
    data.val_mask = torch.tensor(dataset.val_mask)
    data.train_mask = torch.tensor(dataset.train_mask)
    data.test_mask = torch.tensor(dataset.test_mask)
    data.num_classes = labels.max().item() + 1
    return [data]

def matrixToedge(matrix):
    '''
    Convert sparse matrix to edge_index in Data
    :param matrix:
    :return: edge_index
    '''
    matrix = matrix.tocoo()
    return torch.LongTensor([np.array(matrix.row),np.array(matrix.col)]),matrix.data

def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct1 = correct.sum()
    return correct, correct1 / len(labels)

def getChangeNodes(adv_adj, ori_adj):
    """

    :param adv_adj: numpy矩阵，干扰后的邻接矩阵
    :param ori_adj: numpy矩阵，原始的邻接矩阵
    :return:
    """
    adj_changes = adv_adj - ori_adj
    adj_changes = np.nonzero(adj_changes)
    return adj_changes

def get_second_order_similarity(adj):

    """

    :param adj: numpy矩阵：邻接矩阵
    :return: S：numpy矩阵：记录二阶相似度
    """
    degrees = adj.sum(0)
    S = np.dot(adj, adj) - np.diag(degrees)
    D = np.diag(degrees)
    S = np.dot(np.linalg.inv(np.power(D, 0.5)), S, np.linalg.inv(np.power(D, 0.5)))

    return S


def experiment2_save(perturbed_adj, cleaned_adj, adj_ori):
    changed_adj = perturbed_adj.A - adj_ori
    totla_clean_adj = perturbed_adj.A - cleaned_adj

    no_clean_adj = changed_adj * cleaned_adj
    right_clean_adj = changed_adj * (perturbed_adj.A - cleaned_adj)

    l1 = np.nonzero(totla_clean_adj)
    l2 = np.nonzero(changed_adj)

    l3 = np.nonzero(right_clean_adj)
    l4 = np.nonzero(no_clean_adj)

    return l4[0].size / 2, l3[0].size / 2, l1[0].size / 2 - l3[0].size / 2