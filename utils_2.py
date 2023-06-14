# _*_codeing=utf-8_*_
# @Time:2022/7/9  10:10
# @Author:mazhixiu
# @File:utils.py
#
# import sys
# sys.path.append('/home/mzx/Pure_GNN/')
import torch
import numpy as np
from deeprobust.graph.data import Dpr2Pyg,Pyg2Dpr
from deeprobust.graph.data.dataset import get_train_val_test
from utils.set_args import *

from scipy import sparse as sp

from torch_geometric.datasets import CitationFull, Coauthor,Amazon # EmailEUCore,LastFMAsia,DeezerEurope,Actor,Airports,FacebookPagePage,GitHub,WikiCS,Twitch

import torch_geometric.utils as pygUtils
import yaml
import os

def load_param(filename):
    stream = open(filename, 'r',encoding='utf-8')
    docs = yaml.load_all(stream.read())
    param_dict = dict()

    for doc in docs:
        for k, v in doc.items():
            param_dict[k] = v

    return param_dict

from torch_scatter import scatter
def load_subgraph_1_hop_features(data,dataset,attack=None,ptb=None,save_dir='/home/mzx/Code/Pure_GNN_1/subgraph/'):

    if attack==None:
        path = save_dir
        judge_dir(save_dir)
    else:
        path = save_dir+attack+'/'+ptb
        judge_dir(path)

    nodes_list = np.arange(data.features.shape[0])
    pyg_data = Dpr2Pyg(data).data
    features = data.features

    if not os.path.exists(path+'/subgraph_features_{}.txt'.format(dataset)):
        # neighbors_features = []
        subset_list = []
        for node in nodes_list:
            # print(node)
            subset, edge_index, mapping, edge_mask = pygUtils.k_hop_subgraph(int(node), 1, pyg_data.edge_index)
            # print(subset)
            neighbor_feature_set = data.features[subset]
            # print(neighbor_feature_set)
            # t = torch.tensor(neighbor_feature_set.todense())
            # print(t.shape)
            subset_x = drop_perturb_edges(threshold=0.01,features=neighbor_feature_set,
                                               nodes_list=subset,center_node=mapping)
            # print(fea_x.shape)
            # neighbors_features.append(fea_x)
            # print(subset_x)

            subset_list.append(subset_x)
            # break

        edge_index_pre = merge_subgraph(subset_list)
        # print(edge_index_pre)
        edge_index_fea = merge_subgraph_features(edge_index_pre,features)
        # print(edge_index_fea.shape)
        # print(path)
        # print(type(edge_index_pre))
        # print(type(edge_index_fea))
        np.savetxt(path+'/subgraph_features_{}.txt'.format(dataset),edge_index_fea)
        np.savetxt(path+'/subgraph_adj_{}.txt'.format(dataset),edge_index_pre)

        x_pre = scatter(torch.FloatTensor(edge_index_fea), index=torch.LongTensor(edge_index_pre), dim=-2, reduce='sum')
        np.savetxt(path + '/subgraph_x_pre_{}.txt'.format(dataset), x_pre)

        return x_pre
    else:
        print('loading subgraph')
        # edge_index_fea = np.loadtxt(path+'/subgraph_features_{}.txt'.format(dataset))
        # edge_index_pre = np.loadtxt(path + '/subgraph_adj_{}.txt'.format(dataset))
        # x_pre = scatter(torch.FloatTensor(edge_index_fea), index=torch.LongTensor(edge_index_pre), dim=-2, reduce='sum')
        # np.savetxt(path + '/subgraph_x_pre_{}.txt'.format(dataset), x_pre)
        # print('start')
        x_pre = np.loadtxt(path + '/subgraph_x_pre_{}.txt'.format(dataset))
        return x_pre


# 带了自环
def merge_subgraph(subset_list):
    edge_index = []
    # print(subset_list)
    for i in subset_list:
        edge_index.extend(i)
    # print(edge_index)
    return edge_index

def merge_subgraph_features(edge_index_pre,features):

    features_j = []
    # print(edge_index_pre)
    # print(features.shape)
    for i in edge_index_pre:

        if type(features) is not np.ndarray:
            # print('csr')
            features_j.append(features[i].todense())
            # features_j =  np.array(features_j).squeeze(axis=1)
        else:
            # print('np')
            # print(features[i].shape)
            features_j.append(features[i])
            # print(np.array(features_j.shape))
            # features_j =  np.array(features_j)

    if type(features) is not np.ndarray:
        return np.array(features_j).squeeze(axis=1)

    else:
        return np.array(features_j)

def drop_perturb_edges(threshold,features,nodes_list,center_node,binary_features=True):
    #
    # print(type(features))
    if type(features) is not np.ndarray:
        features = features.toarray()
    center_node_feature = features[center_node]
    # fea_x = features
    # subset_x = []
    # print(nodes_list)
    all_nodes_list = nodes_list.tolist()
    nodes_list = nodes_list.tolist()
    # print(nodes_list)
    for node in range(len(nodes_list)):
        if binary_features:
            # print(center_node_feature.shape)
            # print(features[node].shape)
            J = jaccard_similarity(center_node_feature,features[node])
            if J < threshold:
                nodes_list.remove(all_nodes_list[node])
        else:
            C = cosine_similarity(center_node_feature,features[node])
            if C < threshold:
                nodes_list.remove(nodes_list[node])
    # print(nodes_list)
    return nodes_list

def jaccard_similarity(a,b):

    intersection = np.multiply(a,b)
    intersection = np.count_nonzero(intersection)
    a_no = np.count_nonzero(a)
    b_no = np.count_nonzero(b)
    if a_no+b_no==intersection:
        J = 0
    else:
        J = intersection*1.0/(a_no+b_no-intersection)
    return J

def cosine_similarity(a,b):
    inner_product = (a*b).sum()
    C = inner_product/(np.sqrt(np.square(a).sum())*np.sqrt(np.square(b).sum())+1e-10)
    return C

# def features_t(data):
#     features = data.features
#     pyg_data = Dpr2Pyg(data).data
#     edge_index = pyg_data.edge_index
#     edge_index_features = features[]

def judge_dir(path):
    if os.path.exists(path):
        print(path)
    else:
        os.makedirs(path)


def dir_(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

experiment_attention_dir = './attention_distribution/'

def save_aggr_distribution(model,dataset,layer,aggregators,attention,aggr_max,attack=None,ptb=None):
    """
    保存某个数据集的聚合器的注意力分布
    :param dataset: 哪个数据集下的分布
    :param layer: 第几层的
    :param aggregators: 那几个聚合器的分布
    :param attention: 哪个注意力机制下的
    :param aggr_max: 分布是什么
    :param attack: 攻击
    :param ptb: 扰动比率
    :return:
    """
    # print(aggr_max)
    # print(model)
    path = experiment_attention_dir+model
    if not os.path.exists(path):
        os.makedirs(path)
    if attack == None:
        np.savetxt("{}/{}_{}_{}_{}.txt".format(path, model,dataset, layer,aggregators,attention), np.array(aggr_max))
    else:
        np.savetxt("{}/{}_{}_{}_{}_{}_{}.txt".format(path, model,dataset, layer,aggregators,attention,
                                                     attack,ptb), np.array(aggr_max))


experiment_output_dir = './output_representation'
def save_output(output,d_name,attention,aggregators,attack=None,ptb=None):

    if not os.path.exists(experiment_output_dir):
        os.makedirs(experiment_output_dir)
    if attack==None:
        np.save("{}/{}_{}_{}".format(experiment_output_dir, d_name, attention, aggregators),output)
    else:
        np.save("{}/{}_{}_{}_{}_{}".format(experiment_output_dir, d_name, attention, aggregators,attack,ptb), output)


def from_scipy_sparse_matrix(A):
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


# clean_dataset_dir ='/home/mzx/CodeSet/DATA/clean_data/'
# # attack_data_dir ='/home/mzx/CodeSet/DATA/attack_data/'
#
# # clean_dataset_dir ='/home/mzx/Code/MAGNET_1/clean_data/'
# attack_data_dir ='/home/mzx/Code/MAGNET_v3/attack_data_1204/'


class Dataset_Attack():
    def __init__(self,dataset):
        self.dataset = dataset
        self.adj = None
        self.features = None
        self.labels= None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2},labels={3},idx_train={4},idx_val={5},idx_test={6})'.format(
            'Dataset_Attack:'+self.dataset, self.adj.shape, self.features.shape,
            self.labels.shape,self.idx_train.shape,self.idx_val.shape,self.idx_test.shape)

from utils.set_args import *
# from cut_data import cutDataSet
from deeprobust.graph.data import Dataset
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph

class Dataset_():
    """
    # 模仿dpr写的数据集读取形式
    """
    def __init__(self,clean_dir=clean_dataset_dir,
                 dataset='cora',
                 attack=None,
                 attack_dir=attack_data_dir,
                 ptb=None,
                 largest_component=True,
                 is_cut=True,
                 num_nodes_dict=cut_n_dict_2):
        """
        数据集读取类（clean/attack）
        :param clean_dir: 干净数据集保存目录
        :param dataset: 数据集
        :param attack: 攻击类型
        :param attack_dir: 攻击数据集保存目录
        :param largest_component: 是否读取最大联通组件
        :param ptb: 攻击扰动比率
        :param is_cut: 是否切割数据集
        :param num_nodes_dict: 保存的节点数量
        """
        # print(attack)
        # print(attack_dir)
        # print(clean_dir)
        self.clean_dir = clean_dir
        self.dataset = dataset.lower()

        self.attack = attack
        self.attack_dir = attack_dir
        self.ptb = ptb

        self.largest_component = largest_component

        self.is_cut = is_cut
        self.num_nodes_dict = num_nodes_dict

        assert self.dataset in ['acm', 'cora', 'citeseer', 'cora_ml',  'pubmed','photo',
                                'polblogs','blogcatalog', 'uai', 'flickr','physics','dblp','computers',
                                # 'cs','actor'
                                ], \
                'Currently only support  pubmed, acm, cora, citeseer, cora_ml,' # ' polblogs, blogcatalog, flickr'

        if attack==None:
            self.adj, self.features, self.labels,self.idx_train,self.idx_val,self.idx_test = self.load_clean_data(self.clean_dir,
                                                                             self.dataset)
        else:
            if ptb==None:
                assert 'please specify attack ptb'
            self.adj, self.features, self.labels,self.idx_train,self.idx_val,\
                                        self.idx_test = self.load_attack_data(self.dataset,
                                                                              self.attack,
                                                                              self.attack_dir,
                                                                              self.ptb)
    def is_largest_component(self,pyg_data):
        nx_data = to_networkx(pyg_data, to_undirected=True)
        flag = nx.is_connected(nx_data)
        print(flag)
        return nx_data,flag

    def load_largest_component(self,nx_data,data):

        largest_cc = max(nx.connected_components(nx_data), key=len)
        largest_cc = list(largest_cc)
        num_nodes = len(largest_cc)
        features = np.array(data.x[largest_cc])
        # print(features.shape)
        row, col = np.nonzero(features)
        # print(len(row))
        values = features[row, col]
        # print(len(values))
        # print(len())
        features = sp.csr_matrix((values, (row, col)), shape=features.shape)

        labels = np.array(data.y[largest_cc])
        # print(len(labels))
        # print(len(data.edge_index[0]))
        subset = torch.LongTensor(largest_cc)
        edge_index, edge_mask = subgraph(subset=subset, edge_index=data.edge_index,relabel_nodes=True)
        # print(len(edge_index[0]))
        idx_train, idx_val, idx_test = get_train_val_test(
            len(labels), val_size=0.1, test_size=0.8, stratify=labels, seed=15)
        adj = pygUtils.to_scipy_sparse_matrix(edge_index).tocsr()

        return adj, features,labels, idx_train, idx_val, idx_test

    def load_clean_data(self,clean_dataset_dir,name):

        if name =='dblp':
            pyg_data = CitationFull(root=clean_dataset_dir, name=name)
        elif name == 'cs' or name=='physics':
            pyg_data = Coauthor(root=clean_dataset_dir,name=name)
        elif name =='computers' or name=='photo':
            pyg_data = Amazon(root=clean_dataset_dir,name=name)
            # print(pyg_data[0])
            # print(pyg_data[0].edge_index[0)
            # print(len(pyg_data[0].edge_index[0]) / 2)


        # elif name == 'emaileucore':
        #     pyg_data = EmailEUCore(root=clean_dataset_dir + '/emaileucore').data
        # elif name == 'deezereurope':
        #     pyg_data = DeezerEurope(root=clean_dataset_dir + '/deezereurope').data
        #     print(pyg_data)
        # elif name == 'usa' or name == 'brazil' or name == 'europe':
        #     pyg_data = Airports(root=clean_dataset_dir + '/airports', name=name).data
        #     print(pyg_data)
        # elif name == "de" or name == "en" or name == "es" or name == "fr" or name == "pt" or name == "ru":
        #     name = name.upper()
        #     print(clean_dataset_dir + './twitch')
        #     pyg_data = Twitch(root=clean_dataset_dir + './twitch', name=name).data
        #     print(pyg_data)
        #
        #
        # elif name == 'facebookpagepage':
        #     pyg_data = FacebookPagePage(root=clean_dataset_dir + '/facebookpagepage').data
        #     print(pyg_data)
        # elif name == 'lastfmasia':
        #     pyg_data = LastFMAsia(root=clean_dataset_dir + '/lastfmasia').data
        # elif name == 'wikics':
        #     pyg_data = WikiCS(root=clean_dataset_dir + '/wikics').data
        # elif name=='actor':
        #     pyg_data = Actor(root=clean_dataset_dir+'/actor').data
        #     print(pyg_data)
        # elif name=='github':
        #     pyg_data = GitHub(root=clean_dataset_dir+'/github').data
        #     print(pyg_data)

        elif name =='cora' or name=='cora_ml'or name=='citeseer'or name=='polblogs'or name=='pubmed'\
                or name =='acm' or name=='flicker' or name=='uai' or name=='flickr' or name=='blogcatalog':
            dpr_data = Dataset(root=clean_dataset_dir,name=name,seed=15)
            pyg_data = Dpr2Pyg(dpr_data)
            # print(pyg_data[0])
            # print(len(pyg_data[0].edge_index[0])/2)
        else:
            assert name + "is not be supported！"

        dpr_data = Pyg2Dpr(pyg_data)
        print('loading '+ self.dataset+' dataset.....')
        print(self.is_cut)
        if self.is_cut:

            if self.attack=='Metattack' and self.dataset=='photo':
                name = 'meta_photo'

            adj, features, labels, idx_train, idx_val, idx_test = self.cutDataSet(dpr_data, self.num_nodes_dict[name])

            row, col = np.nonzero(features)
            values = features[row, col]
            features = sp.csr_matrix((values, (row, col)), shape=features.shape)

            if self.largest_component:
                data = self.toPygData(adj, features, labels, idx_train, idx_val, idx_test)
                nx_data, flag = self.is_largest_component(data)
                if not flag:
                    adj, features, labels, idx_train, idx_val, idx_test \
                        = self.load_largest_component(nx_data,data)

            row, col = np.diag_indices_from(adj)
            adj[row, col] = 1

            return adj, features, labels, idx_train, idx_val, idx_test

        elif self.largest_component:
                features = dpr_data.features
                labels = dpr_data.labels
                adj = dpr_data.adj
                idx_train, idx_val, idx_test = get_train_val_test(
                len(dpr_data.labels), val_size=0.1, test_size=0.8, stratify=dpr_data.labels, seed=15)

                data = self.toPygData(dpr_data.adj, dpr_data.features, dpr_data.labels, idx_train, idx_val, idx_test)
                nx_data, flag = self.is_largest_component(data)
                if not flag:
                    adj, features, labels, idx_train, idx_val, idx_test \
                        = self.load_largest_component(nx_data,data)

                row, col = np.diag_indices_from(adj)
                adj[row, col] = 1
                return adj, features, labels, idx_train, idx_val, idx_test
        else:
            return dpr_data.adj,dpr_data.features,dpr_data.idx_train,dpr_data.idx_val,dpr_data.idx_test


    def index_to_mask(self,index, size):
        mask = torch.zeros((size,), dtype=torch.bool)
        mask[index] = 1
        return mask

    def toPygData(self,adj, features, labels, idx_train, idx_val, idx_test):
        # Dpr2Pyg
        edge_index = torch.LongTensor(adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(features):
            x = torch.FloatTensor(features.todense()).float()
        else:
            x = torch.FloatTensor(features).float()
        y = torch.LongTensor(labels)

        data = Data(x=x, edge_index=edge_index, y=y)
        # train_mask = self.index_to_mask(idx_train, size=y.size(0))
        # val_mask = self.index_to_mask(idx_val, size=y.size(0))
        # test_mask = self.index_to_mask(idx_test, size=y.size(0))
        # data.train_mask = train_mask
        # data.val_mask = val_mask
        # data.test_mask = test_mask
        return data

    def load_attack_data(self,name, attack, attack_dir,ptb):
        adj, features, labels, idx_train, idx_val, idx_test = self.load_clean_data(clean_dataset_dir, name)

        if attack =='Metattack' or attack=='Nettack' or  attack=='SGAttack' or attack=='Dice':
            data_attack = Dataset_Attack(name)
            data_attack.adj = adj
            data_attack.features = features
            data_attack.labels = labels
            data_attack.idx_train = idx_train
            data_attack.idx_val = idx_val
            data_attack.idx_test = idx_test
            adj, features, labels, idx_train, idx_val, idx_test = self.cutDataSet(data_attack,self.num_nodes_dict[name])

        features = features
        path = attack_dir + '/' + attack + '/' + name
        perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.npz'.format(path, attack, name, float(ptb)))
        adj = perturbed_adj
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        labels = torch.LongTensor(labels)

        row, col = np.diag_indices_from(adj)
        adj[row, col] = 1

        return adj,features, labels, idx_train, idx_val, idx_test

    def cutDataSet(self,data, k):
        """

        :param data: 传入的数据集
        :return:
        """
        print(k)
        num_node = len(data.labels)
        k = min(k, num_node)

        keep_nodes = np.array([i for i in range(k)])
        features = data.features[0:k]
        labels = data.labels[0:k]

        edges = data.adj.nonzero()
        e0 = np.array(edges[0])
        e1 = np.array(edges[1])
        edge_index_array = np.array(list(zip(e0, e1)))
        edge_index = torch.from_numpy(edge_index_array.T)
        nodes = torch.LongTensor(keep_nodes)

        edge_index = pygUtils.subgraph(nodes, torch.LongTensor(edge_index.long()))[0]

        idx_train, idx_val, idx_test = get_train_val_test(
            len(labels), val_size=0.1, test_size=0.8, stratify=labels, seed=15)

        features = features
        labels = labels
        adj = pygUtils.to_scipy_sparse_matrix(edge_index).tocsr()
        idx_train = idx_train
        idx_val = idx_val
        idx_test = idx_test

        return adj,features,labels,idx_train,idx_val,idx_test

    def heterophily_handle(self,pyg_data,):
        n = pyg_data.num_nodes
        self.idx_train = self.mask_to_index(pyg_data.train_mask,n)
        self.idx_val = self.mask_to_index(pyg_data.val_mask, n)
        self.idx_test = self.mask_to_index(pyg_data.test_mask, n)

    def mask_to_index(self,index, size):
        all_idx = np.arange(size)
        return all_idx[index]

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2},labels={3},idx_train={4},idx_val={5},idx_test={6})'.format(
            'Dataset_PureGNN:'+self.dataset, self.adj.shape, self.features.shape,
            self.labels.shape,self.idx_train.shape,self.idx_val.shape,self.idx_test.shape)


import json
def load_attack_nodes(attack,dataset):
    with open(attack_data_dir+attack+'/'+dataset+'_attacked_nodes_1.json', "r", encoding="utf-8") as f1:
        attack_nodes_json = json.load(f1)
        attack_nodes = attack_nodes_json['attacked_test_nodes']
        attack_nodes_sort = np.sort(attack_nodes)
        print(attack_nodes_sort)
        return attack_nodes_sort

if __name__=='__main__':
    """
    各个工具函数的测试
    :return:
    """
    import warnings
    warnings.filterwarnings('ignore')
    # print('test')
    # ACM”, “BlogCatalog”, “Flickr”, “UAI”, “Flickr
    # ['acm', 'blogcatalog', 'uai', 'flickr']
    # dataset_list = ['cs', 'physics','dblp','computers','photo']
    # dataset_list  =  ['acm', 'blogcatalog', 'uai', 'flickr']
    # dataset_list = [
    #                  # 'cora',
    #                     # 'cora_ml','citeseer.yaml','polblogs','pubmed',
    #                 'acm', 'blogcatalog', 'uai', 'flickr',
    #                 # 'cs', 'physics','dblp', 'computers', 'photo'
    #                 ]
    # dataset_list = ['acm','blogcatalog']
    # dataset_list= ['lastfmasia', 'facebookpagepage', 'deezereurope',  'github',]
    # dataset_list = ['cora','cora_ml','citeseer','polblogs','pubmed','acm','uai']
    # dataset_list = ['uai']
    dataset_list = [ 'cora_ml','citeseer','pubmed','acm','uai','photo']
    # dataset_list = ["DE", "EN", "ES", "FR", "PT", "RU"]
    # dataset_list = ['usa','brazil']
    # dataset_list = ['pubmed','photo']
    for dataset in dataset_list:
        data = Dataset_(dataset=dataset)
        # print(data.adj[0][0])
        # print(data.adj[10][10])
        print(data)
        # print()
        print('双向边：',len(data.adj.nonzero()[0]))

        print('减去自环的单项边：',(len(data.adj.nonzero()[0])-len(data.labels))/2)

        print('加上自环的单项边：', (len(data.adj.nonzero()[0]) - len(data.labels)) / 2+len(data.labels))
        # print(le)
        # print(np.unique(data.labels))
        # data = Dataset_(dataset=dataset)
        # data = Dataset_(dataset=dataset,attack='Metattack',ptb='0.25',largest_component=False)
        # for attack in ['Nettack','SGAttack','RND']:
        #     data = Dataset_(dataset=dataset, attack=attack, ptb='5.0')
        #     print(data)
        #     print(data.features.shape)
        #     # print(type(data.features))
        #     print(type(data.adj))
        #     # print(len(data.adj.nonzero()[0]))
        #     # t = np.unique(data.labels)
        #     # print(t)
        #
        # for attack in ['Dice', 'Random', 'Metattack']:
        #
        #     data = Dataset_(dataset=dataset, attack=attack, ptb='0.25')
        #     print(data)
        #     print(data.features.shape)
        #     # print(type(data.features))
        #     print(type(data.adj))

    # find_attack_nodes_in_global()
    #
    # # test_1()
    # print('\n\n')
    # test_2()
    # find_attack_nodes_in_global_1()