
import sys
sys.path.append(r'../defense')
sys.path.append(r'../defense/hgcn')
from MD import MD
import scipy.sparse as sp

from utils import *
import os.path as osp
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_list = ['cora', 'citeseer', 'polblogs', 'cora_ml']
defense_list = ['GCN', 'GCNSVD', 'Jaccard', 'ProGNN', 'RGCN', 'GAT', 'CFS', 'CfsGAT', 'CfsRGCN']
metric_list = ['Cfs_meta', 'Cs_meta', 'Cfs', 'Cs']
attack_list = ['meta', 'nettack', 'random', 'dice']

dataset_name = 'citeseer'

metric = 'Cfs'
attack = 'meta'

# load clean graph data
data = Dataset(root='data/', name=dataset_name, seed=15, require_mask=True)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

if attack == 'nettack':
    json_file = osp.join('adv_adj/{}_nettacked_nodes.json'.format(dataset_name))
    with open(json_file, 'r') as f:
        idx = json.loads(f.read())
    attacked_test_nodes = idx["attacked_test_nodes"]
    idx_test = attacked_test_nodes
    print(idx_test)

acc = []
acc_var = []
for i in range(6):
    sum = 0
    var = []
    if i == 0:
        perturbed_adj = adj
    else:
        if attack == 'nettack':
            perturbed_adj = sp.load_npz('adv_adj/{}_nettack_adj_{}.0.npz'.format(dataset_name, i))
            print('---------{}accack----------'.format(i))
        elif attack == 'random':
            perturbed_adj = sp.load_npz('adv_adj/{}_random_adj_{}.npz'.format(dataset_name, round((i) * 0.2, 3)))
            print('---------{}accack----------'.format(round((i) * 0.2, 3)))
        else:
            perturbed_adj = sp.load_npz('adv_adj/{}_meta_adj_{}.npz'.format(dataset_name, round((i) * 0.05, 3)))
            print('---------{}accack----------'.format(round((i) * 0.05, 3)))

    modified_adj_np = MD(features, perturbed_adj, metric, 0)
    if attack == 'meta':
        sp.save_npz('Cfs-adj-0.05/{}_{}_adj_{}'.format(dataset_name, attack, round((i) * 0.05, 3)), sp.csr_matrix(modified_adj_np))
    if attack == 'nettack':
        sp.save_npz('Cfs-adj-0.05/{}_{}_adj_{}'.format(dataset_name, attack, i), sp.csr_matrix(modified_adj_np))
    if attack == 'random':
        sp.save_npz('Cfs-adj-0.05/{}_{}_adj_{}'.format(dataset_name, attack, round((i) * 0.2, 3)), sp.csr_matrix(modified_adj_np))

