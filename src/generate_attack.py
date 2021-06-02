import torch
import numpy as np
import random
import json
from scipy import sparse as sp

from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, Random, DICE, PGDAttack
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import preprocess

# from utils import  *
from utils1 import *


def generate_meta_adj(dataset, seed = 15, data_root='data/', save_dir = "my_adv_adj/" ):
    # mettack攻击
    data = Dataset(root=data_root, name=dataset, seed=seed)

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    idx_unlabeled = np.union1d(idx_val, idx_test)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                    with_relu=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)

    for i in range(5):

        model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=device)
        model = model.to(device)
        perturbations = int(0.05 * (i+1) * (adj.sum() // 2))
        # print('my_adv_adj_15/cora_ml_meta_adj_{}'.format(round((i + 2) * 0.05, 3)))
        model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
        modified_adj = model.modified_adj

        gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, with_relu=True, device=device)

        # 找出被攻击过的节点
        adj_ori = adj.A
        modified_adj_np = sp.csr_matrix(modified_adj.cpu().numpy())

        print(np.nonzero(modified_adj.cpu().numpy()))
        # adj_changes = getChangeNodes(modified_adj_np, adj_ori)
        # np.savez('../result/attack_adj/mettack/cora_attack_{}'.format(0.1), modified_adj.cpu(), adj_changes)
        sp.save_npz('{}/{}_meta_adj_{}'.format(save_dir, dataset, round((i + 1) * 0.05, 3)), modified_adj_np)

def generate_nettack_adj(dataset, seed = 15, root='data/', save_dir = "my_adv_adj/"):

    data = Dataset(root=root, name=dataset, seed=seed)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    degrees = adj.A.sum(0)

    adj1 = adj
    po_edges = []

    for i in range(idx_test.size):
        if degrees[i] > 10:
            po_edges.append(i)

    po_edges = random.sample(po_edges, int(len(po_edges) * 0.5))
    with open('{}/{}_nettacked_nodes.json'.format(save_dir, dataset), 'w') as fp:
        json.dump({"attacked_test_nodes": po_edges}, fp)

    for j in range(5):

        for i in range(len(po_edges)):
            # Setup Surrogate model
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                                 nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
            surrogate.fit(features, adj1, labels, idx_train, idx_val, patience=30)
            # Setup Attack Model
            target_node = po_edges[i]
            model = Nettack(surrogate, nnodes=adj1.shape[0], attack_structure=True, attack_features=False, device='cpu').to(
                'cpu')
            # Attack
            model.attack(features, adj1, labels, target_node, n_perturbations=j+1)
            modified_adj = model.modified_adj
            modified_features = model.modified_features
            adj1 = sp.csr_matrix(modified_adj.A)
            # c = c + sp.csr_matrix(modified_adj.A - adj.A)

        sp.save_npz('{}/polblogs_nettack_adj_{}.0'.format(save_dir, j+1), adj1)


def generate_random_adj(dataset, seed = 15, root='data/', save_dir = "my_adv_adj/"):

    for i in range(5):
        data = Dataset(root=root, name=dataset, seed=seed)
        adj, features, labels = data.adj, data.features, data.labels
        model = Random()
        model.attack(adj, n_perturbations=int(0.2 * (i+1) * (adj.sum() // 2)))
        modified_adj = model.modified_adj
        sp.save_npz('{}/{}_random_adj_{}'.format(save_dir, dataset, round((i + 1) * 0.2, 3)), sp.csr_matrix(modified_adj.A))


def generate_dice_adj(dataset, seed = 15, root='data/', save_dir = "my_adv_adj/"):
    for i in range(5):
        data = Dataset(root=root, name=dataset, seed=seed)
        adj, features, labels = data.adj, data.features, data.labels
        model = DICE()
        model.attack(adj, labels, n_perturbations=int(0.05 * (i + 1) * (adj.sum() // 2)))
        modified_adj = model.modified_adj
        sp.save_npz('{}/{}_dice_adj_{}'.format(save_dir, dataset, round((i + 1) * 0.05, 3)), sp.csr_matrix(modified_adj.A))


def generate_PGDAttack_adj(dataset, seed = 15, root='data/', save_dir = "my_adv_adj/"):
    for i in range(5):
        data = Dataset(root=root, name=dataset, seed=seed)
        adj, features, labels = data.adj, data.features, data.labels
        adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        # Setup Victim Model
        victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                           nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
        victim_model.fit(features, adj, labels, idx_train)
        # Setup Attack Model
        model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
        model.attack(features, adj, labels, idx_train, n_perturbations=int(0.05 * (i + 1) * (adj.sum() // 2)))
        modified_adj = model.modified_adj
        sp.save_npz('{}/{}_pgd_adj_{}'.format(save_dir, dataset, round((i + 1) * 0.05, 3)), sp.csr_matrix(modified_adj.A))



