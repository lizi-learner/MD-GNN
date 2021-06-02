import os.path as osp
from copy import deepcopy

from deeprobust.graph import utils
import torch
import torch.nn.functional as F
# from utils1 import *
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]
# data = Dataset(root='data/', name='cora', seed=15)
# dataset = dataConver('cora', '../data/')[0]
# data = dataConver('cora', '../data/')[0]
# adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

class Net(torch.nn.Module):
    def __init__(self, nfeat, nclass):
        super(Net, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.conv1 = GATConv(self.nfeat, 16, heads=8, dropout=0.5)
        self.conv2 = GATConv(16 * 8, self.nclass, heads=1, concat=False, dropout=0.5)
        self.output = 0

    def forward(self, data):
        x = F.dropout(data.x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

    def fit(self, data, train_iters):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        best_loss_val = 100
        best_acc_val = 0
        for epoch in range(1, train_iters+1):
            self.train()
            optimizer.zero_grad()
            F.nll_loss(self.forward(data)[data.train_mask], data.y[data.train_mask]).backward()
            optimizer.step()

            accs = self.test(data)
            loss_val = F.nll_loss(self.forward(data)[data.val_mask], data.y[data.val_mask])
            acc_val = accs[1]

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = accs[2]

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                # print(best_acc_val)
                self.output = accs[2]

        return self.output

    def test(self, data):
        self.eval()
        logits, accs = self.forward(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, data = Net(data.num_features, data.num_classes).to(device), data.to(device)
# model.fit(data, 200)
#
# log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
# print(log.format(200, *model.test(data)))

