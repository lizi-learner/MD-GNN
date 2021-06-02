import argparse

import  torch.nn as nn
import torch
from manifolds.poincare import PoincareBall
import model.layer as layer
from model.hgcn import HGCN
import model.hgcn as hgcn
import torch.nn.functional as F
import parameters
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = PoincareBall()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = HGCN(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = hgcn.LinearDecoder(self.c,args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'

        # self.weights = torch.Tensor([1.] * args.n_classes)
        # if not args.cuda == -1:
        #     self.weights = self.weights.to(args.device)

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output, dim=1)

    def forward(self, data):
        x= data.x
        adj = data.norm_adj
        embedings = self.encode(x, adj)
        return self.decode(embedings,adj)

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]
def call(data,name,num_features,num_classes,curv_path):
    device = parameters.device
    data.to(device)
    config_args = {
        'training_config': {
            'lr': (0.01, 'learning rate'),
            'dropout': (0.5, 'dropout probability'),
            'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
            'epochs': (200, 'maximum number of epochs to train for'),
            'weight-decay': (0.0005, 'l2 regularization strength'),
            'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
            'momentum': (0.999, 'momentum in optimizer'),
            'patience': (100, 'patience for early stopping'),
            'seed': (15, 'seed for training'),
            'log-freq': (5, 'how often to compute print train/val metrics (in epochs)'),
            'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
            'save': (0, '1 to save model and logs and 0 otherwise'),
            'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
            'sweep-c': (0, ''),
            'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
            'gamma': (0.5, 'gamma for lr scheduler'),
            'print-epoch': (True, ''),
            'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
            'min-epochs': (100, 'do not early stop before min-epochs')
        },
        'model_config': {
            'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
            'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN]'),
            'dim': (16, 'embedding dimension'),
            'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
            'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
            'r': (2., 'fermi-dirac decoder parameter for lp'),
            't': (1., 'fermi-dirac decoder parameter for lp'),
            'pretrained-embeddings': (
            None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
            'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
            'num-layers': (2, 'number of hidden layers in encoder'),
            'bias': (1, 'whether to use bias (1) or not (0)'),
            'act': (None, 'which activation function to use (or None for no activation)'),
            'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
            'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
            'double-precision': ('0', 'whether to use double precision'),
            'use-att': (0, 'whether to use hyperbolic attention or not'),
            'local-agg': (0, 'whether to local tangent space aggregation or not')
        },
        'data_config': {
            'dataset': ('cora', 'which dataset to use'),
            'val-prop': (0.05, 'proportion of validation edges for link prediction'),
            'test-prop': (0.1, 'proportion of test edges for link prediction'),
            'use-feats': (1, 'whether to use node features or not'),
            'normalize-feats': (1, 'whether to normalize input node features'),
            'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
            'split-seed': (15, 'seed for data splits (train/test/val)'),
        }
    }
    parser = argparse.ArgumentParser()
    for _, config_dict in config_args.items():
        parser = add_flags_from_config(parser, config_dict)
    args = parser.parse_args()
    args.device = parameters.device
    # args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.n_nodes = data.x.shape[0]
    args.feat_dim=num_features
    args.n_classes = num_classes
    return NCModel(args).to(parameters.device),data

def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser