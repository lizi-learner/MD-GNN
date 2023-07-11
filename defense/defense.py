# _*_codeing=utf-8_*_
# @Time:2022/4/10  12:05
# @Author:mazhixiu
# @File:adj_changes.py
import sys
sys.path.append('/home/mzx/Pure_GNN/')

from deeprobust.graph.defense import GCN, GAT, RGCN, GCNSVD, GCNJaccard, SimPGCN, MedianGCN,ProGNN
from deeprobust.graph.utils import preprocess
from compare_defense.elastic_gnn import ElasticGNN
import torch_geometric.transforms as T
import time
from utils.utils_2 import *
from compare_defense.GNNGuard import GNNGuard



def GCN_(data, device, arg):
    # Setup GCN Model
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    pyg_data = Dpr2Pyg(data)
    model = GCN(nfeat=features.shape[1], nhid=128, nclass=int(labels.max()) + 1,
                device=device,lr=0.01,dropout=0.0,weight_decay=5e-4)
    model = model.to(device)

    # print(mod)
    # # using validation to pick conv
    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=200, verbose=False,with_bias=False)
    end = time.perf_counter()
    model.eval()
    output = model.test(idx_test)
    return output,end-start


def GAT_(data, device, arg):
    # start = time.perf_counter()
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    pyg_data = Dpr2Pyg(data).data
    # print(pyg_data[0])
    gat = GAT(nfeat=features.shape[1],
              nhid=16, heads=8,
              nclass=labels.max().item() + 1,
              dropout=0., device=device,lr=0.01,weight_decay=5e-4)
    gat = gat.to(device)
    start = time.perf_counter()
    gat.fit(pyg_data, train_iters=200,verbose=False)  # train with earlystopping
    end = time.perf_counter()
    output = gat.test()
    return output,end-start


def GCN_SVD(data, device, arg):
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1,
                   nhid=128, device=device,lr=0.01,with_bias=False,weight_decay=5e-4,dropout=0)

    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val,train_iters=200,k=arg, verbose=False)
    end = time.perf_counter()

    model.eval()
    output = model.test(idx_test)
    return output,end-start

def GCN_Jaccard(data, device, arg):
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max() + 1,nhid=128,
                       dropout=0.,device=device,lr=0.01,with_bias=False,weight_decay=5e-4)

    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val,train_iters=200,k=arg, verbose=False)
    end = time.perf_counter()

    model.eval()
    output = model.test(idx_test)
    return output,end-start

def R_GCN(data, device, arg):
    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # print(type(adj))
    # print(type(features))
    features = sp.csr_matrix(features)
    model = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                 nhid=128, device=device,lr=0.01,dropout=0.0)
    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=200, verbose=False)
    end = time.perf_counter()
    output = model.test(idx_test)
    return output,end-start


def GNN_Guard(data, device, arg):
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    if torch.is_tensor(adj):
        # print('test')
        adj = sp.csr_matrix(adj.cpu().numpy())
        # print(type(adj))
    ''' testing conv '''

    # print(args.modelname)
    # print(device)
    model = GNNGuard(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1,
                     dropout=0.0, device=device,lr=0.01,weight_decay=5e-4)
    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, train_iters=200,
              idx_val=idx_val,
              idx_test=idx_test,
              verbose=False, attention=True)  # idx_val=idx_val, idx_test=idx_test , model_name=model_name
    end = time.perf_counter()
    model.eval()
    output = model.test(idx_test)
    return output,end-start


def test_():
    import warnings
    warnings.filterwarnings('ignore')
    data = Dataset_(dataset='cora')
    GNN_Guard(data,device='cuda:0',arg=None)


def Pro_GCN(data, device, arg):
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = GCN(nfeat=features.shape[1],
                nhid=128,
                nclass=labels.max().item() + 1,
                dropout=True, device=device)

    start = time.perf_counter()
    perturbed_adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)
    prognn = ProGNN(model, arg, device)
    prognn.fit(features, perturbed_adj, labels, idx_train, idx_val)
    end = time.perf_counter()
    output = prognn.test(features, labels, idx_test)
    return output,end-start


def Median_GCN(data, device, arg):
    # print(data)
    features, labels, adj = data.features, data.labels, data.adj
    pyg_data = Dpr2Pyg(data).data
    model = MedianGCN(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1,
                      dropout=0.0, device=device,lr=0.01)
    model = model.to(device)

    start = time.perf_counter()
    model.fit(pyg_data=pyg_data,train_iters=200, verbose=False)
    end = time.perf_counter()

    model.eval()
    output = model.test(pyg_data)
    return output,end-start


def Simp_GCN(data, device, arg):

    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    print(arg)
    model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1], nhid=128,
                    nclass=int(labels.max()) + 1, device=device,lr=0.01,dropout=0.,
                    lambda_ = arg.get('lambda_'),gamma=arg.get('gama_')
                    )
    model = model.to(device)

    start = time.perf_counter()
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=200, verbose=False)
    end = time.perf_counter()

    output = model.test(idx_test)
    return output,end-start


def Elastic_GCN(data, device, arg):
    adj, features, labels = data.adj, data.features, data.labels

    model = ElasticGNN(nfeat=features.shape[1], nhid=128, nclass=labels.max().item() + 1,
                       device=device,lr=0.01,dropout=0.0,lambda_1=arg.get('lambda_1'),lambda_2=arg.get('lambda_2'),)
    model = model.to(device)
    row, col = np.diag_indices_from(adj)
    data.adj[row,col] = 0
    pyg_data = Dpr2Pyg(data).data

    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    data1 = transform(pyg_data)

    start = time.perf_counter()
    model.fit(data1,train_iters=200)  # train with earlystopping
    end = time.perf_counter()

    model.eval()
    output = model.test(data1)
    return output,end-start


from STABLE import get_contrastive_emb,preprocess_adj,get_logger,to_tensor,to_scipy,sparse_mx_to_sparse_tensor,get_reliable_neighbors,adj_new_norm

logger = get_logger('./try.log')


def STABLE(data, device, args):
    start = args.start

    adj, features, labels = data.adj, data.features, data.labels
    # print(data.adj.shape)
    print('process:',len(data.adj.nonzero()[0]))
    logger.info('===train ours on perturbed graph===')
    adj_temp = sparse_mx_to_sparse_tensor(adj)
    # add k new neighbors to each node

    get_reliable_neighbors(adj_temp, features, k=args.k, degree_threshold=args.threshold,device=device)
    model = GCN(nfeat=features.shape[1], nhid=128, nclass=int(labels.max()) + 1,
                device=device, lr=0.01, dropout=0.0, weight_decay=5e-4)
    model = model.to(device)

    adj_temp = adj_new_norm(adj_temp, args.alpha,device)
    adj_temp = to_scipy(adj_temp)

    # features = to_tensor()
    model.fit(features, adj_temp, labels, data.idx_train, data.idx_val,
              train_iters=200, verbose=False,with_bias=False,device=device)  # train with earlystopping
    end = time.time()

    model.eval()
    output = model.test(idx_test=data.idx_test)
    return output,end-start

from Mid_GCN import middle_normalize_adj,DeepGCN

def Mid_GCN(data, device, arg):

    features, labels, adj = data.features, data.labels, data.adj
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    model = DeepGCN(nfeat=features.shape[1], nhid=128, nclass=int(labels.max()) + 1,dropout=0.0,
                device=device,lr=0.01,weight_decay=5e-4)
    model = model.to(device)

    start = time.perf_counter()
    adj = middle_normalize_adj(data.adj,arg.get('alpha'))
    model.fit(features, adj, labels, idx_train, idx_val, train_iters=200, verbose=False)
    end = time.perf_counter()
    model.eval()
    output = model.test(idx_test)
    return output,end-start


def judge_dir(path):
    if os.path.exists(path):
        print(path)
    else:
        os.makedirs(path)

def test_clean():

    param_dict = load_param('defense.yaml')
    device = param_dict['device']
    defense_dict= param_dict.get('defense_dict')
    dataset_list = param_dict.get('dataset_list')
    save_dir = param_dict.get('save_clean_dir')
    print(save_dir)

    for defense, function in defense_dict.items():
        var_list = []
        acc_list = []
        time_list = []
        path_1 = save_dir + '/acc/'
        path_2 = save_dir + '/var/'
        path_3 = save_dir + '/time/'

        judge_dir(path_1)
        judge_dir(path_2)
        judge_dir(path_3)


        for dataset in dataset_list:
            epoch = 5
            sum = 0
            all_time = 0
            var = []
            for i in range(epoch):
                data = Dataset_(dataset=dataset)

                print("defense:", defense)
                args = None
                if defense == 'Jaccard':
                    args = random.randint(1, 5) * 0.02

                if defense == 'GCNSVD':
                    k_arr = [5, 10, 15, 50, 100, 200]
                    args = k_arr[random.randint(0, 5)]

                if defense == 'SimpGCN':
                    args = {}
                    lambda_arr = [0.1, 0.5, 1, 5, 10, 50, 100]
                    args['lambda_'] = lambda_arr[random.randint(0, 6)]
                    args['gama_'] = random.uniform(0.01, 0.1)

                if defense =='Mid_GCN':
                    args = {}
                    alpha_arr = [0.2, 0.3, 0.5, 0.55, 0.6, 2.0]
                    args['alpha'] = alpha_arr[random.randint(0, 5)]
                    print(args)


                if defense == 'ElasticGCN':
                    args = {}
                    lambda_ = [0,3,6,9]

                    args['lambda_1'] = lambda_[random.randint(0, 3)]
                    args['lambda_2'] = lambda_[random.randint(0, 3)]
                    print(args)

                if defense =='STABLE':
                    from STABLE import args

                    perturbed_adj_sparse = data.adj
                    adj_pre = preprocess_adj(data.features, perturbed_adj_sparse, logger, threshold=args.jt)
                    adj_delete = perturbed_adj_sparse - adj_pre
                    _, features = to_tensor(perturbed_adj_sparse, data.features)
                    print('===start getting contrastive embeddings===')
                    embeds, _ = get_contrastive_emb(logger, adj_pre, features.unsqueeze(dim=0).to_dense(),
                                                    adj_delete=adj_delete,
                                                    lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta,
                                                    device=device)

                    embeds = embeds.squeeze(dim=0)

                    embeds = embeds.to('cpu')
                    embeds = to_scipy(embeds)

                    # prune the perturbed graph by the representations
                    adj_clean = preprocess_adj(embeds, perturbed_adj_sparse, logger, jaccard=False,
                                               threshold=args.cos)
                    data.adj = adj_clean

                output, times = eval(function)(data, device, args)

                all_time = all_time + times
                sum = sum + output
                var.append(output)

            acc_var = np.std(var)
            mean_acc = sum / epoch
            mean_time = all_time / epoch
            acc_list.append(mean_acc)
            var_list.append(acc_var)
            time_list.append(mean_time)
        print(acc_list)
        np.savetxt("{}/{}_{}.txt".format(save_dir + '/acc', 'clean', defense), np.array(acc_list))
        np.savetxt("{}/{}_{}.txt".format(save_dir + '/var', 'clean', defense), np.array(var_list))
        np.savetxt("{}/{}_{}.txt".format(save_dir + '/time', 'clean', defense), np.array(time_list))


from utils.set_args import *

def test_global():

    import random
    param_dict = load_param('defense.yaml')
    device = param_dict['device']
    global_attack = param_dict.get('global_attack')
    global_ptb_list = param_dict.get('global_ptb_list')
    defense_dict= param_dict.get('defense_dict')
    dataset_list = param_dict.get('dataset_list')
    save_dir = param_dict.get('save_attack_dir')


    for defense, function in defense_dict.items():
        path_1 = save_dir + '/acc'
        path_2 = save_dir + '/var'
        path_3 = save_dir + '/time'
        judge_dir(path_1)
        judge_dir(path_2)
        judge_dir(path_3)

        for dataset in dataset_list:
            for attack in global_attack:
                var_list = []
                acc_list = []
                time_list = []
                for ptb in global_ptb_list:
                    kwargs = {}
                    kwargs['model'] = defense
                    kwargs['dataset'] = dataset
                    kwargs['attack'] = attack
                    kwargs['ptb'] = ptb
                    print(kwargs)

                    epoch = 5
                    sum = 0
                    all_time = 0
                    var = []

                    for i in range(epoch):
                        if attack == 'Metattack' and dataset=='photo':
                            data = Dataset_(dataset=dataset, attack=attack, ptb=ptb, largest_component=False)
                        else:
                            print('Metattack')
                            data = Dataset_(dataset=dataset, attack=attack, ptb=ptb, )
                            print(data)
                        arg = None
                        if defense == 'Jaccard':
                            arg = random.randint(1, 5) * 0.02

                        if defense == 'GCNSVD':
                            k_arr = [5, 10, 15, 50, 100, 200]
                            arg = k_arr[random.randint(0, 5)]
                            # arg = 5

                        if defense == 'SimpGCN':
                            arg = {}
                            lambda_arr = [0.1,0.5,1,5,10,50,100]
                            arg['lambda_'] = lambda_arr[random.randint(0, 6)]
                            # arg['lambda_'] = 100
                            # gama_arr = [0.01,0.1]
                            # arg['gama_']= gama_arr[random.randint(0,9)]
                            arg['gama_']=random.uniform(0.01, 0.1)
                            # arg['gama_'] = 0.01

                        if defense == 'ElasticGCN':
                            arg = {}
                            lambda_ = [0, 3, 6, 9]

                            arg['lambda_1'] = lambda_[random.randint(0, 3)]
                            arg['lambda_2'] = lambda_[random.randint(0, 3)]
                            print(arg)

                        if defense == 'Mid_GCN':
                            arg = {}
                            alpha_arr = [0.2, 0.3, 0.5, 0.55,0.6, 2.0]
                            arg['alpha'] = alpha_arr[random.randint(0, 5)]
                            # arg['alpha'] = 0.2
                            # print(arg)

                        if defense == 'ProGCN':
                            from Pro_GCN import args
                            arg = args

                        if defense == 'STABLE':
                            from STABLE import args
                            jt_list = {
                                0.05: 0.1,
                                0.10: 0.2,
                                0.15: 0.3,
                                0.20: 0.4,
                                0.25: 0.5
                            }

                            k_list = {
                                0.05: 1,
                                0.10: 2,
                                0.15: 3,
                                0.20: 4,
                                0.25: 5
                            }

                            alpha_list = {
                                0.05: 0.1,
                                0.10: 0.2,
                                0.15: 0.3,
                                0.20: 0.4,
                                0.25: 0.5
                            }
                            args.jt = jt_list[ptb]
                            args.k = k_list[ptb]
                            args.alpha = alpha_list[ptb]

                            args.start = time.time()

                            perturbed_adj_sparse = data.adj
                            adj_pre = preprocess_adj(data.features, perturbed_adj_sparse, logger, threshold=args.jt)
                            adj_delete = perturbed_adj_sparse - adj_pre
                            _, features = to_tensor(perturbed_adj_sparse, data.features)
                            print('===start getting contrastive embeddings===')
                            embeds, _ = get_contrastive_emb(logger, adj_pre, features.unsqueeze(dim=0).to_dense(),
                                                            adj_delete=adj_delete,
                                                            lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta,
                                                            device=device)

                            embeds = embeds.squeeze(dim=0)

                            embeds = embeds.to('cpu')
                            embeds = to_scipy(embeds)

                            # prune the perturbed graph by the representations
                            adj_clean = preprocess_adj(embeds, perturbed_adj_sparse, logger, jaccard=False,
                                                       threshold=args.cos)
                            data.adj = adj_clean

                            arg = args

                        print(function)
                        output, times = eval(function)(data, device, arg)
                        all_time = all_time + times
                        sum = sum + output
                        var.append(output)

                    acc_var = np.std(var)
                    mean_acc = sum / epoch
                    mean_time = all_time / epoch
                    acc_list.append(mean_acc)
                    var_list.append(acc_var)
                    time_list.append(mean_time)

                print("acc_list:", acc_list)
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/acc', defense, attack, dataset),
                           np.array(acc_list))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/var', defense, attack, dataset),
                           np.array(var_list))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/time', defense, attack, dataset),
                           np.array(time_list))


def test_target():

    import random
    from utils.utils_2 import judge_dir, load_param

    param_dict = load_param('defense.yaml')
    device = param_dict['device']
    targeted_attack = param_dict.get('targeted_attack')
    target_ptb_list = param_dict.get('target_ptb_list')
    defense_dict= param_dict.get('defense_dict')
    dataset_list = param_dict.get('dataset_list')
    save_dir = param_dict.get('save_attack_dir')
    print(save_dir)
    for defense, function in defense_dict.items():

        path_1 = save_dir + '/acc'
        path_2 = save_dir + '/var'
        path_3 = save_dir + '/time'
        judge_dir(path_1)
        judge_dir(path_2)
        judge_dir(path_3)

        for dataset in dataset_list:

            for attack in targeted_attack:
                var_list = []
                acc_list = []
                time_list = []
                for ptb in target_ptb_list:
                    kwargs = {}
                    kwargs['model'] = defense
                    kwargs['dataset'] = dataset
                    kwargs['attack'] = attack
                    kwargs['ptb'] = ptb
                    print(kwargs)

                    epoch = 5
                    sum = 0
                    all_time = 0
                    var = []
                    for i in range(epoch):
                        data = Dataset_(dataset=dataset, attack=attack, ptb=ptb)

                        arg = None
                        if defense == 'Jaccard':
                            arg = random.randint(1, 5) * 0.02
                        if defense == 'GCNSVD':
                            k_arr = [5, 10, 15, 50, 100, 200]
                            arg = k_arr[random.randint(0, 5)]
                        if defense == 'SimpGCN':
                            arg = {}
                            lambda_arr = [0.1,0.5,1,5,10,50,100,200]
                            arg['lambda_'] = lambda_arr[random.randint(0, 6)]
                            arg['gama_']=random.uniform(0.01,0.1)

                        if defense == 'Mid_GCN':
                            arg = {}
                            alpha_arr = [0.2, 0.3, 0.5, 0.55, 0.6, 2.0]
                            arg['alpha'] = alpha_arr[random.randint(0, 5)]
                            print(arg)

                        if defense == 'ElasticGCN':
                            arg = {}
                            lambda_ = [0, 3, 6, 9]

                            arg['lambda_1'] = lambda_[random.randint(0, 3)]
                            arg['lambda_2'] = lambda_[random.randint(0, 3)]
                            print(arg)

                        if defense == 'STABLE':
                            from STABLE import args
                            jt_list = {
                                1.0: 0.1,
                                2.0: 0.2,
                                3.0: 0.3,
                                4.0: 0.4,
                                5.0:0.5
                            }

                            k_list = {
                                1.0: 1,
                                2.0: 2,
                                3.0: 3,
                                4.0: 4,
                                5.0: 5
                            }

                            alpha_list = {
                                1.0: 0.1,
                                2.0: 0.2,
                                3.0: 0.3,
                                4.0: 0.4,
                                5.0:0.5
                            }
                            args.jt = jt_list[ptb]
                            args.k = k_list[ptb]
                            args.alpha = alpha_list[ptb]

                            args.start = time.time()


                            perturbed_adj_sparse = data.adj
                            adj_pre = preprocess_adj(data.features, perturbed_adj_sparse, logger, threshold=args.jt)
                            adj_delete = perturbed_adj_sparse - adj_pre
                            # print(adj_delete)
                            _, features = to_tensor(perturbed_adj_sparse, data.features)

                            print('===start getting contrastive embeddings===')
                            embeds, _ = get_contrastive_emb(logger, adj_pre, features.unsqueeze(dim=0).to_dense(),
                                                            adj_delete=adj_delete,
                                                            lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta,
                                                            device=device)

                            embeds = embeds.squeeze(dim=0)

                            embeds = embeds.to('cpu')
                            embeds = to_scipy(embeds)

                            # prune the perturbed graph by the representations
                            # print()
                            adj_clean = preprocess_adj(embeds, perturbed_adj_sparse, logger, jaccard=False,
                                                       threshold=args.cos)
                            # print(len(adj_clean.nonzero()))
                            data.adj = adj_clean
                            arg = args
                            # STABLE(data,device,arg)

                        print(function)
                        output, times = eval(function)(data, device, arg)
                        all_time = all_time + times
                        sum = sum + output
                        var.append(output)

                    acc_var = np.std(var)
                    mean_acc = sum / epoch
                    mean_time = all_time / epoch
                    acc_list.append(mean_acc)
                    var_list.append(acc_var)
                    time_list.append(mean_time)

                print("acc_list:", acc_list)
                print("{}/{}_{}_{}.txt".format(save_dir + '/acc', defense, attack, dataset))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/acc', defense, attack, dataset),
                           np.array(acc_list))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/var', defense, attack, dataset),
                           np.array(var_list))
                np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/time', defense, attack, dataset),
                           np.array(time_list))

if __name__=="__main__":
    import random
    import warnings
    warnings.filterwarnings('ignore')
    # test_()
    # test_1()
    test_clean()
    # test_global()
    # test_target()


"""
data->denfese
"""

# data = load_Dataset('Cora')
# GNN_Guard(data=data,device=device,k='1')
