import random
import scipy.sparse as sp
import os.path
import os.path as osp
from src.defense import *

def isRoot(root, root1, save_dir):
    if not os.path.isdir(root):
        print('data not Found!')
        return False
    if not os.path.isdir(root1):
        print('adversarial samples not Found!')
        return False
    if not os.path.isdir(save_dir):
        print('save_dir not Found!')
        return False
    return True


def experiment0(dataset_list, attack_list, root, root1, save_dir):
    """
    方案的实验0部分，分析扰动数据和正常数据的特征区别，包括了四个指标
    :param dataset_list: (list) 存放需要实验的数据集名称，只支持cora, citeseer, polblogs, cora_ml
    :param attack_list: (list) 存放需要实验的对抗样本名称，只支持meta, nettack, random, dice
    :param root: 获取对抗样本的路径
    :param save_dir: 存放实验图片的路径
    :return:
    """

    # 判断路径是否存在
    if (not isRoot(root, root1, save_dir)):
        return

    pro_list = ['cn', 'jaccard', 'cos', 'dis']

    for dataset_name in dataset_list:
        for attack in attack_list:
            data = Dataset(root=root, name=dataset_name, seed=15)
            adj, features, labels = data.adj, data.features, data.labels

            features = features.A
            for pro in pro_list:

                for j in range(1, 6):
                    # 这里没有加上pgd
                    if attack == 'meta':
                        perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.npz'.format(root1, dataset_name, attack, round((j) * 0.05, 3)))
                    elif attack == 'nettack':
                        perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.0.npz'.format(root1, dataset_name, attack, j))
                    elif attack == 'random':
                        perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.npz'.format(root1, dataset_name, attack, round((j) * 0.2, 3)))
                    else:
                        # dice attack
                        perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.0.npz'.format(root1, dataset_name, attack, round((j) * 0.05, 3)))


                    adj_changes = np.triu(perturbed_adj.A - adj.A, k=1)
                    nodes = np.nonzero(adj.A)

                    node_changes = np.nonzero(adj_changes)

                    para = []
                    para1 = []
                    if pro == 'cn':
                        xlabel = 'Common Neighbors'
                        S = np.dot(adj.A, adj.A)
                        S1 = np.dot(perturbed_adj.A, perturbed_adj.A)

                        for i in range(nodes[0].size):
                            para.append(S[nodes[0][i]][nodes[1][i]])

                        for i in range(node_changes[0].size):
                            para1.append(S1[node_changes[0][i]][node_changes[1][i]])

                    if pro == 'jaccard':
                        xlabel = 'Jaccard Similarity'
                        for i in range(nodes[0].size):
                            a, b = features[nodes[0][i]], features[nodes[1][i]]
                            intersection = np.count_nonzero(a * b)
                            para.append(intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection))

                        for i in range(node_changes[0].size):
                            a, b = features[node_changes[0][i]], features[node_changes[1][i]]
                            intersection = np.count_nonzero(a * b)
                            para1.append(intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection))

                    if pro == 'cos':
                        xlabel = 'Cosine Similarity'
                        for i in range(nodes[0].size):
                            a, b = features[nodes[0][i]], features[nodes[1][i]]
                            intersection = np.count_nonzero(a * b)
                            para.append(intersection * 1.0 / np.sqrt((np.count_nonzero(a) * np.count_nonzero(b))))

                        for i in range(node_changes[0].size):
                            a, b = features[node_changes[0][i]], features[node_changes[1][i]]
                            intersection = np.count_nonzero(a * b)
                            para1.append(intersection * 1.0 / np.sqrt((np.count_nonzero(a) * np.count_nonzero(b))))

                    if pro == 'dis':
                        xlabel = 'Euclidean Distance'
                        for i in range(nodes[0].size):
                            a, b = features[nodes[0][i]], features[nodes[1][i]]
                            para.append(np.linalg.norm(a - b, 2))

                        for i in range(node_changes[0].size):
                            a, b = features[node_changes[0][i]], features[node_changes[1][i]]
                            para1.append(np.linalg.norm(a - b, 2))

                    plt.hist(para, bins=20, label="Normal Edges", rwidth=2, weights=np.ones(len(para)) / len(para),
                             alpha=0.5)
                    plt.hist(para1, bins=20, label="Adversarial Edges", rwidth=2,
                             weights=np.ones(len(para1)) / len(para1), alpha=0.5)
                    plt.ylabel('Percentage(%)', fontsize=20)
                    plt.xlabel(xlabel, fontsize=20)
                    plt.legend(loc=2, fontsize=18)

                    ax = plt.gca();  # 获得坐标轴的句柄
                    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
                    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
                    plt.xticks(fontsize=18)
                    plt.yticks(fontsize=20)
                    plt.tight_layout()
                    sns.despine()
                    plt.savefig('{}/{}_{}_{}_{}.png'.format(save_dir, dataset_name, attack, j, pro))
                    plt.close()

def experiment1(dataset_list, attack_list, defense_list, root, root1, save_dir, metric = 'Cfs', epoch=10):
    """
    方案的实验1部分，将获取各个方案的实验结果并存放
    :param dataset_list: (list) 存放需要实验的数据集名称，只支持cora, citeseer, polblogs, cora_ml
    :param attack_list: (list) 存放需要实验的对抗样本名称，只支持meta, nettack, random, dice
    :param root: 获取对抗样本的路径
    :param save_dir: 存放实验图片的路径
    :return:
    """
    # 判断路径是否存在
    if (not isRoot(root, root1, save_dir)):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_list:

        if dataset_name == "polblogs":
            metric = 'Cs'

        data = Dataset(root=root, name=dataset_name, seed=15, require_mask=True)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        for defense in defense_list:
            print('-----------------------defense {}------------------------'.format(defense))
            for attack in attack_list:
                print('-----------------------attack {}------------------------'.format(attack))
                acc_var = []
                acc = []

                data.idx_test = idx_test

                if attack == 'nettack':
                    json_file = osp.join('{}/{}_nettacked_nodes.json'.format(root1, dataset_name))
                    with open(json_file, 'r') as f:
                        idx = json.loads(f.read())
                    attacked_test_nodes = idx["attacked_test_nodes"]
                    data.idx_test = attacked_test_nodes
                    print(data.idx_test)

                    mask = np.zeros(data.labels.shape[0], dtype=np.bool)
                    mask[data.idx_test] = 1
                    data.test_mask = mask

                for i in range(6):
                    if i != 0:
                        if attack == 'nettack':
                            data.adj = sp.load_npz('{}/{}_nettack_adj_{}.0.npz'.format(root1, dataset_name, i))

                        elif attack == 'random':
                            data.adj = sp.load_npz('{}/{}_random_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.2, 3)))

                        elif attack == 'meta':
                            data.adj = sp.load_npz('{}/{}_meta_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.05, 3)))

                        else:
                            data.adj = sp.load_npz('{}/{}_{}_adj_{}.0.npz'.format(root1, dataset_name, attack, round((i) * 0.05, 3)))
                    else:
                        data.adj = adj

                    data1 = data
                    if defense == 'GAT':
                        data1 = dataConver1(data, data.adj)[0]

                    sum = 0
                    var = []


                    for j in range(epoch):
                        arg = metric
                        if defense == 'Jaccard':
                            arg = random.randint(1, 5) * 0.02
                        if defense == 'GCNSVD':
                            k_arr = [5, 10, 15, 50, 100, 200]
                            arg = k_arr[random.randint(0, 5)]
                        if defense == 'GAT':
                            arg = attack
                        output, _ = eval(defense + '1')(data1, device, arg)

                        sum = sum + output
                        var.append(output.cpu().numpy())

                    acc_var1 = np.std(var)
                    mean_acc = sum / epoch

                    acc.append(mean_acc)
                    acc_var.append(acc_var1)

                if defense == 'CFS':
                    np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/acc', dataset_name, attack, metric), np.array(acc))
                    np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/var', dataset_name, attack, metric), np.array(acc_var))
                else:
                    np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/acc', dataset_name, attack, defense), np.array(acc))
                    np.savetxt("{}/{}_{}_{}.txt".format(save_dir + '/var', dataset_name, attack, defense), np.array(acc_var))

def experiment2(dataset_list, attack_list, threshold_list, root, root1, save_dir):

    """
    这是实验2，生成的是jaccard和CFS(cfs)净化图数据的效果数据
    :param dataset_list:
    :param attack_list:
    :param root:
    :param root1:
    :param save_dir:
    :return:
    """

    # 判断路径是否存在
    if (not isRoot(root, root1, save_dir)):
        return

    for dataset_name in dataset_list:

        metric = 'Cfs'
        if dataset_name == "Polblogs":
            metric = 'Cs'

        for attack in attack_list:

            data = Dataset(root=root, name=dataset_name, seed=15)
            adj, features, labels = data.adj, data.features, data.labels
            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            adj_ori = adj.A


            for threshold in threshold_list:

                part1 = []
                part2 = []
                part3 = []


                for i in range(6):
                    perturbed_adj = adj
                    if i != 0:
                        if attack == 'nettack':
                            perturbed_adj = sp.load_npz('{}/{}_nettack_adj_{}.0.npz'.format(root1, dataset_name, i))
                            print('---------{}accack----------'.format(i))
                        elif attack == 'random':
                            perturbed_adj = sp.load_npz('{}/{}_random_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.2, 3)))
                            print('---------{}accack----------'.format(round((i) * 0.2, 3)))
                        elif attack == 'meta':
                            perturbed_adj = sp.load_npz('{}/{}_meta_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.05, 3)))
                            print('---------{}accack----------'.format(round((i) * 0.05, 3)))
                        else:
                            perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.0.npz'.format(root1, dataset_name, attack, round((i) * 0.05, 3)))

                    model = GCNJaccard(nfeat=features.shape[1],
                                       nhid=16,
                                       nclass=labels.max().item() + 1,
                                       dropout=0.5, device=device)
                    model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=threshold)

                    cleaned_adj = model.modified_adj.cpu().to_dense().numpy()

                    l1, l2, l3 = experiment2_save(perturbed_adj, cleaned_adj, adj_ori)

                    part1.append(l1)
                    part2.append(l2)
                    part3.append(l3)

                np.savetxt("{}/{}_{}_{}_{}_0.txt".format(save_dir, dataset_name, attack, 'Jaccard', threshold), np.array(part1))
                np.savetxt("{}/{}_{}_{}_{}_1.txt".format(save_dir, dataset_name, attack, 'Jaccard', threshold), np.array(part2))
                np.savetxt("{}/{}_{}_{}_{}_2.txt".format(save_dir, dataset_name, attack, 'Jaccard', threshold), np.array(part3))

            part1 = []
            part2 = []
            part3 = []

            for i in range(6):
                perturbed_adj = adj
                if i != 0:
                    if attack == 'nettack':
                        perturbed_adj = sp.load_npz('{}/{}_nettack_adj_{}.0.npz'.format(root1, dataset_name, i))
                        print('---------{}accack----------'.format(i))
                    elif attack == 'random':
                        perturbed_adj = sp.load_npz(
                            '{}/{}_random_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.2, 3)))
                        print('---------{}accack----------'.format(round((i) * 0.2, 3)))
                    elif attack == 'meta':
                        perturbed_adj = sp.load_npz(
                            '{}/{}_meta_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.05, 3)))
                        print('---------{}accack----------'.format(round((i) * 0.05, 3)))
                    else:
                        perturbed_adj = sp.load_npz(
                            '{}/{}_{}_adj_{}.0.npz'.format(root1, dataset_name, attack, round((i) * 0.05, 3)))

                cleaned_adj = MD(features, perturbed_adj, metric, 0)

                l1, l2, l3 = experiment2_save(perturbed_adj, cleaned_adj, adj_ori)

                part1.append(l1)
                part2.append(l2)
                part3.append(l3)

            np.savetxt("{}/{}_{}_{}_0.txt".format(save_dir, dataset_name, attack, 'CFS'), np.array(part1))
            np.savetxt("{}/{}_{}_{}_1.txt".format(save_dir, dataset_name, attack, 'CFS'), np.array(part2))
            np.savetxt("{}/{}_{}_{}_2.txt".format(save_dir, dataset_name, attack, 'CFS'), np.array(part3))

def experiment3(dataset_list, attack_list, root, root1, save_dir, threshold_list = [0.01, 0.03, 0.05, 0.1, 0.15]):
    """
    实验三：该实验对比了我们的方案和jaccard在不同阈值下的准确度，
    注：其中我们的原方案是不带有阈值的，但是为了分析简便加上了阈值，原方案即等价于将阈值设置为无穷
    :param dataset_list: 数据集
    :param attack_list: 攻击方式
    :param root: 数据集存储路径
    :param root1: 对抗样本存储路径
    :param save_dir: 结果保存路径
    :param threshold_list: 阈值的设置
    """

    # 判断路径是否存在
    if (not isRoot(root, root1, save_dir)):
        return

    metric = 'Cfs'

    for dataset_name in dataset_list:
        for attack in attack_list:
            data = Dataset(root=root, name=dataset_name, seed=15)
            adj, features, labels = data.adj, data.features, data.labels
            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if attack == 'nettack':
                json_file = osp.join('adv_adj/{}_nettacked_nodes.json'.format(dataset_name))
                with open(json_file, 'r') as f:
                    idx = json.loads(f.read())
                attacked_test_nodes = idx["attacked_test_nodes"]
                idx_test = attacked_test_nodes
                print(idx_test)

            for threshold in threshold_list:
                print('-------------------------threshold {}-----------------------------'.format(threshold))
                acc1 = []
                acc2 = []
                for i in range(6):
                    perturbed_adj = adj
                    if i != 0:
                        if attack == 'nettack':
                            perturbed_adj = sp.load_npz('{}/{}_nettack_adj_{}.0.npz'.format(root1, dataset_name, i))
                        elif attack == 'random':
                            perturbed_adj = sp.load_npz('{}/{}_random_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.2, 3)))

                        elif attack == 'meta':
                            perturbed_adj = sp.load_npz('{}/{}_meta_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.05, 3)))
                        else:
                            perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.0.npz'.format(root1, dataset_name, attack, round((i) * 0.05, 3)))
                    sum1 = 0
                    sum2 = 0

                    for i in range(5):
                        model = GCNJaccard(nfeat=features.shape[1],
                                           nhid=16,
                                           nclass=labels.max().item() + 1,
                                           dropout=0.5, device=device)
                        model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=threshold, verbose=False)
                        model.eval()
                        output1 = model.test(idx_test)
                        sum1 += output1

                        threshold1 = threshold
                        if threshold == threshold_list[-1]:
                            threshold1 = 0

                        modified_adj = MD(features, perturbed_adj, metric, 0)
                        model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
                        model = model.to(device)
                        model.fit(features, modified_adj, labels, idx_train, idx_val)
                        model.eval()
                        output2 = model.test(idx_test)
                        sum2 += output2

                    mean_acc1 = sum1 / 5
                    mean_acc2 = sum2 / 5
                    acc1.append(mean_acc1)
                    acc2.append(mean_acc2)

                np.savetxt("{}/{}_{}_{}_{}.txt".format(save_dir, dataset_name, attack, 'Jaccard', threshold),np.array(acc1))
                np.savetxt("{}/{}_{}_{}_{}.txt".format(save_dir, dataset_name, attack, 'CFS', threshold),np.array(acc2))


def experiment4(dataset_list, attack_list, defense_list, root, root1, save_dir, metric='Cfs'):
    """
    记录各个方案的时间
    :param dataset_list: (list) 存放需要实验的数据集名称，只支持cora, citeseer, polblogs, cora_ml
    :param attack_list: (list) 存放需要实验的对抗样本名称，只支持meta, nettack, random, dice
    :param root: 获取对抗样本的路径
    :param save_dir: 存放实验图片的路径
    :return:
    """
    # 判断路径是否存在
    if (not isRoot(root, root1, save_dir)):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_list:

        if dataset_name == "polblogs":
            metric = 'Cs'

        data = Dataset(root=root, name=dataset_name, seed=15, require_mask=True)
        adj, idx_test = data.adj, data.idx_test

        for defense in defense_list:
            print('-----------------------defense {}------------------------'.format(defense))
            for attack in attack_list:


                time_cos_list = []
                data.idx_test = idx_test

                if attack == 'nettack':
                    json_file = osp.join('{}/{}_nettacked_nodes.json'.format(root1, dataset_name))
                    with open(json_file, 'r') as f:
                        idx = json.loads(f.read())
                    attacked_test_nodes = idx["attacked_test_nodes"]
                    data.idx_test = attacked_test_nodes
                    print(data.idx_test)

                    mask = np.zeros(data.labels.shape[0], dtype=np.bool)
                    mask[data.idx_test] = 1
                    data.test_mask = mask


                for i in range(6):
                    if i != 0:
                        if attack == 'nettack':
                            data.adj = sp.load_npz('{}/{}_nettack_adj_{}.0.npz'.format(root1, dataset_name, i))

                        elif attack == 'random':
                            data.adj = sp.load_npz('{}/{}_random_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.2, 3)))

                        elif attack == 'meta':
                            data.adj = sp.load_npz('{}/{}_meta_adj_{}.npz'.format(root1, dataset_name, round((i) * 0.05, 3)))

                        else:
                            data.adj = sp.load_npz('{}/{}_{}_adj_{}.0.npz'.format(root1, dataset_name, attack, round((i) * 0.05, 3)))
                    else:
                        data.adj = adj

                    data1 = data
                    if defense == 'GAT':
                        data1 = dataConver1(data, data.adj)[0]



                    arg = metric
                    if defense == 'Jaccard':
                        arg = random.randint(1, 3) * 0.02
                    if defense == 'GCNSVD':
                        k_arr = [5, 10, 15, 50, 100, 200]
                        arg = k_arr[random.randint(0, 5)]
                    if defense == 'GAT':
                        arg = attack

                   
                    _, time_cos = eval(defense + '1')(data1, device, arg)
                    print('{} {} {} time: {}'.format(dataset_name, attack, i, time_cos))

                    time_cos_list.append(time_cos)



                if defense == 'CFS':
                    np.savetxt("{}/{}_{}_{}.txt".format(save_dir, dataset_name, attack, metric), np.array(time_cos_list))
                else:
                    np.savetxt("{}/{}_{}_{}.txt".format(save_dir, dataset_name, attack, defense), np.array(time_cos_list))



def experiment5(dataset_list, attack_list, lr_list, root, root1, save_dir, metric = 'Cfs', epoch=10):
    """
    测试学习率
    :param dataset_list: (list) 存放需要实验的数据集名称，只支持cora, citeseer, polblogs, cora_ml
    :param attack_list: (list) 存放需要实验的对抗样本名称，只支持meta, nettack, random, dice
    :param root: 获取对抗样本的路径
    :param save_dir: 存放实验图片的路径
    :return:
    """
    # 判断路径是否存在
    if (not isRoot(root, root1, save_dir)):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_list:

        data = Dataset(root=root, name=dataset_name, seed=15, require_mask=True)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        for attack in attack_list:
            print('-----------------------attack {}------------------------'.format(attack))

            data.idx_test = idx_test

            if attack == 'nettack':
                json_file = osp.join('{}/{}_nettacked_nodes.json'.format(root1, dataset_name))
                with open(json_file, 'r') as f:
                    idx = json.loads(f.read())
                attacked_test_nodes = idx["attacked_test_nodes"]
                data.idx_test = attacked_test_nodes
                print(data.idx_test)

            if attack == 'nettack':
                data.adj = sp.load_npz('{}/{}_nettack_adj_5.0.npz'.format(root1, dataset_name))

            if attack == 'meta':
                data.adj = sp.load_npz('{}/{}_meta_adj_0.25.npz'.format(root1, dataset_name))

            acc = []
            for lr in lr_list:
                sum = 0
                for i in range(epoch):
                    features = data.features
                    labels = data.labels

                    modified_adj = MD(data.features, data.adj, metric, 0)
                    model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
                    model = model.to(device)
                    model.fit(features, modified_adj, labels, idx_train, idx_val)
                    model.eval()

                    output = model.test(data.idx_test)
                    sum += output

                acc.append(sum / epoch)

            np.savetxt("{}/{}_{}_{}.txt".format(save_dir, dataset_name, attack, metric), np.array(acc))














