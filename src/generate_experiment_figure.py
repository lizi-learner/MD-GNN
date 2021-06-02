from src.generate_experiment_data import *

experiment_index = [1, 2, 3, 4, 5]
index = 6
color_n3 = ['#7fa6b6', '#385989', '#d32026']
color_n4 = ['#2b5481', '#44bd9d', '#f04f75', '#fdcd6e']
color_n5 = ['#025699', '#f9c20c', '#f3764c', '#60c7cb', '#4d5a6d']

color_n8 = ['#adadad', '#6794a7', '#014d64', '#01a2d9', '#7ad2f6', '#00887d', '#76c0c1', '#d32026']
colormap = list(plt.get_cmap('tab10')(np.linspace(0, 1, 10)))

def figure0(root='adv_adj/', save_dir='experiment/figure/0'):

    if not os.path.exists(root):
        print('Date not Found!')
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_name = 'cora'
    attack = 'meta'
    pro_list = ['cn', 'jaccard', 'cos', 'dis']

    data = Dataset(root='data/', name=dataset_name, seed=15)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    features = features.A
    for pro in pro_list:

        for j in range(1, 6):
            if attack == 'meta':
                perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.npz'.format(root, dataset_name, attack, round((j) * 0.05, 3)))
            else:
                perturbed_adj = sp.load_npz('{}/{}_{}_adj_{}.0.npz'.format(root, dataset_name, attack, j))


            adj_changes = np.triu(perturbed_adj.A - adj.A, k=1)
            nodes = np.nonzero(adj.A)
            nodes1 = np.nonzero(perturbed_adj.A)
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

            plt.hist(para, bins=20, label="Normal Edges", rwidth=2, weights=np.ones(len(para)) / len(para), alpha=0.5)
            plt.hist(para1, bins=20, label="Adversarial Edges", rwidth=2, weights=np.ones(len(para1)) / len(para1), alpha=0.5)
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


def figure1(root='experiment/data/1/acc/', save_dir ='experiment/figure/1'):
    """
    绘制MD清洗后的对抗样本在GAT、RGCN、HGCN的准确度
    :param save_dir: 图片保存地址
    :return:
    """
    if not os.path.exists(root):
        print('Date not Found!')
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_list = ['cora', 'citeseer', 'polblogs', 'cora_ml']
    defense_list = ['GCN', 'CfsGCN', 'GAT', 'CfsGAT']
    attack_list = ['meta', 'nettack', 'random']
    metric_list = ['MD-GCN', 'MD-GAT']
    metric_list1 = ['MD-GCNs', 'MD-GATs']

    for dataset in dataset_list:

        defense_list[1] = 'Cfs'
        if dataset == 'polblogs':
            defense_list[1] = 'Cs'

        for attack in attack_list:
            acc1 = np.loadtxt('{}/{}_{}_{}.txt'.format(root, dataset, attack, defense_list[0]), dtype=np.float64)
            acc1 = acc1[1:]
            acc2 = np.loadtxt('{}/{}_{}_{}.txt'.format(root, dataset, attack, defense_list[1]), dtype=np.float64)
            acc2 = acc2[1:]
            acc3 = np.loadtxt('{}/{}_{}_{}.txt'.format(root, dataset, attack, defense_list[2]), dtype=np.float64)
            acc3 = acc3[1:]
            acc4 = np.loadtxt('{}/{}_{}_{}.txt'.format(root, dataset, attack, defense_list[3]), dtype=np.float64)
            acc4 = acc4[1:]

            N = 5

            font_size1 = 20
            font_size2 = 18

            x = np.arange(N)
            if attack == 'meta':
                plt.xticks(x, ('5', '10', '15', '20', '25'))
                plt.xlabel('Perturbation Rate(%)', fontsize=font_size1)
            elif attack == 'nettack':
                plt.xticks(x, ('1', '2', '3', '4', '5'))
                plt.xlabel('Numbers of Perturbation Per Node', fontsize=font_size1)
            else:
                plt.xticks(x, ('20', '40', '60', '80', '100'))
                plt.xlabel('Perturbation Rate(%)', fontsize=font_size1)
            plt.ylabel('Test Accuracy', fontsize=font_size1)

            labels = metric_list
            if dataset == 'polblogs':
                labels = metric_list1

            ln5 = plt.plot(x, acc1, label="GCN", c=color_n3[0], linewidth=1.5, linestyle='--', marker='*')
            ln6 = plt.plot(x, acc2, label=labels[0], c=color_n3[2], linewidth=1.5, marker='*')
            ln7 = plt.plot(x, acc3, label="GAT", c=color_n3[1], linewidth=1.5, linestyle='--', marker='o')
            ln8 = plt.plot(x, acc4, label=labels[1], c=color_n3[2], linewidth=1.5, marker='o')

            sns.despine()
            # plt.ylim(0.925, 0.975)
            plt.legend()
            plt.legend(loc=3, ncol=2)
            # 设置刻度字体大小
            plt.xticks(fontsize=font_size1)
            plt.yticks(fontsize=font_size1)
            # 设置图例字体大小
            plt.legend(loc=3, fontsize=font_size2)

            ax = plt.gca();  # 获得坐标轴的句柄
            ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细

            plt.tight_layout()
            plt.savefig('{}/{}_{}.png'.format(save_dir, dataset, attack))
            plt.close()

def figure2(root='experiment/data/3/', save_dir ='experiment/figure/2'):
    """
    不同阈值绘图
    :param save_dir:
    :return:
    """

    if not os.path.exists(root):
        print('Date not Found!')
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_list = ['cora', 'citeseer']
    attack_list = ['meta', 'nettack']
    defense = 'CFS'
    for dataset_name in dataset_list:
        for attack in attack_list:

            threshold_list = [0.01, 0.03, 0.05, 0.1]

            acc1 = np.loadtxt("{}/{}_{}_{}_{}.txt".format(root, dataset_name, attack, defense, threshold_list[0]), dtype=np.float64)
            acc1 = acc1[1:]
            acc2 = np.loadtxt("{}/{}_{}_{}_{}.txt".format(root, dataset_name, attack, defense, threshold_list[1]), dtype=np.float64)
            acc2 = acc2[1:]
            acc3 = np.loadtxt("{}/{}_{}_{}_{}.txt".format(root, dataset_name, attack, defense, threshold_list[2]), dtype=np.float64)
            acc3 = acc3[1:]
            acc4 = np.loadtxt("{}/{}_{}_{}_{}.txt".format(root, dataset_name, attack, defense, threshold_list[3]), dtype=np.float64)
            acc4 = acc4[1:]

            font_size = 20
            N = 5
            x = np.arange(N)
            if attack == 'meta':
                plt.xticks(x, ('5', '10', '15', '20', '25'))
                plt.xlabel('Perturbation Rate(%)', fontsize=font_size)
            elif attack == 'nettack':
                plt.xticks(x, ('1', '2', '3', '4', '5'))
                plt.xlabel('Numbers of Perturbation Per Node', fontsize=font_size)
            else:
                plt.xticks(x, ('20', '40', '60', '80', '100'))
                plt.xlabel('Perturbation Rate(%)', fontsize=font_size)
            plt.ylabel('Test Accuracy', fontsize=font_size)

            ln1 = plt.plot(x, acc1, label="b = {}".format(threshold_list[0]), c=colormap[0], linewidth=2, linestyle='--', marker='o')
            ln2 = plt.plot(x, acc2, label="b = {}".format(threshold_list[1]), c=colormap[1], linewidth=2, linestyle='--', marker='v')
            ln3 = plt.plot(x, acc3, label="b = {}".format(threshold_list[2]), c=colormap[2], linewidth=2, linestyle='--', marker='^')
            ln4 = plt.plot(x, acc4, label="b = {}".format(threshold_list[3]), c=colormap[3], linewidth=2, linestyle='--', marker='*')

            sns.despine()
            plt.legend(loc=3, fontsize=18)

            ax = plt.gca();  # 获得坐标轴的句柄
            ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细

            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.tight_layout()

            plt.savefig("{}/{}_{}_{}.png".format(save_dir, dataset_name, attack, defense))
            plt.close()

def figure3(root='experiment/data/5/', save_dir='experiment/figure/3'):
    """
    绘制学习率参数的图
    :param save_dir:
    :return:
    """

    if not os.path.exists(root):
        print('Date not Found!')
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    attack_list = ['meta', 'nettack']
    defense = 'Cfs'
    for attack in attack_list:
        acc1 = np.loadtxt("{}/cora_{}_{}.txt".format(root, attack, defense), dtype=np.float64)
        acc2 = np.loadtxt("{}/citeseer_{}_{}.txt".format(root, attack, defense), dtype=np.float64)

        font_size = 20
        N = 5
        x = np.arange(N)

        plt.xticks(x, ('0.001', '0.005', '0.01', '0.05', '0.1'))
        plt.xlabel('Learning Rate', fontsize=font_size)
        plt.ylabel('Test Accuracy', fontsize=font_size)

        ln1 = plt.plot(x, acc1, label="Cora", c=colormap[0], linewidth=2, linestyle='--', marker='o')
        ln2 = plt.plot(x, acc2, label="Citeseer", c=colormap[1], linewidth=2, linestyle='--', marker='v')

        sns.despine()
        plt.legend(loc=4, fontsize=18)

        ax = plt.gca();  # 获得坐标轴的句柄
        ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
        plt.ylim(0.6,)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout()

        plt.savefig("{}/{}_{}.png".format(save_dir, attack, defense))
        plt.close()

def figure4(root='experiment/data/1/acc/', save_dir='experiment/figure/4'):

    if not os.path.exists(root):
        print('Date not Found!')
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_list = ['cora', 'citeseer']
    attack_list = ['meta', 'nettack']
    metric_list = ['Cfs', 'Cfs1', 'Cfs2', 'Cfs3']
    metric_list1 = ['Jaccard', 'Cosine', 'Euclidean', 'Manhattan']

    for dataset in dataset_list:
        for attack in attack_list:
            acc1 = np.loadtxt('{}/{}_{}_{}.txt'.format(root, dataset, attack, metric_list[0]), dtype=np.float64)
            acc1 = acc1[1:]
            acc2 = np.loadtxt('{}/{}_{}_{}.txt'.format(root, dataset, attack, metric_list[1]), dtype=np.float64)
            acc2 = acc2[1:]
            acc3 = np.loadtxt('{}/{}_{}_{}.txt'.format(root, dataset, attack, metric_list[2]), dtype=np.float64)
            acc3 = acc3[1:]
            acc4 = np.loadtxt('{}/{}_{}_{}.txt'.format(root, dataset, attack, metric_list[3]), dtype=np.float64)
            acc4 = acc4[1:]

            font_size = 20
            N = 5
            x = np.arange(N)
            if attack == 'meta':
                plt.xticks(x, ('5', '10', '15', '20', '25'))
                plt.xlabel('Perturbation Rate(%)', fontsize=font_size)
            elif attack == 'nettack':
                plt.xticks(x, ('1', '2', '3', '4', '5'))
                plt.xlabel('Numbers of Perturbation Per Node', fontsize=font_size)
            else:
                plt.xticks(x, ('20', '40', '60', '80', '100'))
                plt.xlabel('Perturbation Rate(%)', fontsize=font_size)
            plt.ylabel('Test Accuracy', fontsize=font_size)

            ln1 = plt.plot(x, acc1, label=metric_list1[0], c=colormap[0], linewidth=2, linestyle='--', marker='o')
            ln2 = plt.plot(x, acc2, label=metric_list1[1], c=colormap[1], linewidth=2, linestyle='--', marker='v')
            ln3 = plt.plot(x, acc3, label=metric_list1[2], c=colormap[2], linewidth=2, linestyle='--', marker='^')
            ln4 = plt.plot(x, acc4, label=metric_list1[3], c=colormap[3], linewidth=2, linestyle='--', marker='*')

            sns.despine()
            plt.legend(loc=3, fontsize=18)

            ax = plt.gca();  # 获得坐标轴的句柄
            ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细

            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.tight_layout()

            plt.savefig("{}/{}_{}.png".format(save_dir, dataset, attack))
            plt.close()
