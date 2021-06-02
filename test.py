from src.generate_experiment_data import *
from src.generate_experiment_figure import *
from src.save_excel import *
# dataset_list = ['cora', 'citeseer', 'polblogs', 'cora_ml']
# defense_list = ['GCN', 'GCNSVD', 'Jaccard', 'ProGNN', 'RGCN', 'GAT', 'CFS', 'CfsGAT', 'CfsRGCN', 'GNNGuard', 'HGCN', 'CfsHGCN']
# metric_list = ['Cfs', 'Cfs1', 'Cs', 'Cs1', 'Jaccard1']
# attack_list = ['meta', 'nettack', 'random', 'dice']

# dataset_list = ['cora', 'citeseer']
# attack_list = ['meta', 'nettack']
# defense_list = ['CFS']
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/1/'
#
# experiment1(dataset_list, attack_list, defense_list, root, root1, save_dir, 'Cfs2', 10)
# experiment1(dataset_list, attack_list, defense_list, root, root1, save_dir, 'Cfs3', 10)

# dataset_list = ['cora']
# attack_list = ['meta']
# threshold_list = [0.03, 0.05]
#
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/2/'
#
# experiment2(dataset_list, attack_list, threshold_list, root, root1, save_dir)

# dataset_list = ['cora']
# attack_list = ['meta']
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/3/'
# a = [0.05, 0.1]
# experiment3(dataset_list, attack_list, root, root1, save_dir, a)

# dataset_list = ['cora', 'citeseer', 'polblogs', 'cora_ml']
# attack_list = ['meta', 'nettack', 'random']
# defense_list = ['HGCN', 'GCN', 'GCNSVD', 'Jaccard', 'RGCN', 'GAT', 'CFS']
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/4/'
# experiment4(dataset_list, attack_list, defense_list, root, root1, save_dir, 'Cfs')

# dataset_list = ['polblogs']
# attack_list = ['random']
# defense_list = ['HGCN', 'CfsHGCN']
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/1/'
#
# experiment1(dataset_list, attack_list, defense_list, root, root1, save_dir, 'Cfs', 10)

# 测试学习率
# dataset_list = ['cora_ml']
# attack_list = ['meta', 'nettack']
# defense_list = ['CFS']
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/5/'
# lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
#
# experiment5(dataset_list, attack_list, lr_list, root, root1, save_dir, 'Cfs', 10)

# dataset_list = ['cora', 'citeseer', 'cora_ml']
# attack_list = ['meta', 'nettack', 'random']
# defense_list = ['CFS']
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/6/'
#
# experiment1(dataset_list, attack_list, defense_list, root, root1, save_dir, 'Cfs4', 10)

# dataset_list = ['cora']
# attack_list = ['meta']
# defense_list = ['CFS']
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/6/'
# experiment4(dataset_list, attack_list, defense_list, root, root1, save_dir, 'Cfs4')

# dataset_list = ['cora', 'citeseer', 'cora_ml']
# attack_list = ['meta', 'nettack', 'random']
# defense_list = ['CfsHGCN']
# root = 'data/'
# root1 = 'adv_adj/'
# save_dir = 'experiment/data/0/'
#
# threshold_list = [0.03, 0.05]
# a = [0.05, 0.1]
# lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]

# experiment1(dataset_list, attack_list,defense_list, root, root1, save_dir)
# experiment2(dataset_list, attack_list, threshold_list, root, root1, save_dir)
# experiment3(dataset_list, attack_list, root, root1, save_dir, a)
# experiment4(dataset_list, attack_list, defense_list, root, root1, save_dir, 'Cfs4')
# experiment5(dataset_list, attack_list, lr_list, root, root1, save_dir, 'Cfs', 10)

# figure0()
# result_time()

#对图进行预处理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = Dataset(root='data/', name='cora', seed=15, require_mask=True)
data.adj = sp.load_npz('adv_adj/cora_meta_adj_0.25.npz')
output, time = CFS1(data, device, 'Cfs')