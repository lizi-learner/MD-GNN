from src.generate_attack import *

dataset_list  = ['cora', 'citeseer', 'polblogs', 'cora_ml']

#生成random攻击
generate_random_adj("cora")
generate_meta_adj("cora")
generate_nettack_adj("cora")
generate_dice_adj("cora")
generate_PGDAttack_adj("cora")