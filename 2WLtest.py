from math import e
from scipy.sparse import data
from sklearn import utils
import random
import numpy as np
from model import LocalWLNet, WLNet, FWLNet, LocalFWLNet
from datasets import load_dataset, dataset
from impl import train
import torch
from torch.optim import Adam
from ogb.linkproppred import Evaluator
import yaml
import time
import os

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def testparam(device="cpu", dsname="Celegans"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)
    bg = load_dataset(dsname, args.pattern, args.test_hits)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    max_degree = torch.max(bg.x[2])

    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))
    if trn_ds.na != None:
        print("use node feature")
        trn_ds.na = trn_ds.na.to(device)
        val_ds.na = val_ds.na.to(device)
        tst_ds.na = tst_ds.na.to(device)
        use_node_attr = True
    else:
        use_node_attr = False


    def valparam(**kwargs):
        lr = kwargs.pop('lr')
        epoch = kwargs.pop('epoch')
        if args.epoch > 0:
            epoch = args.epoch
        if args.pattern == '2wl':
            mod = WLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2wl_l':
            mod = LocalWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2fwl':
            mod = FWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2fwl_l':
            mod = LocalFWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train.train_routine(args.dataset, mod, opt, trn_ds, val_ds, tst_ds, epoch, args.pattern, args.test_hits, args.test_split, res_dir, verbose=args.verbose)

    #with open(f"config/{args.pattern}/{args.dataset}.yaml") as f:
    #    params = yaml.safe_load(f)
    #collab_params = {'act0': True, 'act1': True, 'channels_1wl': 64, 'channels_2wl': 16, 'depth1': 2, 'depth2': 1, 'dp_1wl0': 0.0,
    # 'dp_1wl1': 0.0, 'dp_2wl': 0.0, 'dp_emb': 0.0, 'dp_lin0': 0.0, 'dp_lin1': 0.0, 'epoch': 1000, 'lr': 0.01}
    #params = {'act': True, 'dp1': 0.5, 'dp2': 0.5, 'dp3': 0.5, 'epoch': 1500, 'hidden_dim_1': 64, 'hidden_dim_2': 64,
    # 'layer1': 2, 'layer2': 2, 'lr': 0.005}
    #print(params)
    return valparam(**(params))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pattern', type=str, default="2wl_l")
    parser.add_argument('--dataset', type=str, default="USAir")
    parser.add_argument('--test_hits', action='store_true')
    parser.add_argument('--test_split', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default="opt/")
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--check', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--res_path', type=str, default="")
    args = parser.parse_args()
    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)

    if args.res_path == '':
        args.res_path = '_' + time.strftime("%m%d%H%M%S")
    res_dir = os.path.join('records/{}/{}{}'.format(args.pattern, args.dataset, args.res_path))
    if not os.path.exists('records/{}'.format(args.pattern)):
        os.makedirs('records/{}'.format(args.pattern))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print(args.device)
    with open(f"config/{args.pattern}/{args.dataset}.yaml") as f:
        params = yaml.safe_load(f)
    print(params)
    #Cele_2fwl
    #params = {'hidden_dim_1': 64, 'hidden_dim_2': 64,
    #          'layer1': 2, 'layer2': 2,
    #          # 'act_1wl0': False, 'act_1wl1': False,
    #          'gn_1wl0': True, 'gn_1wl1': True,
    #          'dp_1wl0': 0.3, 'dp_1wl1': 0.3, 'dp_2wl0': 0.3, 'dp_2wl1': 0.3, 'dp_emb': 0.3,
    #          'epoch': 1500, 'lr': 0.005}
    #USAir#2fwl
    #params = {'hidden_dim_1': 32, 'hidden_dim_2': 24,
    #          'layer1': 1, 'layer2': 2,
    #          'act_1wl0': False, 'act_1wl1': False,
    #          'gn_1wl0': False, 'gn_1wl1': False,
    #          'dp_1wl0': 0.1, 'dp_1wl1': 0.2, 'dp_2wl0': 0.1, 'dp_2wl1': 0.0, 'dp_emb': 0.4,
    #          'epoch': 200, 'lr': 0.005}
    # Citeseer#2f_l
    '''
    params = {'hidden_dim_1': 256, 'hidden_dim_2': 16,
              'layer1': 2, 'layer2': 2, 'layer3': 1,
              'act_2wl0': True, 'act_2wl1': True, 'act_lin': False,
              'ln_lin': True, 'ln_2wl0': False, 'ln_2wl1': True,
              'gn_2wl1': False,# 'gn_1wl': False,
              'dp_lin0': 0.4, 'dp_lin1': 0.7, 'dp_2wl0': 0.1, 'dp_2wl1': 0.1,
              'epoch': 200, 'lr': 0.0005,
              'use_degree': False, 'use_appnp': True,
              'reduce_feat': True, 'sum_pooling': True,}
    '''
    # Power#2f_l
    # params = {'hidden_dim_1': 32, 'hidden_dim_2': 32,
    #          'layer1': 2, 'layer2': 3, 'layer3': 3,
    #          'act_1wl': True,
    #          'ln_1wl': False, 'ln_2wl0': False, 'ln_2wl1': True,
    #          'gn_2wl1': True, 'gn_1wl': True,
    #          'dp_emb': 0.0, 'dp_1wl': 0.0, 'dp_2wl0': 0.3, 'dp_2wl1': 0.5,
    #          'epoch': 4000, 'lr': 0.01,}

    # Pubmed#2f_l
    #params = {'hidden_dim_1': 96, 'hidden_dim_2': 96,
    #          'layer1': 2, 'layer2': 2, 'layer3': 2,
    #          'act_1wl': True, 'act_2wl0': True, 'act_2wl1': True,
    #          'ln_1wl': False, 'ln_2wl0': False, 'ln_2wl1': False,
    #          'gn_2wl1': True, 'gn_1wl': True,
    #          'dp_emb': 0.1, 'dp_1wl': 0.0, 'dp_2wl0': 0.8, 'dp_2wl1': 0.1,
    #          'epoch': 1000, 'lr': 0.05, }

    # Cora#2fwl
    '''
    params = {'hidden_dim_1': 64, 'hidden_dim_2': 32,
              'layer1': 2, 'layer2': 1,
              'act_1wl0': False, 'act_1wl1': False,
              'ln_1wl0': True, 'ln_1wl1': True,
              'gn_1wl0': True, 'gn_1wl1': True,
              'dp_1wl0': 0.7, 'dp_1wl1': 0.0, 'dp_2wl0': 0.1, 'dp_2wl1': 0.0, 'dp_emb': 0.2,
              'epoch': 1000, 'lr': 0.01}
              '''
    #params = {'act_1wl0': False, 'act_1wl1': False, 'dp_emb': 0.3, 'dp_1wl0': 0.5,
    #          'dp_1wl1': 0.0, 'dp_2wl0': 0.0, 'dp_2wl1': 0.0, 'epoch': 1500,
    #          'hidden_dim_1': 64, 'hidden_dim_2': 32, 'layer1': 2, 'layer2': 1,
    #          'lr': 0.005}
    #params = {'act': True, 'dp1': 0.5, 'dp2': 0.5, 'dp3': 0.5, 'epoch': 1500, 'hidden_dim_1': 64, 'hidden_dim_2': 64, 'layer1': 2, 'layer2': 2, 'lr': 0.005}

    if args.pattern == '2fwl_l':
        params = {
            'USAir': {
                'hidden_dim_1': 64, 'hidden_dim_2': 32,
                'layer1': 2, 'layer2': 1, 'layer3': 1,
                'dp_emb': 0.0,
                'dp_1wl': 0.0, 'dp_2wl0': 0.0, 'dp_2wl1': 0.0,
                'epoch': 1000, 'lr': 0.01,
            },
            'NS': {
                'hidden_dim_1': 24, 'hidden_dim_2': 32,
                'layer1': 2, 'layer2': 2, 'layer3': 2,
                'dp_emb': 0.3,
                'dp_1wl': 0.0, 'dp_2wl0': 0.0, 'dp_2wl1': 0.0,
                'epoch': 1000, 'lr': 0.05,
            },
            'PB': {
                'hidden_dim_1': 64, 'hidden_dim_2': 24,
                'layer1': 3, 'layer2': 2, 'layer3': 2,
                'dp_emb': 0.0,
                'dp_1wl': 0.0, 'dp_2wl0': 0.1, 'dp_2wl1': 0.1,
                'epoch': 1000, 'lr': 0.005,
            },
            'Yeast': {
                'hidden_dim_1': 32, 'hidden_dim_2': 24,
                'layer1': 2, 'layer2': 2, 'layer3': 2,
                'dp_emb': 0.3,
                'dp_1wl': 0.0, 'dp_2wl0': 0.1, 'dp_2wl1': 0.1,
                'epoch': 1000, 'lr': 0.01,
            },
            'Celegans': {
                'hidden_dim_1': 32, 'hidden_dim_2': 64,
                'layer1': 3, 'layer2': 1, 'layer3': 1,
                'dp_emb': 0.2,
                'dp_1wl': 0.0, 'dp_2wl0': 0.1, 'dp_2wl1': 0.1,
                'epoch': 1000, 'lr': 0.01,
            },
            'Router': {
                'hidden_dim_1': 64, 'hidden_dim_2': 24,
                'layer1': 3, 'layer2': 1, 'layer3': 1,
                'dp_emb': 0.2,
                'dp_1wl': 0.0, 'dp_2wl0': 0.0, 'dp_2wl1': 0.0,
                'epoch': 1000, 'lr': 0.005,
            },
            'Power': {
                'hidden_dim_1': 32, 'hidden_dim_2': 32,
                'layer1': 2, 'layer2': 3, 'layer3': 3,
                'act_1wl': True,
                'ln_1wl': False, 'ln_2wl0': False, 'ln_2wl1': True,
                'gn_2wl1': True, 'gn_1wl': True,
                'dp_emb': 0.0, 'dp_1wl': 0.0, 'dp_2wl0': 0.3, 'dp_2wl1': 0.5,
                'epoch': 4000, 'lr': 0.01,
            },
            'Ecoli': {
                'hidden_dim_1': 64, 'hidden_dim_2': 24,
                'layer1': 2, 'layer2': 1, 'layer3': 1,
                'dp_emb': 0.1,
                'dp_1wl': 0.0, 'dp_2wl0': 0.1, 'dp_2wl1': 0.1,
                'epoch': 1000, 'lr': 0.005,
            },

            'Cora': {
                'hidden_dim_1': 256, 'hidden_dim_2': 16,
                'layer1': 2, 'layer2': 1, 'layer3': 1,
                'act_2wl0': True, 'act_2wl1': True, 'act_lin': False,
                'ln_lin': True, 'ln_2wl0': True, 'ln_2wl1': False,
                'gn_lin': True,
                'gn_2wl1': True,  # 'gn_1wl': False,
                # 'dp_emb': 0.0,
                'dp_lin0': 0.3, 'dp_lin1': 0.2, 'dp_2wl0': 0.0, 'dp_2wl1': 0.2,
                'epoch': 650, 'lr': 0.01,
                'use_degree': False, 'use_appnp': True,
                'reduce_feat': True, 'sum_pooling': True,
                'alpha': 0.09, 'gn_app': False,
            },
            'Citeseer': {
                'hidden_dim_1': 256, 'hidden_dim_2': 16,
                'layer1': 2, 'layer2': 1, 'layer3': 1,
                'act_2wl0': True, 'act_2wl1': True, 'act_lin': False,
                'ln_lin': True, 'ln_2wl0': False, 'ln_2wl1': False,
                'gn_lin': True,
                'gn_2wl1': True,# 'gn_1wl': False,
                #'dp_emb': 0.0,
                'dp_lin0': 0.5, 'dp_lin1': 0.9, 'dp_2wl0': 0.1, 'dp_2wl1': 0.0,
                'epoch': 200, 'lr': 0.005,
                'use_degree': False, 'use_appnp': True,
                'reduce_feat': True, 'sum_pooling': True,
                'alpha': 0.05
            },
            'Pubmed': {
                'hidden_dim_1': 96, 'hidden_dim_2': 96,
                'layer1': 2, 'layer2': 2, 'layer3': 2,
                'act_1wl': True, 'act_2wl0': True, 'act_2wl1': True,
                'ln_1wl': False, 'ln_2wl0': False, 'ln_2wl1': False,
                'gn_2wl1': True, 'gn_1wl': True,
                'dp_emb': 0.1, 'dp_1wl': 0.0, 'dp_2wl0': 0.8, 'dp_2wl1': 0.1,
                'epoch': 1000, 'lr': 0.05,
            }
        }
    if args.pattern == '2fwl':
        params = {
            'USAir': {
                'hidden_dim_1': 32, 'hidden_dim_2': 24,
                'layer1': 1, 'layer2': 2,
                'act_1wl0': False, 'act_1wl1': False,
                'gn_1wl0': False, 'gn_1wl1': False,
                'dp_1wl0': 0.1, 'dp_1wl1': 0.2, 'dp_2wl0': 0.1, 'dp_2wl1': 0.0, 'dp_emb': 0.4,
                'epoch': 2000, 'lr': 0.005
            },
            'NS': {
                'act_1wl0': True, 'act_1wl1': True,
                'dp_emb': 0.1, 'dp_1wl0': 0.1,
                'dp_1wl1': 0.1, 'dp_2wl0': 0.1, 'dp_2wl1': 0.1,
                'hidden_dim_1': 16, 'hidden_dim_2': 16,
                'layer1': 3, 'layer2': 1,
                'epoch': 1500, 'lr': 0.005
            },
            'Yeast': {
                'act_1wl0': True, 'act_1wl1': True,
                'dp_emb': 0.4, 'dp_1wl0': 0.4,
                'dp_1wl1': 0.4, 'dp_2wl0': 0.4, 'dp_2wl1': 0.4,
                'hidden_dim_1': 64, 'hidden_dim_2': 32,
                'layer1': 1, 'layer2': 2,
                'epoch': 1500, 'lr': 0.005
            },
            'PB': {
                'act_1wl0': True, 'act_1wl1': True,
                'dp_emb': 0.4, 'dp_1wl0': 0.4,
                'dp_1wl1': 0.4, 'dp_2wl0': 0.4, 'dp_2wl1': 0.4,
                'hidden_dim_1': 32, 'hidden_dim_2': 32,
                'layer1': 3, 'layer2': 3,
                'epoch': 1500, 'lr': 0.005
            },
            'Celegans': {
                'hidden_dim_1': 64, 'hidden_dim_2': 32,
                'layer1': 3, 'layer2': 2,
                # 'act_1wl0': False, 'act_1wl1': False,
                'gn_1wl0': False, 'gn_1wl1': False,
                'dp_1wl0': 0.4, 'dp_1wl1': 0.2, 'dp_2wl0': 0.1, 'dp_2wl1': 0.0, 'dp_emb': 0.2,
                'epoch': 2000, 'lr': 0.005
            },
            'Router': {
                'act_1wl0': True, 'act_1wl1': True,
                'dp_emb': 0.4, 'dp_1wl0': 0.0,
                'dp_1wl1': 0.0, 'dp_2wl0': 0.0, 'dp_2wl1': 0.0,
                'hidden_dim_1': 16, 'hidden_dim_2': 16,
                'layer1': 2, 'layer2': 1,
                'epoch': 1500, 'lr': 0.01
            },
            'Power': {
                'act_1wl0': True, 'act_1wl1': True,
                'dp_emb': 0.3, 'dp_1wl0': 0.0,
                'dp_1wl1': 0.1, 'dp_2wl0': 0.1, 'dp_2wl1': 0.1,
                'hidden_dim_1': 64, 'hidden_dim_2': 32,
                'layer1': 1, 'layer2': 2,
                'epoch': 1500, 'lr': 0.005
            },
            'Ecoli': {
                'act_1wl0': True, 'act_1wl1': True,
                'dp_emb': 0.3, 'dp_1wl0': 0.0,
                'dp_1wl1': 0.0, 'dp_2wl0': 0.4, 'dp_2wl1': 0.4,
                'hidden_dim_1': 32, 'hidden_dim_2': 32,
                'layer1': 3, 'layer2': 1,
                'epoch': 1500, 'lr': 0.005
            },
            'Cora': {
                'act_1wl0': False, 'act_1wl1': False,
                'dp_emb': 0.3, 'dp_1wl0': 0.5,
                'dp_1wl1': 0.0, 'dp_2wl0': 0.0, 'dp_2wl1': 0.0,
                'hidden_dim_1': 64, 'hidden_dim_2': 32,
                'layer1': 2, 'layer2': 1,
                'epoch': 1000, 'lr': 0.005
            },
            'Citeseer': {
                'act_1wl0': False, 'act_1wl1': False,
                'dp_emb': 0.1, 'dp_1wl0': 0.2,
                'dp_1wl1': 0.0, 'dp_2wl0': 0.4, 'dp_2wl1': 0.4,
                'hidden_dim_1': 24, 'hidden_dim_2': 16,
                'layer1': 1, 'layer2': 1,
                'epoch': 1500, 'lr': 0.001
            },
        }
    if args.pattern == '2wl_l':
        params = {
            'USAir': {
                'hidden_dim_1': 32, 'hidden_dim_2': 64,
                'layer1': 3, 'layer2': 3,
                'act_1wl0': True, 'act_1wl1': True, 'act_2wl': False,
                'ln_emb': True,
                'ln_1wl0': True, 'ln_1wl1': False, 'ln_2wl': True,
                #'dp_lin0': 0.5, 'dp_lin1': 0.2,
                'dp_emb': 0.1,
                'dp_1wl0': 0.0, 'dp_1wl1': 0.1, 'dp_2wl': 0.4,
                'epoch': 4000, 'lr': 0.005
            },
            'Celegans': {
                'hidden_dim_1': 32, 'hidden_dim_2': 24,
                'layer1': 2, 'layer2': 2,
                'act_1wl0': False, 'act_1wl1': False, 'act_2wl': False,
                'ln_emb': True,
                'ln_1wl0': False, 'ln_1wl1': True, 'ln_2wl': False,
                #'dp_lin0': 0.5, 'dp_lin1': 0.2,
                'dp_emb': 0.1,
                'dp_1wl0': 0.1, 'dp_1wl1': 0.0, 'dp_2wl': 0.0,
                'epoch': 8000, 'lr': 0.005
            },
            'Yeast': {
                'hidden_dim_1': 32, 'hidden_dim_2': 32,
                'layer1': 2, 'layer2': 2,
                'act_1wl0': True, 'act_1wl1': True, 'act_2wl': False,
                'ln_emb': True,
                'ln_1wl0': True, 'ln_1wl1': False, 'ln_2wl': False,
                # 'dp_lin0': 0.5, 'dp_lin1': 0.2,
                'dp_emb': 0.3,
                'dp_1wl0': 0.0, 'dp_1wl1': 0.2, 'dp_2wl': 0.0,
                'epoch': 4000, 'lr': 0.005
            },
            'NS': {
                'hidden_dim_1': 64, 'hidden_dim_2': 64,
                'layer1': 2, 'layer2': 1,
                'act_1wl0': True, 'act_1wl1': True, 'act_2wl': False,
                'ln_emb': True,
                'ln_1wl0': False, 'ln_1wl1': True, 'ln_2wl': False,
                'gn_1wl0': True, 'gn_1wl1': True,
                # 'dp_lin0': 0.5, 'dp_lin1': 0.2,
                'dp_emb': 0.2,
                'dp_1wl0': 0.0, 'dp_1wl1': 0.1, 'dp_2wl': 0.0,
                'epoch': 4000, 'lr': 0.005
            },
            'PB': {
                'hidden_dim_1': 96, 'hidden_dim_2': 64,
                'layer1': 2, 'layer2': 2,
                'act_1wl0': True, 'act_1wl1': False, 'act_2wl': True,
                'ln_emb': False,
                'ln_1wl0': False, 'ln_1wl1': True, 'ln_2wl': True,
                # 'dp_lin0': 0.5, 'dp_lin1': 0.2,
                'dp_emb': 0.4,
                'dp_1wl0': 0.0, 'dp_1wl1': 0.0, 'dp_2wl': 0.0,
                'epoch': 4000, 'lr': 0.01
            },
            'Router': {
                'hidden_dim_1': 48, 'hidden_dim_2': 48,
                'layer1': 3, 'layer2': 2,
                'act_1wl0': True, 'act_1wl1': True, 'act_2wl': True,
                'ln_emb': False,
                'ln_1wl0': False, 'ln_1wl1': False, 'ln_2wl': False,
                'gn_emb': True,
                'gn_1wl0': True, 'gn_1wl1': True, 'gn_2wl': True,
                # 'dp_lin0': 0.5, 'dp_lin1': 0.2,
                'dp_emb': 0.1,
                'dp_1wl0': 0.1, 'dp_1wl1': 0.1, 'dp_2wl': 0.1,
                'epoch': 4000, 'lr': 0.01
            },
            'Power': {
                'hidden_dim_1': 96, 'hidden_dim_2': 64,
                'layer1': 3, 'layer2': 3,
                'act_1wl0': True, 'act_1wl1': False, 'act_2wl': True,
                'ln_emb': False,
                'ln_1wl0': False, 'ln_1wl1': False, 'ln_2wl': False,
                'gn_emb': True,
                'gn_1wl0': True, 'gn_1wl1': True, 'gn_2wl': True,
                # 'dp_lin0': 0.5, 'dp_lin1': 0.2,
                'dp_emb': 0.0,
                'dp_1wl0': 0.1, 'dp_1wl1': 0.2, 'dp_2wl': 0.2,
                'epoch': 4000, 'lr': 0.01
            },
            'Ecoli': {
                'hidden_dim_1': 32, 'hidden_dim_2': 32,
                'layer1': 2, 'layer2': 1,
                'act_1wl0': True, 'act_1wl1': False, 'act_2wl': False,
                'ln_emb': False,
                'ln_1wl0': True, 'ln_1wl1': False, 'ln_2wl': False,
                # 'dp_lin0': 0.5, 'dp_lin1': 0.2,
                'dp_emb': 0.3,
                'dp_1wl0': 0.3, 'dp_1wl1': 0.0, 'dp_2wl': 0.2,
                'epoch': 4000, 'lr': 0.01
            },
            'Citeseer': {
                'hidden_dim_1': 64, 'hidden_dim_2': 96,
                'layer1': 2, 'layer2': 2,
                'act_1wl0': False, 'act_1wl1': True, 'act_2wl': True,
                'ln_1wl0': True, 'ln_1wl1': False, 'ln_2wl': False,
                'gn_1wl0': True, 'gn_1wl1': True,
                'dp_lin0': 0.7, 'dp_lin1': 0.5,
                # 'dp_emb': 0.2,
                'dp_1wl0': 0.0, 'dp_1wl1': 0.0, 'dp_2wl': 0.0,
                'epoch': 4000, 'lr': 0.005
            },
            'Cora': {
                'hidden_dim_1': 64, 'hidden_dim_2': 96,
                'layer1': 1, 'layer2': 1,
                'act_1wl0': True, 'act_1wl1': False, 'act_2wl': False,
                'ln_1wl0': False, 'ln_1wl1': False, 'ln_2wl': True,
                'gn_1wl0': True, 'gn_1wl1': True,
                'dp_lin0': 0.8, 'dp_lin1': 0.0,
                # 'dp_emb': 0.2,
                'dp_1wl0': 0.4, 'dp_1wl1': 0.1, 'dp_2wl': 0.1,
                'epoch': 4000, 'lr': 0.01
            },
            'Pubmed': {
                'hidden_dim_1': 128, 'hidden_dim_2': 128,
                'layer1': 2, 'layer2': 2,
                'act_1wl0': False, 'act_1wl1': True, 'act_2wl': True,
                'ln_1wl0': False, 'ln_1wl1': False, 'ln_2wl': False,
                'gn_1wl0': True, 'gn_1wl1': True, 'gn_2wl': True,
                'dp_lin0': 0.4, 'dp_lin1': 0.1,
                # 'dp_emb': 0.2,
                'dp_1wl0': 0.3, 'dp_1wl1': 0.0, 'dp_2wl': 0.1,
                'epoch': 4000, 'lr': 0.01
            },
        }
    if args.pattern != '2wl':
        params = params[args.dataset]

    '''
    params = {
                'hidden_dim_1': 96, 'hidden_dim_2': 64,
                'layer1': 2, 'layer2': 2,
                'act_1wl0': False, 'act_1wl1': False, 'act_2wl': True,
                'ln_1wl0': False, 'ln_1wl1': False, 'ln_2wl': False,
                'dp_lin0': 0.8, 'dp_lin1': 0.3,
                # 'dp_emb': 0.2,
                'dp_1wl0': 0.1, 'dp_1wl1': 0.0, 'dp_2wl': 0.1,
                'epoch': 4000, 'lr': 0.001
            }'''

    print(params)
    for key, result in params.items():
        with open(f'./{res_dir}/record.txt', 'a') as f:
            f.write(f'{key}: ' + str(result) + '   ')

    auc_list, hits2k_list, hits5k_list, hits10k_list = [], [], [], []
    for i in range(args.run):
        #set_seed(i + args.seed)
        results = testparam(args.device, args.dataset)

        auc_list.append(results['test_auc'])
        hits2k_list.append(results['Hits@2k'])
        hits5k_list.append(results['Hits@5k'])
        hits10k_list.append(results['Hits@10k'])
        auc,h2,h5,h10 = results['test_auc'], results['Hits@2k'], results['Hits@5k'], results['Hits@10k']
        print(f'Trial: {i+1}: AUC {auc * 100:.2f}, '
              f'H20 {h2 * 100:.2f}, '
              f'H50 {h5 * 100:.2f}, '
              f'H100 {h10 * 100:.2f}')
        with open(f'./{res_dir}/record.txt', 'a') as f:
            f.write('trial: ' + str(i) + '   ' + 'AUC:' + str(round(results['test_auc'], 4)) + '   ' + 'val:' + str(
                round(results['val_auc'], 4)) + '   ' + 'Hits@2k:' + str(
                round(results['Hits@2k'], 4)) + '   ' + 'Hits@5k:' + str(
                round(results['Hits@5k'], 4)) + '   ' + 'Hits@10k:' + str(
                round(results['Hits@10k'], 4)) + '   ' + '\n')
        if i>0:
            with open(f'./{res_dir}/record.txt', 'a') as f:
                f.write('AUC: ' + str(round(np.mean(auc_list) * 100, 2)) + '+' + str(
                    round(np.std(auc_list) * 100, 2)) + '   ' +
                        'Hits@2k: ' + str(round(np.mean(hits2k_list) * 100, 2)) + '+' + str(
                    round(np.std(hits2k_list) * 100, 2)) + '   ' +
                        'Hits@5k: ' + str(round(np.mean(hits5k_list) * 100, 2)) + '+' + str(
                    round(np.std(hits5k_list) * 100, 2)) + '   ' +
                        'Hits@10k: ' + str(round(np.mean(hits10k_list) * 100, 2)) + '+' + str(
                    round(np.std(hits10k_list) * 100, 2)) + '\n')

    print(f'Final Test: AUC {np.mean(auc_list)*100:.2f}+{np.std(auc_list)*100:.2f}, '
          f'H20 {np.mean(hits2k_list)*100:.2f}+{np.std(hits2k_list)*100:.2f}, '
          f'H50 {np.mean(hits5k_list)*100:.2f}+{np.std(hits5k_list)*100:.2f}, '
          f'H100 {np.mean(hits10k_list) * 100:.2f}+{np.std(hits10k_list) * 100:.2f}, ')

    with open(f'./{res_dir}/record.txt', 'a') as f:
        f.write('AUC: ' + str(round(np.mean(auc_list)*100, 2)) + '+'+str(round(np.std(auc_list)*100, 2))+'   ' +
                'Hits@2k: ' + str(round(np.mean(hits2k_list)*100, 2)) + '+'+str(round(np.std(hits2k_list)*100, 2))+'   ' +
                'Hits@5k: ' + str(round(np.mean(hits5k_list)*100, 2)) + '+'+str(round(np.std(hits5k_list)*100, 2))+'   ' +
                'Hits@10k: ' + str(round(np.mean(hits10k_list)*100, 2)) + '+'+str(round(np.std(hits10k_list)*100, 2))+'\n')
