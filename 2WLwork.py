import random
import numpy as np
import torch
from torch.optim import Adam
import os

import optuna
import pandas as pd
from impl import train
from datasets import load_dataset, dataset
from model import LocalWLNet, WLNet, FWLNet, LocalFWLNet


def work(device="cpu", dsname="Celegans"):
    device = torch.device(device)
    bg = load_dataset(dsname, args.pattern)
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

    def selparam(trial):
        nonlocal bg, trn_ds, val_ds, tst_ds
        if random.random() < 0.1:
            bg = load_dataset(dsname, args.pattern)
            bg.to(device)
            bg.preprocess()
            bg.setPosDegreeFeature()
            trn_ds = dataset(*bg.split(0))
            val_ds = dataset(*bg.split(1))
            tst_ds = dataset(*bg.split(2))
            if trn_ds.na != None:
                print("use node feature")
                trn_ds.na = trn_ds.na.to(device)
                val_ds.na = val_ds.na.to(device)
                tst_ds.na = tst_ds.na.to(device)

        lr = trial.suggest_categorical("lr", [0.01, 0.05, 0.1])
        layer1 = trial.suggest_int("l1", 2, 2)
        #layer2 = trial.suggest_int("l2", 1, 2)
        layer3 = trial.suggest_int("l3", 1, 1)
        hidden_dim_1wl = trial.suggest_categorical("h1", [256])
        hidden_dim_2wl = trial.suggest_categorical("h2", [16, 32])
        dp_lin0 = trial.suggest_float("dp0_0", 0.0, 0.9, step=0.1)
        dp_lin1 = trial.suggest_float("dp0_1", 0.0, 0.9, step=0.1)
        #dp_emb = trial.suggest_float("dp0", 0.0, 0.8, step=0.1)
        #dp_1wl0 = trial.suggest_float("dp1", 0.0, 0.6, step=0.1)
        #dp_1wl1 = trial.suggest_float("dp2", 0.0, 0.6, step=0.1)
        dp_2wl0 = trial.suggest_float("dp3", 0.0, 0.5, step=0.1)
        dp_2wl1 = trial.suggest_float("dp4", 0.0, 0.5, step=0.1)
        alpha = trial.suggest_float("alpha", 0.05, 0.1, step=0.01)
        act_lin = trial.suggest_categorical("al", [True, False])
        #act_1wl0 = trial.suggest_categorical("a0", [True, False])
        #act_1wl1 = trial.suggest_categorical("a1", [True, False])
        act_2wl0 = trial.suggest_categorical("a2", [True, False])
        act_2wl1 = trial.suggest_categorical("a3", [True, False])
        ln_lin = trial.suggest_categorical("lnl", [True, False])
        #ln_emb = trial.suggest_categorical("lne", [True, False])
        #ln_1wl0 = trial.suggest_categorical("ln0", [True, False])
        #ln_1wl1 = trial.suggest_categorical("ln1", [True, False])
        ln_2wl0 = trial.suggest_categorical("ln2", [True, False])
        ln_2wl1 = trial.suggest_categorical("ln3", [True, False])
        gn_lin = trial.suggest_categorical("gl", [True, False])
        #gn_1wl0 = trial.suggest_categorical("g1", [True, False])
        gn_2wl1 = trial.suggest_categorical("g2", [True, False])
        gn_app = trial.suggest_categorical("ga", [False])
        setting = {'dp_lin0': dp_lin0, 'dp_lin1': dp_lin1,
                   #'dp_1wl0': dp_1wl0, 'dp_1wl1': dp_1wl1,
                   'dp_2wl0': dp_2wl0, 'dp_2wl1': dp_2wl1,
                   #'dp_emb': dp_emb,
                   'hidden_dim_1': hidden_dim_1wl, 'hidden_dim_2': hidden_dim_2wl,
                   'layer1': layer1, 'layer2': 1,
                   'layer3': layer3,
                   'lr': lr,
                   'act_lin': act_lin,
                   #'act_1wl0': act_1wl0, 'act_1wl1': act_1wl1,
                   'act_2wl0': act_2wl0, 'act_2wl1': act_2wl1,
                   'ln_lin': ln_lin,
                   #'ln_emb': ln_emb,
                   #'ln_1wl0': ln_1wl0, 'ln_1wl1': ln_1wl1,
                   'ln_2wl0': ln_2wl0, 'ln_2wl1': ln_2wl1,
                   'gn_lin': gn_lin,
                   #'gn_1wl0': gn_1wl0, 'gn_1wl1': gn_1wl1,
                   'gn_2wl1': gn_2wl1,
                   'alpha': alpha, 'gn_app': gn_app,
                   'use_degree': False, 'use_appnp': True,
                   'reduce_feat': True, 'sum_pooling': True,
                   }
        return valparam(setting)

    def valparam(kwargs):
        lr = kwargs.pop('lr')
        epoch = args.epoch
        max_degree = torch.max(tst_ds.x)
        if args.pattern == '2wl':
            mod = WLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2wl_l':
            mod = LocalWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2fwl':
            mod = FWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        elif args.pattern == '2fwl_l':
            mod = LocalFWLNet(max_degree, use_node_attr, trn_ds.na, **kwargs).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train.train_routine(args.dataset, mod, opt, trn_ds, val_ds, tst_ds, epoch, args.pattern, verbose=args.verbose)['val_auc']

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + args.path + dsname +
                                ".db",
                                study_name=dsname,
                                load_if_exists=True)
    print(
        f"storage {'sqlite:///' + args.path + dsname + '.db'} study_name {dsname}"
    )
    study.optimize(selparam, n_trials=args.episode)


def check(dsname):
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + args.path + dsname +
                                        ".db",
                                study_name=dsname,
                                load_if_exists=True)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = study.trials_dataframe().drop(['state', 'duration', 'number'], axis=1)
    print(df)
    print('\n')
    best_study_index = np.argmax(df['value'])
    best_study = df.iloc[best_study_index]
    #print(f'Best trial: {best_study_index}')
    print(best_study)
    import pdb
    pdb.set_trace()
    exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="USAir")
    parser.add_argument('--pattern', type=str, default="2wl_l")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--episode', type=int, default=200)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default="Opt/")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--check', action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    args.path = args.path + args.pattern + "/"
    if not os.path.exists(args.path):
        os.mkdir(args.path)


    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)
    print(args.dataset, args.device)
    if args.check:
        check(args.dataset)
    else:
        work(args.device, args.dataset)
