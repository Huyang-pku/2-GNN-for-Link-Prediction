from math import e
from scipy.sparse import data
from sklearn import utils
from model import WLGNN, Model_HY
from datasets import load_dataset, dataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import time
from utils import sample_block, double
import optuna


def train(mod, opt, dataset, batch_size, mask0):
    perm1 = torch.randperm(dataset.ei.shape[1]//2, device=dataset.x.device)
    perm2 = torch.randperm((dataset.pos1.shape[0] - dataset.ei.shape[1])//2,
                           device=dataset.x.device)
    out = []
    scores = []
    mod.train()
    pos_batchsize = batch_size // 2
    neg_batchsize = (dataset.pos1.shape[0] - dataset.ei.shape[1])//(dataset.ei.shape[1]//pos_batchsize)
    mask1 = ~mask0

    for i in range(perm1.shape[0] // pos_batchsize):
        idx1 = perm1[i * pos_batchsize:(i + 1) * pos_batchsize]
        idx2 = perm2[i * neg_batchsize:(i + 1) * neg_batchsize]
        y = torch.cat((torch.ones_like(idx1, dtype=torch.float),
                       torch.zeros_like(idx2, dtype=torch.float)),
                      dim=0).unsqueeze(-1)

        idx1 = double(idx1, for_index = True)
        idx2 = double(idx2, for_index = True)
        length = idx1.shape[0] + idx2.shape[0]
        #import pdb
        #pdb.set_trace()
        mask0 = mask0[:length]
        mask1 = mask1[:length]

        new_ei, new_x, pos_pos, new_ei2 = sample_block(idx1, dataset.ea, dataset.x.shape[0], dataset.ei, dataset.ei2)
        opt.zero_grad()
        pos2 = torch.cat((idx1, dataset.ei.shape[1] + idx2), dim=0)
        pred = mod(new_x, new_ei, new_ei2, dataset.pos1, pos2, mask0, mask1)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        opt.step()
        out.append(loss.item())
        with torch.no_grad():
            sig = pred.sigmoid().cpu().numpy()
            score = roc_auc_score(y.cpu().numpy(), sig)
        scores.append(score)
    print(f"trn score {sum(scores)/len(scores)}", end=" ")
    return sum(out) / len(out)


@torch.no_grad()
def test(mod, dataset, mask0):
    mod.eval()
    mask0 = mask0[:dataset.y.shape[0]]
    mask1 = ~mask0
    pred = mod(
        dataset.x, dataset.ei, dataset.ei2, dataset.pos1, dataset.ei.shape[1] +
        torch.arange(dataset.y.shape[0], device=dataset.x.device), mask0, mask1)
    sig = pred.sigmoid().cpu().numpy()
    return roc_auc_score(dataset.y[mask0].cpu().numpy(), sig)


def main(device="cpu", dsname="Celegans", mod_params=(32, 2, 1, 0.0), lr=3e-4):
    device = torch.device(device)
    bg = load_dataset(dsname)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    mod = WLGNN(torch.max(bg.x), *mod_params).to(device)
    opt = Adam(mod.parameters(), lr=lr)
    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))
    train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=True)


def train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    trn_ds.pos1 = trn_ds.pos1.to(torch.long)
    val_ds.pos1 = val_ds.pos1.to(torch.long)
    tst_ds.pos1 = tst_ds.pos1.to(torch.long)
    batch_size = val_ds.y.shape[0]
    vprint(f"batch size{batch_size}")

    length = max(2 * batch_size, tst_ds.y.shape[0], val_ds.y.shape[0])
    even_mask = torch.zeros((length,), dtype=torch.bool)
    for i in range(length // 2):
        even_mask[i * 2] = 1

    best_val = 0
    tst_score = 0
    early_stop = 0
    for i in range(2000):
        t1 = time.time()
        loss = train(mod, opt, trn_ds, batch_size, even_mask)
        t2 = time.time()
        val_score = test(mod, val_ds, even_mask)
        vprint(f"trn: time {t2-t1:.2f} s, loss {loss:.4f} val {val_score:.4f}",
               end=" ")
        early_stop += 1
        if val_score > best_val:
            early_stop = 0
            best_val = val_score
            if verbose:
                tst_score = test(mod, tst_ds, even_mask)
            vprint(f"tst {tst_score:.4f}")
        else:
            vprint()
        if early_stop > 200:
            break
    vprint(f"end test {tst_score:.3f} time {(t2-t1)/8:.3f} s")
    return val_score


def work(device="cpu", dsname="Celegans"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)
    bg = load_dataset(dsname)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))

    def selparam(trial):
        lr = trial.suggest_categorical("lr",
                                       [0.0005, 0.001, 0.005, 0.01, 0.05])
        layer1 = trial.suggest_int("layer1", 1, 3)
        layer2 = trial.suggest_int("layer2", 1, 3)
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 48, 64])
        dp = trial.suggest_float("dp", 0.0, 0.9, step=0.05)
        return valparam(layer1, layer2, dp, hidden_dim, lr)

    def valparam(layer1, layer2, dp, hidden_dim, lr):
        mod = WLGNN(torch.max(bg.x),
                    *(hidden_dim, layer1, layer2, dp)).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train_routine(mod, opt, trn_ds, val_ds, tst_ds)

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + args.path + dsname +
                                ".db",
                                study_name=dsname,
                                load_if_exists=True)
    study.optimize(selparam, n_trials=200)


def testparam(device="cpu", dsname="Celegans"):  # mod_params=(32, 2, 1, 0.0), lr=3e-4
    device = torch.device(device)
    bg = load_dataset(dsname)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))

    def valparam(layer1, layer2, dp, hidden_dim, lr):
        mod = WLGNN(torch.max(bg.x),
                    *(hidden_dim, layer1, layer2, dp)).to(device)
        opt = Adam(mod.parameters(), lr=lr)
        return train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=True)

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + args.path + dsname +
                                ".db",
                                study_name=dsname,
                                load_if_exists=True)
    print("best param", study.best_params)
    valparam(**(study.best_params))    
    


def reproduce(device, ds):
    device = torch.device(device)
    bg = load_dataset(ds)
    bg.to(device)
    bg.preprocess()
    bg.setPosDegreeFeature()
    trn_ds = dataset(*bg.split(0))
    val_ds = dataset(*bg.split(1))
    tst_ds = dataset(*bg.split(2))
    
    lr = 1e-2
    if ds == "PB":
        hidden_dim = 96
    elif ds == "Yeast":
        hidden_dim = 64
    elif ds == "Celegans":
        hidden_dim = 32
    elif ds == "Power":
        hidden_dim = 64
        #lr = 1e-2
    else:
        raise NotImplementedError

    mod = Model_HY(torch.max(bg.x[2]), *(hidden_dim, 2, 3, 0)).to(device)
    opt = Adam(mod.parameters(), lr=lr)
    train_routine(mod, opt, trn_ds, val_ds, tst_ds, verbose=True)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="Yeast")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default="opt/")
    parser.add_argument('--test', action="store_true", default=False)
    parser.add_argument('--reproduce', action="store_true", default=True)
    args = parser.parse_args()
    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:" + str(args.device)
    print(args.device)
    if args.test:
        testparam(args.device, args.dataset)
    elif args.reproduce:
        reproduce(args.device, args.dataset)
    else:
        work(args.device, args.dataset)
    # main(args.device, args.dataset, lr=1e-3)