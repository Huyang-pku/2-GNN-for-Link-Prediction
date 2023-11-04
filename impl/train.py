import torch
from utils import sample_block, double, get_ei2
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling, add_self_loops
from model import LocalWLNet
from tqdm import tqdm
from ogb.linkproppred import Evaluator
import time


def eval_hits(y_pred_pos, y_pred_neg, k1=2000, k2=5000, k3=10000):

    kth_score_in_negative_edges = torch.topk(y_pred_neg, k1)[0][-1]
    hits20 = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    kth_score_in_negative_edges = torch.topk(y_pred_neg, k2)[0][-1]
    hits50 = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    kth_score_in_negative_edges = torch.topk(y_pred_neg, k3)[0][-1]
    hits100 = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    hits = {
        'Hits@2k': hits20,
        'Hits@5k': hits50,
        'Hits@10k': hits100,
    }
    return hits


def train(mod, opt, dataset, batch_size, i):
    mod.train()
    global perm1, perm2, pos_batchsize, neg_batchsize
    if i == 0:
        pos_batchsize = batch_size // 2
        neg_batchsize = batch_size // 2
        perm1 = torch.randperm(dataset.ei.shape[1] // 2, device=dataset.x.device)
        perm2 = torch.randperm((dataset.pos1.shape[0] - dataset.ei.shape[1]) // 2,
                               device=dataset.x.device)

    idx1 = perm1[i * pos_batchsize:(i + 1) * pos_batchsize]
    idx2 = perm2[i * neg_batchsize:(i + 1) * neg_batchsize]
    y = torch.cat((torch.ones_like(idx1, dtype=torch.float),
                   torch.zeros_like(idx2, dtype=torch.float)),
                  dim=0).unsqueeze(-1)

    idx1 = double(idx1, for_index=True)
    idx2 = double(idx2, for_index=True) + dataset.ei.shape[1]

    ei_new, x_new, ei2_new = sample_block(idx1, dataset.x.shape[0], dataset.ei, dataset.ei2)
    pos2 = torch.cat((idx1, idx2), dim=0)
    opt.zero_grad()
    if isinstance(mod, LocalWLNet):
        pred = mod(x_new, ei_new, dataset.pos1, pos2, ei2_new)
    else:
        pred_pos = dataset.pos1[idx1][:, 0].reshape(-1, 2)
        if mod.use_feat:
            edge_index, _ = add_self_loops(dataset.ei)
            pred_neg = negative_sampling(
                dataset.ei,
                num_nodes=dataset.x.shape[0],
                num_neg_samples=neg_batchsize,
            ).t().to(dataset.x.device)
        else:
            pred_neg = dataset.pos1[idx2][:, 0].reshape(-1, 2)
        pred_links = torch.cat([pred_pos, pred_neg], 0)
        pred = mod(x_new, ei_new, pred_links, ei2_new)
    loss = F.binary_cross_entropy_with_logits(pred, y)
    loss.backward()
    opt.step()
    with torch.no_grad():
        sig = pred.sigmoid().cpu().numpy()
        score = roc_auc_score(y.cpu().numpy(), sig)

    i += 1
    if (i + 1) * pos_batchsize > perm1.shape[0]:
        i = 0
    return loss.item(), score, i


@torch.no_grad()
def test(mod, dataset, test_hits=False, divide=1, test=False, epoch=0):
    mod.eval()
    if isinstance(mod, LocalWLNet):
        pred = mod(
            dataset.x,
            dataset.ei,
            dataset.pos1,
            dataset.ei.shape[1] + torch.arange(dataset.y.shape[0], device=dataset.x.device),
            dataset.ei2,
            True)
        if test_hits:
            batch_size = dataset.all_neg_idx.size(0)//2//divide * 2
            pred_all_neg = None
            for split in tqdm(range(divide)):
                neg_idx = dataset.all_neg_idx[batch_size*split:batch_size*(split+1)]
                neg_ei2 = get_ei2(dataset.x.size(0), dataset.ei,
                        neg_idx.t())
                pred_neg = mod(
                    dataset.x,
                    dataset.ei,
                    torch.cat([dataset.ei.t(), neg_idx], 0),
                    dataset.ei.shape[1] + torch.arange(batch_size, device=dataset.x.device),
                    neg_ei2,
                    True).squeeze().sigmoid().cpu()
                pred_all_neg = torch.cat([pred_all_neg, pred_neg], dim=0) if pred_all_neg!=None else pred_neg

    else:
        pred_links = dataset.pos1[dataset.ei.shape[1] + torch.arange(dataset.y.shape[0], device=dataset.x.device)][:,0].reshape(-1,2)
        pred = mod(dataset.x,dataset.ei,pred_links,dataset.ei2,True)
        if test_hits:
            batch_size = dataset.all_neg_idx.size(0) // 2 // divide * 2
            pred_all_neg = None
            for split in tqdm(range(divide)):
                neg_idx = dataset.all_neg_idx[batch_size * split:batch_size * (split + 1)]
                pred_neg = mod(
                    dataset.x,
                    dataset.ei,
                    neg_idx,
                    dataset.ei2,
                    True).squeeze().sigmoid().cpu()
                pred_all_neg = torch.cat([pred_all_neg, pred_neg], dim=0) if pred_all_neg != None else pred_neg

    sig = pred.squeeze().sigmoid().cpu()
    mask = torch.cat(
        [torch.ones([1, sig.shape[0]], dtype=bool), torch.zeros([1, sig.shape[0]], dtype=bool)]).t().reshape(
        -1, 1)

    result = roc_auc_score(dataset.y[mask].squeeze().cpu().numpy(), sig)
    y = dataset.y[mask].to(torch.bool)

    if test_hits:
        if isinstance(mod, LocalWLNet):
            hits = eval_hits(sig[y], pred_all_neg, 1000, 2500, 5000)
        else:
            hits = eval_hits(sig[y], pred_all_neg)
        return result, hits
    else:
        return result

def train_routine(dsname, mod, opt, trn_ds, val_ds, tst_ds, epoch, pattern, test_hits=False, split=1, res_dir='', verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    trn_ds.pos1 = trn_ds.pos1.to(torch.long)
    val_ds.pos1 = val_ds.pos1.to(torch.long)
    tst_ds.pos1 = tst_ds.pos1.to(torch.long)
    batch_size = val_ds.y.shape[0]
    vprint(f"batch size{batch_size}")

    best_val = 0
    tst_score = 0
    test_hit = {
        'Hits@2k': 0,
        'Hits@5k': 0,
        'Hits@10k': 0,
    }
    early_stop = 0
    early_stop_thd = epoch // 4
    for i in range(epoch):
        train_idx = 0
        t0 = time.time()
        loss, trn_score, train_idx = train(mod, opt, trn_ds, batch_size, train_idx)
        t1 = time.time()
        val_score = test(mod, val_ds, test_hits=False)
        vprint(f"epoch: {i:03d}, trn: time {t1 - t0:.2f} s, loss {loss:.4f}, trn {trn_score:.4f}, val {val_score:.4f}",
               end=" ")
        early_stop += 1
        if val_score > best_val:
            early_stop = 0
            best_val = val_score
            t0 = time.time()
            tst_score = test(mod, tst_ds, test_hits=False, test=True)
            t1 = time.time()
            #with open(f'./records/{dsname}_{pattern}_record.txt', 'a') as f:
            #    f.write('AUC:' + str(round(tst_score, 4)) + '\n')

            if res_dir != '':
                torch.save(mod.state_dict(), f'{res_dir}/model.pth')
            vprint(f"tst {tst_score:.4f}")
        else:
            vprint()
        if verbose and res_dir != '':
            with open(f'./{res_dir}/record.txt', 'a') as f:
                f.write('Epoch: '+ str(i+1) + '   ' +'AUC:' + str(round(tst_score, 4))+ '   ' + 'val:' + str(round(val_score, 4)) +
                    '   ' + 'Time:' + str(round(t1 - t0, 4)) + '\n')
        if early_stop > early_stop_thd:
            break

    if test_hits:
        mod.load_state_dict(torch.load(f'{res_dir}/model.pth'))
        _, test_hit = test(mod, tst_ds, test_hits, split, test=True)

    vprint(f"end test {tst_score:.3f}")
    results = {
        'val_auc': best_val,
        'test_auc': tst_score,
        'Hits@2k': test_hit['Hits@2k'],
        'Hits@5k': test_hit['Hits@5k'],
        'Hits@10k': test_hit['Hits@10k'],
    } if test_hits else {
        'val_auc': best_val,
        'test_auc': tst_score,
        'Hits@2k': 0,
        'Hits@5k': 0,
        'Hits@10k': 0,
    }
    return results