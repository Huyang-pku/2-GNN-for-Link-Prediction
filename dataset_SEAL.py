import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as ssp
from torch.functional import split
from torch_geometric.utils import negative_sampling, add_self_loops, to_undirected
from torch_geometric.data import Data


def load(args):
    # check whether train and test links are provided
    train_pos, test_pos = None, None
    if args["train_name"] is not None:
        train_idx = np.loadtxt('data_SEAL/{}'.format(args["train_name"]),
                               dtype=int)
        train_pos = (train_idx[:, 0], train_idx[:, 1])
    if args["test_name"] is not None:
        test_idx = np.loadtxt('data_SEAL/{}'.format(args["test_name"]),
                              dtype=int)
        test_pos = (test_idx[:, 0], test_idx[:, 1])

    if args["data_name"] is not None:
        data = sio.loadmat('data_SEAL/{}.mat'.format(args["data_name"]))
        net = data['net']
        if 'group' in data:
            # load node attributes (here a.k.a. node classes)
            attributes = data['group'].toarray().astype('float32')
        else:
            attributes = None
    else:  # build network from train links
        assert (args["train_name"]
                is not None), "must provide train links if not using .mat"
        if args["train_name"].endswith('_train.txt'):
            args["data_name"] = args["train_name"][:-10]
        else:
            args["data_name"] = args["train_name"].split('.')[0]
        max_idx = np.max(train_idx)
        if args["test_name"] is not None:
            max_idx = max(max_idx, np.max(test_idx))
        net = ssp.csc_matrix(
            (np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])),
            shape=(max_idx + 1, max_idx + 1))
        net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
        net[np.arange(max_idx + 1),
            np.arange(max_idx + 1)] = 0  # remove self-loops
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    row, col, _ = ssp.find(net_triu)
    edge_index = torch.stack((torch.tensor(row).flatten(),
                              torch.tensor(col).flatten())).to(torch.long)
    #edge_index = to_undirected(edge_index)
    data = Data(edge_index=edge_index)
    split_edge = do_edge_split(data, args["val_ratio"], args["test_ratio"])
    return split_edge


from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)


def do_edge_split(data, val_ratio=0.05, test_ratio=0.1):
    data = train_test_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(to_undirected(data.train_pos_edge_index))
    data.train_neg_edge_index = negative_sampling(
        edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    #data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    #data.val_pos_edge_index = to_undirected(data.val_pos_edge_index)
    #data.test_pos_edge_index = to_undirected(data.test_pos_edge_index)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index
    split_edge['train']['edge_neg'] = data.train_neg_edge_index
    split_edge['valid']['edge'] = data.val_pos_edge_index
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index
    split_edge['test']['edge'] = data.test_pos_edge_index
    split_edge['test']['edge_neg'] = data.test_neg_edge_index
    return split_edge