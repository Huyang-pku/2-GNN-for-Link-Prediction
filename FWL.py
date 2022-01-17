import os
import os.path as osp
import sys
import argparse

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import Embedding, BCEWithLogitsLoss
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv  # noqa
from torch_scatter import scatter_mean
import random
from copy import deepcopy
import warnings
import time

warnings.filterwarnings("ignore")


def node2link(p, q, s):
    if p <= q:
        return s * (s - 1) - (s - p - 2) * (s - p - 1) - 2 * (s - q)
    else:
        return s * (s - 1) - (s - q - 2) * (s - q - 1) - 2 * (s - p) + 1


def link2node(l, s):
    rev = True if l > l // 2 * 2 else False
    l = l // 2
    tem = s * (s - 1) // 2 - l - 1
    t = int(np.sqrt(2 * tem))
    if tem < t * (t + 1) // 2:
        t -= 1
    p = s - t - 2
    q = s - 1 - tem + t * (t + 1) // 2
    if rev:
        return q, p
    else:
        return p, q


def avg_pool(x, assignment):
    row, col = assignment
    return scatter_mean(x[row], col, dim=0)


def random_split(data, train_size, val_size, test_size):
    n = tailed_size * (tailed_size - 1) // 2
    train_mask = np.zeros(num_new_node, dtype=bool)
    val_mask = np.zeros(num_new_node, dtype=bool)
    test_mask = np.zeros(num_new_node, dtype=bool)
    train_size = train_size // 2
    val_size = val_size // 2
    test_size = test_size // 2
    l1 = np.arange(0, num_new_node // 2, 1)[(B == 1)[even_mask]].tolist()
    l0 = np.arange(0, num_new_node // 2, 1)[(B == 0)[even_mask]].tolist()
    l1 = np.random.permutation(l1)
    l0 = np.random.permutation(l0)
    l1 = l1.tolist()
    l0 = l0.tolist()
    train_label = l1[:train_size] + l0[:train_size]
    val_label = l1[train_size:train_size + val_size] + l0[train_size:train_size + val_size]
    test_label = l1[train_size + val_size:train_size + val_size + test_size] \
                 + l0[train_size + val_size:train_size + val_size + test_size]

    for i in range(len(train_label)):
        train_mask[train_label[i] * 2] = 1
        train_mask[train_label[i] * 2 + 1] = 1
    for i in range(len(val_label)):
        val_mask[val_label[i] * 2] = 1
        val_mask[val_label[i] * 2 + 1] = 1
    for i in range(len(test_label)):
        test_mask[test_label[i] * 2] = 1
        test_mask[test_label[i] * 2 + 1] = 1

    return torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)


def random_train_mask(train_size, pool):
    train_size = train_size // 2
    n = len(pool)
    train_mask = np.zeros(num_new_node, dtype=bool)

    l1 = np.random.permutation(l1_init)
    l0 = np.random.permutation(l0_init)

    l1 = l1.tolist()
    l0 = l0.tolist()

    train_label = l1[:train_size] + l0[:train_size]

    for i in range(len(train_label)):
        train_mask[train_label[i]] = 1
        train_mask[train_label[i] + 1] = 1
    return torch.from_numpy(train_mask)


def compute_hyperedge():
    if True:
        if two_wl_flow == 'local_dir':
            edge0 = []
            edge1 = []
            for i in range(data.edge_index.shape[1]):
                p = data.edge_index[0, i]
                q = data.edge_index[1, i]
                if p >= q: continue
                link1 = node2link(p, q, tailed_size)
                link2 = node2link(q, p, tailed_size)
                for r in range(tailed_size):
                    if r == p or r == q:
                        continue
                    link3 = node2link(p, r, tailed_size)
                    link4 = node2link(q, r, tailed_size)
                    edge0.append(link1)
                    edge1.append(link4)
                    edge0.append(link2)
                    edge1.append(link3)
            edge_index_new = np.array([edge0, edge1])
        np.save('hyperedge_{}_{}.npy'.format(dataset, two_wl_flow), edge_index_new)
    return edge_index_new

def to_directed(attr, type = None):
    pool = entire_pool[attr[:,0]!=0] if len(attr.shape)>1 else entire_pool[attr!=0]
    attr0 = torch.zeros((num_new_node0, attr.shape[-1])) if len(attr.shape)>1 else torch.zeros((num_new_node0,))
    #print(len(mask))
    for i in range(len(pool)):
        e = pool[i]
        if e % 2 == 1: continue
        p, q = link2node(e, tailed_size)
        attr0[p * tailed_size + q] = attr[e]
        attr0[q * tailed_size + p] = attr[e]
    attr0 = attr0.to(type) if type != None else attr0
    return attr0

def cut_edge(initial_edge_attr, initial_x, initial_edge, A, B_ob, node_mask, pruning, device):
    if two_wl_flow == 'local_dir':
        t0 = time.time()

        edge_mask = torch.zeros((initial_edge.shape[1],), dtype=torch.bool)
        output_A = A

        t1 = time.time()

        for i in range(initial_edge.shape[1]):
            p = initial_edge[0, i]
            q = initial_edge[1, i]
            if node_mask[node2link(p, q, tailed_size)]:
                edge_mask[i] = 1

        edges = entire_pool[node_mask]
        output_x = deepcopy(initial_x)
        t2 = time.time()

        if True:
            for i in range(len(edges) // 2):
                tem = 2 * i
                if B_ob[edges[tem]] == 0:
                    continue
                p, q = link2node(edges[tem], tailed_size)
                output_x[p] -= 1
                output_x[q] -= 1
                output_A[p, q] = 0
                output_A[q, p] = 0
                if output_x[p] < 0 or output_x[q] < 0: sys.exit(0)

        output_edge_attr = deepcopy(initial_edge_attr)
        t3 = time.time()
        if use_edge_attr:
            for i in range(len(edges)):
                output_edge_attr[edges[i]] = 0
        output_x = output_x.to(device)
        t4 = time.time()
        if output_edge_attr != None:
            output_edge_attr = output_edge_attr.to(device)
        output_edge_index = initial_edge[:, ~edge_mask].to(device)
        #output_edge_index_2 = initial_edge_2[:, ~edge_mask_2].to(device)
        t5 = time.time()
        #print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)

    return output_edge_attr, output_x, output_edge_index, output_A


def half(mask):
    mask0 = mask
    for i in range(len(mask) // 2):
        mask0[2 * i + 1] = False
    return mask0


def tail(data, s):
    data.x = data.x[:s, :]
    l = [i for i in range(data.edge_index.shape[1]) if (data.edge_index[0, i] < s) and (data.edge_index[1, i] < s)]
    edge_index = np.zeros((2, len(l)), dtype=int)
    for i in range(len(l)):
        edge_index[0, i] = data.edge_index[0, l[i]]
        edge_index[1, i] = data.edge_index[1, l[i]]
    data.edge_index = edge_index
    return data


if True:
    np.random.seed(100)
    torch.manual_seed(100)

    # datasets = 'Cora'
    path = osp.join('dataset', 'Cora')

    datasets = Planetoid(path, 'Cora', transform=T.NormalizeFeatures())
    data = datasets[0]

    dataset = 'NS'
    print(dataset)

    two_wl_flow = 'local_dir'  # 'global', 'local_k', 'local', 'local_undir', 'local_dir'

    node_undirect = False
    edge_undirect = False

    if dataset == 'USAir':
        f = open("./dataset/USAir.txt")
        line = f.readlines()
        f.close()
        usair_edge_index = np.zeros((2, len(line)), dtype=int)
        edge_feature = np.zeros(len(line, ), dtype=float)
        for i in range(len(line)):
            usair_edge_index[0, i] = int(line[i][:7])
            usair_edge_index[1, i] = int(line[i][7:14])
            edge_feature[i] = float(line[i][14:24])
        f.close()

        undirect = True
        use_node_feature = False
        use_edge_feature = True
        use_edge_exist = True
        use_node_degree = True
        use_edge_dist = False
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        train_size = 500
        val_size = 212
        test_size = 424
        latent_size = 24
        embedding_size = 16
        # num_layer = 4

        tailed_size = int(usair_edge_index.max())
        # data = tail(data, tailed_size)
        # data.edge_index = np.array([[0,1,1,3,3,1,2,3,4,5],[1,2,3,4,5,0,1,1,3,3]], dtype = int)
        data.edge_index = usair_edge_index - 1
        data.edge_attr = None
        data.x = np.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'Cele':
        f = open("./dataset/Cele.txt")
        line = f.readlines()
        f.close()
        edge_index = np.zeros((2, len(line)), dtype=int)
        for i in range(len(line)):
            if line[i][1] == ' ':
                edge_index[0, i] = int(line[i][:1])
                edge_index[1, i] = int(line[i][2:5])
            elif line[i][2] == ' ':
                edge_index[0, i] = int(line[i][:2])
                edge_index[1, i] = int(line[i][3:6])
            else:
                edge_index[0, i] = int(line[i][:3])
                edge_index[1, i] = int(line[i][4:7])
        f.close()

        undirect = True
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        train_size = 450
        val_size = 236
        test_size = 472
        latent_size = 32
        embedding_size = 16

        tailed_size = int(edge_index.max()) + 1
        # data = tail(data, tailed_size)
        # data.edge_index = np.array([[0,1,1,3,3,1,2,3,4,5],[1,2,3,4,5,0,1,1,3,3]], dtype = int)
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = np.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = True

    if dataset == 'Cora':
        undirect = False
        use_node_feature = False
        use_edge_feature = False
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        train_size = 1000
        val_size = 526
        test_size = 1054
        latent_size = 64
        embedding_size = 16

        tailed_size = len(data.x)
        #tailed_size = 500
        #data = tail(data, tailed_size)
        data.feature = data.x
        data.x = np.zeros(tailed_size)
        compute_new_edge = False
        compute_all_edge = False
        remove_edge = False

    if dataset == 'NS':
        f = open("./dataset/NS.txt")
        line = f.readlines()
        f.close()
        edge_index = np.zeros((2, len(line)), dtype=int)
        edge_feature = np.zeros(len(line, ), dtype=float)
        for i in range(len(line)):
            p = 0
            while line[i][p] != ' ':
                p += 1
            q = p + 1
            while line[i][q] != ' ':
                q += 1
            edge_index[0, i] = int(line[i][0:p]) - 1
            edge_index[1, i] = int(line[i][p:q]) - 1
            edge_feature[i] = float(line[i][q:])
        f.close()

        undirect = True
        use_node_feature = False
        use_edge_feature = True
        use_edge_exist = True
        use_node_degree = True
        use_node_attr = use_node_feature | use_node_degree
        use_edge_attr = use_edge_feature | use_edge_exist
        train_size = 500
        val_size = 274
        test_size = 548

        tailed_size = int(edge_index.max()) + 1
        # data = tail(data, tailed_size)
        # data.edge_index = np.array([[0,1,1,3,3,1,2,3,4,5],[1,2,3,4,5,0,1,1,3,3]], dtype = int)
        data.edge_index = edge_index
        data.edge_attr = None
        data.x = np.zeros(tailed_size)
        compute_new_edge = True
        compute_all_edge = False
        remove_edge = False

    if undirect:
        edge_index_r = np.concatenate((np.expand_dims(data.edge_index[1], 0), np.expand_dims(data.edge_index[0], 0)),
                                      axis=0)
        data.edge_index = np.concatenate((data.edge_index, edge_index_r), axis=1)
        if use_edge_feature:
            edge_feature = np.concatenate((edge_feature, edge_feature), axis=0)
        # data.edge_attr = np.concatenate((data.edge_attr, data.edge_attr), axis = 0)

if True:
    if True:
        p = 0
        q = 0
        num_new_node = tailed_size * (tailed_size-1)
        num_new_node0 = tailed_size * tailed_size
        entire_pool = torch.from_numpy(np.arange(0, num_new_node, 1)).to(torch.long)
        initial_edge = data.edge_index
        #num_new_edge = len(data.x) * (len(data.x) - 1) * (len(data.x) - 2)
        x_new = np.zeros(tailed_size, dtype=int)
        y_new = np.zeros(num_new_node, dtype=bool)
        y_new0 = np.zeros(num_new_node0, dtype=bool)
        #assignment = np.zeros((2, num_new_node * 2), dtype=int)
        deg = np.zeros(tailed_size, dtype=int)
        # edge_index_new = np.zeros((2,num_new_edge), dtype = int)

        A = torch.zeros((tailed_size, tailed_size), dtype=torch.bool)
        B = torch.zeros((num_new_node,), dtype=torch.bool)
        C = [[] for _ in range(tailed_size)]
        t = 0

        one_mask = torch.ones((num_new_node,), dtype=torch.bool)
        if not node_undirect:
            odd_mask = torch.zeros((num_new_node,), dtype=torch.bool)
            even_mask = torch.zeros((num_new_node,), dtype=torch.bool)
            for i in range(num_new_node // 2):
                odd_mask[2 * i + 1] = 1
                even_mask[2 * i] = 1

        for i in range(data.edge_index.shape[1]):
            p = data.edge_index[0, i]
            q = data.edge_index[1, i]
            A[p, q] = 1
            B[node2link(p, q, tailed_size)] = 1
            C[p].append(q)

        row = torch.zeros([1,tailed_size], dtype = torch.long)
        col = torch.zeros([tailed_size,1], dtype = torch.long)
        for i in range(1, tailed_size):
            row = torch.cat([row, torch.ones([1, tailed_size], dtype = torch.long)*i], 0)
            col = torch.cat([col, torch.ones([tailed_size, 1], dtype = torch.long)*i], 1)

        all_edges = torch.cat([row.unsqueeze(0), col.unsqueeze(0)],0)
        all_edges_res = all_edges.reshape(2,-1)
        for i in range(tailed_size):
            A[i, i] = 2

        if remove_edge:
            flag = torch.zeros((len(data.x), len(data.x)), dtype=bool)
            mask = torch.ones((data.edge_index.shape[1],), dtype=bool)
            for i in range(data.edge_index.shape[1]):
                p = data.edge_index[0, i]
                q = data.edge_index[1, i]
                if flag[p, q]: mask[i] = 0
                flag[p, q] = 1
            data.edge_index = data.edge_index[:, mask]

        for i in range(data.edge_index.shape[1]):
            p = data.edge_index[0, i]
            q = data.edge_index[1, i]
            if p >= q:
                continue
            if use_edge_feature:
                if 'edge_attr' not in data.keys:
                    data.edge_attr = torch.zeros((num_new_node, 1)) if len(edge_feature.shape) == 1 else torch.zeros(
                        (num_new_node, edge_feature.shape[1]))
                data.edge_attr[node2link(p, q, tailed_size)] = edge_feature[i]
                data.edge_attr[node2link(p, q, tailed_size) + 1] = edge_feature[i]
            y_new[node2link(p, q, tailed_size)] = 1
            y_new[node2link(p, q, tailed_size) + 1] = 1
            y_new0[p * tailed_size + q] = 1
            y_new0[q * tailed_size + p] = 1
            deg[p] = deg[p] + 1
            deg[q] = deg[q] + 1

        if use_edge_exist:
            if 'edge_attr' not in data.keys:
                data.edge_attr = B.unsqueeze(1)
            else:
                data.edge_attr = torch.cat([data.edge_attr, B.unsqueeze(1)], 1)

    for i in range(tailed_size):
        x_new[i] = deg[i]
        # if A[i, j] == 1:

    # data.feature = torch.tensor(data.x) if use_node_feature else None
    data.edge_attr = torch.tensor(data.edge_attr, dtype=torch.float) if use_edge_attr else None

    data.x = torch.tensor(x_new, dtype=torch.long)
    data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)
    #data.edge_index_2 = torch.tensor(edge_index_new, dtype=torch.long)
    #data.assignment = torch.tensor(assignment, dtype=torch.long)
    data.y = torch.tensor(y_new0, dtype=torch.long)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initial_edge = data.edge_index
    #initial_edge_2 = data.edge_index_2
    initial_x = deepcopy(data.x)
    initial_edge_attr = data.edge_attr

    train_mask, val_mask, test_mask = random_split(data, 0, val_size, test_size)
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    train_pool = np.arange(0, num_new_node, 1)[~val_mask & ~test_mask]

    l1_init = [train_pool[x] for x in range(len(train_pool)) if y_new[train_pool[x]] == 1 and train_pool[x] % 2 == 0]
    # l0_init = [train_pool[x] for x in range(len(train_pool)) if y_new[train_pool[x]] == 0 and train_pool[x] % 2 == 0]
    l0_init = [train_pool[x] for x in range(len(train_pool)) if y_new[train_pool[x]] == 0 and train_pool[x] % 2 == 0]
    # node_mask = train_mask | val_mask | test_mask

    # print(data.x.sum())
    node_mask = val_mask | test_mask
    B_train = (B == 1) & ~node_mask


    if use_node_feature:
        node_feature_size = 1 if len(data.feature.shape) == 1 else data.feature.shape[1]
    if use_edge_attr:
        edge_attr_size = data.edge_attr.shape[1]


if True:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    initial_edge = initial_edge.to(device)
    #initial_edge_2 = initial_edge_2.to(device)
    initial_x = initial_x.to(device)
    if use_edge_attr:
        initial_edge_attr = initial_edge_attr.to(device)


    # all_edges = all_edges.to(device)
    class Net(torch.nn.Module):
        def __init__(self, latent_size=20, embedding_size=20):
            super(Net, self).__init__()

            input_node_size = 0
            if use_node_feature: input_node_size += node_feature_size
            if use_node_degree: input_node_size += embedding_size

            self.embedding = Embedding(200, embedding_size)
            # self.nlin1 = torch.nn.Linear(input_node_size, latent_size)
            self.nconv1 = SAGEConv(input_node_size, latent_size)
            self.nconv2 = SAGEConv(latent_size, latent_size)
            # self.lin0 = torch.nn.Linear(data.x.shape[1], embedding_size)

            input_edge_size = latent_size
            if use_edge_attr: input_edge_size += edge_attr_size


            self.mlp0_1 = torch.nn.Linear(input_edge_size, latent_size)
            self.mlp0_2 = torch.nn.Linear(input_edge_size, latent_size)
            self.mlp0_3 = torch.nn.Linear(latent_size + input_edge_size, latent_size)

            self.mlp1_1 = torch.nn.Linear(latent_size, latent_size)
            self.mlp1_2 = torch.nn.Linear(latent_size, latent_size)
            self.mlp1_3 = torch.nn.Linear(latent_size * 2, latent_size)

            self.lin_dir = torch.nn.Linear(latent_size, 1)


        def forward(self, edge_mask, use_logsoftmax=False):
            x, edge_index= data.x, data.edge_index
            if use_node_feature: feature = data.feature
            if use_edge_attr: edge_attr = data.edge_attr
            edge_weight = None

            if use_node_degree: x = self.embedding(x)

            if use_node_feature:
                feature = feature.unsqueeze(1) if len(feature.shape) == 1 else feature
                x = torch.cat([x, feature.to(torch.float)], 1) if use_node_degree else feature.to(torch.float)
            # x = F.relu(self.nlin1(x))

            x = F.relu(self.nconv1(x, edge_index))

            # x = F.dropout(x, training=self.training)
            x = self.nconv2(x, edge_index)

            x = x[all_edges_res[0]] * x[all_edges_res[1]]

            if use_edge_attr: x = torch.cat([x, edge_attr], 1)


            x1 = F.relu(self.mlp0_1(x)).reshape(tailed_size, tailed_size, -1).permute(2, 0, 1)
            x2 = F.relu(self.mlp0_2(x)).reshape(tailed_size, tailed_size, -1).permute(2, 0, 1)
            x = torch.cat([x, torch.bmm(x1, x2).permute(1, 2, 0).reshape(tailed_size ** 2, -1)], -1)
            x = F.relu(self.mlp0_3(x))

            x1 = F.relu(self.mlp1_1(x)).reshape(tailed_size, tailed_size, -1).permute(2, 0, 1)
            x2 = F.relu(self.mlp1_2(x)).reshape(tailed_size, tailed_size, -1).permute(2, 0, 1)
            x = torch.cat([x, torch.bmm(x1, x2).permute(1, 2, 0).reshape(tailed_size ** 2, -1)], -1)
            x = F.relu(self.mlp1_3(x))

            x = x.reshape(tailed_size, tailed_size, -1)
            x = (x * x.permute(1, 0, 2)).reshape(tailed_size * tailed_size, -1)

            x = x[edge_mask]

            x = self.lin_dir(x)
            return x


    node_mask = val_mask | test_mask
    initial_train_edge_attr, initial_train_x, initial_train_edge, A_train = cut_edge(
        initial_edge_attr, initial_x, initial_edge, A, B, node_mask, False,
        device)  # , edge2)
    initial_val_edge_attr, initial_val_x, initial_val_edge, A_val = cut_edge(initial_edge_attr,
                                                                                          initial_x, initial_edge,
                                                                                          A, B,
                                                                                          node_mask, True,
                                                                                          device)  # , edge2)
    initial_test_edge_attr, initial_test_x, initial_test_edge, A_test = cut_edge(initial_edge_attr,
                                                                                              initial_x,
                                                                                              initial_edge,
                                                                                              A, B,
                                                                                              test_mask,
                                                                                              True,
                                                                                              device)  # , edge2)

    model, data = Net(64, 16).to(device), data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)

    Criteria = BCEWithLogitsLoss()

if True:
    def train():
        model.train()
        optimizer.zero_grad()
        if True:
            output = model(train_mask0).view(-1)
            loss = Criteria(output, data.y[train_mask0].to(torch.float))
            loss.backward()
        optimizer.step()
        return loss


    def initialize_val():
        data.x = deepcopy(initial_val_x).to(device)
        if initial_edge_attr != None:
            data.edge_attr = deepcopy(initial_val_edge_attr).to(device)
        data.edge_index = deepcopy(initial_val_edge).to(device)
        #data.edge_index_2 = deepcopy(initial_val_edge_2).to(device)
        # print(data.x.sum())


    def initialize_test():
        data.x = deepcopy(initial_test_x).to(device)
        if initial_edge_attr != None:
            data.edge_attr = deepcopy(initial_test_edge_attr).to(device)
        data.edge_index = deepcopy(initial_test_edge).to(device)
        #data.edge_index_2 = deepcopy(initial_test_edge_2).to(device)
        # print(data.x.sum())


    @torch.no_grad()
    def test():
        model.eval()
        aucs = []

        # print('train')
        logits = model(train_mask0)
        sig = logits.sigmoid().cpu().numpy()
        label = data.y[train_mask0].cpu().numpy()
        mask1 = (B0 == 1)[train_mask0]
        train1, train0 = sig[mask1].mean(), sig[~mask1].mean()
        aucs.append(roc_auc_score(label, sig))



        # print('val')
        initialize_val()
        if use_edge_exist:
            data.edge_attr = to_directed(data.edge_attr).to(device)

        logits = model(val_mask0)
        sig = logits.sigmoid().cpu().numpy()
        label = data.y[val_mask0].cpu().numpy()
        mask1 = (B0 == 1)[val_mask0]
        # print(sig[mask1].mean(), sig[~mask1].mean())
        aucs.append(roc_auc_score(label, sig))

        # logits = model(one_mask).sigmoid()
        # print(logits[(B==1)&~val_mask].mean().cpu().numpy(), logits[(B==0)&~val_mask].mean().cpu().numpy())

        # print('test')
        initialize_test()
        if use_edge_exist:
            data.edge_attr = to_directed(data.edge_attr).to(device)

        logits = model(test_mask0)
        sig = logits.sigmoid().cpu().numpy()
        label = data.y[test_mask0].cpu().numpy()
        mask1 = (B0 == 1)[test_mask0]
        print(train1, train0, sig[mask1].mean(), sig[~mask1].mean())
        aucs.append(roc_auc_score(label, sig))
        aucs.append(sig[mask1].mean())
        aucs.append(sig[~mask1].mean())

        # logits = model(one_mask).sigmoid()
        # print(logits[(B==1)&~test_mask].mean().cpu().numpy(), logits[(B==0)&~test_mask].mean().cpu().numpy())

        return aucs


    if True:
        # model.load_state_dict(torch.load('./model/model_1221109.pt'))
        best_val_auc = test_auc = 0
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        val_mask0, test_mask0, B0 = to_directed(val_mask, torch.bool), to_directed(test_mask, torch.bool), to_directed(B, torch.bool)
        for epoch in tqdm(range(1, 1001)):
            t0 = time.time()

            if True:
                train_mask = random_train_mask(train_size, train_pool)
                train_mask0 = to_directed(train_mask, torch.bool)
            data.edge_attr, data.x, data.edge_index, data.A = cut_edge(initial_train_edge_attr,
                                                                                  initial_train_x,
                                                                                  initial_train_edge,
                                                                                  A_train, B_train,
                                                                                  train_mask, True,
                                                                                  device)
            if use_edge_exist:
                data.edge_attr = to_directed(data.edge_attr).to(device)

            data.train_mask = train_mask

            # print(data.x.sum())
            loss = train()
            training_auc, val_auc, tmp_test_auc, pred1, pred0 = test()
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                test_auc = tmp_test_auc
                print('Best!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            log1 = 'Epoch: {:03d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            # print(log1.format(epoch, training_auc, train_auc, best_val_auc, test_auc))
            print(log1.format(epoch, loss, training_auc, val_auc, tmp_test_auc))
            with open(f'./records/{dataset}_auc_record.txt', 'a') as f:
                f.write(
                    'Epoch:' + str(epoch) + '   ' + 'Loss:' + str(
                        round(loss.item(), 4)) + '   ' + 'Train:' + str(
                        round(training_auc, 4)) + '   ' + 'Val:' + str(
                        round(val_auc, 4)) + '   ' + 'Test:' + str(round(tmp_test_auc, 4)) + '   ' + 'Pred1:' + str(
                        round(pred1, 4)) + '   ' + 'Pred0:' + str(round(pred0, 4)) + '\n')

        import datetime

        i = datetime.datetime.now()
        # print ("dd/mm/yyyy 格式是  %s%s" % (i.month, i.day) )
        torch.save(model.state_dict(), './model/model_%s_%s%s%s%s.pt' % (dataset, i.month, i.day, i.hour, i.minute))