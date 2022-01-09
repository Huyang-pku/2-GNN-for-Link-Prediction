from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv


class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class WLGNN(nn.Module):
    def __init__(self, max_x, latent_size=32, depth1=1, depth2=1, dropout=0.5):
        super().__init__()
        block_fn = lambda insize, outsize: Seq([
            GCNConv(insize, outsize),
            GraphNorm(outsize),
            Dropout(p=dropout, inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.emb = nn.Sequential(nn.Embedding(max_x + 1, latent_size),
                                 GraphNorm(latent_size),
                                 Dropout(p=dropout, inplace=True))
        self.conv1s = nn.ModuleList(
            [block_fn(latent_size, latent_size)] +
            [block_fn(latent_size, latent_size) for _ in range(depth1 - 1)])
        self.conv2s = nn.ModuleList(
            [block_fn(2 * latent_size, latent_size)] +
            [block_fn(latent_size, latent_size) for _ in range(depth2 - 1)])
        self.pred = nn.Linear(latent_size, 1)

    def forward(self, x, edge1, edge2, pos1, pos2):
        x = self.emb(x).squeeze()
        for conv1 in self.conv1s:
            x = conv1(x, edge1)
        x = x[pos1].reshape(pos1.shape[0], -1)
        for conv2 in self.conv2s:
            x = conv2(x, edge2)
        x = x[pos2]
        x = self.pred(x)
        return x

class Model_HY(nn.Module):
    def __init__(self, max_x, latent_size=32, depth1=1, depth2=1, dropout=0.5):
        super().__init__()
        block_fn = lambda insize, outsize: Seq([
            SAGEConv(insize, outsize),
            nn.ReLU(inplace=True)
        ])
        self.emb = nn.Sequential(nn.Embedding(max_x + 1, latent_size),
                                 GraphNorm(latent_size),
                                 Dropout(p=dropout, inplace=True))
        self.conv1s = nn.ModuleList(
            [block_fn(latent_size, latent_size)] +
            [block_fn(latent_size, latent_size) for _ in range(depth1 - 1)])
        self.conv2s = nn.ModuleList(
            [block_fn(2*latent_size, latent_size)] +
            [block_fn(latent_size, latent_size) for _ in range(depth2 - 1)])
        self.pred = nn.Linear(2*latent_size, 1)

    def forward(self, x, edge1, edge2, pos1, pos2, mask0, mask1):
        x = self.emb(x).squeeze()
        for conv1 in self.conv1s:
            x = conv1(x, edge1)
        
        #x = x[pos1[:,0]] * x[pos1[:,1]]
        x = x[pos1].reshape(pos1.shape[0], -1)
        for conv2 in self.conv2s:
            x = conv2(x, edge2)

        x = x[pos2]
        #x = x[mask0] * x[mask1]
        x = torch.cat([x[mask0], x[mask1]],1)
        x = self.pred(x)

        return x