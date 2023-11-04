from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv, APPNP
from utils import reverse, sparse_bmm, sparse_cat, add_zero, edge_list
import time

class WLNet(torch.nn.Module):
    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 hidden_dim_1=20,
                 hidden_dim_2=20,
                 layer1=2,
                 layer2=1,
                 layer3=1,
                 dp0_0 = 0.0,
                 dp0_1 = 0.0,
                 dp1=0.0,
                 dp2=0.0,
                 dp3=0.0,
                 ln0=True,
                 ln1=True,
                 ln2=True,
                 ln3=True,
                 ln4=True,
                 act0=False,
                 act1=False,
                 act2=False,
                 act3=True,
                 act4=True,
                 ):
        super(WLNet, self).__init__()

        self.use_feat = use_feat
        self.feat = feat
        use_affine = False

        relu_lin = lambda a, b, dp, lnx, actx: Seq([
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity()])
        if feat is not None:
            self.lin1 = nn.Sequential(
                nn.Dropout(dp0_0),
                relu_lin(feat.shape[1], hidden_dim_1, dp0_1, ln0, act0)
            )

        Convs = lambda a, b, dp, lnx, actx: Seq([
            SAGEConv(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity()])

        self.embedding = torch.nn.Sequential(torch.nn.Embedding(max_x + 1, hidden_dim_1),
                                             torch.nn.Dropout(p=dp1))
        #self.embedding = nn.Embedding(max_x + 1, latent_size_1)

        self.nconvs = nn.ModuleList([Convs(hidden_dim_1, hidden_dim_1, dp2, ln1, act1)] +
                                    [Convs(hidden_dim_1, hidden_dim_1, dp2, ln2, act2) for _ in range(layer1 - 1)]
                                    )

        input_edge_size = hidden_dim_1

        self.h_1 = Seq([relu_lin(input_edge_size + 1, hidden_dim_2, dp3, ln3, act3)] +
                       [relu_lin(hidden_dim_2, hidden_dim_2, dp3, ln3, act3) for _ in range(layer2 - 1)])

        self.g_1 = Seq([relu_lin(hidden_dim_2 * 2 + input_edge_size + 1, hidden_dim_2, dp3, ln4, act4)] +
                       [relu_lin(hidden_dim_2, hidden_dim_2, dp3, ln4, act4) for _ in range(layer3 - 1)])

        self.lin_dir = torch.nn.Linear(hidden_dim_2, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, ei, pos, ei2=None, test=False):
        edge_index = ei
        n = x.shape[0]
        if self.use_feat:
            x = self.feat
            x = self.lin1(x)
        else:
            x = self.embedding(x)
        # x = F.relu(self.nlin1(x))

        for conv in self.nconvs:
            x = conv(x, edge_index)
        colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)
        rowx = x.unsqueeze(1).expand(-1, n, -1).reshape(n * n, -1)
        x = rowx * colx
        x = x.reshape(n, n, -1)
        eim = torch.zeros((n * n,), device=x.device)
        eim[edge_index[0] * n + edge_index[1]] = 1
        eim = eim.reshape(n, n, 1)
        x = torch.cat((x, eim), dim=-1)
        x = mataggr(x, self.h_1, self.g_1)
        x = (x * x.permute(1, 0, 2)).reshape(n * n, -1)
        x = x[pos[:, 0] * n + pos[:, 1]]
        x = self.lin_dir(x)
        return x

class LocalWLNet(nn.Module):
    def __init__(self,
                 max_x,
                 use_node_feat,
                 node_feat,
                 hidden_dim_1=256,
                 hidden_dim_2=32,
                 layer1=1,
                 layer2=1,
                 dp_lin0 = 0.7,
                 dp_lin1 = 0.7,
                 dp_emb = 0.5,
                 dp_1wl0=0.5,
                 dp_2wl=0.5,
                 dp_1wl1=0.5,
                 ln_emb=False,
                 ln_1wl0=False,
                 ln_1wl1=False,
                 ln_2wl=False,
                 gn_emb=False,
                 gn_1wl0=False,
                 gn_1wl1=False,
                 gn_2wl=False,
                 act_1wl0 = True,
                 act_1wl1 = True,
                 act_2wl=True,
                 reduce_feat=True,
                 ):
        super().__init__()

        use_affine = False

        relu_lin = lambda a, b, dp, lnx, actx: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity())

        relu_conv = lambda insize, outsize, dp, lnx, gnx, act: Seq([
            GCNConv(insize, outsize),
            nn.LayerNorm(outsize, elementwise_affine=use_affine) if lnx else nn.Identity(),
            GraphNorm(outsize) if gnx else nn.Identity(),
            Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity()
        ])

        self.max_x = max_x
        self.use_node_feat = use_node_feat
        self.node_feat = node_feat
        if use_node_feat:
            if reduce_feat:
                self.lin1 = nn.Sequential(
                    nn.Dropout(dp_lin0),
                    relu_lin(node_feat.shape[-1], hidden_dim_1, dp_lin1, True, False)
                )
            else:
                self.lin1 = nn.Identity()
        else:
            self.emb = nn.Sequential(nn.Embedding(max_x + 1, hidden_dim_1),
                                     nn.LayerNorm(hidden_dim_1, elementwise_affine=use_affine) if ln_emb else nn.Identity(),
                                     GraphNorm(hidden_dim_1) if gn_emb else nn.Identity(),
                                     Dropout(p=dp_emb, inplace=True))

        self.conv1s = nn.ModuleList(
            [relu_conv(hidden_dim_1, hidden_dim_1, dp_1wl0, ln_1wl0, gn_1wl0, act_1wl0) for _ in range(layer1 - 1)] +
            [relu_conv(hidden_dim_1, hidden_dim_2, dp_1wl1, ln_1wl1, gn_1wl1, act_1wl1)])

        self.conv2s = nn.ModuleList(
            [relu_conv(hidden_dim_2, hidden_dim_2, dp_2wl, ln_2wl, gn_2wl, act_2wl) for _ in range(layer2)])
        self.conv2s_r = nn.ModuleList(
            [relu_conv(hidden_dim_2, hidden_dim_2, dp_2wl, ln_2wl, gn_2wl, act_2wl) for _ in range(layer2)])
        self.pred = nn.Linear(hidden_dim_2, 1)

    def forward(self, x, edge1, pos, idx = None, ei2 = None, test = False):
        edge2, edge2_r = reverse(ei2)

        #import pdb
        #pdb.set_trace()
        x = self.lin1(self.node_feat) if self.use_node_feat else self.emb(x).squeeze()
        for conv1 in self.conv1s:
            x = conv1(x, edge1)

        x = x[pos[:, 0]] * x[pos[:, 1]]
        for i in range(len(self.conv2s)):
            x = self.conv2s[i](x, edge2) + self.conv2s_r[i](x, edge2_r)
        x = x[idx]
        mask = torch.cat(
            [torch.ones([1, x.shape[0] // 2], dtype=bool),
             torch.zeros([1, x.shape[0] // 2], dtype=bool)]).t().reshape(-1)
        x = x[mask] * x[~mask]
        x = self.pred(x)
        return x

class FWLNet(nn.Module):
    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 hidden_dim_1=20,
                 hidden_dim_2=20,
                 layer1=2,
                 layer2=1,
                 act_1wl0=True,
                 act_1wl1=True,
                 dp_emb=0.0,
                 dp_1wl0=0.0,
                 dp_1wl1=0.0,
                 dp_2wl0=0.0,
                 dp_2wl1=0.0,
                 ln_1wl0=False,
                 ln_1wl1=False,
                 gn_1wl0=True,
                 gn_1wl1=True,
                 mul_pool=True,
                 use_ea=False,
                 easize=None,):
        super(FWLNet, self).__init__()
        self.mul_pool = mul_pool
        self.use_ea = use_ea
        self.use_feat = use_feat
        self.layer1 = layer1
        self.layer2 = layer2
        input_node_size = hidden_dim_1
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)

        self.embedding = nn.Sequential(nn.Embedding(max_x + 1, hidden_dim_1),
                                       nn.Dropout(p=dp_emb))
        relu_sage = lambda a, b, dp, ln, gn, act: Seq([
            GCNConv(a, b),
            nn.LayerNorm(b) if ln else nn.Identity(),
            GraphNorm(b) if gn else nn.Identity(),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity()
        ])
        relu_sage_end = lambda a, b, dp, ln, gn, act: Seq([
            GCNConv(a, b),
            nn.LayerNorm(b) if ln else nn.Identity(),
            GraphNorm(b) if gn else nn.Identity(),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True) if act else nn.Identity()
        ])
        if True:
            self.nconvs = nn.ModuleList(
                [relu_sage(input_node_size, hidden_dim_1, dp_1wl0, ln_1wl0, gn_1wl0, act_1wl0)] + [
                    relu_sage_end(hidden_dim_1, hidden_dim_1, dp_1wl1, ln_1wl1, gn_1wl1, act_1wl1)
                    for i in range(layer1 - 1)
                ])

        input_edge_size = hidden_dim_1
        if use_ea:
            input_edge_size += easize.shape[1]

        relu_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_1 = nn.ModuleList(
            [relu_lin(input_edge_size + 1, hidden_dim_2, dp_2wl0)] + [
                relu_lin(hidden_dim_2, hidden_dim_2, dp_2wl0)
                for i in range(layer2 - 1)
            ])
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size + 1, hidden_dim_2, dp_2wl0)] + [
                relu_lin(hidden_dim_2, hidden_dim_2, dp_2wl0)
                for i in range(layer2 - 1)
            ])
        relu_norm_lin = lambda a, b, dp: nn.Sequential(
            nn.Linear(a, b), GraphNorm(b), nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True))
        self.mlps_3 = nn.ModuleList(
            [relu_norm_lin(hidden_dim_2 + input_edge_size + 1, hidden_dim_2, dp_2wl1)] +
            [
                relu_norm_lin(hidden_dim_2 * 2, hidden_dim_2, dp_2wl1)
                for i in range(layer2 - 1)
            ])

        self.lin_dir = nn.Linear(hidden_dim_2, 1)

    def forward(self, x, ei, pos, ei2=None, test=False):
        edge_index = ei
        x = self.embedding(x)
        if self.use_feat:
            x = torch.cat((x, self.feat), dim=1)
        n = x.shape[0]
        for i in range(self.layer1):
            x = self.nconvs[i](x, edge_index)
        colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)
        rowx = x.unsqueeze(1).expand(-1, n, -1).reshape(n * n, -1)
        x = rowx * colx
        if self.use_ea:
            print("ERROR")
            exit(0)

        eim = torch.zeros((n*n,), device = x.device)
        eim[ei[0] * n + ei[1]] = 1
        eim = eim.reshape(n, n)
        add_chan = torch.eye(n, device = x.device)

        x = x.reshape(n, n, -1)
        for i in range(1):
            add_chan = torch.mm(add_chan, eim)
            x = torch.cat((x, add_chan.reshape(n, n, -1)), dim=-1)
        #x = localize(x, ei)
        '''
        nl = torch.eye(n, device=x.device).unsqueeze(-1)
        x = torch.cat((x, nl), dim=-1)
        '''
        for i in range(self.layer2):
            #xx = deepcopy(x)
            x1 = self.mlps_1[i](x).permute(2, 0, 1)
            x2 = self.mlps_2[i](x).permute(2, 0, 1)
            x = torch.cat([x, (x1 @ x2).permute(1, 2, 0)], -1)
            x = self.mlps_3[i](x)
        x = (x * x.permute(1, 0, 2)).reshape(n * n, -1) if self.mul_pool else (x + x.permute(1, 0, 2)).reshape(n * n, -1)
        import pdb
        #pdb.set_trace()
        x = x[pos[:, 0] * n + pos[:, 1]]
        x = self.lin_dir(x)
        return x

class LocalFWLNet(nn.Module):
    def __init__(self,
                 max_x,
                 use_feat=False,
                 feat=None,
                 use_degree=True,
                 #use_degree=False,
                 use_appnp=False,
                 #use_appnp=True,
                 reduce_feat=False,
                 #reduce_feat=True,
                 sum_pooling=False,
                 #sum_pooling=True,
                 hidden_dim_1=20,
                 hidden_dim_2=20,
                 layer1=2,
                 layer2=1,
                 layer3=1,
                 dp_emb=0.0,
                 dp_lin0=0.0,
                 dp_lin1=0.0,
                 dp_1wl=0.0,
                 dp_2wl0=0.0,
                 dp_2wl1=0.0,
                 alpha=0.1,
                 ln_lin=False,
                 ln_1wl=False,
                 ln_2wl0=False,
                 ln_2wl1=False,
                 gn_lin=False,
                 gn_1wl=True,
                 gn_2wl1=True,
                 gn_app=False,
                 act_lin=False,
                 act_1wl=True,
                 act_2wl0=True,
                 act_2wl1=True,
                 fast_bsmm = False,
                 use_ea=False,
                 easize=None):
        super(LocalFWLNet, self).__init__()
        assert use_feat or use_degree
        self.use_ea = use_ea
        self.max_x = max_x
        self.use_feat = use_feat
        self.use_degree = use_degree
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.fast = fast_bsmm
        self.sum_pooling = sum_pooling
        self.appnp = use_appnp
        relu_sage = lambda a, b, dp, lnx, gnx, actx: Seq([
            GCNConv(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            GraphNorm(b) if gnx else nn.Identity(),
            nn.Dropout(dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity()
        ])
        relu_lin = lambda a, b, dp, lnx, gnx, actx: nn.Sequential(
            nn.Linear(a, b),
            nn.LayerNorm(b, elementwise_affine=use_affine) if lnx else nn.Identity(),
            GraphNorm(b) if gnx else nn.Identity(),
            nn.Dropout(p=dp, inplace=True),
            nn.ReLU(inplace=True) if actx else nn.Identity())
        use_affine = False
        input_node_size = hidden_dim_1 if use_degree else 0
        if use_feat:
            input_node_size += feat.shape[1]
            self.feat = nn.parameter.Parameter(feat, requires_grad=False)
        self.embedding = nn.Sequential(
            nn.Embedding(max_x + 1, hidden_dim_1),
            nn.Dropout(p=dp_emb))
        if not self.appnp:
            self.nconvs = nn.ModuleList(
                [relu_sage(input_node_size, hidden_dim_1, dp_1wl, ln_1wl, gn_1wl, act_1wl)] + [
                    relu_sage(hidden_dim_1, hidden_dim_1, dp_1wl, ln_1wl, gn_1wl, act_1wl)
                    for _ in range(layer1 - 1)
                ])
        else:
            self.nconvs = APPNP(layer1, alpha)
            self.gn_app = GraphNorm(hidden_dim_1) if gn_app else nn.Identity()
        if reduce_feat:
            assert self.use_feat
            self.lin1 = nn.Sequential(
                nn.Dropout(dp_lin0),
                relu_lin(input_node_size, hidden_dim_1, dp_lin1, ln_lin, gn_lin, act_lin)
            )
        else:
            self.lin1 = nn.Identity()
        input_edge_size = hidden_dim_1

        self.mlps_1 = relu_lin(input_edge_size * 2, hidden_dim_2, dp_2wl0, ln_2wl0, False, act_2wl0)
        self.mlps_2 = nn.ModuleList(
            [relu_lin(input_edge_size * 2, hidden_dim_2, dp_2wl0, ln_2wl0, False, act_2wl0)] + [
             relu_lin(input_edge_size * 2, hidden_dim_2, dp_2wl0, ln_2wl0, False, act_2wl0)
             for _ in range(layer2 - 1)
            ])
        self.mlps_3 = nn.ModuleList(
            [relu_lin(hidden_dim_2 + 1, hidden_dim_2, dp_2wl1, ln_2wl1, gn_2wl1, act_2wl1)
             for _ in range(layer3)
            ])
        if not sum_pooling:
            self.lin_dir = nn.Linear(hidden_dim_1 + hidden_dim_2, 1)

    def forward(self, x, ei, pos, ei2=None, test=False):
        edge_index = ei
        #pos = pos1[pos2][:, 0].reshape(-1, 2)
        n = x.shape[0]

        x = self.embedding(x)
        if self.use_feat:
            x = torch.cat((x, self.feat), dim=1)
            if not self.use_degree:
                x = self.feat
        x = self.lin1(x)
        if not self.appnp:
            for i in range(self.layer1):
                x = self.nconvs[i](x, edge_index)
        else:
            x = self.nconvs(x, edge_index)
            x = self.gn_app(x)
        xx = x[pos[:, 0]] * x[pos[:, 1]]

        val = torch.cat([x[edge_index[0]], x[edge_index[1]]], 1)#colx = x.unsqueeze(0).expand(n, -1, -1).reshape(n * n, -1)

        x = val.clone()
        x = self.mlps_1(x)
        current_edges = edge_index

        for i in range(self.layer3):
            if i < self.layer2:
                mul = self.mlps_2[i](val)
            x = sparse_bmm(current_edges, x, edge_index, mul, n, fast=self.fast)
            current_edges, value = sparse_cat(x, edge_index, torch.ones((edge_index.shape[1], 1), device=x.device))
            x = self.mlps_3[i](value)
        sm = torch.sparse.FloatTensor(torch.cat([current_edges[1].unsqueeze(0), current_edges[0].unsqueeze(0)], 0), x,
                                      torch.Size([n, n, x.shape[-1]])).coalesce().values()
        x = x * sm
        x = add_zero(x, pos.t().cpu().numpy(), current_edges)
        pred_list = edge_list(current_edges, pos.t(), n)
        x = x[pred_list]
        x = torch.cat([x, xx], 1)
        x = self.lin_dir(x) if not self.sum_pooling else torch.sum(x, dim=-1, keepdim=True)
        return x


class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

def mataggr(A, h, g):
    '''
    A (n, n, d). n is number of node, d is latent dimension
    h, g are mlp
    '''
    B = h(A)
    #C = f(A)
    n, d = A.shape[0], A.shape[1]
    vec_p = (torch.sum(B, dim=1, keepdim=True)).expand(-1, n, -1)
    vec_q = (torch.sum(B, dim=0, keepdim=True)).expand(n, -1, -1)
    D = torch.cat([A, vec_p, vec_q], -1)
    return g(D)