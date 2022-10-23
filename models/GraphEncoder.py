import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class RGTLayer(nn.Module):
    def __init__(self, d_model, n_head=1, drop=0.1):
        super(RGTLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.msg_fc = nn.Linear(self.d_model * 2, self.n_head * self.d_model, bias=False)

        self.qw = nn.Linear(self.d_model, self.n_head * self.d_model, bias=False)
        self.kw = nn.Linear(self.d_model * 2, self.n_head * self.d_model, bias=False)
        self.temp = self.d_model ** 0.5

        self.output_fc = nn.Linear(self.n_head * self.d_model, self.d_model, bias=False)

        self.layer_norm1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.dropout = nn.Dropout(drop)

        nn.init.xavier_uniform_(self.qw.weight)
        nn.init.xavier_uniform_(self.kw.weight)
        nn.init.xavier_uniform_(self.msg_fc.weight)

    def msg_func(self, edges):
        msg = self.msg_fc(torch.cat([edges.src['h'], edges.data['h']], dim=-1))
        msg = F.leaky_relu(msg)
        msg = self.dropout(msg)
        q = self.qw(edges.data['qrh']) / self.temp
        k = self.kw(torch.cat([edges.src['h'], edges.data['h']], dim=-1))
        msg = msg.view(-1, self.n_head, self.d_model)
        q = q.view(-1, self.n_head, self.d_model)
        k = k.view(-1, self.n_head, self.d_model)
        att = torch.sum(q * k, dim=-1).unsqueeze(-1)  # [-1, n_head, 1]
        return {'msg': msg, 'att': att}

    def reduce_func(self, nodes):
        res = nodes.data['h']
        alpha = self.dropout(F.softmax(nodes.mailbox['att'], dim=1))
        h = torch.sum(alpha * nodes.mailbox['msg'], dim=1).view(-1, self.n_head * self.d_model)
        h = self.dropout(F.leaky_relu(self.output_fc(h)))
        # print(nodes.data['type'])
        # print(alpha.squeeze(-1))

        h = h + res
        h = self.layer_norm1(h)
        return {'h': h}

    def forward(self, g):
        g.update_all(self.msg_func, self.reduce_func)
        return g

class RGTEncoder(nn.Module):
    def __init__(self, d_model, drop=0.1, n_head=4):
        super(RGTEncoder, self).__init__()
        self.layer1 = RGTLayer(d_model, n_head, drop)
        self.layer2 = RGTLayer(d_model, n_head, drop)
        # self.layer3 = RGTLayer(d_model, n_head, drop)
        # self.layer4 = RGTLayer(d_model, n_head, drop)

    def forward(self, g):
        self.layer1(g)
        self.layer2(g)
        # self.layer3(g)
        # self.layer4(g)
        return g.ndata['h']


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=False,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.self_loop = self_loop
        self.activation = activation
        self.out_feat = out_feat

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))


        if self.bias:
            self.bias_parm = nn.Parameter(torch.zeros(out_feat))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h']}

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            loop_message = self.dropout(loop_message)

        self.propagate(g)

        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias_parm
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        return g

class RGCNEncoder(nn.Module):
    def __init__(self, ent_dim, num_rels, num_bases, dropout=0.0):
        super(RGCNEncoder, self).__init__()
        self.layer1 = RGCNLayer(ent_dim, ent_dim, num_rels, num_bases, True, torch.nn.functional.relu, True, dropout)
        self.layer2 = RGCNLayer(ent_dim, ent_dim, num_rels, num_bases, True, None, True, dropout)

    def forward(self, g):
        self.layer1(g)
        self.layer2(g)
        return g.ndata['h']
