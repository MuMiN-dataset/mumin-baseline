'''Implementation of a heterogeneous GraphSAGE model'''

from dgl.utils import expand_as_pair
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable

from heterographconv import HeteroGraphConv


class HeteroGraphSAGE(nn.Module):
    def __init__(self,
                 input_dropout: float,
                 dropout: float,
                 hidden_dim: int,
                 feat_dict: Dict[Tuple[str, str, str],
                                 Tuple[int, int, int]],
                 task: str = 'claim'):
        super().__init__()
        self.feat_dict = feat_dict
        self.hidden_dim = hidden_dim
        self.task = task

        self.conv1 = HeteroGraphConv(
            {rel: SAGEConv(in_feats=(feats[0], feats[1]),
                           out_feats=hidden_dim,
                           activation=F.gelu,
                           input_dropout=input_dropout,
                           dropout=dropout)
             for rel, feats in feat_dict.items()},
            aggregate='sum')

        self.conv2 = HeteroGraphConv(
            {rel: SAGEConv(in_feats=hidden_dim,
                           out_feats=hidden_dim,
                           activation=F.gelu,
                           input_dropout=dropout,
                           dropout=dropout)
             for rel, _ in feat_dict.items()},
            aggregate='sum')

        # self.conv3 = HeteroGraphConv(
        #     {rel: SAGEConv(in_feats=hidden_dim,
        #                    out_feats=hidden_dim,
        #                    activation=F.gelu,
        #                    input_dropout=dropout,
        #                    dropout=dropout)
        #      for rel, _ in feat_dict.items()},
        #     aggregate='sum')

        self.clf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    # def _agg_func(self, inputs, dsttype):
    #     if len(inputs) == 0:
    #         return None
    #     stacked = torch.stack(inputs, dim=0)
    #     return fn(stacked, dim=0)

    def forward(self, blocks, h_dict: dict) -> dict:
        h_dict = self.conv1(blocks[0], h_dict)
        h_dict = {k: self.norm(v) for k, v in h_dict.items()}
        h_dict = self.conv2(blocks[1], h_dict)
        h_dict = {k: self.norm(v) for k, v in h_dict.items()}
        #h_dict = self.conv3(blocks[2], h_dict)
        #h_dict = {k: self.norm(v) for k, v in h_dict.items()}
        return self.clf(h_dict[self.task])


class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 input_dropout: float,
                 dropout: float,
                 activation: Optional[Callable] = None):
        super().__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.src_fc = nn.Linear(self._in_src_feats, self._in_src_feats)
        self.fc = nn.Linear(self._in_src_feats + self._in_dst_feats, out_feats)
        self.input_dropout = nn.Dropout(input_dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = (lambda x: x) if activation is None else activation
        # self.norm_concat = ((lambda x: x)
        #                     if activation is None
        #                     else nn.LayerNorm(out_feats))

    def _message(self, edges):
        src_feats = edges.src['h']
        src_feats = self.input_dropout(src_feats)
        src_feats = self.src_fc(src_feats)
        return {'m': src_feats}

    def _reduce(self, nodes):
        messages = nodes.mailbox['m']
        return {'neigh': messages.mean(dim=1)}

    def _apply_node(self, nodes):
        h_dst = nodes.data['h']
        h_neigh = nodes.data['neigh']
        h = torch.cat((h_dst, h_neigh), dim=-1)
        h = self.dropout(h)
        h = self.fc(h)
        h = self.activation(h)
        # h = self.norm_concat(h)
        return {'h':  h}

    def forward(self, graph, feat):
        h_src, h_dst = expand_as_pair(feat)

        graph.srcdata['h'] = h_src
        graph.dstdata['h'] = h_dst
        graph.update_all(message_func=self._message,
                         reduce_func=self._reduce,
                         apply_node_func=self._apply_node)
        return graph.dstdata['h']
