'''Implementation of a heterogeneous GraphSAGE model'''

import dgl.function as dglfn
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
                 feat_dict: Dict[Tuple[str, str, str],
                                 Tuple[int, int, int]]):
        super().__init__()
        self.feat_dict = feat_dict

        self.conv1 = HeteroGraphConv(
            {rel: SAGEConv(in_feats=(feats[0], feats[2]),
                           hidden_feats=feats[1],
                           out_feats=feats[1],
                           activation=F.gelu,
                           input_dropout=input_dropout,
                           dropout=dropout)
             for rel, feats in feat_dict.items()},
            aggregate='sum')

        self.conv2 = HeteroGraphConv(
            {rel: SAGEConv(in_feats=feats[1],
                           hidden_feats=feats[1],
                           out_feats=feats[1],
                           activation=F.gelu,
                           input_dropout=dropout,
                           dropout=dropout)
             for rel, feats in feat_dict.items()},
            aggregate='sum')

        self.conv3 = HeteroGraphConv(
            {rel: SAGEConv(in_feats=feats[1],
                           hidden_feats=feats[1],
                           out_feats=1,
                           input_dropout=dropout,
                           dropout=dropout)
             for rel, feats in feat_dict.items()},
            aggregate='sum')

    def forward(self, blocks, input_dict: dict) -> dict:
        h_dict = self.conv1(blocks[0], input_dict)
        h_dict = self.conv2(blocks[1], h_dict)
        h_dict = self.conv3(blocks[2], h_dict)
        return h_dict


class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats: int,
                 hidden_feats: int,
                 out_feats: int,
                 input_dropout: float,
                 dropout: float,
                 activation: Optional[Callable] = None):
        super().__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.proj_src = nn.Linear(self._in_src_feats, hidden_feats)
        self.proj_dst = nn.Linear(self._in_dst_feats, hidden_feats)
        self.norm_src = nn.LayerNorm(hidden_feats)
        self.norm_dst = nn.LayerNorm(hidden_feats)
        self.norm_concat = nn.LayerNorm(hidden_feats)
        self.fc = nn.Linear(2 * hidden_feats, out_feats)
        self.input_dropout = nn.Dropout(input_dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = (lambda x: x) if activation is None else activation

    def _message(self, edges):
        src_feats = edges.src['h']
        src_feats = self.input_dropout(src_feats)
        src_feats = self.proj_src(src_feats)
        src_feats = F.gelu(src_feats)
        src_feats = self.norm_src(src_feats)
        return {'m': src_feats}

    def _reduce(self, nodes):
        messages = nodes.mailbox['m']
        return {'neigh': messages.sum(dim=1)}

    def _apply_node(self, nodes):
        h_dst = nodes.data['h']
        h_dst = self.proj_dst(h_dst)
        h_dst = F.gelu(h_dst)
        h_dst = self.norm_dst(h_dst)
        h_neigh = nodes.data['neigh']

        h = torch.cat((h_dst, h_neigh), dim=-1)
        h = self.dropout(h)
        h = self.fc(h)
        h = self.activation(h)
        h = self.norm_concat(h)
        return {'h':  h}

    def forward(self, graph, feat):
        h_src, h_dst = expand_as_pair(feat)

        graph.srcdata['h'] = h_src
        graph.dstdata['h'] = h_dst
        graph.update_all(message_func=self._message,
                         reduce_func=self._reduce,
                         apply_node_func=self._apply_node)
        return graph.dstdata['h']
