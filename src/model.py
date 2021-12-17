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
                           out_feats=feats[1],
                           activation=F.relu,
                           dropout=input_dropout)
             for rel, feats in feat_dict.items()},
            aggregate='max')

        self.conv2 = HeteroGraphConv(
            {rel: SAGEConv(in_feats=feats[1],
                           out_feats=feats[1],
                           dropout=dropout)
             for rel, feats in feat_dict.items()},
            aggregate='max')

        self.conv3 = HeteroGraphConv(
            {rel: SAGEConv(in_feats=feats[1], out_feats=1, dropout=dropout)
             for rel, feats in feat_dict.items()},
            aggregate='max')

    def forward(self, blocks, input_dict: dict) -> dict:
        h_dict = self.conv1(blocks[0], input_dict)
        h_dict = self.conv2(blocks[1], h_dict)
        h_dict = self.conv3(blocks[2], h_dict)
        return h_dict


class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 dropout: float,
                 activation: Optional[Callable] = None):
        super().__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.proj_dst = nn.Linear(self._in_dst_feats, out_feats)
        self.fc = nn.Linear(self._in_src_feats + self._in_dst_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.activation = (lambda x: x) if activation is None else activation

    @staticmethod
    def _reduce(nodes):
        messages = nodes.mailbox['m']
        breakpoint()
        return {'neigh': messages.max(dim=1)[0]}

    def forward(self, graph, feat):
        h_src, h_dst = expand_as_pair(feat)

        graph.srcdata['h'] = h_src
        graph.update_all(message_func=dglfn.copy_u('h', 'm'),
                         reduce_func=self._reduce)
        h_neigh = graph.dstdata['neigh']

        h = torch.cat((h_dst, h_neigh), dim=-1)
        h = self.dropout(h)
        rst = self.fc(h)
        return self.proj_dst(h_dst) + self.activation(rst)
