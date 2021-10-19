'''Implementation of a heterogeneous GraphSAGE model'''

import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class HeteroGraphSAGE(nn.Module):
    def __init__(self, feat_dict: Dict[Tuple[str, str, str],
                                       Tuple[int, int, int]]):
        super().__init__()
        self.feat_dict = feat_dict

        self.conv1 = dglnn.HeteroGraphConv(
            {rel[1]: dglnn.GraphConv(*feats[:2])
             for rel, feats in feat_dict.items()},
            aggregate='sum')

        self.conv2 = dglnn.HeteroGraphConv(
            {rel[1]: dglnn.GraphConv(*feats[1:])
             for rel, feats in feat_dict.items()},
            aggregate='sum')

    def forward(self, g, input_dict: dict) -> dict:
        h_dict = self.conv1(g, input_dict)
        h_dict = {rel: F.relu(val) for rel, val in h_dict.items()}
        h_dict = self.conv2(g, h_dict)
        return h_dict
