'''Implementation of a heterogeneous GraphSAGE model'''

import dgl.nn.pytorch as dglnn
import dgl.function as dglfn
from dgl.utils import expand_as_pair
from dgl import DGLError

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple, Optional, Callable
from functools import partial


class HeteroGraphSAGE(nn.Module):
    def __init__(self, feat_dict: Dict[Tuple[str, str, str],
                                       Tuple[int, int, int]]):
        super().__init__()
        self.feat_dict = feat_dict

        self.conv1 = HeteroGraphConv(
            {rel[1]: DivFeatConv(in_feats=(feats[0], feats[2]),
                                 out_feats=feats[1],
                                 activation=F.relu)
             for rel, feats in feat_dict.items()},
            aggregate='sum')

        print(self.conv1)

        self.conv2 = HeteroGraphConv(
            {rel[1]: DivFeatConv(in_feats=feats[1], out_feats=1)
             for rel, feats in feat_dict.items()},
            aggregate='sum')

    def forward(self, g, input_dict: dict) -> dict:
        h_dict = self.conv1(g, (input_dict, input_dict))
        h_dict = self.conv2(g, (h_dict, h_dict))
        return h_dict


class DivFeatConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation: Optional[Callable] = None):
        super().__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)

        self.activation = (lambda x: x) if activation is None else activation

    def forward(self, graph, feat):
        h_src, h_dst = expand_as_pair(feat)
        print(self._in_src_feats, self._in_dst_feats, self._out_feats)

        breakpoint()

        graph.srcdata['h'] = h_src
        graph.update_all(dglfn.copy_u('h', 'm'), dglfn.mean('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']

        print(h_dst.shape, h_neigh.shape)
        print(self.fc_self, self.fc_neigh)
        print(self.fc_self(h_dst).shape, self.fc_neigh(h_neigh).shape)

        rst = self.fc_self(h_dst) + self.fc_neigh(h_neigh)
        return self.activation(rst)


class HeteroGraphConv(nn.Module):
    def __init__(self, mods, aggregate='sum'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict({'_'.join(etype): val
                                   for etype, val in mods.items()})
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v,
                                                  'set_allow_zero_in_degree',
                                                  None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLHeteroGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[f'{stype}_{etype}_{dtype}'](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[f'{stype}_{etype}_{dtype}'](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts

def _max_reduce_func(inputs, dim):
    return torch.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return torch.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return torch.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return torch.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return torch.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, fn): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = torch.stack(inputs, dim=0)
    return fn(stacked, dim=0)

def get_aggregate_fn(agg):
    """Internal function to get the aggregation function for node data
    generated from different relations.

    Parameters
    ----------
    agg : str
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.

    Returns
    -------
    callable
        Aggregator function that takes a list of tensors to aggregate
        and returns one aggregated tensor.
    """
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        raise DGLError('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)
