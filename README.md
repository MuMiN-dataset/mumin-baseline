# MuMiN Baselines

This repository contains implementations of baseline models on the MuMiN
dataset, introduced in the following paper:

> Nielsen and McConville: _MuMiN: A Large-Scale Multilingual Multimodal
> Fact-Checked Misinformation Social Network Dataset_ (2022).


## Reproducing paper results

To perform the baselines we have centralised all the training scripts into the `src/train.py` script. This can be called with many different parameters, of which the mandatory ones are the following:

- `model_type`: This picks the type of model you want to benchmark. Can be 'claim', 'tweet', 'image' or 'graph.
- `size`: The size of the MuMiN dataset to perform the benchmark.
- `task`: Only relevant if `model_type=='graph'`, in which case it determines whether you want to benchmark the graph model on the claim classification task or the tweet classification task.

Call `python src/train.py --help` for a more detailed list of all the arguments
that can be used.
