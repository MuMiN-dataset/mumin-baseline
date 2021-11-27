'''Training scripts'''

from data import load_mumin_graph
from model import HeteroGraphSAGE

from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics as tm
from dgl.data.utils import save_graphs, load_graphs
import dgl
import logging
from termcolor import colored
import json
from typing import Tuple
import datetime as dt


logger = logging.getLogger(__name__)


def train(num_epochs: int,
          hidden_dim: int,
          dropout: float = 0.0,
          size: str = 'small',
          task: str = 'claim',
          initial_lr: float = 5e-5,
          lr_factor: float = 0.8,
          lr_patience: int = 10,
          betas: Tuple[float, float] = (0.8, 0.998),
          pos_weight: float = 20.):
    '''Train a heterogeneous GraphConv model on the MuMiN dataset.

    Args:
        num_epochs (int):
            The number of epochs to train for.
        hidden_dim (int):
            The dimension of the hidden layer.
        dropout (float, optional):
            The amount of dropout. Defaults to 0.0.
        size (str, optional):
            The size of the dataset to use. Defaults to 'small'.
        task (str, optional):
            The task to consider, which can be either 'tweet' or 'claim',
            corresponding to doing thread-level or claim-level node
            classification. Defaults to 'claim'.
        initial_lr (float, optional):
            The initial learning rate. Defaults to 5e-5.
        lr_factor (float, optional):
            The factor by which to reduce the learning rate. Defaults to 0.8.
        lr_patience (int, optional):
            The number of epochs to wait before reducing the learning rate.
            Defaults to 10.
        betas (Tuple[float, float], optional):
            The coefficients for the Adam optimizer. Defaults to (0.8, 0.998).
        pos_weight (float, optional):
            The weight to give to the positive examples. Defaults to 20.
    '''
    # Set random seeds
    torch.manual_seed(4242)
    dgl.seed(4242)

    # Set config
    config = dict(hidden_dim=hidden_dim,
                  dropout=dropout,
                  size=size,
                  task=task,
                  initial_lr=initial_lr,
                  lr_factor=lr_factor,
                  lr_patience=lr_patience,
                  betas=betas,
                  pos_weight=pos_weight)

    # Set up PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up graph path
    graph_path = Path(f'dgl-graph-{size}.bin')

    if graph_path.exists():
        # Load the graph
        graph = load_graphs(str(graph_path))[0][0]

    else:
        # Load dataset
        graph = load_mumin_graph(size=size).to(device)

        # Save graph to disk
        save_graphs(str(graph_path), [graph])

    # Store labels and masks
    labels = graph.nodes[task].data['label']
    train_mask = graph.nodes[task].data['train_mask'].bool()
    val_mask = graph.nodes[task].data['val_mask'].bool()

    # Store node features
    feats = {node_type: graph.nodes[node_type].data['feat'].float()
             for node_type in graph.ntypes}

    # Initialise dictionary with feature dimensions
    dims = {ntype: graph.nodes[ntype].data['feat'].shape[-1]
            for ntype in graph.ntypes}
    feat_dict = {rel: (dims[rel[0]], hidden_dim, dims[rel[2]])
                 for rel in graph.canonical_etypes}

    # Initialise model
    model = HeteroGraphSAGE(dropout=dropout, feat_dict=feat_dict)
    model.to(device)
    model.train()

    # Set up pos_weight
    pos_weight_tensor = torch.tensor(pos_weight).to(device)

    # Set up path to state dict
    datetime = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    Path('models').mkdir(exist_ok=True)
    model_dir = Path('models') / f'{datetime}-{task}-model-{size}'
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'model.pt'
    config_path = model_dir / 'config.json'

    # Initialise optimiser
    opt = optim.AdamW(model.parameters(), lr=initial_lr, betas=betas)

    # Initialise learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer=opt,
                                  factor=lr_factor,
                                  patience=lr_patience)

    # Initialise scorer
    scorer = tm.F1(num_classes=2, average='none')

    # Initialise best validation loss
    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        # Reset the gradients
        opt.zero_grad()

        # Forward propagation
        logits = model(graph, feats)
        logits = logits[task].squeeze(1)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(logits[train_mask],
                                                  labels[train_mask].float(),
                                                  pos_weight=pos_weight_tensor)

        with torch.no_grad():

            # Compute validation loss
            val_loss = F.binary_cross_entropy_with_logits(
                logits[val_mask],
                labels[val_mask].float(),
                pos_weight=pos_weight_tensor
            )

            # Compute training metrics
            train_scores = scorer(logits[train_mask].ge(0), labels[train_mask])
            train_misinformation_f1 = train_scores[0]
            train_factual_f1 = train_scores[1]

            # Compute validation metrics
            val_scores = scorer(logits[val_mask].ge(0), labels[val_mask])
            val_misinformation_f1 = val_scores[0]
            val_factual_f1 = val_scores[1]

        # Backward propagation
        loss.backward()

        # Update gradients
        opt.step()

        # Update learning rate
        scheduler.step(val_loss)

        # Gather statistics to be logged
        stats = [
            ('train_loss', loss.item()),
            ('train_misinformation_f1', train_misinformation_f1.item()),
            ('train_factual_f1', train_factual_f1.item()),
            ('val_loss', val_loss.item()),
            ('val_misinformation_f1', val_misinformation_f1.item()),
            ('val_factual_f1', val_factual_f1.item()),
            ('learning_rate', opt.param_groups[0]['lr'])
        ]

        # Report and log statistics
        log = f'Epoch {epoch}\n'
        config['epoch'] = epoch
        for statistic, value in stats:
            log += f'> {statistic}: {value}\n'
            config[statistic] = value
        logger.info(log)

        # Save model and config
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(model_path))
            with config_path.open('w') as f:
                json.dump(config, f)


if __name__ == '__main__':
    # Set up logging
    fmt = (colored('%(asctime)s [%(levelname)s] <%(name)s>\n↳ ', 'green') +
           colored('%(message)s', 'yellow'))
    logging.basicConfig(level=logging.INFO, format=fmt)

    # Train the model
    train(num_epochs=10_000, hidden_dim=500, dropout=0.5, task='claim')
