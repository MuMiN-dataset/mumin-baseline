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
from dgl.dataloading.neighbor import MultiLayerFullNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
import dgl
import logging
import json
from typing import Tuple
import datetime as dt


logger = logging.getLogger(__name__)


def train(num_epochs: int,
          hidden_dim: int,
          input_dropout: float = 0.0,
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
        input_dropout (float, optional):
            The amount of dropout of the inputs. Defaults to 0.0.
        dropout (float, optional):
            The amount of dropout of the hidden layers. Defaults to 0.0.
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
                  input_dropout=input_dropout,
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
        graph = load_mumin_graph(size=size)

        # Ensure that Boolean tensors are not present in the graph, as saving
        # fails in that case
        for ntype in ['claim', 'tweet']:
            for split in ['train', 'val', 'test']:
                split_tensor = graph.nodes[ntype].data[f'{split}_mask']
                graph.nodes[ntype].data[f'{split}_mask'] = split_tensor.int()

        # Save graph to disk
        save_graphs(str(graph_path), [graph])

    # Store labels and masks
    train_mask = graph.nodes[task].data['train_mask'].bool()
    val_mask = graph.nodes[task].data['val_mask'].bool()

    # Initialise dictionary with feature dimensions
    dims = {ntype: graph.nodes[ntype].data['feat'].shape[-1]
            for ntype in graph.ntypes}
    feat_dict = {rel: (dims[rel[0]], hidden_dim, dims[rel[2]])
                 for rel in graph.canonical_etypes}

    # Initialise model
    model = HeteroGraphSAGE(input_dropout=input_dropout,
                            dropout=dropout,
                            feat_dict=feat_dict)
    model.to(device)
    model.train()

    # Set up the training and validation node IDs, being the node indexes where
    # `train_mask` and `val_mask` is True, respectively
    node_enum = torch.arange(graph.num_nodes(task))
    train_nids = {task: node_enum[train_mask].int()}
    val_nids = {task: node_enum[val_mask].int()}

    # Set up the sampler
    sampler = MultiLayerFullNeighborSampler(n_layers=2)

    # Set up the dataloaders
    train_dataloader = NodeDataLoader(g=graph,
                                      nids=train_nids,
                                      block_sampler=sampler,
                                      batch_size=2048,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=8)
    val_dataloader = NodeDataLoader(g=graph,
                                    nids=val_nids,
                                    block_sampler=sampler,
                                    batch_size=2048,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=8)

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
    scorer = tm.F1(num_classes=2, average='none').to(device)

    # Initialise best validation loss
    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        # Reset metrics
        train_loss = 0.0
        train_misinformation_f1 = 0.0
        train_factual_f1 = 0.0
        val_loss = 0.0
        val_misinformation_f1 = 0.0
        val_factual_f1 = 0.0

        # Train model
        for _, _, blocks in train_dataloader:

            # Reset the gradients
            opt.zero_grad()

            # Ensure that `blocks` are on the correct device
            blocks = [block.to(device) for block in blocks]

            # Get the input features and the output labels
            input_feats = {n: feat.float()
                           for n, feat in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['label'][task].to(device)

            # Forward propagation
            logits = model(blocks, input_feats)
            logits = logits[task].squeeze(1)

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(
                input=logits,
                target=output_labels.float(),
                pos_weight=pos_weight_tensor
            )

            # Compute training metrics
            scores = scorer(logits.ge(0), output_labels)
            misinformation_f1 = scores[0]
            factual_f1 = scores[1]

            # Backward propagation
            loss.backward()

            # Update gradients
            opt.step()

            # Store the training metrics
            train_loss += float(loss)
            train_misinformation_f1 += float(misinformation_f1)
            train_factual_f1 += float(factual_f1)

        # Divide the training metrics by the number of batches
        train_loss /= len(train_dataloader)
        train_misinformation_f1 /= len(train_dataloader)
        train_factual_f1 /= len(train_dataloader)

        # Evaluate model
        for _, _, blocks in val_dataloader:
            with torch.no_grad():

                # Ensure that `blocks` are on the correct device
                blocks = [block.to(device) for block in blocks]

                # Get the input features and the output labels
                input_feats = {n: f.float()
                               for n, f in blocks[0].srcdata['feat'].items()}
                output_labels = blocks[-1].dstdata['label'][task].to(device)

                # Forward propagation
                logits = model(blocks, input_feats)
                logits = logits[task].squeeze(1)

                # Compute validation loss
                loss = F.binary_cross_entropy_with_logits(
                    input=logits,
                    target=output_labels.float(),
                    pos_weight=pos_weight_tensor
                )

                # Compute validation metrics
                scores = scorer(logits.ge(0), output_labels)
                misinformation_f1 = scores[0]
                factual_f1 = scores[1]

                # Store the validation metrics
                val_loss += float(loss)
                val_misinformation_f1 += float(misinformation_f1)
                val_factual_f1 += float(factual_f1)

        # Divide the validation metrics by the number of batches
        val_loss /= len(val_dataloader)
        val_misinformation_f1 /= len(val_dataloader)
        val_factual_f1 /= len(val_dataloader)

        # Gather statistics to be logged
        stats = [
            ('train_loss', train_loss),
            ('train_misinformation_f1', train_misinformation_f1),
            ('train_factual_f1', train_factual_f1),
            ('val_loss', val_loss),
            ('val_misinformation_f1', val_misinformation_f1),
            ('val_factual_f1', val_factual_f1),
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

        # Update learning rate
        scheduler.step(val_loss)


if __name__ == '__main__':
    config = dict(num_epochs=10_000,
                  hidden_dim=1024,
                  input_dropout=0.0,
                  dropout=0.0,
                  size='small',
                  task='claim',
                  initial_lr=2e-5,
                  lr_factor=0.8,
                  lr_patience=20,
                  betas=(0.8, 0.998),
                  pos_weight=10.)
    train(**config)
