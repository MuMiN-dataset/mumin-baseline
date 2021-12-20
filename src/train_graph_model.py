'''Training scripts'''

from data import load_mumin_graph
from model import HeteroGraphSAGE

from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import torchmetrics as tm
from dgl.data.utils import save_graphs, load_graphs
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
import dgl
import logging
import json
from typing import Tuple
import datetime as dt
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


def train(num_epochs: int,
          batch_size: int,
          hidden_dim: int,
          input_dropout: float,
          dropout: float,
          size: str,
          task: str,
          lr: float,
          betas: Tuple[float, float],
          pos_weight: float):
    '''Train a heterogeneous GraphConv model on the MuMiN dataset.

    Args:
        num_epochs (int):
            The number of epochs to train for.
        batch_size (int):
            The batch size.
        hidden_dim (int):
            The dimension of the hidden layer.
        input_dropout (float):
            The amount of dropout of the inputs.
        dropout (float):
            The amount of dropout of the hidden layers.
        size (str):
            The size of the dataset to use.
        task (str):
            The task to consider, which can be either 'tweet' or 'claim',
            corresponding to doing thread-level or claim-level node
            classification.
        lr (float):
            The learning rate.
        betas (Tuple[float, float]):
            The coefficients for the Adam optimizer.
        pos_weight (float):
            The weight to give to the positive examples.
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
                  lr=lr,
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
    test_mask = graph.nodes[task].data['test_mask'].bool()

    # Initialise dictionary with feature dimensions
    dims = {ntype: graph.nodes[ntype].data['feat'].shape[-1]
            for ntype in graph.ntypes}
    feat_dict = {rel: (dims[rel[0]], dims[rel[2]])
                 for rel in graph.canonical_etypes}

    # Initialise model
    model = HeteroGraphSAGE(input_dropout=input_dropout,
                            dropout=dropout,
                            hidden_dim=hidden_dim,
                            feat_dict=feat_dict)
    model.to(device)
    model.train()

    # Set up the training and validation node IDs, being the node indexes where
    # `train_mask` and `val_mask` is True, respectively
    node_enum = torch.arange(graph.num_nodes(task))
    train_nids = {task: node_enum[train_mask].int()}
    val_nids = {task: node_enum[val_mask].int()}
    test_nids = {task: node_enum[test_mask].int()}

    # Set up the sampler
    sampler = MultiLayerNeighborSampler([50, 50, 50], replace=False)

    # Set up the dataloaders
    train_dataloader = NodeDataLoader(g=graph,
                                      nids=train_nids,
                                      block_sampler=sampler,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=4)
    val_dataloader = NodeDataLoader(g=graph,
                                    nids=val_nids,
                                    block_sampler=sampler,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=4)
    test_dataloader = NodeDataLoader(g=graph,
                                     nids=test_nids,
                                     block_sampler=sampler,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=4)

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
    opt = optim.AdamW(model.parameters(), lr=lr, betas=betas)

    # Initialise learning rate scheduler
    scheduler = LinearLR(optimizer=opt,
                         start_factor=1.,
                         end_factor=1e-7 / lr,
                         total_iters=100)

    # Initialise scorer
    scorer = tm.F1(num_classes=2, average='none').to(device)

    # Initialise best score
    best_factual_f1 = 0.0

    # Initialise progress bar
    epoch_pbar = tqdm(range(num_epochs), desc='Training')

    for epoch in epoch_pbar:

        # Reset metrics
        train_loss = 0.0
        train_misinformation_f1 = 0.0
        train_factual_f1 = 0.0
        val_loss = 0.0
        val_misinformation_f1 = 0.0
        val_factual_f1 = 0.0

        # Train model
        model.train()
        for _, _, blocks in tqdm(train_dataloader,
                                 desc=f'Epoch {epoch}',
                                 leave=False):

            # Reset the gradients
            opt.zero_grad()

            # Ensure that `blocks` are on the correct device
            blocks = [block.to(device) for block in blocks]

            # Get the input features and the output labels
            input_feats = {n: feat.float()
                           for n, feat in blocks[0].srcdata['feat'].items()}
            output_labels = blocks[-1].dstdata['label'][task].to(device)

            # Forward propagation
            logits = model(blocks, input_feats).squeeze(1)

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
            if misinformation_f1 == misinformation_f1:
                train_misinformation_f1 += float(misinformation_f1)
            if factual_f1 == factual_f1:
                train_factual_f1 += float(factual_f1)

        # Divide the training metrics by the number of batches
        train_loss /= len(train_dataloader)
        train_misinformation_f1 /= len(train_dataloader)
        train_factual_f1 /= len(train_dataloader)

        # Evaluate model
        model.eval()
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
                if misinformation_f1 == misinformation_f1:
                    val_misinformation_f1 += float(misinformation_f1)
                if factual_f1 == factual_f1:
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
        #log = f'Epoch {epoch}\n'
        config['epoch'] = epoch
        for statistic, value in stats:
            #log += f'> {statistic}: {value}\n'
            config[statistic] = value
        #logger.info(log)

        # Update progress bar description
        desc = (f'Training - '
                f'loss {train_loss:.3f} - '
                f'train_f1 {train_factual_f1:.3f} - '
                f'val_loss {val_loss:.3f} - '
                f'val_f1 {val_factual_f1:.3f}')
        epoch_pbar.set_description(desc)

        # Save model and config
        if val_factual_f1 > best_factual_f1:
            best_factual_f1 = val_factual_f1
            torch.save(model.state_dict(), str(model_path))
            with config_path.open('w') as f:
                json.dump(config, f)

        # Update learning rate
        scheduler.step()

    # Close progress bar
    epoch_pbar.close()

    # Load best model
    model.load_state_dict(torch.load(str(model_path)))

    # Reset metrics
    val_loss = 0.0
    val_misinformation_f1 = 0.0
    val_factual_f1 = 0.0
    test_loss = 0.0
    test_misinformation_f1 = 0.0
    test_factual_f1 = 0.0

    # Final evaluation on the validation set
    model.eval()
    for _, _, blocks in tqdm(val_dataloader, desc='Evaluating'):
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
            if misinformation_f1 == misinformation_f1:
                val_misinformation_f1 += float(misinformation_f1)
            if factual_f1 == factual_f1:
                val_factual_f1 += float(factual_f1)

    # Divide the validation metrics by the number of batches
    val_loss /= len(val_dataloader)
    val_misinformation_f1 /= len(val_dataloader)
    val_factual_f1 /= len(val_dataloader)

    # Final evaluation on the test set
    model.eval()
    for _, _, blocks in tqdm(test_dataloader, desc='Evaluating'):
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

            # Store the test metrics
            test_loss += float(loss)
            if misinformation_f1 == misinformation_f1:
                test_misinformation_f1 += float(misinformation_f1)
            if factual_f1 == factual_f1:
                test_factual_f1 += float(factual_f1)

    # Divide the validation metrics by the number of batches
    test_loss /= len(test_dataloader)
    test_misinformation_f1 /= len(test_dataloader)
    test_factual_f1 /= len(test_dataloader)

    # Gather statistics to be logged
    stats = [
        ('val_loss', val_loss),
        ('val_misinformation_f1', val_misinformation_f1),
        ('val_factual_f1', val_factual_f1),
        ('test_loss', test_loss),
        ('test_misinformation_f1', test_misinformation_f1),
        ('test_factual_f1', test_factual_f1),
    ]

    # Report statistics
    log = 'Final evaluation\n'
    for statistic, value in stats:
        log += f'> {statistic}: {value}\n'
    logger.info(log)


if __name__ == '__main__':
    config = dict(num_epochs=500,
                  batch_size=32,
                  hidden_dim=128,
                  input_dropout=0.1,
                  dropout=0.2,
                  size='small',
                  task='claim',
                  lr=2e-5,
                  betas=(0.9, 0.999),
                  pos_weight=1.)
    train(**config)
