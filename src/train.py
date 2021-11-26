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


# Set up logging
fmt = (colored('%(asctime)s [%(levelname)s] <%(name)s>\n↳ ', 'green') +
       colored('%(message)s', 'yellow'))
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)


def train(num_epochs: int,
          hidden_dim: int,
          size: str = 'small',
          task: str = 'claim'):
    '''Train a heterogeneous GraphConv model on the MuMiN dataset.

    Args:
        num_epochs (int):
            The number of epochs to train for.
        hidden_dim (int):
            The dimension of the hidden layer.
        size (str, optional):
            The size of the dataset to use. Defaults to 'small'.
        task (str, optional):
            The task to consider, which can be either 'tweet' or 'claim',
            corresponding to doing thread-level or claim-level node
            classification. Defaults to 'claim'.
    '''
    # Set random seeds
    torch.manual_seed(4242)
    dgl.seed(4242)

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

    # Store node features
    feats = {node_type: graph.nodes[node_type].data['feat'].float()
             for node_type in graph.ntypes}

    # Store labels and masks
    labels = graph.nodes[task].data['label']
    train_mask = graph.nodes[task].data['train_mask']
    val_mask = graph.nodes[task].data['val_mask']

    # Initialise dictionary with feature dimensions
    dims = {ntype: graph.nodes[ntype].data['feat'].shape[-1]
            for ntype in graph.ntypes}
    feat_dict = {rel: (dims[rel[0]], hidden_dim, dims[rel[2]])
                 for rel in graph.canonical_etypes}

    # Initialise model
    model = HeteroGraphSAGE(feat_dict)
    model.to(device)
    model.train()

    # Set up path to state dict, and load model weights if they exist
    model_path = Path(f'{task}-model-{size}-{hidden_dim}.pt')
    if model_path.exists():
        model.load_state_dict(torch.load(str(model_path)))

    # Initialise optimiser
    opt = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.8, 0.998))

    # Initialise learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer=opt, factor=0.8, patience=10)

    # Initialise scorer
    scorer = tm.F1(num_classes=2, average='none')

    for epoch in range(num_epochs):

        # Reset the gradients
        opt.zero_grad()

        # Forward propagation
        logits = model(graph, feats)
        logits = logits[task].squeeze(1)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(logits[train_mask],
                                                  labels[train_mask].float(),
                                                  pos_weight=torch.tensor(20.))

        with torch.no_grad():

            # Compute validation loss
            val_loss = F.binary_cross_entropy_with_logits(
                logits[val_mask],
                labels[val_mask].float(),
                pos_weight=torch.tensor(20.)
            )

            # Compute training metrics
            train_scores = scorer(logits[train_mask].ge(0), labels)
            train_misinformation_f1 = train_scores[0]
            train_factual_f1 = train_scores[1]

            # Compute validation metrics
            val_scores = scorer(logits[val_mask].ge(0), labels)
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
            ('Train loss', loss.item()),
            ('Train misinformation F1', train_misinformation_f1.item()),
            ('Train factual F1', train_factual_f1.item()),
            ('Validation loss', val_loss.item()),
            ('Validation misinformation F1', val_misinformation_f1.item()),
            ('Validation factual F1', val_factual_f1.item()),
            ('Learning rate', opt.param_groups[0]['lr'])
        ]

        # Report statistics
        log = f'Epoch {epoch}\n'
        for statistic, value in stats:
            log += f'> {statistic}: {value}\n'
        logger.info(log)

        # Save model
        torch.save(model.state_dict(), str(model_path))


if __name__ == '__main__':
    # Train model
    train(num_epochs=1000, hidden_dim=500, task='claim')
