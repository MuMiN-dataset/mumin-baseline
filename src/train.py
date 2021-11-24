'''Training scripts'''

from data import load_mumin_graph
from model import HeteroGraphSAGE

from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm
from dgl.data.utils import save_graphs, load_graphs


def train(num_epochs: int, hidden_dim: int, task: str = 'tweet'):
    '''Train a heterogeneous GraphConv model on the MuMiN dataset.

    Args:
        num_epochs (int):
            The number of epochs to train for.
        hidden_dim (int):
            The dimension of the hidden layer.
        task (str, optional):
            The task to consider, which can be either 'tweet' or 'claim',
            corresponding to doing thread-level or claim-level node
            classification.
    '''
    # Set up graph path
    graph_path = Path('data.bin')

    if graph_path.exists():
        # Load the graph
        graph = load_graphs(str(graph_path))[0][0]

    else:
        # Load dataset
        graph = load_mumin_graph()#.to('cuda')

        # Save graph to disk
        save_graphs(str(graph_path), [graph])

    # Store node features
    feats = {node_type: graph.nodes[node_type].data['feat'].float()
             for node_type in graph.ntypes}

    # Store labels and masks
    labels = graph.nodes[task].data['label']
    train_mask = graph.nodes[task].data['train_mask'].long()
    val_mask = graph.nodes[task].data['val_mask'].long()

    # Initialise dictionary with feature dimensions
    dims = dict(claim=870, user=774, tweet=811, reply=824, image=768,
                article=1536, hashtag=1)
    feat_dict = {rel: (dims[rel[0]], hidden_dim, dims[rel[2]])
                 for rel in graph.canonical_etypes}

    # Initialise model
    model = HeteroGraphSAGE(feat_dict)#.to('cuda')
    model.train()

    # Initialise optimiser
    opt = optim.Adam(model.parameters(), lr=2e-5)

    # Initialise scorer
    scorer = tm.F1(num_classes=2, average='none')

    for epoch in range(num_epochs):

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
            train_scores = scorer(logits[train_mask].ge(0), labels[train_mask])
            train_misinformation_f1 = train_scores[0]
            train_factual_f1 = train_scores[1]

            # Compute validation metrics
            val_scores = scorer(logits[val_mask].ge(0), labels[val_mask])
            val_misinformation_f1 = val_scores[0]
            val_factual_f1 = val_scores[1]

        # Backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Report loss and score
        print(f'Epoch {epoch}')
        print('> Train loss:', loss.item())
        print('> Train misinformation F1:', train_misinformation_f1.item())
        print('> Train factual F1:', train_factual_f1.item())
        print('> Val loss:', val_loss.item())
        print('> Val misinformation F1:', val_misinformation_f1.item())
        print('> Val factual F1:', val_factual_f1.item())
        print()

        # Save model
        # TODO


if __name__ == '__main__':
    train(num_epochs=1000, hidden_dim=100, task='claim')
