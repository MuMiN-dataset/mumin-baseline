'''Training scripts'''

from data import load_mumin_graph
from model import HeteroGraphSAGE

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm


def train(num_epochs: int, hidden_dim: int):
    '''Train a heterogeneous GraphConv model on the MuMiN dataset.

    Args:
        num_epochs (int):
            The number of epochs to train for.
        hidden_dim (int):
            The dimension of the hidden layer.
    '''
    # Load dataset
    graph = load_mumin_graph()#.to('cuda')

    # Store node features
    feats = {node_type: graph.nodes[node_type].data['feat'].float()
             for node_type in graph.ntypes}

    # Store labels and masks
    labels = graph.nodes['tweet'].data['label']
    train_mask = graph.nodes['tweet'].data['train_mask']
    val_mask = graph.nodes['tweet'].data['val_mask']

    # Initialise dictionary with feature dimensions
    dims = dict(claim=768, user=6, tweet=3, reply=3, image=1, article=1,
                hashtag=1, place=2)
    feat_dict = {rel: (dims[rel[0]], hidden_dim, dims[rel[2]])
                 for rel in graph.canonical_etypes}

    # Initialise model
    model = HeteroGraphSAGE(feat_dict)#.to('cuda')
    model.train()

    # Initialise optimiser
    opt = optim.Adam(model.parameters(), lr=3e-4)

    # Initialise scorer
    scorer = tm.F1(num_classes=2, average='none')

    for epoch in range(num_epochs):

        # Forward propagation
        logits = model(graph, feats)
        logits = logits['tweet'].squeeze(1)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(logits[train_mask],
                                                  labels[train_mask].float(),
                                                  pos_weight=torch.tensor(20.))

        # Compute validation score
        with torch.no_grad():
            scores = scorer(logits[val_mask].ge(0), labels[val_mask])
            misinformation_f1 = scores[0]
            factual_f1 = scores[1]

        # Backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Report loss and score
        print(f'Epoch {epoch}')
        print('> Loss:', float(loss))
        print('> Misinformation F1:', float(misinformation_f1))
        print('> Factual F1:', float(factual_f1))
        print()

        # Save model
        # TODO


if __name__ == '__main__':
    train(num_epochs=1000, hidden_dim=100)
