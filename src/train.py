'''Training scripts'''

from data import load_mumin_graph
from graphsage import HeteroGraphSAGE

import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm


def train(num_epochs: int, hidden_dim: int):
    '''Train a heterogeneous GraphSAGE model on the MuMiN dataset.

    Args:
        num_epochs (int):
            The number of epochs to train for.
        hidden_dim (int):
            The dimension of the hidden layer.
    '''
    breakpoint()

    # Load dataset
    graph = load_mumin_graph().to('cuda')

    # Store node features
    feats = {node_type: graph.nodes[node_type].data['feat']
             for node_type in graph.ntypes()}

    # Store labels and masks
    labels = graph.nodes['tweet'].data['label']
    train_mask = graph.nodes['tweet'].data['train_mask']
    val_mask = graph.nodes['tweet'].data['val_mask']

    # Initialise dictionary with feature dimensions
    dims = dict(user=1, tweet=1, reply=1, image=1, article=1, hashtag=1)
    feat_dict = {rel: (dims[rel[0]], hidden_dim, dims[rel[2]])
                 for rel in graph.etypes()}

    # Initialise model
    model = HeteroGraphSAGE(feat_dict).to('cuda')
    model.train()

    # Initialise optimiser
    opt = optim.Adam(model.parameters())

    # Initialise scorer
    scorer = tm.F1(num_classes=2, threshold=0.5, average='macro')

    for epoch in range(num_epochs):

        # Forward propagation
        logits = model(graph, feats)['tweet']

        # Compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute validation score
        score = scorer(logits[train_mask].argmax(dim=-1), labels[train_mask])

        # Backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Report loss and score
        print('Loss:', loss)
        print('Score:', score)

        # Save model
        # TODO


if __name__ == '__main__':
    train(num_epochs=100, hidden_dim=10)
