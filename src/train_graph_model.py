'''Training scripts'''

from data import load_mumin_graph
from model import HeteroGraphSAGE

from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
from torch.optim.lr_scheduler import LinearLR
import torchmetrics as tm
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
import dgl
import logging
import datetime as dt
from tqdm.auto import tqdm
from mumin import load_dgl_graph, save_dgl_graph


logger = logging.getLogger(__name__)


def train_graph_model(task: str,
                      size: str,
                      num_epochs: int = 300,
                      random_split: bool = False,
                      **_):
    '''Train a heterogeneous GraphConv model on the MuMiN dataset.

    Args:
        task (str):
            The task to consider, which can be either 'tweet' or 'claim',
            corresponding to doing thread-level or claim-level node
            classification.
        size (str):
            The size of the dataset to use.
        num_epochs (int, optional):
            The number of epochs to train for. Defaults to 300.
        random_split (bool, optional):
            Whether a random train/val/test split of the data should be
            performed (with a fixed random seed). If not then the claim cluster
            splits will be used. Defaults to False.
    '''
    # Set random seeds
    torch.manual_seed(4242)
    dgl.seed(4242)

    # Set config
    config = dict(hidden_dim=1024,
                  input_dropout=0.2,
                  dropout=0.2,
                  size=size,
                  task=task,
                  lr=3e-4,
                  betas=(0.9, 0.999),
                  pos_weight=20.)

    # Set up PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up graph path
    graph_path = Path(f'dgl-graph-{size}.bin')

    # Load the graph if it exists
    if graph_path.exists():
        graph = load_dgl_graph(graph_path)

    # Otherwise, build the graph and save it
    else:
        # Build the graph
        graph = load_mumin_graph(size=size)

        # Save the graph
        save_dgl_graph(graph, graph_path)

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
    model = HeteroGraphSAGE(input_dropout=0.2,
                            dropout=0.2,
                            hidden_dim=1024,
                            feat_dict=feat_dict,
                            task=task)
    model.to(device)
    model.train()

    # Enumerate the nodes with the labels, for performing train/val/test splits
    node_enum = torch.arange(graph.num_nodes(task))

    # If we are performing a random split then split the dataset into a
    # 80/10/10 train/val/test split, with a fixed random seed
    if random_split:

        # Set a random seed through a PyTorch Generator
        torch_gen = torch.Generator().manual_seed(4242)

        # Compute the number of train/val/test samples
        num_train = int(0.8 * graph.num_nodes(task))
        num_val = int(0.1 * graph.num_nodes(task))
        num_test = graph.num_nodes(task) - (num_train + num_val)
        nums = [num_train, num_val, num_test]

        # Split the data, using the PyTorch generator for reproducibility
        train_nids, val_nids, test_nids = D.random_split(dataset=node_enum,
                                                         lengths=nums,
                                                         generator=torch_gen)

        # Store the resulting node IDs
        train_nids = {task: train_nids}
        val_nids = {task: val_nids}
        test_nids = {task: test_nids}

    # If we are not performing a random split we're performing a split based on
    # the claim clusters of the data. This means that the different splits will
    # belong to different events, thus making the task harder.
    else:
        train_nids = {task: node_enum[train_mask].int()}
        val_nids = {task: node_enum[val_mask].int()}
        test_nids = {task: node_enum[test_mask].int()}

    # Set up the sampler
    sampler = MultiLayerNeighborSampler([100, 100], replace=False)

    # Set up the dataloaders
    train_dataloader = NodeDataLoader(g=graph,
                                      nids=train_nids,
                                      block_sampler=sampler,
                                      batch_size=1024,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=1)
    val_dataloader = NodeDataLoader(g=graph,
                                    nids=val_nids,
                                    block_sampler=sampler,
                                    batch_size=1024,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=1)
    test_dataloader = NodeDataLoader(g=graph,
                                     nids=test_nids,
                                     block_sampler=sampler,
                                     batch_size=1024,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=1)

    # Set up pos_weight
    pos_weight_tensor = torch.tensor(20.).to(device)

    # Set up path to state dict
    datetime = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    Path('models').mkdir(exist_ok=True)
    model_dir = Path('models') / f'{datetime}-{task}-model-{size}'
    model_dir.mkdir(exist_ok=True)

    # Initialise optimiser
    opt = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    # Initialise learning rate scheduler
    scheduler = LinearLR(optimizer=opt,
                         start_factor=1.,
                         end_factor=1e-7 / 3e-4,
                         total_iters=100)

    # Initialise scorer
    scorer = tm.classification.f_beta.F1Score(num_classes=2,
                                              average='none').to(device)


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
            logits = model(blocks, input_feats).squeeze()

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
                logits = model(blocks, input_feats).squeeze()

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
        config['epoch'] = epoch
        for statistic, value in stats:
            config[statistic] = value

        # Update progress bar description
        desc = (f'Training - '
                f'loss {train_loss:.3f} - '
                f'train_f1 {train_factual_f1:.3f} - '
                f'val_loss {val_loss:.3f} - '
                f'val_f1 {val_factual_f1:.3f}')
        epoch_pbar.set_description(desc)

        # Update learning rate
        scheduler.step()

    # Close progress bar
    epoch_pbar.close()

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
            logits = model(blocks, input_feats).squeeze()

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
            logits = model(blocks, input_feats).squeeze()

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
