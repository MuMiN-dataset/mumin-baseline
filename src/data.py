'''Data loading scripts'''

import dgl
from mumin import MuminDataset
from dotenv import load_dotenv
import os

load_dotenv()


def load_mumin_graph(size: str = 'small') -> dgl.DGLHeteroGraph:
    '''Load the MuMiN dataset as a DGL heterogeneous graph.

    Args:
        size (str, optional):
            The size of the dataset to compile. Can be 'small', 'medium' and
            'large'.
    '''
    dataset = MuminDataset(os.getenv('TWITTER_API_KEY'), size=size)
    dataset.compile()
    dataset.add_embeddings()
    return dataset.to_dgl()
