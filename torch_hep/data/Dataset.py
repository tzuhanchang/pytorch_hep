import os
import torch
import shutil
import warnings

from tqdm import tqdm
from typing import List, Optional
from torch_geometric.data import Dataset


class GraphDataset(Dataset):
    r"""The :obj:`GraphDataset` stores a set of graphs and additional non-graph
    information. This :class: is built on the Dataset base class see `here
    <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html>`.

    .. code-block:: python
        from torch_hep.data import GraphDataset
        from torch_geometric.loader import DataLoader

        dataset = GraphDataset(graphs=[...])
        dataset.save_to('./my_graphs')

        dataset = GraphDataset(root='./my_graphs')
        loader = DataLoader(dataset, batch_size=2048)

    Args:
        root (str, optional): Root directory where the dataset is saved. (default: :obj:`None`)
        graphs (List, optional): List of :obj:`torch_geometric.data.Data`. (default: :obj:`None`)
    """
    def __init__(self, root: Optional[str] = None, graphs: Optional[List] = None):
        super(GraphDataset, self).__init__()
        self.root = root
        self.graphs = graphs

        if self.root is None and self.graphs is None:
            raise ValueError("User must provide either dataset directory `root` or a list of `graphs`.")

        if self.root is not None:
            self.batch_size = len(os.listdir(os.path.join(self.root, 'batch_0')))
            self.n_splits = len(os.listdir(self.root))
            self.n_graphs = len([val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(self.root)] for val in sublist])
            if self.graphs is not None:
                warnings.warn("Using dataset in `root`, ignoring list of `graphs`.", UserWarning)
                self.graphs = None

        if self.graphs is not None and self.root is None:
            self.n_graphs = len(self.graphs)

    def save_to(self, path: str, batch_size: Optional[int] = 1000):
        self.batch_size = batch_size

        if self.n_graphs <= 0 or self.batch_size <= 0:
            self.n_splits = 0
        else:
            self.n_splits = -(-self.n_graphs // self.batch_size)

        if self.graphs is None:
            raise ValueError("To save, a list of `graphs` must to be provided.")

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
            warnings.warn(f"{path} already exist, overwriting!", UserWarning)

        idx = 0
        for i in tqdm(range(self.n_splits), desc='Saving the dataset', unit='batch'):
            start = int(i * self.batch_size)
            if i == self.n_splits - 1:
                end = self.n_graphs
            else:
                end = (i + 1) * self.batch_size

            batch_path = os.path.join(path,f'batch_{i}')
            os.makedirs(batch_path)
            for data in self.graphs[start:end]:
                torch.save(data, os.path.join(batch_path, f'data_{idx}.pt'))
                idx += 1

        self.root = path
        self.graphs = None

    def len(self):
        return self.n_graphs

    def get(self, idx):
        if self.graphs is not None:
            data = self.graphs[idx]

        if self.root is not None:
            loc = idx // self.batch_size
            data = torch.load(os.path.join(self.root, f'batch_{loc}', f'data_{idx}.pt'))
        return data