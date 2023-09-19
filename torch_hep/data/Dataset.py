import os
import math
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

    Authors:
        Dr. Callum Jacob Birch-Sykes (University of Manchester)
        Zihan Zhang (University of Manchester)

    Args:
        graphs (List): a list of graph to store.
        df (pandas.DataFrame): a dataframe associated with the graph dataset. (default: :obj:`None`)
        batch_size (Int): number of graph per batch. (default: :obj:`Int`=1000)
    """

    def __init__(self, root: Optional[str] = None, graphs: Optional[List] = None):
        super(GraphDataset, self).__init__()
        self.root = root
        self.graphs = graphs

        if self.root is None and self.graphs is None:
            raise ValueError("User must provide either dataset directory `root` or a list of `graphs`.")
        
        if self.root is not None:
            self.batch   = len(os.listdir(os.path.join(self.root, 'batch_0')))
            self.n_split = len(os.listdir(self.root))
            self.n_graph = len([val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(self.root)] for val in sublist])
            if self.graphs is not None:
                warnings.warn("Using dataset in `root`, ignoring list of `graphs`.", UserWarning)
                self.graphs = None

        if self.graphs is not None and self.root is None:
            self.n_graph = len(self.graphs)

    def save_to(self, path: str, batch_size: Optional[int] = 100000):
        self.batch = batch_size
        if len(self.graphs) <= 0 or self.batch <= 0:
            self.n_split = 0
        else:
            self.n_split = math.ceil(len(self.graphs) / self.batch)

        if self.graphs == None:
            raise ValueError("To save, a list of `graphs` must to be provided.")

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
            warnings.warn("{} already exist, overwriting!".format(path), UserWarning)

        remainder_n = len(self.graphs) %  self.n_split
        len_int = int(len(self.graphs)-remainder_n)

        idx = 0
        for i in tqdm(range(self.n_split), desc='Saving the dataset', unit='batch'):
            if i==self.n_split-1:
                end=len(self.graphs)
            else:
                end=int((i*len_int+len_int)/self.n_split)
            start=int(i*len_int/self.n_split)

            batch_path = os.path.join(path,f'batch_{i}')
            os.makedirs(batch_path)
            for data in self.graphs[start:end]:
                torch.save(data, os.path.join(batch_path, f'data_{idx}.pt'))
                idx += 1

        self.root = path
        self.graphs = None
        self.n_graph = idx

    def len(self):
        return self.n_graph

    def get(self, idx):
        if self.graphs is not None:
            data = self.graphs[idx]

        if self.root is not None:
            loc = math.floor(idx / self.batch)
            data = torch.load(os.path.join(self.root, f'batch_{loc}', f'data_{idx}.pt'))
        return data
