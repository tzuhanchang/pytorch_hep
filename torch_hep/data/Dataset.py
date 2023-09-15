import os
import pickle
import shutil
import warnings
import math

from subprocess import Popen

from tqdm import tqdm

from typing import List, Optional

from pandas import DataFrame, concat, read_json

from torch_geometric.data import Dataset


class GraphDataset(Dataset):
    r"""The :class:`GraphDataset` stores a set of graphs and additional non-graph
    information. This :class: is built on the Dataset base class see `here
    <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html>`.

    Authors:
        Dr. Callum Jacob Birch-Sykes (University of Manchester)
        Zihan Zhang (University of Manchester)

    .. code-block:: python
        from torch_hep.data import GraphDataset
        
        dataset = GraphDataset([Data1, Data2, ...])
        dataset.save_to('./myData')

    Args:
        graphs (List, optional): a list of graph to store. (default: :obj:`List`=[])
        df (pandas.DataFrame): a dataframe associated with the graph dataset. (default: :obj:`None`)
        batch_size (Int): number of graph per batch. (default: :obj:`Int`=1000)
    """
    def __init__(self, graphs: Optional[List] = [], df: Optional[DataFrame] = None, batch_size: Optional[int] = 100000):
        super(GraphDataset, self).__init__()
        self.graphs = graphs
        self.df = df
        self.batch = batch_size

        if len(self.graphs) <= 0 or self.batch <= 0:
            self.n_split = 0
        else:
            self.n_split = math.ceil(len(self.graphs) / self.batch)

    def save_to(self, path: str, graphs_only: Optional[bool] = False, compression: Optional[bool] = False):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
            warnings.warn("{} already exist, overwriting!".format(path), UserWarning)

        remainder_n = len(self.graphs) % self.n_split
        len_int = int(len(self.graphs)-remainder_n)

        for i in tqdm(range(self.n_split), desc='Saving the dataset', unit=' batch'):
            if i==self.n_split-1:
                end=len(self.graphs)
            else:
                end=int((i*len_int+len_int)/self.n_split)
            start=int(i*len_int/self.n_split)

            # saving df
            if isinstance(self.df, type(None)) == False and graphs_only == False:
                self.df.iloc[start:end].to_json('{}/df_batch-{}.json'.format(path, i),orient="index")

            # saving graphs
            with open('{}/graphs_batch-{}.pkl'.format(path, i), 'wb') as f:
                pickle.dump(self.graphs[start:end], f)
        
        if compression is True:
            where = os.path.split(path)
            tar = Popen(['tar', '-czvf', str(where[0])+'/'+str(where[1])+'.tar.gz', str(path)])
            tar.wait()

    def download_from(self, path: str, graphs_only: Optional[bool] = False):
        list_of_files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) and file.find('df')==-1]
        self.n_split = max([int(file.split('-')[1].replace(".pkl", "")) for file in list_of_files])+1

        if 'df' in ''.join(os.listdir(path)) == True and graphs_only == False:
            self.df=DataFrame()

        # downloading dataset
        self.graphs = []
        for i in tqdm(range(self.n_split), desc='Downloading the dataset', unit=' batch'):
            if graphs_only == False:
                try:
                    self.df = concat([self.df, read_json('{}/df_batch-{}.json'.format(path, i),orient="index")], axis=0)
                except:
                    warnings.warn("`df` files cannot be found, download graph only.", UserWarning)

            with open('{}/graphs_batch-{}.pkl'.format(path, i), 'rb') as f:
                self.graphs = self.graphs + pickle.load(f)

    def concat(self, obj: Dataset):
        if isinstance(self.df, type(None)) == False and isinstance(obj.df, type(None)) == False:
            self.df = concat([self.df, obj.df],axis=0).reset_index(drop=True)
        elif isinstance(self.df, type(None)) == True and isinstance(obj.df, type(None)) == False:
            self.df = DataFrame()
            self.df = concat([self.df, obj.df],axis=0).reset_index(drop=True)
        elif isinstance(self.df, type(None)) == False and isinstance(obj.df, type(None)) == True:
            raise ValueError("Concating to a dataset with no df!")
        self.graphs  = self.graphs + obj.graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    def get_df(self, idx):
        if self.df != None:
            return self.df.iloc[idx]
        else:
            raise ValueError("df does not exit in this dataset.")