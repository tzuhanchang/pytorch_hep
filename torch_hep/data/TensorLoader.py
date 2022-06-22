import torch

from torch_geometric.data import Data


class TensorLoader():
    r"""A `TensorLoader` which splits full `torch.tensor` attributes into
    multiple mini-batches.
    
    Args:
        batch_size (int): number of tensors per batch to load (default: 1).
        **kwargs (optional): additional attributes.
    """
    batches = list
    dataset = dict
    data    = dict

    def __init__(self, batch_size: int=1, **kwargs):
        self.batches = self.batches()
        self._dataset = self.dataset()
        self.batch_size = batch_size

        self._itercount = 0
        
        for key, value in kwargs.items():
            if torch.is_tensor(value) == False:
                raise ValueError("'TensorLoader only takes 'torch.tensor' inputs")

            self._dataset.update({key:list(x for x in torch.split(value, batch_size))})
            self._num_batches = len(self._dataset[key])
            self._data_size = value.size(dim=0)
        
        for i in range(0,self._num_batches):
            self._data = self.data()
            for key, value in self._dataset.items():
                self._data.update({key:value[i]})
            self.batches.append(Data(**self._data))


    def __getitem__(self, key):
        return self.data[key]
    

    def __iter__(self):
        return self
    

    def __next__(self):
        if self._itercount < self._num_batches:
            self._itercount += 1
            return self.batches[self._itercount-1]
        else:
            raise StopIteration
            