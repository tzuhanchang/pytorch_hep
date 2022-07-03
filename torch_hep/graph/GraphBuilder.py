import torch

from torch import Tensor
from torch_geometric.data import Data
from typing import Callable


class GraphBuilder():
    r""":class:`GraphBuilder`
    """
    _graph = dict[Tensor]
    def __init__(self, device='cpu'):
        self._graph = self._graph()
        self.values = self._graph
        self.device = device

    def __getitem__(self, key):
        return self._graph[key]

    def __setitem__(self, key, value):
        self._graph[key] = value

    def add_asGlobal(self, key: str, dtype: Callable=torch.float64, **attr):
        if key not in self._graph:
            if len(attr.items()) == 1 and hasattr([x for x in attr.values()][0],'__len__') == False:
                self._graph.update({key: torch.tensor([x for x in attr.values()][0],dtype=dtype,device=self.device)})
            else:
                self._graph.update({key: torch.tensor([[x for x in attr.values()]],dtype=dtype,device=self.device)})
                self._graph.update({key+'_features': [x for x in attr.keys()]})
        elif key in self._graph:
            self[key] = torch.cat((self[key],torch.tensor([[x for x in attr.values()]],dtype=dtype,device=self.device)),dim=1)
            self[key+'_features'] += [x for x in attr.keys()]
        setattr(self,key,self[key])

    def add_asNode(self, key: str, dtype: Callable=torch.float64, **attr):
        if key not in self._graph:
            self._graph.update({key: torch.tensor([list(x) for x in attr.values()],dtype=dtype,device=self.device).transpose(1,0)})
            self._graph.update({key+'_features': [x for x in attr.keys()]})
        elif key in self._graph:
            self[key] = torch.cat((self[key],torch.tensor([list(x) for x in attr.values()],dtype=dtype,device=self.device).transpose(1,0)),dim=1)
            self[key+'_features'] += [x for x in attr.keys()]
        setattr(self,key,self[key])

    def add_asEdge(self, key: str, dtype: Callable=torch.float64, **attr):
        if 'edge_index' not in self._graph:
            if 'index' not in attr:
                raise ValueError("'index' must be included")
            else:
                self._graph.update({'edge_index': torch.tensor(attr.get('index'),device=self.device).transpose(0,1)})
                del attr['index']
        if key not in self._graph:
            self._graph.update({key: torch.tensor([list(x) for x in attr.values()],dtype=dtype,device=self.device).transpose(1,0)})
            self._graph.update({key+'_features': [x for x in attr.keys()]})
        elif key in self._graph:
            self[key] = torch.cat((self[key],torch.tensor([list(x) for x in attr.values()],dtype=dtype,device=self.device).transpose(1,0)),dim=1)
            self[key+'_features'] += [x for x in attr.keys()]
        setattr(self,key,self[key])

    def to_Data(self):
        return Data(**self._graph)
