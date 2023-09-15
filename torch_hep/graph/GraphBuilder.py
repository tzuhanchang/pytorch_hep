import torch

from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Callable


class GraphBuilder():
    r"""A fast graph dataset building tool.
    :class:`torch_hep.graph.GraphBuilder` converts standard :class:`dict` object
    into :class:`torch_geometric.data.Data` format.

    .. code-block:: python

        from torch_hep.graph import GraphBuilder
        from itertools import permutations

        num_nodes = 2
        nodes = {'node_feat_1': [0,1],'node_feat_2': [3,2], 'node_feat_3': [1,9]}
        edge_index = list(permutations(range(0,num_nodes),2))
        edge_attrs = {'edge_feat_1': [1,2],'edge_feat_2': [3,4]}
        globals = {'global_feat_1': 3, 'global_feat_2': 9}

        G = GraphBuilder()
        G.add_asNode(key='x', **nodes)
        G.add_asEdge(key='edge_attrs', index=edge_index, **edge_attrs)
        G.add_asGlobal(key='num_nodes',num_nodes=int(num_nodes),dtype=torch.int64)
        G.add_asGlobal(key='u', **globals)

        graph = G.to_Data()

    where :obj:`nodes` contains node features, each node feature should have `dim = num_nodes`.
    :obj:`edge_attrs` contains edge features, each has a `dim = len(edge_index)`.
    All :obj:`globals` features should be a scalar.

    Args:
        device (device, optional): device to store the graphs. (default, :obj:`str`='cpu')
    """
    _graph = dict[Tensor]
    def __init__(self, device: Optional[str] = 'cpu'):
        self._graph = self._graph()
        self.values = self._graph
        self.device = device

    def __getitem__(self, key):
        return self._graph[key]

    def __setitem__(self, key, value):
        self._graph[key] = value

    def add_asGlobal(self, key: str, dtype: Optional[Callable] = torch.float32, **attr):
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

    def add_asNode(self, key: str, dtype: Optional[Callable] = torch.float32, **attr):
        if key not in self._graph:
            self._graph.update({key: torch.tensor([list(x) for x in attr.values()],dtype=dtype,device=self.device).transpose(1,0)})
            self._graph.update({key+'_features': [x for x in attr.keys()]})
        elif key in self._graph:
            self[key] = torch.cat((self[key],torch.tensor([list(x) for x in attr.values()],dtype=dtype,device=self.device).transpose(1,0)),dim=1)
            self[key+'_features'] += [x for x in attr.keys()]
        setattr(self,key,self[key])

    def add_asEdge(self, key: str, dtype: Optional[Callable] = torch.float32, **attr):
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
