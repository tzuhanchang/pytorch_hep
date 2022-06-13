import torch

from torch import Tensor


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

    def add_asGlobal(self, key: str, **attr):
        if key not in self._graph:
            self._graph.update({key: torch.tensor([*attr.values()],device=self.device)})
            self._graph.update({key+'_features': [*attr.keys()]})
        elif key in self._graph:
            self[key] = torch.cat((self[key],torch.tensor([*attr.values()],device=self.device)),dim=0)
            self[key+'_features'] += [*attr.keys()]

    def add_asNode(self, key: str, **attr):
        if key not in self._graph:
            self._graph.update({key: torch.tensor([*attr.values()],device=self.device).transpose(1,0)})
            self._graph.update({key+'_features': [*attr.keys()]})
        elif key in self._graph:
            self[key] = torch.cat((self[key],torch.tensor([*attr.values()],device=self.device).transpose(1,0)),dim=1)
            self[key+'_features'] += [*attr.keys()]

    def add_asEdge(self, key: str, **attr):
        if 'edge_index' not in self._graph:
            if 'index' not in attr:
                raise ValueError("'index' must be included")
            else:
                self._graph.update({'edge_index': torch.tensor(attr.get('index'),device=self.device).transpose(0,1)})
                del attr['index']
        if key not in self._graph:
            self._graph.update({key: torch.tensor([*attr.values()],device=self.device).transpose(1,0)})
            self._graph.update({key+'_features': [*attr.keys()]})
        elif key in self._graph:
            self[key] = torch.cat((self[key],torch.tensor([*attr.values()],device=self.device).transpose(1,0)),dim=1)
            self[key+'_features'] += [*attr.keys()]

