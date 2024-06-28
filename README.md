[pypi-image]: https://badge.fury.io/py/torch-hep.svg
[pypi-url]: https://pypi.org/project/torch-hep

<p align="center">
  <img height="150" src=".github/logo/pyTorch-hep_logo.png" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]

**PyTorch HEP** is a python package built upon [PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io) (PyTorch Geometric) for easy HEP event graph builiding, storing and loading.

PyTorch HEP is currently a personal project, contributions to this project are warmly welcomed. If you are interested in contributing, please submit a pull request with your implemented new features or fixed bugs.


* [Quick Start](#quick-start)
* [Event&rarr;Graph](#event-&rarr;-graph)

## Quick Start

### Installation
```
pip install torch-hep
```

## Event &rarr; Graph
To initialise the builder:
```
from torch_hep.graph import GraphBuilder
G = GraphBuilder()
```
At reconstructed level, each event can have multiple final states, such as jets, electrons, muons or missing ET. These are commonly represented as **nodes** (or _vertices_). We can assign the four-momentum (but not limited to) of each final state object as the **features**.
```
p4 = {'pt': [...], 'eta': [...], 'phi': [...], 'e': [...]}

G.add_asNode(key='x', **p4)

# It is equivalent to:
G.add_asNode(key='x', pt = p4['pt'], eta = p4['eta'], ...)
```
where the entries of `p4['pt'], p4['eta'], ...` are the kinematics information of corresponding final state of a single event.

**Edges** describle the binary connections between paired nodes. For a fully connected **digraph** (_directed graph_), the edges can be computed as follow
```
from itertools import permutations
n_nodes = len(p4['pt'])

G.add_asEdge(key='edge_attrs', index=list(permutations(range(n_nodes),2)), dR=[...], ...)
```
where `dR=[...]` is one of the edge feature, which has entries corresponding to each directed edge. `index` must be included when add edge like object (once). For convenience, we use `list(permutations(range(n_nodes),2))` as edge index for a fully connected graph.

For variables that cannot be represented by a graph, a **global** can be included:
```
G.add_asGlobal(key='u', nJet=3, nBtagged=2, ...)
```
If you are performing a graph classification, of course you can label your target:
```
G.add_asGlobal(key='u_t', IsSIG=1)
```
You can do the same for edge/node classification by define desired objects with corresponding 'add' function.
For multiple events, append G to a dataset for each event.
```
dataset = []

for i in range(num_events):
    G = torch_hep.graph.GraphBuilder()
    G.add_asNode(key='x', pt = dic_tthh['pt'][i], eta = dic_tthh['eta'][i], phi = dic_tthh['phi'][i],...) # dic_tthh['pt'][i] takes the pt values for each particle in the i th event
    G.add_asEdge(key='edge_attrs', index=list(itertools.permutations(range(len(dic_tthh['pt'][i])),2)))
    G.add_asGlobal(key='u_t', IsSIG=1)

    dataset.append(G.to_Data())
```
This can the be converted into a loader.
```
from torch_geometric.data import DataLoader
loader = DataLoader(dataset, batch_size=batch_size)
```
`GraphBuilder` is interfaced with `torch_geometric.data.Data`. Users can covert information stored in `GraphBuilder` to `torch_geometric.data.Data` using `G.to_Data()`.
