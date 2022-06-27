# PyTorch HEP
<br />
PyTorch HEP is a project built upon PyTorch to act as a foundation for high performance computing for High Energy Physics. PyTorch HEP currently includes tensor based Lorentz manipulation for particle reconstruction and graph builder for Graph Neural Network (GNN).
<br />

## Installation
To install Pytorch HEP, run:
```
pip install torch-hep
```
Install requires:
```
python >= 3.7
torch
torch_geometric
```

## GraphBuilder
### Build a graph from HEP event
In a HEP event, e.g. a jet, whoes kinematics are described by variables such as "jet_pt=[125000.6, 290000.4, 223422.8]", "jet_eta = [3.4, -2.5, 1.3]" etc, in this case there are 3 jets. To build its graph we use `add_asNode` to add object which is node-like (such as node features or node weights):
```ruby
G = GraphBuilder(device='cpu') #initialize builder
G.add_asNode(key='x', jet_pt=[125000.6, 290000.4, 223422.8], jet_eta = [3.4, -2.5, 1.3])
```
it builds 3 nodes automatically in this case. <br />

To add edges:
```ruby
G.add_asEdge(key='e','index'=list(permutations(range(0,3),2)),'Is_SIG'=[0,0,0,1,0])
```
For convenience, we use `list(permutations(range(0,3),2))` as edge index for a fully connected graph. 'index' must be included when add edge like object (once). <br />

Similarly, to add global-like object:
```ruby
G.add_asGlobal(key='u', nJet=3, nBtagged=2)
```

`GraphBuilder` stores input informations as tensors on provided device (cpu or gpu), to get values:
```
>>> G.x  # or G['x']
torch.tensor([[125000.6,3.4],[290000.4,-2.5],[223422.8,1.3]])
>>> G.x_features
['jet_pt','jet_eta']
```
Users can choose to covert informations stored in `GraphBuilder` to `torch_geometric.data.Data` using `G.to_Data()`, which allows users to use it as inputs of custom dataset.


## LorentzTensor
### Tensor to LorentzTensor
Define a `LorentzTensor` with single 4-vector:
```ruby
vector4 = torch.tensor([112849, 35192.7, -44507.4, 94562.1])
LorentzTensor(vector4)
```
`LorentzTensor` fully supports GPU computing, upon input set `device='cuda'` (default `cpu`) for `vector4` tensor. In need of high precision set `dtype=torch.float64` (default `dtype=toch.float32`).<br />
<br />
Define a `LorentzTensor` with 4-vector vector space:
```ruby
vector4Space = torch.tensor([[112849, 35192.7, -44507.4, 94562.1],[82849.2, 12143.8, 83553.1, 21007.5],...])
LorentzTensor(vector4Space)
```
`LorentzTensor` computes properties of every 4-vectors in this vector space, it does not involve `numpy` so included tools fully support GPU computing.<br />

### Lorentz vector Operations in LorentzTensor
It supports all lorentz vector basic operations, including sum, subtract, dot product, multiply and divide:
```ruby
a = LorentzTensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
b = LorentzTensor([[2,2,2,2],[2,2,2,2],[2,2,2,2]])
```
```
>>> (a+b).values
torch.tensor([[3,3,3,3],[3,3,3,3],[3,3,3,3]])
>>> (a-b).values
torch.tensor([[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1]])
>>> a.dot(b)
torch.tensor([[-4],[-4],[-4]])
>>> (5*a).values
torch.tensor([[5,5,5,5],[5,5,5,5],[5,5,5,5]])
>>> (a/2).values
torch.tensor([[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]])
```

## MomentumTensor
To define a `MomentumTensor`: 
```ruby
MomentumTensor(torch.tensor([e,px,py,pz]))
MomentumTensor.EEtaPhiPt(torch.tensor([e,eta,phi,pt]))
MomentumTensor.MEtaPhiPt(torch.tensor([m,eta,phi,pt]))
```
same operations apply like a `LorentzTensor` does.<br />
`MomentumTensor` also includes calculations of common 4-momenta properties:
```ruby
momentum = MomentumTensor(torch.tensor([[111549, 35202.7, -46507.4, 94552.1],[86549.2, 12443.8, 81453.1, 25407.5],[86799.1, 12423.2, 81499.2, 25411.3]]))
```
get transverse momenta 'pT' and mass 'm' from MomentumTensor:
```
>>> momentum.pt
torch.tensor([[58328.10936461082],[82398.15324417168],[82440.61801612115]])
>>> momentum.m
torch.tensor([[10045.468856155943],[7467.751089852902],[9586.140737544087]])
```