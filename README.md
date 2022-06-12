# PyTorch HEP
<br />
PyTorch HEP is a project built upon PyTorch to act as a foundation for high performance computing for High Energy Physics. PyTorch HEP is currently under development, this includes tensor based Lorentz manipulation for particle reconstruction.
<br />

## Installation
To install Pytorch HEP, run:
```
pip install torch-hep
```

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
LoretnzTensor1 = LorentzTensor(torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]]))
LoretnzTensor2 = LorentzTensor(torch.tensor([[2,2,2,2],[2,2,2,2],[2,2,2,2]]))
LoretnzTensor1+LoretnzTensor2
> LorentzTensor(torch.tensor([[3,3,3,3],[3,3,3,3],[3,3,3,3]]))
LoretnzTensor1-LoretnzTensor2
> LorentzTensor(torch.tensor([[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1]]))
LoretnzTensor1*LoretnzTensor2
> torch.tensor([[-4],[-4],[-4]])
5*LoretnzTensor1
> LorentzTensor(torch.tensor([[5,5,5,5],[5,5,5,5],[5,5,5,5]]))
LoretnzTensor1/2
> LorentzTensor(torch.tensor([[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]]))
```

## MomentumTensor
To define a `MomentumTensor`: `MomentumTensor(torch.tensor([e,px,py,pz]))` or use method `MomentumTensor.EEtaPhiPt(torch.tensor([e,eta,phi,pt]))` or `MomentumTensor.MEtaPhiPt(torch.tensor([m,eta,phi,pt]))`. A `MomentumTensor` operates like a `LorentzTensor`, all methods apply in the same way.<br />
Beside all `LorentzTensor` properties, `MomentumTensor` also includes, such as:
```ruby
momentum = MomentumTensor(torch.tensor([[111549, 35202.7, -46507.4, 94552.1],[86549.2, 12443.8, 81453.1, 25407.5],[86799.1, 12423.2, 81499.2, 25411.3]]))
momentum.pt
> torch.tensor([[58328.10936461082],[82398.15324417168],[82440.61801612115]])
momentum.m
> torch.tensor([[10045.468856155943],[7467.751089852902],[9586.140737544087]])
```