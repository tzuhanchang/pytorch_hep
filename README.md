# PyTorch HEP
<br />
PyTorch HEP is a project built upon PyTorch to act as a foundation for high performance computing for High Energy Physics. PyTorch HEP is currently under development, this includes tensor based Lorentz manipulation for particle reconstruction.
<br />

## LorentzTensor
### Tensor to LorentzTensor
Define a `LorentzTensor` with single 4-vector:
```
vector4 = torch.tensor([112849, 35192.7, -44507.4, 94562.1])
LorentzTensor(vector4)
```
`LorentzTensor` fully supports GPU computing, upon input set `device='cuda'` (default `cpu`) for `vector4` tensor. In need of high precision set `dtype=torch.float64` (default `dtype=toch.float32`).<br />
<br />
Define a `LorentzTensor` with 4-vector vector space:
```
vector4Space = torch.tensor([[112849, 35192.7, -44507.4, 94562.1],[82849.2, 12143.8, 83553.1, 21007.5],...])
LorentzTensor(vector4Space)
```
`LorentzTensor` computes properties of every 4-vectors in this vector space, it does not involve `numpy` so included tools fully support GPU computing.<br />
<br />
### Lorentz vector Operations in LorentzTensor
Basic operations, sum, subtract, dot product, multiply and divide:
```
LoretnzTensor1 = LorentzTensor(torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]]))
LoretnzTensor2 = LorentzTensor(torch.tensor([[2,2,2,2],[2,2,2,2],[2,2,2,2]]))
LoretnzTensor1+LoretnzTensor2
LoretnzTensor1-LoretnzTensor2
LoretnzTensor1*LoretnzTensor2
5*LoretnzTensor1
LoretnzTensor2/2
```
outputs:
```
LorentzTensor(torch.tensor([[3,3,3,3],[3,3,3,3],[3,3,3,3]]))
LorentzTensor(torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]]))
torch.tensor([[-4],[-4],[-4]])
LorentzTensor(torch.tensor([[5,5,5,5],[5,5,5,5],[5,5,5,5]]))
LorentzTensor(torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]]))
```