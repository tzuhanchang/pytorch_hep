# PyTorch HEP
<br />
PyTorch HEP is a project built using PyTorch to act as a foundation for performing high performance computations for High Energy Physics. PyTorch HEP is currently under development, this includes tensor based Lorentz manipulation for particle reconstruction.
<br />

## LorentzTensor
### Tensor to LorentzTensor
Define a `LorentzTensor` with single 4-vector:
```
vector4 = torch.tensor([112849, 35192.7, -44507.4, 94562.1])
LorentzTensor(vector4)
```
`LorentzTensor` supports GPU tensor computation, set `device='cuda'` (default `cpu`). `torch.tensor` default `dtype=toch.float32`, for precisional computing set `dtype=torch.float64`.<br />
<br />
Define a `LorentzTensor` with 4-vector vector space:
```
vector4Space = torch.tensor([[112849, 35192.7, -44507.4, 94562.1],[82849.2, 12143.8, 83553.1, 21007.5],...])
LorentzTensor(vector4Space)
```
`LorentzTensor` computes properties of every 4-vectors in this vector space tensor wise. It does not involve `numpy` so included tools fully support GPU computing.<br />
<br />
