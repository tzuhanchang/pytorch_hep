import torch

from torch_hep.data import TensorLoader


def test_TensorLoader():
    a = torch.tensor([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    b = torch.tensor([0,0,0]).reshape(-1,1)

    tensorloader = TensorLoader(a=a, b=b, batch_size=2)

    batch = 0
    for data in tensorloader:
        if batch == 0:
            assert torch.eq(data.a,torch.tensor([[1,2,3,4],[1,2,3,4]]))
            assert torch.eq(data.b,torch.tensor([0,0]).reshape(-1,1))
        else:
            assert torch.eq(data.a,torch.tensor([[1,2,3,4]]))
            assert torch.eq(data.b,torch.tensor([[0]]))

        batch += 1