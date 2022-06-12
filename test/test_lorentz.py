import torch

from torch_hep.lorentz import LorentzTensor, MomentumTensor


def test_LorentzTensor():
    short = LorentzTensor(torch.tensor([111549, 35202.7, -46507.4, 94552.1]))
    long  = LorentzTensor(torch.tensor([[111549, 35202.7, -46507.4, 94552.1],
                                        [86549.2, 12443.8, 81453.1, 25407.5],
                                        [86799.1, 12423.2, 81499.2, 25411.3]]))
    
    assert torch.eq((short*2).values,(short+short).values)
    assert torch.eq((long*2).values,(long+long).values)
    assert torch.eq((short-short/2).values,(short/2).values)
    assert torch.eq((long-long/2).values,(long/2).values)
