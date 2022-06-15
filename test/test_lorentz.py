import torch

from torch_hep.lorentz import LorentzTensor, MomentumTensor


def test_LorentzTensor():
    short = LorentzTensor([111549, 35202.7, -46507.4, 94552.1])
    long  = LorentzTensor([[111549, 35202.7, -46507.4, 94552.1],
                                        [86549.2, 12443.8, 81453.1, 25407.5],
                                        [86799.1, 12423.2, 81499.2, 25411.3]])
    
    assert torch.all(torch.eq((short*2).values,(short+short).values))
    assert torch.all(torch.eq((long*2).values,(long+long).values))
    assert torch.all(torch.isclose((short-short/2).values,(short/2).values,1.))
    assert torch.all(torch.isclose((long-long/2).values,(long/2).values,1.))


def test_MomentumTensor():
    short = MomentumTensor([111549, 35202.7, -46507.4, 94552.1])
    long  = MomentumTensor([[111549, 35202.7, -46507.4, 94552.1],
                            [86549.2, 12443.8, 81453.1, 25407.5],
                            [86799.1, 12423.2, 81499.2, 25411.3]])
    short_EEtaPhiPt = MomentumTensor.EEtaPhiPt([111549.0, 1.2600811625519022, -0.9228767507929038, 58328.10936461082])
    long_EEtaPhiPt  = MomentumTensor.EEtaPhiPt([[111549.0, 1.2600811625519022, -0.9228767507929038, 58328.10936461082],
                                                [86549.2, 0.3036619846021553, 1.4191959214152075, 82398.15324417168],
                                                [86799.1, 0.30355425211172465, 1.4195273814629104, 82440.61801612115]])
    short_MEtaPhiPt = MomentumTensor.MEtaPhiPt([10045.468856155943, 1.2600811625519022, -0.9228767507929038, 58328.10936461082])
    long_MEtaPhiPt  = MomentumTensor.MEtaPhiPt([[10045.468856155943, 1.2600811625519022, -0.9228767507929038, 58328.10936461082],
                                                [7467.751089852902, 0.3036619846021553, 1.4191959214152075, 82398.15324417168],
                                                [9586.140737544087, 0.30355425211172465, 1.4195273814629104, 82440.61801612115]])

    assert torch.all(torch.isclose(short[0],short_EEtaPhiPt.e))
    assert torch.all(torch.isclose(long[0],long_EEtaPhiPt.e))
    assert torch.all(torch.isclose(short[1],short_EEtaPhiPt.px))
    assert torch.all(torch.isclose(long[1],long_EEtaPhiPt.px))
    assert torch.all(torch.isclose(short[2],short_EEtaPhiPt.py))
    assert torch.all(torch.isclose(long[2],long_EEtaPhiPt.py))
    assert torch.all(torch.isclose(short[3],short_EEtaPhiPt.pz))
    assert torch.all(torch.isclose(long[3],long_EEtaPhiPt.pz))

    assert torch.all(torch.isclose(short[0],short_MEtaPhiPt.e))
    assert torch.all(torch.isclose(long[0],long_MEtaPhiPt.e))
    assert torch.all(torch.isclose(short[1],short_MEtaPhiPt.px))
    assert torch.all(torch.isclose(long[1],long_MEtaPhiPt.px))
    assert torch.all(torch.isclose(short[2],short_MEtaPhiPt.py))
    assert torch.all(torch.isclose(long[2],long_MEtaPhiPt.py))
    assert torch.all(torch.isclose(short[3],short_MEtaPhiPt.pz))
    assert torch.all(torch.isclose(long[3],long_MEtaPhiPt.pz))

    assert torch.isclose(short.p,torch.tensor([111095.76029921215]))
    assert torch.all(torch.isclose(long.p,torch.tensor([[111095.76029921215],[86226.42700645783],[86268.12659708102]])))

    assert torch.isclose(short.pt,torch.tensor([58328.10936461082]))
    assert torch.all(torch.isclose(long.pt,torch.tensor([[58328.10936461082],[82398.15324417168],[82440.61801612115]])))

    assert torch.isclose(short.m,torch.tensor([10045.468856155943]))
    assert torch.all(torch.isclose(long.m,torch.tensor([[10045.468856155943],[7467.751089852902],[9586.140737544087]])))
