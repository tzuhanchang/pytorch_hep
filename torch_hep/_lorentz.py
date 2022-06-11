import torch

from torch import Tensor
from torch_scatter import scatter

from pylorentz import Momentum4


class LorentzTensor(object):
    r"""Convert a `torch.Tensor` with size `torch.Size([N,4])` into a
    `LorentzTensor`. Each `dim=0` is a 4 vector, `torch.Tensor` allows
    high performance Lorentz vector manipulation over entire vector space,
    it also enables GPU computation.

    Args:
        input (torch.tensor): input tensor with size `torch.Size([N,4])`.
    """
    def __init__(self, input: Tensor):
        if input.ndim == 1:
            if input.size() == torch.Size([4]):
                self.values = input
                self.device = input.device
                self.type = 'single'
                self.size = 1
            else:
                raise TypeError("For 1 dimension tensor, expect size 4.")
        else:
            if input.size(dim=1) == 4:
                self.values = input
                self.device = input.device
                self.type = 'long'
                self.size = self.values.size(dim=0)
            else:
                raise TypeError("Expect size 4 in dimension 1.")

    def __add__(self, other):
        return self.__class__(torch.add(self.values, other.values))

    def __sub__(self, other):
        return self.__class__(torch.sub(self.values, other.values))

    def __mul__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return __class__(torch.multiply(other,self.values))
        elif (other.values).size() == (self.values).size():
            indices = torch.arange(0,self.size,1,device=self.device,dtype=torch.int64).expand(4,self.size).transpose(0,1).reshape(1,4*self.size)
            METRIC = torch.diag(torch.tensor([1,-1,-1,-1],device=self.device).repeat(1,self.size)[0])
            scattered = torch.diagonal(METRIC*self.values.reshape(1,4*self.size)*other.values.reshape(1,4*self.size),0)
            return scatter(scattered,index=indices[0],reduce='sum').reshape(self.size,1)
        else:
            raise TypeError("Expect two Lorentz Tensor are same size.")
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return __class__(torch.div(self.values,other,rounding_mode='floor'))
        else:
            raise TypeError("Division only possible by scalars.")
    
    def __floordiv__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return __class__(torch.div(self.values,other,rounding_mode='floor'))
        else:
            raise TypeError("Floor division only possible by scalars.")
    
    def __div__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return self.__truediv__(other)

    def __getitem__(self, i):
        return self.values[i]

    def dot(self, other):
        if (other.values).size() == torch.Size([4]):
            return self*other

    def to_tensor(self):
        return self.values

    def to_list(self):
        return self.values.tolist()

    @property
    def mag2(self):
        return self * self

    @property
    def mag(self):
        return torch.sqrt(torch.abs(self.mag2))
    
    @property
    def trans(self):
        return torch.sqrt(self.values.select(1,1)**2+self.values.select(1,2)**2).reshape(self.size,1)

    @property
    def phi(self):
        return torch.atan2(self.values.select(1,2),self.values.select(1,1)).reshape(self.size,1)
        
    @property
    def theta(self):
        return torch.atan2(self.trans,self.values.select(1,3)).reshape(self.size,1)

    @property
    def eta(self):
        return -torch.log(torch.tan(self.theta/ 2)).reshape(self.size,1)


# class MomentumTensor(LorentzTensor):
#     @staticmethod
#     def EEtaPhiPt(e_eta_phi_pt: Tensor):
#         px = torch.cos(e_eta_phi_pt.select(-1,2)) * 
#         return MomentumTensor(torch.cat(()))


"""
TEST
"""
b1 = Momentum4(111549, 35202.7, -46507.4, 94552.1)
b2 = Momentum4(86549.2, 12443.8, 81453.1, 25407.5)
lep_pos = Momentum4(96806.2, 29784.7, -14348.2, 90985.8)
lep_neg = Momentum4(27209.1, 26173.4, 3376.81, 6623.67)
met_met = 39915.24609375
met_phi = -1.7860441207885742

print(lep_pos.mag2, lep_neg.mag2)

b = LorentzTensor(torch.tensor([[111549, 35202.7, -46507.4, 94552.1],[86549.2, 12443.8, 81453.1, 25407.5]],device='cpu',dtype=torch.float64))
lep = LorentzTensor(torch.tensor([[96806.2, 29784.7, -14348.2, 90985.8],[27209.1, 26173.4, 3376.81, 6623.67]],device='cpu',dtype=torch.float64))

print((lep).mag2)

