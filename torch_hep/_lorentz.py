import torch

from torch import Tensor
from torch_scatter import scatter

from pylorentz import Momentum4


class LorentzTensor():
    def __init__(self, input: Tensor):
        if input.ndim == 1:
            if input.size() == torch.Size([4]):
                self.values = input
                self.device = input.get_device()
                self.type = 'single'
            else:
                raise TypeError("For 1 dimension tensor, expect size 4.")
        else:
            if input.size(dim=1) == 4:
                self.values = input
                self.device = input.get_device()
                self.type = 'long'
            else:
                raise TypeError("Expect size 4 in dimension 1.")

    def size(self, dim: int=0):
        if self.type == 'long':
            return self.values.size(dim)
        elif self.type == 'single':
            return 1

    def __add__(self, other):
        return __class__(torch.add(self.values, other.values))

    def __sub__(self, other):
        return __class__(torch.sub(self.values, other.values))

    def __mul__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return torch.multiply(other,self.values)
        elif (other.values).size() == (self.values).size():
            indices = torch.arange(0,self.size(),1,device='cuda').expand(4,self.size()).transpose(0,1).reshape(1,4*self.size()).to(torch.long)
            METRIC = torch.diag(torch.tensor([1,-1,-1,-1],device='cuda').repeat(1,self.size())[0])
            scattered = torch.diagonal(METRIC*self.values.reshape(1,4*self.size())*other.values.reshape(1,4*self.size()),0)
            return scatter(scattered,index=indices[0],reduce='sum')
        else:
            raise TypeError("Expect two Lorentz Tensor are same size.")
    
    def __truediv__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return 
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
        return torch.sqrt(self.values.select(1,1)**2+self.values.select(1,2)**2)

    @property
    def phi(self):
        return torch.arctan2(self.values.select(1,2),self.values.select(1,1))
        
    @property
    def theta(self):
        return torch.arctan2(self.trans,self.values.select(1,3))



a = LorentzTensor(torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]],device='cuda'))
b = LorentzTensor(torch.tensor([[3,3,3,3],[2,1,2,3],[2,1,2,3]],device='cuda'))
c = LorentzTensor(torch.tensor([3,3,3,3],device='cuda'))
d = LorentzTensor(torch.tensor([2,1,2,3],device='cuda'))
a_1_m = Momentum4(1,1,1,1)
a_2_m = Momentum4(2,2,2,2)
b_1_m = Momentum4(3,3,3,3)
b_2_m = Momentum4(2,1,2,3)
print(b.theta)
print(b_2_m.theta)

# print((b*3))
# print(b_1_m*3)
# print(b_2_m*3)
# print(torch.range(0,1,1).expand(4,2).transpose(0,1).reshape(1,8))
# print(torch.range(0,1,1).expand(4*8,2).transpose(0,1).reshape(8,8))
# lenth = torch.arange(0,2,1,device='cuda').expand(4*8,2).transpose(0,1).reshape(8,8).to(torch.long)
# lenth2 =torch.arange(0,2,1,device='cuda').expand(4,2).transpose(0,1).reshape(1,8).to(torch.long)
# from torch_scatter import scatter, segment_coo, segment_csr
# b=torch.tensor([[3,3,3,3],[2,1,2,3]],device='cuda').reshape(1,8)
# a=torch.tensor([[1,1,1,1],[2,2,2,2]],device='cuda').reshape(1,8)
# print(torch.tensor([1,-1,-1,-1],device='cuda').repeat(1,2)[0])
# print(torch.diagonal(scatter(segment_coo(torch.diag(torch.tensor([1,-1,-1,-1],device='cuda').repeat(1,2)[0])*b*a,
#               index=lenth2[0],
#               reduce='sum'),
#               index=lenth2[0],
#               reduce='sum')),0)
# diaa = torch.diagonal((torch.diag(torch.tensor([1,-1,-1,-1],device='cuda').repeat(1,2)[0])*b*a),0)
# print(scatter(diaa,index=lenth2[0],reduce='sum'))