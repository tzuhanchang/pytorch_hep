import torch

from torch import Tensor

from pylorentz import Momentum4


METRIC = torch.diag(torch.tensor([1,-1,-1,-1]))


class LorentzTensor():
    def __init__(self, input: Tensor):
        if input.size() == torch.Size([4]):
            self.values = input
            self.device = input.get_device()
        else:
            raise TypeError("Expect a size=[1,4] tensor")

    def size(self, dim: int=0):
        return self.values.size(dim)

    def __add__(self, other):
        return __class__(torch.add(self.values, other.values))

    def __sub__(self, other):
        return __class__(torch.sub(self.values, other.values))

    def __mul__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return torch.multiply(other,self.values)
        elif (other.values).size() == torch.Size([4]):
            return ((METRIC.to(self.device)*other.values)*self.values).sum()
        else:
            raise TypeError("Expect a scalar or Lorentz Tensor")
    
    def __truediv__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return __class__(self.values/other)
        else:
            raise TypeError("Expect a scalar.")
    
    def __floordiv__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return __class__(torch.div(self.values,other,rounding_mode='floor'))
        else:
            raise TypeError("Expect a scalar for floor divide")
    
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
        return torch.sqrt(self[1]**2+self[2]**2)

    @property
    def phi(self):
        return torch.arctan2(self[2],self[1])
        
    @property
    def theta(self):
        return torch.arctan2(self.trans,self[3])


        

print((LorentzTensor(torch.tensor([1,1,1,1],device='cuda'))).to_list())
print( torch.tensor([3,3,3,3],device='cuda').ndim )
print(torch.tensor([[3,3,3,3],[2,1,2,3]],device='cuda').size())

b = Momentum4(3,3,3,3)
# print(a.theta)
# print(b.theta)