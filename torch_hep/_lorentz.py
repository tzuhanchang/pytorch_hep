import torch

from torch import Tensor

from pylorentz import Momentum4


METRIC = torch.diag(torch.tensor([1,-1,-1,-1]))

def apply_along_axis(function: callable, tensor: Tensor, dim: int=0) -> Tensor:
    r"""This is equivalent to `numpy.apply_along_axis`, where the `tensor` is 
    sliced along a given `axis` then apply function. This function only works with
    CPU tensors and should not be used in code sections that require high performance.
    
    Args:
        function (callable): This function should only accept only one variable.
        tensor (torch.tensor): tensor to apply function.
        axis (int): axis along which arr is sliced.

    :rtype: :class:`torch.tensor`
    """
    return torch.stack([torch.tensor(function(x)) for x in torch.unbind(tensor, dim=dim)], dim=dim)


def Dot1(tensor):
    tensor1 = tensor[0:4]
    tensor2 = tensor[4:8]
    return (torch.mul(torch.mul(METRIC.to(tensor.device),tensor2),tensor1)).sum()


def Trans1(tensor):
    return torch.sqrt(tensor[1]**2+tensor[2]**2)

def Phi1(tensor):
    return torch.arctan2(tensor[2],tensor[1])

def Theta1(tensor):
    return torch.arctan2(Trans1(tensor),tensor[3])


class LorentzTensor():
    def __init__(self, input: Tensor):
        if input.ndim == 1:
            if input.size() == torch.Size([4]):
                self.values = input
                self.device = input.get_device()
                self.dim = 1
            else:
                raise TypeError("For 1 dimension tensor, expect size 4.")
        else:
            if input.size(dim=1) == 4:
                self.values = input
                self.device = input.get_device()
                self.dim = -1
            else:
                raise TypeError("Expect size 4 in dimension 1.")

    def size(self, dim: int=0):
        return self.values.size(dim)

    def __add__(self, other):
        return __class__(torch.add(self.values, other.values))

    def __sub__(self, other):
        return __class__(torch.sub(self.values, other.values))

    def __mul__(self, other):
        if hasattr(other, "__len__") == False and hasattr(other, "size") == False:
            return torch.multiply(other,self.values)
        elif self.dim == -1 and (other.values).size(dim=1) == 4:
            return apply_along_axis(Dot1, torch.cat((self.values,other.values),dim=1), dim=0)
        elif self.dim == 1 and (other.values).size() == torch.Size([4]):
            return ((METRIC.to(self.device)*other.values)*self.values).sum()
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
        if self.dim == -1:
            return apply_along_axis(Trans1,self.values,dim=0)
        elif self.dim == 1:
            return Trans1(self)

    @property
    def phi(self):
        if self.dim == -1:
            return apply_along_axis(Phi1,self.values,dim=0)
        elif self.dim == 1:
            return Phi1(self)
        
    @property
    def theta(self):
        if self.dim == -1:
            return apply_along_axis(Theta1,self.values,dim=0)
        elif self.dim == 1:
            return Theta1(self)



a = LorentzTensor(torch.tensor([[1,1,1,1],[2,2,2,2]],device='cuda'))
b = LorentzTensor(torch.tensor([[3,3,3,3],[2,1,2,3]],device='cuda'))
c = LorentzTensor(torch.tensor([3,3,3,3],device='cuda'))
d = LorentzTensor(torch.tensor([2,1,2,3],device='cuda'))
a_1_m = Momentum4(1,1,1,1)
a_2_m = Momentum4(2,2,2,2)
b_1_m = Momentum4(3,3,3,3)
b_2_m = Momentum4(2,1,2,3)

print((b*3))
print(b_1_m*3)
print(b_2_m*3)