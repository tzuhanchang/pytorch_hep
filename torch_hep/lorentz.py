import torch

from torch import Tensor
from torch_scatter import scatter

from pylorentz import Momentum4


class LorentzTensor(object):
    r""":class:`LorentzTensor`

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
            return scatter(scattered,index=indices[0],reduce='sum').reshape(-1,1)
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
        return self.values.select(-1,i).reshape((-1,1))

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
        return torch.sqrt(self[1]**2+self[2]**2).reshape(-1,1)

    @property
    def phi(self):
        return torch.atan2(self[2],self[1]).reshape(-1,1)
        
    @property
    def theta(self):
        return torch.atan2(self.trans,self[3]).reshape(-1,1)

    @property
    def eta(self):
        return -torch.log(torch.tan(self.theta/ 2)).reshape(-1,1)


class MomentumTensor(LorentzTensor):
    r""":class:`MomentumTensor`

    Args:
        input (torch.tensor): input tensor with size `torch.Size([N,4])`.
    """
    @staticmethod
    def EEtaPhiPt(input: Tensor):
        Px = (torch.cos(input.select(-1,2))*input.select(-1,3)).reshape([-1,1])
        Py = (torch.sin(input.select(-1,2))*input.select(-1,3)).reshape([-1,1])
        Pz = torch.div(input.select(-1,3),torch.tan(2*torch.arctan(torch.exp(-input.select(-1,1))))).reshape([-1,1])
        return MomentumTensor(torch.cat((input.select(-1,0).reshape([-1,1]),Px,Py,Pz),dim=1))
    
    @staticmethod
    def MEtaPhiPt(input: Tensor):
        Px = (torch.cos(input.select(-1,2))*input.select(-1,3)).reshape([-1,1])
        Py = (torch.sin(input.select(-1,2))*input.select(-1,3)).reshape([-1,1])
        Pz = torch.div(input.select(-1,3),torch.tan(2*torch.arctan(torch.exp(-input.select(-1,1))))).reshape([-1,1])
        E  = torch.sqrt((input.select(-1,0).reshape([-1,1]))**2+Px**2+Py**2+Pz**2)
        return MomentumTensor(torch.cat((E,Px,Py,Pz),dim=1))
    
    @property
    def e(self):
        return self[0]

    @property
    def px(self):
        return self[1]

    @property
    def py(self):
        return self[2]
    
    @property
    def pz(self):
        return self[3]
    
    @property
    def p2(self):
        return self.px**2 + self.py**2 + self.pz**2

    @property
    def p(self):
        return torch.sqrt(self.p2)
    
    @property
    def pt(self):
        return self.trans

    @property
    def m2(self):
        return self.mag2
    
    @property
    def m(self):
        return self.mag

