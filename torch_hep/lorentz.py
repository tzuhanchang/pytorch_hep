import torch

from torch import Tensor


class LorentzTensor(object):
    r""":class:`LorentzTensor`

    Args:
        input (torch.tensor): input tensor with size `torch.Size([N,4])`.
        device (str): device is used to store tensors (default 'cpu').
    """
    def __init__(self, input, **kwargs):
        if torch.is_tensor(input) == True:
            self.values = input.clone()
        else:
            self.values = torch.tensor(input,**kwargs)
        self.device = self.values.device
        self._minkowski = torch.tensor([1,-1,-1,-1],device=self.device)
        if self.values.ndim == 1:
            if self.values.size() == torch.Size([4]):
                self.type = 'single'
                self.size = 1
            else:
                raise TypeError("For 1 dimension tensor, expect size 4.")
        else:
            if self.values.size(dim=1) == 4:
                self.type = 'long'
                self.size = self.values.size(dim=0)
            else:
                raise TypeError("Expect size 4 in dimension 1.")

    def __add__(self, other):
        return self.__class__(torch.add(self.values, other.values))

    def __sub__(self, other):
        return self.__class__(torch.sub(self.values, other.values))

    def __mul__(self, other):
        return self.__class__(torch.multiply(other,self.values))
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self.__class__(torch.div(self.values,other,rounding_mode='floor'))
    
    def __floordiv__(self, other):
        return self.__class__(torch.div(self.values,other,rounding_mode='floor'))
    
    def __div__(self, other):
        return self.__truediv__(other)

    def __getitem__(self, index):
        return self.select(0, index)
    
    def __len__(self):
        return self.size

    def select(self, dim, index):
        if self.type == 'single':
            dim = 0
        
        if dim == 0:
            return self.values.select(0,index)
        else:
            return self.values.select(dim,index).reshape((-1,1))

    def dot(self, other):
        if (other.values).size() == (self.values).size():
            if self.type == 'long':
                return torch.sum(self.values * self._minkowski * other.values, dim=1).reshape(-1,1)
            else:
                return torch.sum(self.values * self._minkowski * other.values, dim=0)
        else:
            raise ValueError("two 'LorentzTensor' must have same size in dim=0")

    def to_tensor(self):
        return self.values

    def to_list(self):
        return self.values.tolist()

    @property
    def mag2(self):
        return self.dot(self)

    @property
    def mag(self):
        return torch.sqrt(torch.abs(self.mag2))
    
    @property
    def trans(self):
        if self.type == 'long':
            return torch.sqrt(self.select(-1,1)**2+self.select(-1,2)**2).reshape(-1,1)
        else:
            return torch.sqrt(self.select(-1,1)**2+self.select(-1,2)**2)

    @property
    def phi(self):
        if self.type == 'long':
            return torch.atan2(self.select(-1,2),self.select(-1,1)).reshape(-1,1)
        else:
            return torch.atan2(self.select(-1,2),self.select(-1,1))
        
    @property
    def theta(self):
        if self.type == 'long':
            return torch.atan2(self.trans,self.select(-1,3)).reshape(-1,1)
        else:
            return torch.atan2(self.trans,self.select(-1,3))

    @property
    def eta(self):
        if self.type == 'long':
            return -torch.log(torch.tan(self.theta/ 2)).reshape(-1,1)
        else:
            return -torch.log(torch.tan(self.theta/ 2))


class MomentumTensor(LorentzTensor):
    r""":class:`MomentumTensor`

    Args:
        input (torch.tensor): input tensor with size `torch.Size([N,4])`.
    """
    @staticmethod
    def EEtaPhiPt(input, **kwargs):
        if torch.is_tensor(input) != True:
            input = torch.tensor(input,**kwargs)
        if input.ndim == 1:
            if input.size() != torch.Size([4]):
                raise TypeError("For 1 dimension tensor, expect size 4.")
            Px = (torch.cos(input[2])*input[3])
            Py = (torch.sin(input[2])*input[3])
            Pz = torch.div(input[3],torch.tan(2*torch.arctan(torch.exp(-input[1]))))
            return MomentumTensor(torch.cat((input[0].unsqueeze(0),Px.unsqueeze(0),Py.unsqueeze(0),Pz.unsqueeze(0))))
        else:
            if input.size(dim=1) != 4:
                raise TypeError("Expect size 4 in dimension 1.")
            Px = (torch.cos(input.select(-1,2))*input.select(-1,3)).reshape([-1,1])
            Py = (torch.sin(input.select(-1,2))*input.select(-1,3)).reshape([-1,1])
            Pz = torch.div(input.select(-1,3),torch.tan(2*torch.arctan(torch.exp(-input.select(-1,1))))).reshape([-1,1])
            return MomentumTensor(torch.cat((input.select(-1,0).reshape([-1,1]),Px,Py,Pz),dim=1))
    
    @staticmethod
    def MEtaPhiPt(input, **kwargs):
        if torch.is_tensor(input) != True:
            input = torch.tensor(input,**kwargs)
        if input.ndim == 1:
            if input.size() != torch.Size([4]):
                raise TypeError("For 1 dimension tensor, expect size 4.")
            Px = (torch.cos(input[2])*input[3])
            Py = (torch.sin(input[2])*input[3])
            Pz = torch.div(input[3],torch.tan(2*torch.arctan(torch.exp(-input[1]))))
            E  = torch.sqrt(input[0]**2+Px**2+Py**2+Pz**2)
            return MomentumTensor(torch.cat((E.unsqueeze(0),Px.unsqueeze(0),Py.unsqueeze(0),Pz.unsqueeze(0))))
        else:
            if input.size(dim=1) != 4:
                raise TypeError("Expect size 4 in dimension 1.")
            Px = (torch.cos(input.select(-1,2))*input.select(-1,3)).reshape([-1,1])
            Py = (torch.sin(input.select(-1,2))*input.select(-1,3)).reshape([-1,1])
            Pz = torch.div(input.select(-1,3),torch.tan(2*torch.arctan(torch.exp(-input.select(-1,1))))).reshape([-1,1])
            E  = torch.sqrt((input.select(-1,0).reshape([-1,1]))**2+Px**2+Py**2+Pz**2)
            return MomentumTensor(torch.cat((E,Px,Py,Pz),dim=1))
    
    @property
    def e(self):
        return self.select(-1,0)

    @property
    def px(self):
        return self.select(-1,1)

    @property
    def py(self):
        return self.select(-1,2)
    
    @property
    def pz(self):
        return self.select(-1,3)
    
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

