import torch 
import torch.nn as nn
from torch.nn import Parameter
import torch.functional as F
import math


class GCN(nn.Module):
    def __init__(self,in_features,out_features,bias=False):
        super(GCN,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1,1,out_features))
        else:
            self.register_parameter('bias',None)
    
    def reset_parameters(self):
        standard_variance = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-standard_variance,standard_variance)
        if self.bias is not None:
            self.bias.data.uniform_(-standard_variance,standard_variance)
    
    def forward(self,input,adj):
        support = torch.matmul(input,self.weight)
        out = torch.matmul(adj,support)
        if self.bias is not None:
            return out+self.bias
        else:
            return out


#Parameters 
in_features = 2048
out_features = 10
num_nodes = 64



#Forward
y_hat = model(x,adj)

