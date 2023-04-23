  
#Borrow from: https://github.com/talsperre/INFER/blob/master/LayerNorm2D.py
import torch
import torch.nn as nn

from torch.autograd import Variable

class LayerNormConv2d(nn.Module):
    """
    Layer norm the just works on the channel axis for a Conv2d
    Ref:
    - code modified from https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/LayerNorm.py
    - paper: https://arxiv.org/abs/1607.06450
    Usage:
        ln = LayerNormConv(3)
        x = Variable(torch.rand((1,3,4,2)))
        ln(x).size()
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features).cuda()).unsqueeze(-1).unsqueeze(-1)
        self.beta = nn.Parameter(torch.zeros(features).cuda()).unsqueeze(-1).unsqueeze(-1)
        self.eps = eps
        self.features = features

    def _check_input_dim(self, input):
        if input.size(1) != self.gamma.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input.size(1), self.features))

    def forward(self, x):
        self._check_input_dim(x)
        x_flat = x.transpose(1,-1).contiguous().view((-1, x.size(1)))
        mean = x_flat.mean(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        std = x_flat.std(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps) + self.beta.expand_as(x)
