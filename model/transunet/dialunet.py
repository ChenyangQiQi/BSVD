""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet.unet_parts import *
from .layernorm2d import LayerNormConv2d
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_layer == 'bn':
            norm_layer = nn.BatchNorm2d
        if norm_layer == 'in':
            norm_layer = nn.InstanceNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DilatedConvolutions(nn.Module):
    """
    Sequential Dialted convolutions
    """
    def __init__(self, n_channels, n_convolutions, dropout, norm='bn'):
        super(DilatedConvolutions, self).__init__()
        kernel_size = 3
        padding = 1
        self.dropout = nn.Dropout2d(dropout)
        self.non_linearity = nn.ReLU(inplace=True)
        self.strides = [2**(k+1) for k in range(n_convolutions)]
        convs = [nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size,dilation=s, padding=s) for s in self.strides ]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for c in convs:
            self.convs.append(c)
            if norm == 'bn':
                self.bns.append(nn.BatchNorm2d(n_channels))
            else:
                self.bns.append(nn.InstanceNorm2d(n_channels))
            
    def forward(self,x):
        skips = []
        for (c,bn,s) in zip(self.convs,self.bns,self.strides):
            x_in = x
            x = c(x)
            x = bn(x)
            x = self.non_linearity(x)
            x = self.dropout(x)
            x = x_in + x
            skips.append(x)
        return x, skips

class DialUnet(nn.Module):
    def __init__(self, n_channels, n_classes, dim=64, residual_num=16, bilinear=True, norm='bn',dropout=0):
        super(DialUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(dim, dim * 2, norm=norm)
        self.down2 = Down(dim * 2, dim * 4, norm=norm)
        self.down3 = Down(dim * 4, dim * 8, norm=norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim * 16 // factor, norm=norm)
        self.up1 = Up(dim * 16, dim * 8 // factor, bilinear)
        self.up2 = Up(dim * 8, dim * 4 // factor, bilinear)
        self.up3 = Up(dim * 4, dim * 2 // factor, bilinear)
        self.up4 = Up(dim * 2, dim, bilinear)
        self.outc = OutConv(dim, n_classes)
        self.DilatedConvolutions = DilatedConvolutions(dim * 8, 6, dropout=dropout, norm='norm')


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x,skips = self.DilatedConvolutions(x5)
        for d in skips:
            x += d
        x += skips[-1]

        x5 = x
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits