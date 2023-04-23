""" Full assembly of the parts to form the complete network. Light v7: run denoise block twice"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .unet_parts import *
# from .unet_parts_ori import *


class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)
        return x


class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch, in_channel=4):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*in_channel, num_in_frames*self.interm_ch,
                      kernel_size=3, padding=1, groups=num_in_frames, bias=False),
            nn.BatchNorm2d(num_in_frames*self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


# class InputCvBlock2(nn.Module):
#     '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

#     def __init__(self, num_in_frames, out_ch):
#         super(InputCvBlock2, self).__init__()
#         self.interm_ch = 30
#         self.convblock = nn.Sequential(
#             nn.Conv2d(num_in_frames*(3), num_in_frames*self.interm_ch,
#                       kernel_size=3, padding=1, groups=num_in_frames, bias=False),
#             nn.BatchNorm2d(num_in_frames*self.interm_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_in_frames*self.interm_ch, out_ch,
#                       kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.convblock(x)


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch, interm_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(interm_ch, out_ch, kernel_size=3, padding=1, bias=False)
    def forward(self, x, in1):
        out =  self.convblock(x)
        out = self.out_conv(torch.cat([out, in1], axis=1))
        return out


class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, in_channel, out_channel, mid_channel=32):
        super(DenBlock, self).__init__()
        num_input_frames = 1
        self.chs_lyr0 = mid_channel
        self.chs_lyr1 = mid_channel*2
        self.chs_lyr2 = mid_channel*4

        self.inc = InputCvBlock(
            num_in_frames=num_input_frames, out_ch=self.chs_lyr0, in_channel=in_channel)
        # if stage2:
        #     self.inc = InputCvBlock2(
        #         num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        # self.interm_ch = 30
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out_channel, interm_ch=self.chs_lyr0+3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in1):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        x0 = self.inc(in1)
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1+x2)
        # Estimation
        
        x = self.outc(x0+x1, in1[:, :3, :, :])

        # Residual
        # x = in1[:, :3, :, :] - x

        return x


class UNet(nn.Module):
    def __init__(self, channel=4, mid_channel=32):
        super(UNet, self).__init__()
        self.temp1 = DenBlock(channel, out_channel=channel-1,   mid_channel=mid_channel)
        self.temp2 = DenBlock(channel-1, out_channel=channel-1, mid_channel=mid_channel)

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x1 = self.temp1(x)
        x2 = self.temp2(x1)
        return x2