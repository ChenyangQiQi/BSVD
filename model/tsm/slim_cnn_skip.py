import torch
import torch.nn as nn
import torch.nn.functional as F

class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=False)


    def forward(self, x):
        x = self.c1(x)
        x = self.relu1(x)
        x = self.c2(x)
        return x

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, in_channel=4, out_channel=16):
        super(InputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(num_in_frames*self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch, interm_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(interm_ch, out_ch, kernel_size=3, padding=1, bias=False)
    def forward(self, x, in1):
        out =  self.convblock(x)
        out = self.out_conv(torch.cat([out, in1], axis=1))
        return out


class SlimCNN(nn.Module):
    def __init__(self, in_channel=5, out_channel=4, mid_channel=16):
        super(SlimCNN, self).__init__()
        self.inc  = InputCvBlock(in_channel=in_channel, out_channel=mid_channel)
        self.midc = CvBlock(in_ch=mid_channel, out_ch=out_channel)
        self.outc = OutputCvBlock(in_ch=out_channel, out_ch=out_channel, interm_ch=out_channel+in_channel)

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
        x1 = x
        x = self.inc(x)
        x = self.midc(x)
        x = self.outc(x, x1)
        return x