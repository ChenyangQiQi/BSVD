""" Full assembly of the parts to form the complete network. Light v7: run denoise block twice"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .unet_parts import *
# from .unet_parts_ori import *
# input_seq = [1,2,3,4,5,6,7]
# base_tensor = torch.ones(1, 5, 960, 540)
# input_seq = [i*base_tensor for i in input_seq]
# zero_tensor = 0*base_tensor
# zero_tensor = zero_tensor.cuda()

#%%

def extract_dict(ckpt_state, string_name='base_model.temp1.'):
    m_dict = {}
    for k, v in ckpt_state.items():
        if string_name in k:
            m_dict[k.replace(string_name, '')] = v
    return m_dict

class ShiftConv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
        ) -> None:
        super(ShiftConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )
        # import pdb; pdb.set_trace()
        # self.conv.weight    = torch.nn.Parameter(torch.ones_like(self.conv.weight))
        # self.conv.bias      = torch.nn.Parameter(torch.zeros_like(self.conv.bias))
    def forward(self, left, center, right):
        fold_div = 8
        n, c, h, w = center.size()
        fold = c//fold_div
        # import pdb; pdb.set_trace()
        return  self.conv(torch.cat([ right[:, :fold, :, :],
                                     left[:, fold: 2*fold, :, :], 
                                     center[:, 2*fold:, :, :]], dim=1))
        # return  self.conv(torch.cat([left[:, fold: 2*fold, :, :], center[:, 2*fold:, :, :], right[:, :fold, :, :]], dim=1))

class MemShiftConv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
        ) -> None:
        super(MemShiftConv, self).__init__()
        self.op = ShiftConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding
        )
        self.out_channels = out_channels
        self.left = None
        self.zero_tensor = None
        self.center = None
        
    def forward(self, input_right, verbose=False):
        
        # Case1: In the start or end stage, the memory is empty
        if self.center is None:
            self.center = input_right
            # if verbose:
            
            if input_right is not None:
                if self.left is None:
                    # In the start stage, the memory and left tensor is empty
                    self.left = torch.zeros_like(input_right)
                if verbose: print("%f+none+%f = none"%(torch.mean(self.left), torch.mean(input_right)))
            else:
                # in the end stage, both feed in and memory are empty
                if verbose: print("%f+none+none = none"%(torch.mean(self.left)))
                # print("self.center is None")
            return None
        # Case2: Center is not None, but input_right is None
        elif input_right is None:
            # In the last procesing stage, center is 0
            if torch.count_nonzero(self.center) == 0:
                if verbose: print("%f+%f+none = 0"%(torch.mean(self.left), torch.mean(self.center)))
                self.left = self.center
                self.center = input_right
                n, c, h, w = self.left.shape
                return torch.zeros(n, self.out_channels, h, w).cuda()
            if verbose: print("%f+%f+none = 0"%(torch.mean(self.left), torch.mean(self.center)))
            # Stange, what is this case? this case should not be reached
            raise NotImplementedError
            return torch.zeros_like(self.center)
        # Case3: Center, input_right, and left are all valid value
        else:
            
            output =  self.op(self.left, self.center, input_right)
            if verbose: print("%f+%f+%f = %f"%(torch.mean(self.left), torch.mean(self.center), torch.mean(input_right), torch.mean(output)))
            # if output == 57:
                # a = 1
            self.left = self.center
            self.center = input_right
            return output

class MemCvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_channels, out_channels):
        super(MemCvBlock, self).__init__()
        self.c1 = MemShiftConv(in_channels, out_channels, kernel_size=3,
                            padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.c2 = MemShiftConv(out_channels, out_channels, kernel_size=3,
                            padding=1)
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.c1(x)
        if x is not None:
            x = self.relu1(x)
        x = self.c2(x)
        if x is not None:
            x = self.relu2(x)
        return x

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, in_channels=4, out_channels=16):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, self.interm_ch,
                      kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(num_in_frames*self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.interm_ch, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if x is None: 
            return None
        else:
            return self.convblock(x)

    def load(self, state_dict):
        self.load_state_dict(state_dict)
          
class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=2, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.memconv = MemCvBlock(out_ch, out_ch)

    def forward(self, x):
        if x is not None: 
            x = self.convblock(x)
        return self.memconv(x)

    def load(self, ckpt_state):
        self.convblock[0].weight = torch.nn.Parameter(ckpt_state['convblock.0.weight'])
        self.memconv.c1.op.conv.weight = torch.nn.Parameter(ckpt_state['convblock.2.c1.net.weight'])
        self.memconv.c2.op.conv.weight = torch.nn.Parameter(ckpt_state['convblock.2.c2.net.weight'])

class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.memconv = MemCvBlock(in_ch, in_ch)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        # if x is None: return None
        x = self.memconv(x)
        if x is not None:
            x = self.convblock(x)
        return x

    def load(self, ckpt_state):
        self.convblock[0].weight = torch.nn.Parameter(ckpt_state['convblock.1.weight'])
        self.memconv.c1.op.conv.weight = torch.nn.Parameter(ckpt_state['convblock.0.c1.net.weight'])
        self.memconv.c2.op.conv.weight = torch.nn.Parameter(ckpt_state['convblock.0.c2.net.weight'])


class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch, interm_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(interm_ch, out_ch, kernel_size=3, padding=1, bias=False)
    def forward(self, x, in1):
        if x is None: return None
        out =  self.convblock(x)
        out = self.out_conv(torch.cat([out, in1], axis=1))
        return out

    def load(self, state_dict):
        self.load_state_dict(state_dict)
          

class MemSkip(nn.Module):
    def __init__(self):
        super(MemSkip, self).__init__()
        self.mem_list = []
    def push(self, x):
        if x is not None:
            self.mem_list.insert(0,x)
            return 1
        else:
            return 0
    def pop(self, x):
        if x is not None:
            return self.mem_list.pop()
        else:
            return None
            

class DenBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, in_channel, out_channel, mid_channel=32, channels=None):
        super(DenBlock, self).__init__()
        num_input_frames = 1
        self.out_channel = out_channel
        self.chs_lyr0 = mid_channel
        self.chs_lyr1 = mid_channel*2
        self.chs_lyr2 = mid_channel*4

        if channels is not None:
            self.chs_lyr0, self.chs_lyr1, self.chs_lyr2 = channels
            
        self.inc = InputCvBlock( in_channels=in_channel, out_channels=self.chs_lyr0)

        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        # self.interm_ch = 30
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out_channel, interm_ch=self.chs_lyr0+in_channel)
        self.skip1  = MemSkip()
        self.skip2  = MemSkip()
        self.skip3  = MemSkip()

        self.reset_params()

    def load_from(self, ckpt_state):
        # ckpt = torch.load(path)
        # print("load from %s"%path)
        # ckpt_state = ckpt['state_dict']
        # # split the dict here
        self.inc.load(extract_dict(ckpt_state, string_name='inc.'))
        self.downc0.load(extract_dict(ckpt_state, string_name='downc0.'))
        self.downc1.load(  extract_dict(ckpt_state, string_name='downc1.'))
        self.upc2.load(  extract_dict(ckpt_state, string_name='upc2.'))
        self.upc1.load(  extract_dict(ckpt_state, string_name='upc1.'))
        self.outc.load( extract_dict(ckpt_state, string_name='outc.'))


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
        self.skip1.push(in1)
        
        x0 = self.inc(in1)
        self.skip2.push(x0)
        
        # Downsampling
        x1 = self.downc0(x0)
        self.skip3.push(x1)
        
        x2 = self.downc1(x1)
        # None
        # Upsampling
        x2 = self.upc2(x2) # out_seq=N,N,N
        # self.upc2.memconv.c1.center = None
        # if x2 is None: return None
        x1 = self.upc1(self.none_add(x2, self.skip3.pop(x2)))
        # Estimation
        # import pdb; pdb.set_trace()
        # if x1 is None: return None
        x = self.outc(self.none_add(x1, self.skip2.pop(x1)), self.skip1.pop(x1))

        return x

    def none_add(self, x1, x2):
        if x1 is None or x2 is None:
            return None
        else: 
            return x1+x2
            
class UNet(nn.Module):
    def __init__(self, channel=5, mid_channel=32, channels1=[16,32,64], channels2=[16,32,64], pretrain=False):
        super(UNet, self).__init__()
        self.temp1 = DenBlock(channel, out_channel=channel-1,   mid_channel=mid_channel, channels=channels1)
        self.temp2 = DenBlock(channel-1, out_channel=channel-1, mid_channel=mid_channel, channels=channels2)

        self.shift_num = self.count_shift()
        # Init weights
        self.reset_params()
        if pretrain:
            self.load()
        
        # self.shift_num = 
    def load(self, path="./logs/1106_1008_crvd_tswnet16_batch16_lr_1e-4_seq7_nonorm_10_300/ckpt.pth"):
            ckpt = torch.load(path)
            print("load from %s"%path)
            ckpt_state = ckpt['state_dict']
            # split the dict here
            ckpt_state_1 = extract_dict(ckpt_state, string_name='base_model.temp1.')
            ckpt_state_2 = extract_dict(ckpt_state, string_name='base_model.temp2.')
            self.temp1.load_from(ckpt_state_1)
            self.temp2.load_from(ckpt_state_2)
            
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def feedin_one_element(self, x):
        x = self.temp1(x)
        x = self.temp2(x)
        return x
    
    def forward(self, input_seq):
        out_seq = []
        if isinstance(input_seq, torch.Tensor):
            n,c,h,w = input_seq.shape
            input_seq = [input_seq[i:i+1, ...] for i in np.arange(n)]
        assert type(input_seq) == list, "convert the input into a sequence"
        _,c,h,w = input_seq[0].shape
        with torch.no_grad():
            for i, x in enumerate(input_seq):
                print("feed in %d image"%i)
                x_cuda = x.cuda()
                x_cuda = self.feedin_one_element(x_cuda)
                if isinstance(x_cuda, torch.Tensor):
                    out_seq.append(x_cuda.cpu())
                else:
                    out_seq.append(x_cuda)
                    
            end_out = self.feedin_one_element(torch.zeros(1, c, h, w).cuda())
            if isinstance(end_out, torch.Tensor): end_out = end_out.cpu()
            out_seq.append(end_out)
            # end_out = self.feedin_one_element(0)
            # end stage
            while 1:
                # print("feed in none")
                end_out = self.feedin_one_element(None)
                
                if len(out_seq) == (self.shift_num+len(input_seq)):
                    break
                if isinstance(end_out, torch.Tensor): end_out = end_out.cpu()
                out_seq.append(end_out)
            # number of temporal shift is 2, last element is 0
            # TODO fix init and end frames
            out_seq_clip = out_seq[self.shift_num:]
            return torch.cat(out_seq_clip, dim=0)

    def count_shift(self):
        count = 0
        for name, module in self.named_modules():
            print(type(module))
            if "MemShiftConv" in str(type(module)):
                count+=1
        return count

