#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract_dict(ckpt_state, string_name='base_model.temp1.', replace_name=''):
    m_dict = {}
    for k, v in ckpt_state.items():
        if string_name in k:
            m_dict[k.replace(string_name, replace_name)] = v
    return m_dict
        
def replace_dict(ckpt_state, string_name='base_model.temp1.', replace_name=''):
    m_dict = {}
    for k, v in ckpt_state.items():
        # if string_name in k:
        m_dict[k.replace(string_name, replace_name)] = v
    return m_dict

class ShiftConv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias
        ) -> None:
        super(ShiftConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias
        )
        # import pdb; pdb.set_trace()
        # self.conv.weight    = torch.nn.Parameter(torch.ones_like(self.conv.weight))
        # self.conv.bias      = torch.nn.Parameter(torch.zeros_like(self.conv.bias))
    def forward(self, left_fold_2fold, center, right):
        fold_div = 8
        n, c, h, w = center.size()
        fold = c//fold_div
        # import pdb; pdb.set_trace()
        assert left_fold_2fold.size()[1] == fold
        return  self.conv(torch.cat([ right[:, :fold, :, :],
                                     left_fold_2fold, 
                                     center[:, 2*fold:, :, :]], dim=1))
        # return  self.conv(torch.cat([left[:, fold: 2*fold, :, :], center[:, 2*fold:, :, :], right[:, :fold, :, :]], dim=1))

class BiBufferConv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True
        ) -> None:
        super(BiBufferConv, self).__init__()
        self.op = ShiftConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias
        )
        self.out_channels = out_channels
        self.left_fold_2fold = None
        # self.zero_tensor = None
        self.center = None
        
    def reset(self):
        self.left_fold_2fold = None
        self.center = None
        
    def forward(self, input_right, verbose=False):
        fold_div = 8
        if input_right is not None:
            self.n, self.c, self.h, self.w = input_right.size()
            self.fold = self.c//fold_div
        # Case1: In the start or end stage, the memory is empty
        if self.center is None:
            self.center = input_right
            # if verbose:
            
            if input_right is not None:
                if self.left_fold_2fold is None:
                    # In the start stage, the memory and left tensor is empty

                    self.left_fold_2fold = torch.zeros((self.n, self.fold, self.h, self.w), device=torch.device('cuda'))
                if verbose: print("%f+none+%f = none"%(torch.mean(self.left_fold_2fold), torch.mean(input_right)))
            else:
                # in the end stage, both feed in and memory are empty
                if verbose: print("%f+none+none = none"%(torch.mean(self.left_fold_2fold)))
                # print("self.center is None")
            return None
        # Case2: Center is not None, but input_right is None
        elif input_right is None:
            # In the last procesing stage, center is 0
            output =  self.op(self.left_fold_2fold, self.center, torch.zeros((self.n, self.fold, self.h, self.w), device=torch.device('cuda')))
            if verbose: print("%f+%f+none = %f"%(torch.mean(self.left_fold_2fold), torch.mean(self.center), torch.mean(output)))
        else:
            
            output =  self.op(self.left_fold_2fold, self.center, input_right)
            if verbose: print("%f+%f+%f = %f"%(torch.mean(self.left_fold_2fold), torch.mean(self.center), torch.mean(input_right), torch.mean(output)))
            # if output == 57:
                # a = 1
        self.left_fold_2fold = self.center[:, self.fold:2*self.fold, :, :]
        self.center = input_right
        return output

class MemCvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(MemCvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.c1 = BiBufferConv(in_ch, out_ch, kernel_size=3,
                            padding=1,bias=bias)
        self.b1 = norm_fn(out_ch)
        self.relu1 = act_fn(inplace=True)
        self.c2 = BiBufferConv(out_ch, out_ch, kernel_size=3,
                            padding=1,bias=bias)
        self.b2 = norm_fn(out_ch)
        self.relu2 = act_fn(inplace=True)


    def forward(self, x):
        x = self.c1(x)
        if x is not None:
            x = self.b1(x)
            x = self.relu1(x)
        x = self.c2(x)
        if x is not None:
            x = self.b2(x)
            x = self.relu2(x)
        return x
    def load(self, state_dict):
        state_dict = replace_dict(state_dict, 'net.', 'op.conv.')
        self.load_state_dict(state_dict)
    
    def reset(self):
        self.c1.reset()
        self.c2.reset()
    
class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(CvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                            padding=1, bias=bias)
        self.b1 = norm_fn(out_ch)
        self.relu1 = act_fn(inplace=True)
        self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                            padding=1, bias=bias)
        self.b2 = norm_fn(out_ch)
        self.relu2 = act_fn(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.relu2(x)
        return x

def get_norm_function(norm):
    if norm == "bn":
        norm_fn = nn.BatchNorm2d
    elif norm == "in":
        norm_fn = nn.InstanceNorm2d
    elif norm == 'none':
        norm_fn =nn.Identity
    return norm_fn

def get_act_function(act):
    if act == "relu":
        act_fn = nn.ReLU
    elif act == "relu6":
        act_fn = nn.ReLU6
    elif act == 'none':
        act_fn =nn.Identity
    return act_fn

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch, in_ch=4, norm='bn', bias=True, act='relu', interm_ch = 30, blind=False):
        super(InputCvBlock, self).__init__()
        # self.interm_ch = 30
        # if with_sigma: channel_per_frame = 4
        # else: channel_per_frame = 3
        self.interm_ch = interm_ch
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        if blind:
            in_ch = 3
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*in_ch, num_in_frames*self.interm_ch,
                      kernel_size=3, padding=1, groups=num_in_frames, bias=bias),
            norm_fn(num_in_frames*self.interm_ch),
            act_fn(inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch,
                      kernel_size=3, padding=1, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True)
        )

    def forward(self, x):
        if x is not None:
            self.n, self.in_channels, self.h, self.w = x.size()
        if x is None:
            return None
        else:
            return self.convblock(x)
    def load(self, state_dict):
        self.load_state_dict(state_dict)


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(DownBlock, self).__init__()
        self.out_channels = out_ch
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, stride=2, bias=bias),
            norm_fn(out_ch),
            act_fn(inplace=True),
        )
        self.memconv = MemCvBlock(out_ch, out_ch, norm=norm, bias=bias, act=act)
    def reset(self):
        self.memconv.reset()
    def forward(self, x):
        if x is not None: 
            self.n, self.in_channels, self.h, self.w = x.size()
            x = self.convblock(x)
        return self.memconv(x)

    def load(self, ckpt_state):
        self.convblock[0].load_state_dict(extract_dict(ckpt_state,string_name='convblock.0.'))
        self.convblock[1].load_state_dict(extract_dict(ckpt_state,string_name='convblock.1.'))
        self.memconv.load(extract_dict(ckpt_state, string_name='convblock.3.'))


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(UpBlock, self).__init__()
        self.memconv = MemCvBlock(in_ch, in_ch, norm=norm, bias=bias, act=act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2)
        )
        self.out_channels = out_ch
    def reset(self):
        self.memconv.reset()
    def forward(self, x):
        # if x is None: return None
        if x is not None:
            self.n, self.in_channels, self.h, self.w = x.size()
        x = self.memconv(x)
        if x is not None:
            x = self.convblock(x)
        return x

    def load(self, ckpt_state):
        self.convblock[0].load_state_dict(extract_dict(ckpt_state,string_name='convblock.1.'))
        self.memconv.load(extract_dict(ckpt_state, string_name='convblock.0.'))
        



class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch, norm='bn', bias=True, act='relu'):
        super(OutputCvBlock, self).__init__()
        norm_fn = get_norm_function(norm)
        act_fn = get_act_function(act)
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=bias),
            norm_fn(in_ch),
            act_fn(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x):
        if x is None: return None
        if x is not None:
            return self.convblock(x)
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
    """ Definition of the denosing block
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, chns=[32, 64, 128], out_ch=3, in_ch=4, shift_input=False, norm='bn', bias=True,  act='relu', interm_ch=30, blind=False):
        super(DenBlock, self).__init__()
        self.chs_lyr0, self.chs_lyr1, self.chs_lyr2 = chns
        if shift_input:
            self.inc = CvBlock(in_ch=in_ch, out_ch=self.chs_lyr0, norm=norm, bias=bias, act=act)
        else:
            self.inc = InputCvBlock(
                num_in_frames=1, out_ch=self.chs_lyr0, in_ch=in_ch, norm=norm, bias=bias, act=act, interm_ch=interm_ch, blind=blind)

        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, norm=norm, bias=bias, act=act)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, norm=norm, bias=bias, act=act)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, norm=norm, bias=bias,    act=act)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, norm=norm, bias=bias,    act=act)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=out_ch, norm=norm, bias=bias,     act=act)
        self.skip1  = MemSkip()
        self.skip2  = MemSkip()
        self.skip3  = MemSkip()
        self.reset_params()
    def reset(self):
        self.downc0.reset()
        self.downc1.reset()
        self.upc2.reset()
        self.upc1.reset()
    def load_from(self, ckpt_state):
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
        self.skip1.push(self.non_slice(in1))
        x0 = self.inc(in1)
        self.skip2.push(x0)
        # Downsampling
        x1 = self.downc0(x0)
        self.skip3.push(x1)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(self.none_add(x2, self.skip3.pop(x2)))
        # Estimation
        x = self.outc(self.none_add(x1, self.skip2.pop(x1)))

        # Residual
        x = self.none_minus(self.skip1.pop(x), x)

        return x
    def non_slice(self, x):
        if x is None:
            return None
        else:
            return x[:, 0:3, :, :]
    def none_add(self, x1, x2):
        if x1 is None or x2 is None:
            return None
        else: 
            return x1+x2
        
    def none_minus(self, x1, x2):
        if x1 is None or x2 is None:
            return None
        else: 
            x_out = x2
            x_out[:, :3, :, :] = x1[:, :3, :, :] - x_out[:, :3, :, :]
            return x_out
        

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.temp1 = DenBlock()

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
        x = self.temp1(x)
        return x



from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class BSVD(nn.Module):
    """
        Bidirection-buffer based framework with pipeline-style inference
    """
    def __init__(self, chns=[32, 64, 128], mid_ch=3, shift_input=False, in_ch=4, out_ch=3, norm='bn', act='relu', interm_ch=30, blind=False, 
                 pretrain_ckpt='./experiments/pretrained_ckpt/bsvd-64.pth'):
        super(BSVD, self).__init__()
        self.temp1 = DenBlock(chns=chns, out_ch=mid_ch, in_ch=in_ch,  shift_input=shift_input, norm=norm, act=act, blind=blind, interm_ch=interm_ch)
        self.temp2 = DenBlock(chns=chns, out_ch=out_ch, in_ch=mid_ch, shift_input=shift_input, norm=norm, act=act, blind=blind, interm_ch=interm_ch)

        self.shift_num = self.count_shift()
        # Init weights
        self.reset_params()
        if pretrain_ckpt is not None:
            self.load(pretrain_ckpt)
        # self.shift_num = 
        # self.shift_num = 
    def reset(self):
        self.temp1.reset()
        self.temp2.reset()
    def load(self, path):
        ckpt = torch.load(path)
        print("load from %s"%path)
        ckpt_state = ckpt['params']
        # split the dict here
        if 'module' in list(ckpt_state.keys())[0]:
            base_name = 'module.base_model.'
        else:
            base_name = 'base_model.'
        ckpt_state_1 = extract_dict(ckpt_state, string_name=base_name+'nets_list.0.')
        ckpt_state_2 = extract_dict(ckpt_state, string_name=base_name+'nets_list.1.')
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
        x   = self.temp1(x)
        x   = self.temp2(x)
        return x
    
    def forward(self, input, noise_map=None):
        # N, F, C, H, W -> (N*F, C, H, W)
        if noise_map != None:
            input = torch.cat([input, noise_map], dim=2)
        N, F, C, H, W = input.shape
        input = input.reshape(N*F, C, H, W)
        base_out = self.streaming_forward(input)
        NF, C, H, W = base_out.shape
        base_out = base_out.reshape(N, F, C, H, W)
        return base_out
    
    def streaming_forward(self, input_seq):
        """
        pipeline-style inference

        Args:
            Noisy video stream

        Returns:
            Denoised video stream
        """
        out_seq = []
        if isinstance(input_seq, torch.Tensor):
            n,c,h,w = input_seq.shape
            input_seq = [input_seq[i:i+1, ...] for i in np.arange(n)]
        assert type(input_seq) == list, "convert the input into a sequence"
        _,c,h,w = input_seq[0].shape
        with torch.no_grad():
            for i, x in enumerate(input_seq):
                # print("feed in %d image"%i)
                x_cuda = x.cuda()
                x_cuda = self.feedin_one_element(x_cuda)
                # if x_cuda is not None: x_cuda = x_cuda.cpu()
                if isinstance(x_cuda, torch.Tensor):
                    out_seq.append(x_cuda)
                else:
                    out_seq.append(x_cuda)
                # max_mem = torch.cuda.max_memory_allocated()/1024/1024/1024
                # print("max memory required \t\t %.2fGB"%max_mem)
                # print("*****************************************************************************")
            end_out = self.feedin_one_element(None)
            # if end_out is not None: end_out = end_out.cpu()
            # if isinstance(end_out, torch.Tensor): end_out = end_out.cpu()
            out_seq.append(end_out)
            # end_out = self.feedin_one_element(0)
            # end stage
            while 1:
                # print("feed in none")
                end_out = self.feedin_one_element(None)
                # if end_out is not None: end_out = end_out.cpu()
                
                if len(out_seq) == (self.shift_num+len(input_seq)):
                    break
                # if isinstance(end_out, torch.Tensor): end_out = end_out.cpu()
                out_seq.append(end_out)
                # max_mem = torch.cuda.max_memory_allocated()/1024/1024/1024
                # print("max memory required \t\t %.2fGB"%max_mem)
                # print("*****************************************************************************")
            # number of temporal shift is 2, last element is 0
            # TODO fix init and end frames
            out_seq_clip = out_seq[self.shift_num:]
            self.reset()
            return torch.cat(out_seq_clip, dim=0)

    def count_shift(self):
        count = 0
        for name, module in self.named_modules():
            # print(type(module))
            if "BiBufferConv" in str(type(module)):
                count+=1
        return count
#%%
if __name__ == '__main__':
    pass