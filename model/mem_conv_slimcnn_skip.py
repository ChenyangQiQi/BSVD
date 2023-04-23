# This python file implement the idea of buffer as buffer conv
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# input_seq = [1,2,3]
# base_tensor = torch.ones(1, 4, 2000, 2000)
# input_seq = [i*base_tensor for i in input_seq]
# zero_tensor = 0*base_tensor
# zero_tensor = zero_tensor.cuda()

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

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, in_channels=4, out_channels=16):
        super(InputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(num_in_frames*self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if x is None: 
            return None
        else:
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
        if x is None: 
            return None
        else:
            # print(x.shape)
            out =  self.convblock(x)
            out = self.out_conv(torch.cat([out, in1], axis=1))
        return out

class MemCvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_channels, out_channels):
        super(MemCvBlock, self).__init__()
        self.c1 = MemShiftConv(in_channels, out_channels, kernel_size=3,
                            padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.c2 = MemShiftConv(out_channels, out_channels, kernel_size=3,
                            padding=1)


    def forward(self, x):
        x = self.c1(x)
        if x is not None:
            x = self.relu1(x)
        x = self.c2(x)
        return x

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
            
class SlimCNNMemConv(nn.Module):
    def __init__(self, in_channel=5, out_channel=4, mid_channel=16, pretrain=False) -> None:
        super(SlimCNNMemConv, self).__init__()
        self.inc    = InputCvBlock(in_channels=in_channel, out_channels=mid_channel)
        self.midc   = MemCvBlock(in_channels=mid_channel, out_channels=out_channel)
        self.outc   = OutputCvBlock(in_ch=out_channel, out_ch=out_channel, interm_ch=out_channel+in_channel)
        self.skip1  = MemSkip()
        if pretrain:
            self.load_from()
        # self.add3 = BufferAdd()
        # self.add4 = BufferAdd()

    def feedin_one_element(self, x):
        # print("state after feed in ", x)
        self.skip1.push(x)
        x = self.inc(x)
        # if x is not None:
            # print("shape after x = self.inc(x)", x.shape)
        x = self.midc(x)
        # if x is not None:
            # print("shape after x = self.midc(x)", x.shape)
        x = self.outc(x, self.skip1.pop(x))
        
        # self.print_state()
        # x = self.add3.step(x)
        # x = self.add4.step(x)
        return x

    def print_state(self):
        def print_buffer(buffer):
            # print("buffer left is", buffer.left)
            print("buffer center is", buffer.center)
            print("buffer left is", buffer.center)
        print("start of the first buffer")
        # print_buffer(self.add1)
        print_buffer(self.add2)
        # print_buffer(self.add3)
        # print_buffer(self.add4)
    # def end(self):
        
    def forward(self, input_seq):
        out_seq = []
        if isinstance(input_seq, torch.Tensor):
            n,c,h,w = input_seq.shape
            input_seq = [input_seq[i:i+1, ...] for i in np.arange(n)]
        assert type(input_seq) == list, "convert the input into a sequence"
        _,c,h,w = input_seq[0].shape
        with torch.no_grad():
            for x in input_seq:
                # print("feed in %d"%x)
                x_cuda = x.cuda()
                x_cuda = self.feedin_one_element(x_cuda)
                if isinstance(x_cuda, torch.Tensor):
                    out_seq.append(x_cuda.cpu())
                else:
                    out_seq.append(x_cuda)
                    
            end_out = self.feedin_one_element(torch.zeros(1, c, h, w).cuda()).cpu()
            out_seq.append(end_out)
            # end_out = self.feedin_one_element(0)
            # end stage
            while 1:
                # print("feed in none")
                end_out = self.feedin_one_element(None)
                if end_out is None:
                    break
                out_seq.append(end_out.cpu())
            # number of temporal shift is 2, last element is 0
            out_seq_clip = out_seq[2:-1]
            return torch.cat(out_seq_clip, dim=0)
        
    def load_from(self, path="./logs/1108_crvd_slimcnn_skip_batch8_lr_1e-4_seq7/ckpt.pth"):
        ckpt = torch.load(path)
        print("load from %s"%path)
        ckpt_state = ckpt['state_dict']
        self.inc.convblock[0].weight     =   torch.nn.Parameter(ckpt_state['base_model.inc.convblock.0.weight'])
        self.inc.convblock[2].weight     =   torch.nn.Parameter(ckpt_state['base_model.inc.convblock.2.weight'])
        self.midc.c1.op.conv.weight      =   torch.nn.Parameter(ckpt_state['base_model.midc.c1.net.weight'])
        self.midc.c2.op.conv.weight      =   torch.nn.Parameter(ckpt_state['base_model.midc.c2.net.weight'])
        self.outc.convblock[0].weight    =   torch.nn.Parameter(ckpt_state['base_model.outc.convblock.0.weight'])
        self.outc.out_conv.weight        =   torch.nn.Parameter(ckpt_state['base_model.outc.out_conv.weight'])
        
        
        
        

#%%
# import numpy as np
# add_net = SlimCNNMemConv().cuda()
# with torch.profiler.profile(
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('../tb_profile_logger/buffer_slim_conv'),
#     record_shapes=True,
#     with_stack=True,
#     profile_memory=True,
# ) as prof:
# for step in np.arange(4):
    # out_seq = add_net.forward(input_seq)
    # print(out_seq)
    # prof.step()
                
# print(out_seq)
#%%
# for result in out_seq:
#     if isinstance(result, torch.Tensor):
#         print(torch.mean(result))
#     else:
#         print(result)

            
#%% 
