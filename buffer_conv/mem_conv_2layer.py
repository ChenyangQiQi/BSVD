# This python file implement the idea of buffer as buffer conv
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

input_seq = [1,2,3]
base_tensor = torch.ones(1, 8, 2000, 2000)
input_seq = [i*base_tensor for i in input_seq]
zero_tensor = 0*base_tensor
zero_tensor = zero_tensor.cuda()

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
        )
        self.conv.weight    = torch.nn.Parameter(torch.ones_like(self.conv.weight))
        self.conv.bias      = torch.nn.Parameter(torch.zeros_like(self.conv.bias))
    def forward(self, left, center, right):
        fold_div = 8
        n, c, h, w = center.size()
        fold = c//fold_div
        return  self.conv(torch.cat([left[:, fold: 2*fold, :, :], 
                                     center[:, 2*fold:, :, :], 
                                     right[:, :fold, :, :]], dim=1))

class MemShiftConv(nn.Module):
    def __init__(self) -> None:
        super(MemShiftConv, self).__init__()
        self.op = ShiftConv(8, 8, 1, 1, 0)
        self.left = zero_tensor
        self.center = None
        
    def step(self, input_right, verbose=False):
        # order of input / center None need to think
        if self.center is None:
            self.center = input_right
            if verbose:
                if input_right is not None:
                    print("%f+none+%f = none"%(torch.mean(self.left), torch.mean(input_right)))
                else:
                    print("%f+none+none = none"%(torch.mean(self.left)))
                # print("self.center is None")
            return None
        elif input_right is None:
            # print("input_right is None")
            if torch.equal(self.center, zero_tensor):
                if verbose: print("%f+%f+none = 0"%(torch.mean(self.left), torch.mean(self.center)))
                self.left = self.center
                self.center = input_right
                return zero_tensor
            if verbose: print("%f+%f+none = 0"%(torch.mean(self.left), torch.mean(self.center)))
            return zero_tensor

        else:
            output =  self.op(self.left, self.center, input_right)
            if verbose: print("%f+%f+%f = %f"%(torch.mean(self.left), torch.mean(self.center), torch.mean(input_right), torch.mean(output)))
            # if output == 57:
                # a = 1
            self.left = self.center
            self.center = input_right
            return output


class TwoLayerMemConv(nn.Module):
    def __init__(self) -> None:
        super(TwoLayerMemConv, self).__init__()
        self.add1 = MemShiftConv()
        # self.add2 = BufferAdd()
        # self.add3 = BufferAdd()
        # self.add4 = BufferAdd()

    def feedin_one_element(self, x):
        # print("state after feed in ", x)
        x = self.add1.step(x)
        # x = self.add2.step(x)
        
        # self.print_state()
        # x = self.add3.step(x)
        # x = self.add4.step(x)
        return x
    def print_state(self):
        def print_buffer(buffer):
            # print("buffer left is", buffer.left)
            print("buffer center is", buffer.center)
        print("start of the first buffer")
        print_buffer(self.add1)
        # print_buffer(self.add2)
        # print_buffer(self.add3)
        # print_buffer(self.add4)
    # def end(self):
        
    def forward(self, input_seq):
        out_seq = []
        with torch.no_grad():
            for x in input_seq:
                # print("feed in %d"%x)
                x_cuda = x.cuda() # increase memory
                x_cuda = self.feedin_one_element(x_cuda) # increase memory
                if isinstance(x_cuda, torch.Tensor):
                    out_seq.append(x_cuda.cpu())
                else:
                    out_seq.append(x_cuda)
                
                del x_cuda
                torch.cuda.empty_cache()

            end_out = self.feedin_one_element(zero_tensor)
            out_seq.append(end_out)
            # end_out = self.feedin_one_element(0)
            # end stage
            while 1:
                # print("feed in none")
                end_out = self.feedin_one_element(None)
                if end_out is None:
                    break
                out_seq.append(end_out)
        return out_seq
#%%
import numpy as np
add_net = TwoLayerMemConv().cuda()

out_seq = add_net.forward(input_seq)

for result in out_seq:
    if isinstance(result, torch.Tensor):
        print(torch.mean(result))
    else:
        print(result)

        