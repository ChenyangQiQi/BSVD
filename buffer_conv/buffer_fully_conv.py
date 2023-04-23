# This python file implement the idea of buffer as buffer conv
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

input_seq = [1,2,3]
base_tensor = torch.ones(1, 1, 2000, 2000).cuda()
input_seq = [1*base_tensor, 2*base_tensor, 3*base_tensor]
zero_tensor = 0*base_tensor

class OpConv(nn.Module):
    def __init__(self) -> None:
        super(OpConv, self).__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.conv.weight    = torch.nn.Parameter(torch.ones_like(self.conv.weight))
        self.conv.bias      = torch.nn.Parameter(torch.zeros_like(self.conv.bias))
    def forward(self, left, center, right):
        return  self.conv(torch.cat([left, center, right], dim=1))

class BufferAdd(nn.Module):
    def __init__(self) -> None:
        super(BufferAdd, self).__init__()
        self.op = OpConv()
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
            # if torch.equal(self.center, zero_tensor):
            if torch.count_nonzero(self.center) == 0:
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


class AddNet(nn.Module):
    def __init__(self) -> None:
        super(AddNet, self).__init__()
        self.add1 = BufferAdd()
        self.add2 = BufferAdd()
        # self.add3 = BufferAdd()
        # self.add4 = BufferAdd()

    def feedin_one_element(self, x):
        # print("state after feed in ", x)
        x = self.add1.step(x)
        x = self.add2.step(x)
        
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
        print_buffer(self.add2)
        # print_buffer(self.add3)
        # print_buffer(self.add4)
    # def end(self):
        
    def forward(self, input_seq):
        out_seq = []
        for x in input_seq:
            # print("feed in %d"%x)
            out_seq.append(self.feedin_one_element(x))
        
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
add_net = AddNet().cuda()
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('../tb_profile_logger/buffer_fully_conv2000'),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    for step in np.arange(4):
        out_seq = add_net.forward(input_seq)
        # print(out_seq)
        prof.step()
                
print(out_seq)
            
        