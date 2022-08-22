import torch
import torch.nn as nn
import torch.nn.functional as F
from Experimental_root.models import global_queue_buffer

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, shift_type='TSM', inplace=False, enable_past_buffer=True,
                 **kwargs):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.shift_type = shift_type
        self.inplace = inplace
        self.enable_past_buffer = enable_past_buffer
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        if 'TSM' in self.shift_type:
            if self.net.training:
                x = shift(x, self.n_segment, self.shift_type, fold_div=self.fold_div, inplace = self.inplace)
            else:
                x = batch_shift(x, self.shift_type, fold_div=self.fold_div, enable_past_buffer=self.enable_past_buffer)

        return self.net(x)

def shift(x, n_segment, shift_type, fold_div=3, stride=1, inplace=False):
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)

    fold = c // fold_div # 32/8 = 4

    if inplace:
        # Due to some out of order error when performing parallel computing. 
        # May need to write a CUDA kernel.
        print("WARNING: use inplace shift. it has bugs")
        raise NotImplementedError  
        
    else:
        out = torch.zeros_like(x)
        if not 'toFutureOnly' in shift_type:
            out[:, :-stride, :fold] = x[:, stride:, :fold]  # backward (left shift)
            out[:, stride:, fold: 2 * fold] = x[:, :-stride, fold: 2 * fold]  # forward (right shift)
        else:
            out[:, stride:, : 2 * fold] = x[:, :-stride, : 2 * fold] # right shift only
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(nt, c, h, w)


    # Use batch_shift during validating or testing.
def batch_shift(x, shift_type, fold_div=3, stride=1, enable_past_buffer=True):
    nt, c, h, w = x.size()

    fold = c // fold_div
    
    out = torch.zeros_like(x)
    if not 'toFutureOnly' in shift_type: 
        out[:-stride, :fold] = x[stride:, :fold]  # backward (left) shift
        out[stride:, fold: 2 * fold] = x[:-stride, fold: 2 * fold] # forward (right) shift
        
        if enable_past_buffer:
            # memory-based inference
            if global_queue_buffer.get_batch_index() > 0:
                out[:stride, fold: 2 * fold] = global_queue_buffer.get()
            # Keep stride=1, future_buffer_length is abandened
            global_queue_buffer.put(x[-stride-global_queue_buffer.get_future_buffer_length(), fold: 2 * fold])
    else:
        out[stride:, : 2 * fold] = x[:-stride, : 2 * fold] # forward (right) shift only
        
        if enable_past_buffer:
            # memory-based inference
            if global_queue_buffer.get_batch_index() > 0:
                out[:stride, : 2 * fold] = global_queue_buffer.get()
            global_queue_buffer.put(x[-stride-global_queue_buffer.get_future_buffer_length(), : 2 * fold])
    out[:, 2 * fold:] = x[:, 2 * fold:]  # not shift

    
    return out


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
    tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')
