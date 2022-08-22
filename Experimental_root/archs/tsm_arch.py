# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn
import torch
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class TSN(nn.Module):
    """
        Temporal-shift denoiser during training
    """
    def __init__(self, 
                 num_segments=11,
                 base_model='WNet_multistage', 
                 shift_type='TSM', 
                 shift_div=8,
                 inplace=False,
                 net2d_opt={},
                 enable_past_buffer=True,
                 **kwargs):

        super(TSN, self).__init__()

        self.reshape = True
        self.num_segments = num_segments
        self.shift_type = shift_type
        self.shift_div = shift_div
        self.base_model_name = base_model
        self.net2d_opt = net2d_opt
        self.enable_past_buffer = enable_past_buffer        
        self.inplace = inplace
        self._prepare_base_model(base_model)

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))
        ShiftOutputCvBlock = int

        if base_model == 'WNet_multistage':
            from .archs_2d.wnet_models import WNet, CvBlock
            self.base_model = WNet(**self.net2d_opt)

        else:
            print('No such model')
            raise NotImplementedError

        if self.shift_type != 'no_temporal_shift':
            from .temporal_shift_ops.temporal_shift import TemporalShift
            for m in self.base_model.modules():
                if isinstance(m, CvBlock) or isinstance(m, ShiftOutputCvBlock):
                    print('Adding temporal shift... {} {}'.format(m.c1, m.c2))
                    m.c1 = TemporalShift(m.c1, self.num_segments, self.shift_div, self.shift_type,
                                         inplace=self.inplace, enable_past_buffer=self.enable_past_buffer)
                    m.c2 = TemporalShift(m.c2, self.num_segments, self.shift_div, self.shift_type,
                                         inplace=self.inplace, enable_past_buffer=self.enable_past_buffer)

    def forward(self, input, noise_map=None):
        # N, F, C, H, W -> (N*F, C, H, W)
        if noise_map != None:
            input = torch.cat([input, noise_map], dim=2)
        if len(input.shape)==5:
            N, F, C, H, W = input.shape
            model_input = input.reshape(N*F, C, H, W)
        else:
            model_input = input
        base_out = self.base_model(model_input)
        if len(input.shape)==5:
            NF, C, H, W = base_out.shape
            base_out = base_out.reshape(N, F, C, H, W)
        return base_out
    
