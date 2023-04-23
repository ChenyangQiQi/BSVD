# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu


from torch import nn
import torch
import numpy as np

# from .basic_ops import ConsensusModule
# from .transforms import *
# from torch.nn.init import normal_, constant_


class TSN(nn.Module):
    def __init__(self, num_segments,
                 base_model='unet',
                 print_spec=True,
                 is_shift=False, shift_div=8,
                 noise_channel=1,
                 channel_per_frame = 3,
                 model_channel=32,
                 args=None):
        super(TSN, self).__init__()

        self.num_segments = num_segments
        self.reshape = True
        self.print_spec = print_spec
        self.is_shift = is_shift
        self.shift_div = shift_div
        self.base_model_name = base_model
        self.noise_channel = noise_channel
        self.channel =  channel_per_frame
        self.model_channel = model_channel
        self.args = args
        if self.args== None:
            self.args['channels1'] = [model_channel, model_channel*2, model_channel*4]
            self.args['channels2'] = [model_channel, model_channel*2, model_channel*4]
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        num_segments:       {}
            """.format(base_model, self.num_segments)))

        self._prepare_base_model(base_model)
        self.criterion = nn.MSELoss(reduction='sum')
    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if base_model == "unet_model_c5":
            from .unet_model_channel_arg import UNet, DenBlock, CvBlock
            print('build a model with basic channel %d'%self.model_channel)
            # import pdb; pdb.set_trace()
            self.base_model = UNet(channel=self.channel, mid_channel=self.model_channel,
                                   channels1=self.args['channels1'], 
                                   channels2=self.args['channels2'])
        elif base_model == "wnet_in_norm":
            from .wnet_in_norm import UNet, DenBlock, CvBlock
            print("build a %s"%base_model)
            print('build a model with basic channel %d'%self.model_channel)
            # import pdb; pdb.set_trace()
            self.base_model = UNet(channel=self.channel, mid_channel=self.model_channel,
                                   channels1=self.args['channels1'], 
                                   channels2=self.args['channels2'],)
                                   
        elif base_model == "wnet_no_norm":
            from .unet_model_channel_arg_nonorm import UNet, DenBlock, CvBlock
            print("build a %s"%base_model)
            print('build a model with basic channel %d'%self.model_channel)
            # import pdb; pdb.set_trace()
            self.base_model = UNet(channel=self.channel, mid_channel=self.model_channel,
                                   channels1=self.args['channels1'], 
                                   channels2=self.args['channels2'])
                                   
        elif base_model == "single_unet":
            from .unet_model_channel_arg import SingleUNet, DenBlock, CvBlock
            print('build a single_unet model with basic channel %d'%self.model_channel)
            # import pdb; pdb.set_trace()
            self.base_model = SingleUNet(channel=self.args['channel_per_frame'], mid_channel=self.model_channel,
                                   channels1=self.args['channels1'])
        elif base_model == "unet_skip_3": # a bug model that
            from .unet_skip_3 import UNet, DenBlock, CvBlock
            print('build a model with basic channel %d'%self.model_channel)
            # import pdb; pdb.set_trace()
            self.base_model = UNet(channel=5, mid_channel=self.model_channel)
            
        elif base_model == "ts_transunet":
            from .transunet import UNet, DenBlock, CvBlock
            self.base_model = UNet(channel=5)
            # self.channel = 4
        elif base_model == "SlimCNN":
            from .slim_cnn import SlimCNN, CvBlock
            # input -> 16 channel -> output
            self.base_model = SlimCNN(5, 4)
        elif base_model == "SlimCNNSkip":
            from .slim_cnn_skip import SlimCNN, CvBlock
            # input -> 16 channel -> output
            self.base_model = SlimCNN(5, 4)
        
        else:
            import pdb; pdb.set_trace()
            raise ValueError('Unknown base model: {}'.format(base_model))
        if self.is_shift == 'pyram_shift' or self.is_shift == 'shift':
            
            if self.is_shift == 'pyram_shift':
                from .temporal_shift_pyramid import TemporalShift
                print('use pyram_shift')
            elif self.is_shift == 'shift':
                from .temporal_shift import TemporalShift
                print('use left_right_shift')
                
            for m in self.base_model.modules():
                if isinstance(m, CvBlock):
                    if hasattr(m, 'c1'):
                        if self.print_spec:
                            # import pdb; pdb.set_trace()
                            print('Adding temporal shift... {} {}'.format(m.c1, m.c2))
                        m.c1 = TemporalShift(m.c1, n_segment=self.num_segments, n_div=self.shift_div)
                        m.c2 = TemporalShift(m.c2, n_segment=self.num_segments, n_div=self.shift_div)
                    # deeper convblock
                    elif hasattr(m, 'convblock'):
                        for i in np.arange(m.depth):
                            # import pdb; pdb.set_trace()
                            if self.print_spec:
                                # print(m)
                                # print(i)
                                print('Adding temporal shift{}... {}'.format(i, m.convblock[i*3]))
                            m.convblock[i*3] = TemporalShift(m.convblock[i*3], n_segment=self.num_segments, n_div=self.shift_div)




    def forward(self, input, gt_train=None, return_type=None, ):
        N,C, H,W = input.shape
        if (C == self.channel):
            pass
        else:
            assert (C-self.noise_channel)%self.channel == 0, "input channel %d is unexpected"%C
            input = prepare_nf_cp1_from_n_fcp1(input, image_channel=self.channel, share_channel=self.noise_channel)
            
        base_out = self.base_model(input)
        # Following 140-143 not necessary ???
        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        # base_out = base_out[:, 2, ...] # extract central image
        N, L, C, H, W = base_out.shape # 20, 5, 3, 96, 96
        base_out = base_out.view(-1, C, H, W) # 100, 3, 96, 96,
        if gt_train is not None:
            loss = self.criterion(gt_train, base_out) / (N*2)
            # loss.backward()
            if return_type=="loss_and_base_out":
                return (loss, base_out)
            else:
                return loss
        else:
            return base_out


def prepare_nf_cp1_from_n_fcp1(img_input, image_channel=3, share_channel=1):
    N,C, H,W = img_input.shape
    if (C-share_channel)%image_channel!=0:
        import pdb; pdb.set_trace()
    assert (C-share_channel)%image_channel==0, "input image_channel should be num*image_channel+share_channel"

    F = int((C-share_channel) / image_channel)
    img_train = img_input[:, 0:image_channel*F, :, :].reshape(-1, image_channel, H, W)
    if share_channel != 0:
        noise = img_input[:, C-share_channel:, :, :] # n1hw
        noise = torch.cat([noise for _ in range(F)], axis=0).reshape((-1, 1, H, W))
        # import pdb; pdb.set_trace()
        img_input = torch.cat((img_train, noise), dim=1) # n5, 4, h, w
    else:
        img_input = img_train
    N1, C1, H1, W1 = img_input.shape
    assert N1 == N*F and C1 == (image_channel+share_channel) and H1 == H and W1 == W, "reshape is not correct"
    return img_input