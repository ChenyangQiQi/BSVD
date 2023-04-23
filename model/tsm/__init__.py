# from .tsn_2_step import TSN as TSN2Step
from .tsn import TSN
import torch.nn as nn


class ExpTSN():
    def __init__(self) -> None:
        self.names = ['unet_model_outconcat', "unet_model_upsamplecat", 
                      "unet_model_c5", "unet_skip_3", "wnet_no_norm", "SlimCNN", "SlimCNNSkip",
                      "unet_model_c5_pyram_shift", "unet_model_c5_mid_c16",
                      "ts_transunet","single_unet"]
        # self.step_tsm = ["UNet%dstep"%i for i in [1,2,4,8]]+["UNet%dstep_depth%d"%(i, int(4/i)) for i in [1,2,4]]
        # self.names = self.names+self.step_tsm
    def get_model(self, args):
        model = None
        # import pdb; pdb.set_trace()
        # if args['model'] == 'unet_model_c5_pyram_shift':
        #     model = TSN(5,
        #         base_model="unet_model_c5",
        #         is_shift="pyram_shift", shift_div=8,
        #         )            
        if args['model'] in self.names:
            model = TSN(
                num_segments=args['sequence_length'],
                base_model=args['model'],
                is_shift=args['shift'], shift_div=8,
                noise_channel=args['noise_channel'],
                channel_per_frame=args['channel_per_frame'],
                model_channel=args['model_channel'],
                args=args
                )
            # if 'data_parallel' not in args or args['data_parallel'] == True:
            #     model = nn.DataParallel(model, device_ids=[0,1,2,3])
        else: print('name and model mismatches')
        return model

exp_tsn = ExpTSN()