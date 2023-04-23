#%% 
r'''test in server7 py37torch170 
    0511: It is better to test the code on a empty server
    0511: When some program stops, the time cost is smaller
    0511: Haven't find other difference
    
'''
import os, subprocess
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) 
            for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
import torch
import torch.nn as nn

import argparse
import model as Model

# from model.tsn import TSN
#%%
#**********************Step1: Define available device****************************

def get_device(device_name):
    # device_name = "CUDA"
    assert device_name in ["CUDA", "Multi-thread CPU", "Single-thread CPU"]
    if device_name == 'CUDA' and torch.cuda.is_available():
        device = torch.device("cuda")
        print('os.environ["CUDA_VISIBLE_DEVICES"]', os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        device = torch.device("cpu")

        if device_name == "Single-thread CPU":
            os.environ["OMP_NUM_THREADS"] = '1'
            # Use command OMP_NUM_THREADS=1 python operator_profile.py
        elif device_name == "Multi-thread CPU":
            os.environ["OMP_NUM_THREADS"] = '32'
        print('os.environ["OMP_NUM_THREADS"] = '+os.environ["OMP_NUM_THREADS"])
    return device     


#%%
#**********************Step2: Define profiler****************************
from profiler import MyTimeit, MyFlops
#**********************Step3: Input data****************************

class Input():
    def __init__(self, channel=64) -> None:
        # channel = 64
        height = 960
        width = 512
        # if device == torch.device("cpu"):
        #     height = 960
        #     width = 640        

        # Standard Unet use (568, 568, 64) -> (30, 30, 1024)
        # image in denoising (1024, 1024, 64) -> (64, 64, 1024 )

        # size of test data in fastDVDnet
        # channel = 512
        # height = 960*64//512
        # width = 540*64//512

        self.inp_chw        =   torch.randn(1, channel,height,width)
        self.inp_chw_96_54        =   torch.randn(1, channel,96,54)
        self.inp_denoise_5frames  =   torch.randn(1, 1+3*5, height,width)
        self.inp_3d         =   torch.randn(1, 32, 3, 2, 2) # N, C, D, H, W
    
    def get_input(self, name):
        if name == 'inp_denoise_5frames':
            return self.inp_denoise_5frames
        if name == '1_16_960_512':
            return torch.randn(1, 16, 960, 512)
        if name == '1_3_960_512':
            return torch.randn(1, 3, 960, 512)
        if name == '1_3_256_256':
            return torch.randn(1, 3, 256, 256)
        if name == '1_4_960_512':
            return torch.randn(1, 4, 960, 512)
        if name == '1_4_960_540':
            return torch.randn(1, 4, 960, 540)
        if name == '1_20_960_540':
            return torch.randn(1, 20, 960, 540)
        if name == '1_20_960_512':
            return torch.randn(1, 20, 960, 512)
        if name == '1_4_5_960_512':
            return torch.randn(1, 4, 5, 960, 512)
            
            
#%%
#**********************Step3: Define Main****************************


def main(test_name, inp, device):
    
    inp         = inp.to(device)
    test_name   = test_name.to(device)

    print('size of tensor '+str(inp.shape))
    print('use device ', device)

    with torch.no_grad(): out = test_name(inp)
    if isinstance(out, torch.Tensor):
        print('output shape is', out.shape)
    elif isinstance(out, list):
        print('output shape is', out[-1].shape)
    del out
    torch.cuda.empty_cache()

    test_name = test_name.eval()
    print('size of tensor'+str(inp.shape))
    print('use device', device)

    with torch.no_grad():
        # for profiler in (MyTimeit('line'), MyFlops(as_strings=True, print_per_layer_stat=False, verbose=False)):
        
        # for profiler in (MyTimeit('time', device), #):
        # for profiler in (MyTimeit('torchprofile19', device, savepath='../../tb_profile_logger/mem_wnet_nonorm_none_as_end'), #):
        for profiler in (MyTimeit('time', device), #):
        # for profiler in (MyTimeit('torchprofile19', device, savepath='../../tb_profile_logger/inplace_shift'), #):
                        MyFlops(mode='ptflops')):
        # for profiler in (MyFlops(mode='thop'),):
                        # thop ptflops
        # for profiler in (MyTimeit('line'),):
            # for function in test_list:
                print('\n test function name: '+str(test_name.__class__))
                # check if it is a module
                new_function = profiler(test_name)
                out = new_function(inp)
                del out
                torch.cuda.empty_cache()
    max_mem = torch.cuda.max_memory_allocated()/1024/1024/1024
    print("max memory required \t\t %.2fGB"%max_mem)
    print("*****************************************************************************")


# %%
if __name__ == '__main__':

    # from model import involution
    # for channel in [32, 32*64]:
    #     inp     =   torch.randn(1, channel, 960*32//channel, 480*32//channel)
    #     model_list = [
    #                     involution(channel, 3, 1),
    #                     nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
    #                     ]
    #     for test_model in model_list:
    #         # for device_name in ["Single-thread CPU"]:
    #         for device_name in ["Single-thread CPU"]:
    #     # device_name = "CUDA" # ["CUDA", "Multi-thread CPU", "Single-thread CPU"] 
    #             print("device_name", device_name)
    #             device      = get_device(device_name)
    #             print("device", device)

    #             # print("input name", input_name)
    #             # inp         = input_all.get_input(input_name)
    #             # conv_channel = 256
    #             # input_all   = Input(conv_channel)
    #             # inp     =   torch.randn(1, 32, 960, 480) # N, C, D, H, W
    #             # inp     =   torch.randn(1, 32*64, 960//64, 480//64) # N, C, D, H, W
                
    #             main(test_model, inp, device)

#%%
    from model import get_model
    
    # inp = torch.randn(1, 4, 1024, 1024)

    # inp = torch.randn(7, 5, 540, 960).cuda()
    inp = torch.randn(16, 5, 540, 960).cuda()
    # inp = torch.randn(7, 5, 1080, 1920).cuda()
    # args = {
    #     'model': "transunet",
    #     'in_channel': 4,
    #     'out_channel':3,
    #     'dim':64,
    #     'residual_num':8,
    #     'norm':'bn'
    # }
    base=16
    scale=1
    
    args = {
        'model': "wnet_no_norm",
        'channels1': [base*scale,   base*2*scale,   base*4*scale], \
        'channels2': [base,         base*2,         base*4], \
        # 'channels1': [16, 32, 64], \
        # 'channels2': [16, 32, 64], \
            
        'shift':'shift', \
        'channel_per_frame': 5,
        'noise_channel': 0,
        'model_channel': 16
    }
    
    # args = {'model': "mem_wnet_nonorm_none_as_end",
    #         'shift':'shift',
    #         'channel_per_frame': 5,
    #         'noise_channel': 0,
    #         'model_channel': 16
    #         }
    
    # base=46
    # args = {
    #     'model': "single_unet",
    #     'channels1': [base,   base*2,   base*4], \
    #     'shift':'shift', \
    #     'channel_per_frame': 4,
    #     'noise_channel': 0,
    #     'model_channel': 16
    # }
    # inp = torch.randn(7, 5, 540, 960)
    # args = {
    #     'model': "fastdvd_crvd",
    #     'image_channel': 4,
    # }
    # 38.92 GMac for default setting
    test_model = get_model(args)
    device_name = "CUDA"
    # device_name = "Single-thread CPU"
    print("device_name", device_name)
    device      = get_device(device_name)
    print("device", device)
    main(test_model, inp, device)
# %%
