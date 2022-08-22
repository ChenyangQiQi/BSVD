#%% 

import os, subprocess
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax([int(x.split()[2]) 
            for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

import torch
import torch.nn as nn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Experimental_root.scripts.profiler import MyTimeit, MyFlops

def main(test_name, inp, device, opt):
    inp         = inp.to(device)
    if isinstance(test_name, nn.Module):
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
    if isinstance(test_name, nn.Module):
        test_name.eval()
    print('size of tensor'+str(inp.shape))
    print('use device', device)

    with torch.no_grad():
        # for profiler in (MyTimeit('line'),):
        # for profiler in (MyFlops(mode='thop'),):
        # for profiler in (MyTimeit('torchprofile19', path='./torchprofiler/'+opt['name']),):
        for profiler in (MyTimeit('time'),
                        #  MyFlops(mode='ptflops')
                        ):
                print('\n test function name: '+str(test_name.__class__))
                new_function = profiler(test_name)
                out = new_function(inp)
                del out
                torch.cuda.empty_cache()
    max_mem = torch.cuda.max_memory_allocated()/1024/1024/1024
    print("max memory required \t\t %.2fGB"%max_mem)
    print("*****************************************************************************")


# %%

if __name__ == '__main__':
    import yaml
    import os.path as osp
    from basicsr.utils.options import *
    from basicsr.models import build_model
    
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_path = "./options/test/bsvd_c64.yml"

    with open(test_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    opt['dist'] = False
    opt['is_train'] = False
    opt['root_path'] = root_path

    temp_psz=10
    inp = torch.randn(1, temp_psz, 4, 540, 960).cuda()
    opt['val']['temp_psz'] = temp_psz

    device      = torch.device("cuda")
    print("device", device)

    model = build_model(opt)
    print(model)
    net_g_forward = model.net_g.half()
    net_g_forward.eval()    
    
    with torch.cuda.amp.autocast(True):
        main(net_g_forward, inp, device, opt)

