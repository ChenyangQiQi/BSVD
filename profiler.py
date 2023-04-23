#%% 
r'''test in server7 py37torch170 
    0511: It is better to test the code on a empty server
    0511: When some program stops, the time cost is smaller
    0511: Haven't find other difference
    
'''
import torch
import torch.nn as nn
import torchvision.models as models
import os, subprocess
import numpy as np
import time
# lib for profiler
from torchstat import stat
from ptflops import get_model_complexity_info
from thop import profile
from line_profiler import LineProfiler
from functools import wraps
import model as Model

#**********************Step2: Define profiler****************************
class MyProfiler(object):
    def __init__(self):
        pass
    def __call__(self, func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            # log_string = func.__name__ + " was called"
            # print(log_string)
            out = self.profile_module(func, *args, **kwargs)
            return out
        return wrapped_function
    def profile_module(self, func, *args, **kwargs):
        pass    

def profile_best_repeat(module, x, waitting=1.0):
    repeat = 1
    test_time_list = [0]
    first = True
    while(np.mean(test_time_list)*repeat < waitting):
        
        repeat = repeat*10
        best_number = np.ceil(repeat/10.0)
        test_time_list = []
        all_list = []
        for i in range(repeat):
            
            # we only support torch module
            # if next(module.parameters()).device == torch.device("cuda"):
            if first: 
                print("next(module.parameters()).device", next(module.parameters()).device)
            #     print("device == torch.device('cuda'):", next(module.parameters()).device == torch.device("cuda"))
                print("next(model.parameters()).is_cuda", next(module.parameters()).is_cuda)
            first = False
            # import pdb; pdb.set_trace()
            if next(module.parameters()).is_cuda:
                torch.cuda.synchronize()
            st = time.time()
            with torch.no_grad(): out = module(x)
            # if next(module.parameters()).device == torch.device("cuda"):
            if next(module.parameters()).is_cuda:
                torch.cuda.synchronize()
            test_time = time.time()-st
            del out
            torch.cuda.empty_cache()
            all_list.append(test_time)
            
            if len(test_time_list) < best_number:
                test_time_list.append(test_time)
                test_time_list = sorted(test_time_list)
            elif test_time< test_time_list[-1]:
                test_time_list[-1] = test_time
                test_time_list = sorted(test_time_list)
            # all_list.append(test_time)
        # print('len of test_time_list, 'len(test_time_list))
        # import pdb; pdb.set_trace()
        print('%d loops, best of %d: %f sec per loop'%(repeat, best_number, np.mean(test_time_list)))
    # print('test result form time.time', time.time()-st)
    with torch.no_grad(): out = module(x)
    return out, np.mean(test_time_list)

class MyTimeit(MyProfiler):
    
    def __init__(self, mode, device=torch.device("cuda"), savepath='../../tb_profile_logger/tsnet16_wnet_no_norm'):
        self.mode   = mode
        self.device = device
        self.savepath = savepath
        
    def profile_module(self, func, *args, **kwargs):
        out = self.profile_module_time(func, self.mode, *args, **kwargs)
        return out
    def profile_module_time(self, module, mode, *args, **kwargs):
        # if device == torch.device("cpu"):
        #     repeat = 1
        #     print('repeat = %d'%repeat)
        # elif device == torch.device("cuda:0"):
        #     repeat = 1
        # for i in np.arange(repeat):
        #     o = module(*args, **kwargs)
        #     shape = o.shape

        # if mode == 'timeit':
        #     print(timeit.timeit('o = module(*args, **kwargs)', globals=globals()))
        out = None
        if mode == 'torchprofile':
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                out = module(*args, **kwargs)
            print(prof)
        if mode == 'torchprofile19':
            # import pdb; pdb.set_trace()
            print("mode == torchprofile19, save in", self.savepath) 
            with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(self.savepath),
                    record_shapes=True,
                    with_stack=True,
                    # profile_memory=True,
                    with_flops=True
            ) as prof:
                for i in np.arange(5):
                    out = module(*args, **kwargs)
                    prof.step()
                    print("prof.step()")
                    
            print(prof)
        if mode == 'line':
            for i in range(10):
                out = module(*args, **kwargs)
            lp = LineProfiler()
            lp_wrap = lp(module.forward)
            lp_wrap(*args, **kwargs) 
            lp.print_stats() 
            
        if mode == 'time':
            ################################
            # repeat = 1
            # test_time_list = [0]
            # while(np.mean(test_time_list)*repeat < 1):
            #     repeat = repeat*10
            #     best_number = np.ceil(repeat/10.0)
            #     test_time_list = []
            #     for i in range(repeat):
            #         st = time.time()
            #         if self.device == torch.device("cuda"):
            #             torch.cuda.synchronize()
            #         out = module(*args, **kwargs)
            #         if self.device == torch.device("cuda"):
            #             torch.cuda.synchronize()
            #         test_time = time.time()-st
            #         if len(test_time_list) < best_number:
            #             test_time_list.append(test_time)
            #             test_time_list = sorted(test_time_list)
            #         elif test_time< test_time_list[-1]:
            #             test_time_list[-1] = test_time
            #             test_time_list = sorted(test_time_list)
            
            # # print('test result form time.time', time.time()-st)
            # print('%d loops, best of %d: %f sec per loop'%(repeat, best_number, np.mean(test_time_list)))
            #####################
            out, mean_time = profile_best_repeat(module, args[0])
        return out
    @staticmethod
    def mytimeit_test():
        @MyTimeit('time')
        def myfunc1():
            a = 1
            b = a+3
            return a,b
        a,b = myfunc1()


def profile_flops_ptflops(module, input_tensor, print_per_layer_stat=True):
     
    n, channel, height, width = input_tensor.shape
    # 1, 3, 960, 540 
    # 5, 3, 
    # 1, 15, 960, 540 
    # import pdb; pdb.set_trace()
    with torch.no_grad(): 
        import sys
        # sys.stdout = open('test_sys_model.log', 'w') not work for get_model_complexity_info
        # print('i am print in ')
        macs, params = get_model_complexity_info(
        module, ( channel, height, width),
        as_strings              =   True,
        print_per_layer_stat    =   print_per_layer_stat,
        verbose                 =   False)
        sys.stdout = sys.__stdout__ 
    # with torch.no_grad(): 
    #     macs, params = get_model_complexity_info(
    #     module.inc, ( channel, height, width),
    #     as_strings              =   True,
    #     print_per_layer_stat    =   print_per_layer_stat,
    #     verbose                 =   False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params
            
            
class MyFlops(MyProfiler):
    def __init__(self, mode='thop'):
        super(MyFlops).__init__()
        self.mode = mode

        
    def profile_module(self, func, *args, **kwargs):
        # print('in my flops profile module')
        out = self.profile_module_flops(func, *args, **kwargs)
        return out
    
    def profile_module_flops(self, module, *args, **kwargs):
        
        if self.mode == 'ptflops':
            macs, params = profile_flops_ptflops(module, args[0])
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            
        if self.mode == 'thop':
            input_tensor = args[0]
            # macs, params = profile(nn.Sequential(module), inputs=(input_tensor,), 
            #             custom_ops={Model.Conv2d_2x2: self.thop_count_Conv2d_2x2})
            macs, params = profile(nn.Sequential(module), inputs=(input_tensor,))            
            if macs > 1e9:
                macs = str(macs/1e9)+' G'
            print('Computational complexity: '+str(macs)+' MAdds')
        if self.mode == 'torchstat':
            # only work in CPU mode, not in GPU mode
            input_tensor = args[0]
            n, channel, height, width = input_tensor.shape
            stat(module, (channel, height, width))
            def torchstat_test():
                model = models.mobilenet_v2()
                stat(model, (3, 224, 224))
                #not work for gpu
                '''
                output:
                                module name   input shape  output shape     params memory(MB)           MAdd          Flops  MemRead(B)  MemWrite(B) duration[%]    MemR+W(B)
                0              features.0.0     3 224 224    32 112 112      864.0       1.53   21,274,624.0   10,838,016.0    605568.0    1605632.0       1.99%    2211200.0
                1              features.0.1    32 112 112    32 112 112       64.0       1.53    1,605,632.0      802,816.0   1605888.0    1605632.0       1.07%    3211520.0
                6         features.1.conv.1    32 112 112    16 112 112      512.0       0.77   12,644,352.0    6,422,528.0   1607680.0     802816.0       1.49%    2410496.0
                140            classifier.1          1280          1000  1281000.0       0.00    2,559,000.0    1,280,000.0   5129120.0       4000.0       0.69%    5133120.0
                total                                                    3504872.0      74.25  627,687,672.0  320,236,288.0   5129120.0       4000.0     100.00%  170075968.0
                =============================================================================================================================================================
                Total params: 3,504,872
                -------------------------------------------------------------------------------------------------------------------------------------------------------------
                Total memory: 74.25MB
                Total MAdd: 627.69MMAdd
                Total Flops: 320.24MFlops
                Total MemR+W: 162.2MB

                '''
        out = module(*args, **kwargs)
        return out
    # def thop_count_Conv2d_2x2(self, model: Model.Conv2d_2x2, x, y: torch.Tensor):
    #     '''
    #     x: (tensor,)    input N C H W
    #     y: tensor       input N C H W
    #     '''
    #     verbose = False
    #     if verbose:
    #         print(model)
    #         print(type(x))
    #         print(x[0].shape)
    #         print(y.shape)
    #     N, C, H, W = x[0].shape
    #     upsample_ops = 4*N*C*(H+1)*(W+1)
    #     if verbose:
    #         print("count in upsample", upsample_ops)
        
    #     conv2x2 = model.Conv2x2
        
    #     # copy from thop profile code
    #     kernel_ops = torch.zeros(conv2x2.weight.size()[2:]).numel()  # Kw x Kh
    #     bias_ops = 1 if conv2x2.bias is not None else 0
    #     # N x Cout x H x W x  (Cin x Kw x Kh + bias) A pair of Mul and Add is count as 1
    #     conv_ops = y.nelement() * (conv2x2.in_channels // conv2x2.groups * kernel_ops + bias_ops)
    #     total_ops = upsample_ops + conv_ops
    #     model.total_ops += torch.DoubleTensor([int(total_ops)])

    # def ptflops_count_Conv2d_2x2(self, model: Model.Conv2d_2x2, x, y: torch.Tensor):
    #     '''
    #     ptflops accumulate the MACs of non-leaf node and leaf node
    #     x: (tensor,)    input N C H W
    #     y: tensor       input N C H W
    #     '''
    #     verbose = False
    #     if verbose:
    #         print(model)
    #         print(type(x))
    #         print(x[0].shape)
    #         print(y.shape)
    #     N, C, H, W = x[0].shape
    #     upsample_ops = 4*N*C*(H+1)*(W+1)
    #     if verbose:
    #         print("count in upsample", upsample_ops)
    #     # ptflops accumulate the MACs of non-leaf node and leaf node
        
    #     model.__flops__ += int(upsample_ops)
    # # stat(conv2d_2x2pad, (channel, height, width)) do not support tensor on GPU


def profile_in_main():
    print('*******************************profile the model***************************')
    from profiler import profile_best_repeat, profile_flops_ptflops
    x   =   torch.randn(2, sequence_length*(1+args["image_channel"]), 512, 960).cuda()
    # if args["data_shape"] == "nf_4":
    #     x_nf_4 = prepare_nf_4_from_n_f3plus1(x)
    #     out, run_time   =   profile_best_repeat(model, x_nf_4, waitting=0.5)
    # else:
    out, run_time   =   profile_best_repeat(model, x, waitting=0.5)
    print('run time is %f\n'%run_time)
    logger.write('run time is %f\n'%run_time)
    
    # from trash.learn_stdout import config_logging
    # flops lib only accept batch size=1, so reshape in tsm
    # n, c, h, w
    macs, params    =   profile_flops_ptflops(model, x, print_per_layer_stat=False)
    print('{:<30}  {:<8}\n'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}\n'.format('Number of parameters: ', params))
    logger.write('{:<30}  {:<8}\n'.format('Computational complexity: ', macs))
    logger.write('{:<30}  {:<8}\n'.format('Number of parameters: ', params))
    logger.flush()
    print('*******************************Training***************************')
    