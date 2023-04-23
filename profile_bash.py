from profile_main import *
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Profile the model")

    parser.add_argument("--model",  type=str, default='tsn_unet', help="model_to_use")
    parser.add_argument("--device", type=str, default='CUDA', help="device") # ["CUDA", "Multi-thread CPU", "Single-thread CPU"]
    parser.add_argument("--input",  type=str, default='inp_denoise_5frames', help="input data to test")
    # parser.add_argument("--path",   type=str, default='./log_profile', help="path to save profile result")
    # print("save path", argspar.path)
    # os.makedirs(os.path.basename(argspar.path), exist_ok=True)
    argspar = parser.parse_args()


    test_model   = Model.get_model(argspar.model)
    device_name = argspar.device
    input_name  = argspar.input
    
    print("input name", input_name)
    input_all   = Input()
    inp         = input_all.get_input(input_name)
    device      = get_device(device_name)
    main(test_model, inp, device)
    
    # debug time profiling*****************************************************
    # module = test_model.to(device)
    # inp = inp.to(device)
    
    # print('size of tensor '+str(inp.shape))
    # print('use device ', device)


    # with torch.no_grad(): out = module(inp)
    # # del out
    # # torch.cuda.empty_cache()
    # # if isinstance(out, torch.Tensor):
    # #     print('output shape is', out.shape)
    # # elif isinstance(out, list):
    # #     print('output shape is', out[-1].shape)
    # #%%
    # # unet_2stage = test_list[0]
    # # with torch.no_grad():
    # #     out = test_name(inp)
    # #     print(out.shape)
    #     # assert(out.shape == torch.Size([1, 3, 960, 540])), "ouput shape is not correct"
    # #%%
    # # test_list = [OP.conv2d_sp, OP.conv2d_iden, OP.conv2d, OP.conv2d_pointwise, OP.conv2d_depthwise]
    # test_list = [module]
    # test_list \
    # = map(lambda obj: obj.to(device),  test_list)

    # test_list = list(test_list)
    # for i in np.arange(len(test_list)):
    #     test_list[i].eval()
    # #%%

    # print('size of tensor'+str(inp.shape))
    # print('use device', device)
    
    # import time
    # repeat = 10 # 1462MB
    # all_list = []
    # with torch.no_grad(): out = module(inp)
    # del out
    # torch.cuda.empty_cache()
    # for i in range(repeat):
    #     st = time.time()
    #     # we only support torch module
    #     if next(module.parameters()).device == torch.device("cuda"):
    #         torch.cuda.synchronize()
    #     with torch.no_grad(): out = module(inp)
    #     if next(module.parameters()).device == torch.device("cuda"):
    #         torch.cuda.synchronize()
    #     test_time = time.time()-st
    #     del out
    #     torch.cuda.empty_cache()
    #     all_list.append(test_time)
    # print('all list ', all_list)
    # import pdb; pdb.set_trace()
    