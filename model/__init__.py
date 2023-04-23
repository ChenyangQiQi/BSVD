import torch
import torch.nn as nn
from .tsm import exp_tsn

def get_model(args):
    if isinstance(args, str):
        name_string = args
        args = {}
        args['model'] = name_string
    if 'channel' not in args.keys():
        args['channel'] = 64
    if 'kernel_size' not in args.keys():
        args['kernel_size'] = 3
    if 'conv_mode' not in args.keys():
        args['conv_mode'] = None
    # args['sequence_length']
    if 'sequence_length' not in args.keys():
        args['sequence_length'] = 1
    # kernel_size = args['kernel_size']
    print('Loading %s model architecture'%(args['model']))
    # import pdb; pdb.set_trace()
    if args['model'] == 'unet':
        model = UNet(4,3, kernel_size=args['kernel_size'], conv_mode=args['conv_mode'])
    elif args['model'] == 'fastdvd_unet':
        model = FastDVDUnet(num_input_frames=5, residual=False)
    elif args['model'] == 'fastdvd_unet_res_out':
        model = FastDVDUnet(num_input_frames=5, residual=True)
    elif args['model'] == 'fastdvd_unet_res_out_last_stage':
        model = FastDVDUnet(num_input_frames=5, residual=False, output_res=True)
    elif args['model'] == 'fastdvd_unet_res_full':
        model = FastDVDUnet(num_input_frames=5, residual=True, output_res=True)
    elif args['model'] == 'fastdvd':
        if 'image_channel' not in args.keys():
            args['image_channel'] = 3
            # import pdb; pdb.set_trace()
        model = FastDVDnet(args['image_channel'])
        if ('ckpt_file' in args.keys() and 
            args['ckpt_file'] == "/home/chenyangqi/disk1/fast_video/reference/fastdvdnet/model.pth"):
            args["device_ids"] = [0]
            
    elif args['model'] == 'fastdvd_crvd':
        from .fastdvdnet_models_crvd_forward import FastDVDnet
        if 'image_channel' not in args.keys():
            args['image_channel'] = 3
        model = FastDVDnet(args['image_channel'])
        if ('ckpt_file' in args.keys() and 
            args['ckpt_file'] == "/home/chenyangqi/disk1/fast_video/reference/fastdvdnet/model.pth"):
            args["device_ids"] = [0]

    elif args['model'] in exp_tsn.names:
        # print(exp_tsn.names)
        model = exp_tsn.get_model(args)
    elif args['model'] == 'slim_cnn_memconv':
        from .mem_conv_slimcnn import SlimCNNMemConv
        model = SlimCNNMemConv(pretrain=True)
    elif args['model'] == 'slim_skip_cnn_memconv':
        from .mem_conv_slimcnn_skip import SlimCNNMemConv
        model = SlimCNNMemConv(pretrain=True)
    elif args['model'] == 'wnet_memconv':
        from .mem_wnet_model_channel_arg_nonorm import UNet
    elif args['model'] == 'wnet_memconv_center_mem':
        from .mem_wnet_model_channel_arg_nonorm_center_mem import UNet
    elif args['model'] == 'wnet_memconv_center_mem_end':
        from .mem_wnet_model_channel_arg_nonorm_center_mem_end import UNet
    elif args['model'] == 'mem_wnet_nonorm_none_as_end':
        from .mem_wnet_nonorm_none_as_end import UNet
        model = UNet(pretrain=True)
    elif args['model'] == 'transunet':
        from .transunet import TransUnet
        # in_channel = args['in_channel']
        # out_channel = args['out_channel']
        model = TransUnet(args['in_channel'], args['out_channel'], dim=args['dim'], \
                          residual_num=args['residual_num'], norm=args['norm'])
        # print('exp_unet_inv.names', exp_unet_inv.names)
    else:
        raise ValueError("no model named %s" %args['model'])
    if "device_ids" in args.keys() and args["device_ids"] is not None:
        print('number of gpus', torch.cuda.device_count())
        device_ids = args["device_ids"]
        model = nn.DataParallel(model, device_ids=device_ids)
    
    return model
