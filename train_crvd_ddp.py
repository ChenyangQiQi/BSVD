#%%
# this version has a bug that is get stuck when start training...
"""
# SRVD videos plus the scenes 1–6 from CRVD for training, 
# CRVD scenes 7–11 for objective validation.
"""
import os
import random
import numpy as np
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.validate_crvd_sigma_from_noisy import ValDataset
from data.rvidenet_dataloaders import RViDenetDatasetAB_sigma_from_noisy as RViDenetDataset
import model as Model
from loss import get_loss
import pytorch_ssim
from utils.train_utils import prepare_input_cuda, adjust_learning_rate
from utils.utils import batch_psnr
# from validate_util_crvd import validate_and_log, save_out_seq
from log_util import init_logging, resume_training, \
                     save_model_checkpoint
# lib for Distributed Data Parallel
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import warnings
import builtins
import torch.distributed as dist
#%%

# print('initial device', os.environ["CUDA_VISIBLE_DEVICES"])

    
parser = argparse.ArgumentParser(description="Train the denoiser")
# Convenient Training mode
parser.add_argument("--training_mode", type=str, default="all", 
                        choices=["all", "fast", "finetune", "validation", 
                                    "profile", "profile_time_flops"],	
                        help='quick training mode switch, fast or all, or validation.\
                        this argument overwrites the following arguments')

parser.add_argument("--exp_name", type=str, default="default_log", \
                    help='folder that contains log file and result file')

# Model parameters
parser.add_argument("--model", type=str, default='unet3x3', 	\
                help="model_to_use")
parser.add_argument("--model_channel", type=int, default=32, 	\
                help="basic channel num in model ")
parser.add_argument('--channels1', type=int, default=[16,32,64], nargs='+',help='channels for first Unet')
parser.add_argument('--channels2', type=int, default=[16,32,64], nargs='+',help='channels for second Unet')

parser.add_argument("--shift", type=str, default='shift', 	\
                choices=["shift", "pyram_shift", "noshift"])

parser.add_argument("--resume_optimizer", action='store_true',\
                help="model_to_use")
parser.add_argument("--channel", type=int, default=32, 	\
                help="channel of feature in Unet")
parser.add_argument("--ckpt", type=str, default=None, 	\
                help="ckpt of pretrained model")
# Data Preprocessing parameters
parser.add_argument("--batch_size", type=int, default=32, 	\
                help="Training batch size")

parser.add_argument("--input_output_num", type=str, default='5to5',
                        choices=["5to5", "5to1"],	
                        help="model_to_use")
parser.add_argument("--net_input_mode", type=str, default='tsm',
                    choices=["tsm", "fastdvd"],	
                    help="model_to_use")
parser.add_argument("--scene_start", type=int, default=7,
                    help="start of test scene id")    
parser.add_argument("--scene_end", type=int, default=11,
                    help="end of test scene id")    
parser.add_argument("--fix_noisy_index", action='store_true',\
                    help="use only noisy frame 0 to train")

parser.add_argument("--num_train_files", type=int, default=-1,
                    help="end of test scene id")    
parser.add_argument("--verbose", action='store_true',\
                    help="verbose to print path")

parser.add_argument("--patch_size", "--p", type=int, default=128, help="Patch size")
parser.add_argument("--full_res", action='store_true',\
                    help="use full resolution to train the data")
parser.add_argument("--A", type=float, default=1, help="srvd dataset size")
parser.add_argument("--B", type=float, default=10, help="crvd dataset size")

parser.add_argument("--data_shape", type=str, default="n_f3plus1", \
                    help="input shape of network") # prepare_nf_4_from_n_f3plus1

parser.add_argument("--sequence_length", type=int, default=5, \
                    help="temporal sequence length of input for model")

# Training parameters
parser.add_argument("--epochs", "--e", type=int, default=300, \
                    help="Number of total training epochs")
parser.add_argument("--loss", type=str, default="l1", \
                    choices=["l1", "mse"],
                    help="type of loss")
parser.add_argument("--lr", type=float, default=1e-4, \
                    help="Initial learning rate")
parser.add_argument("--cosine_annealing", action='store_true',\
                    help="use cos decay learning rate")
# TODO rename to save_freq, print_freq
parser.add_argument("--save_every", type=int, default=1000,\
                    help="Number of training steps to log psnr and perform \
                    orthogonalization")
parser.add_argument("--save_every_epochs", type=int, default=40,\
                    help="Number of training epochs to save state")


# Distributed Data Parallel parameters
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
    # Distributed training 
parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on [[[gpu]]]s")

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    # ngpus_per_node = 8

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def train_one_epoch(loader_train, model, optimizer, epoch,  args, step, writer, save_out_seq=None,):
    old_time = None
    iter_end_time = time.time()
    for i, data_tuple in enumerate(loader_train, 0):
        gt_train, imgn_train = data_tuple
        if step % args['save_every'] == 0 and args['first_gpu']:
            # print(dataloader time)
            print(gt_train.shape)
            dataloader_time = time.time()-iter_end_time
        if step % args['save_every'] == 0:
            torch.cuda.synchronize()
        iter_time_st = time.time()
        model.train()

        # When optimizer = optim.Optdataimizer(net.parameters()) we only zero the optim's grads
        optimizer.zero_grad()
        
        n, f, c, h, w = gt_train.shape
        # gt_train = gt_train.reshape(-1, c, h, w)
        # imgn_train = imgn_train.reshape(-1, c+1, h, w)

        if args['input_output_num'] == "5to1":
            center = int(args['sequence_length']/2.0)
            gt_train = gt_train[:, center, :, :, :].cuda()
            
        gt_train, imgn_train = \
            prepare_input_cuda(gt_train, imgn_train, mode=args['net_input_mode'])
        
        # print(imgn_train.device)
        # Evaluate model and optimize it
        
        if args['training_mode'] == "profile":
            torch.cuda.synchronize()
        result = model(imgn_train)
        loss = get_loss(args['loss'], gt_train, result)
        
        # (loss_list, result) = model(imgn_train, gt_train, "loss_and_base_out")
        # loss = torch.mean(loss_list)
        loss.backward()
        optimizer.step()

        # Results
        if step % args['save_every'] == 0 and args['first_gpu']: # 66, 
            print('\ninput shape is '+str(imgn_train.shape))
            if np.random.rand()<0.1 or args["training_mode"]=="fast": 
                save_out_seq(imgn_train, gt_train, torch.clamp(result, 0., 1.),
                            save_dir=args['log_dir']+ \
                            '/train_image/%04d/%05d_%06d/'%(epoch, i, step),
                            sigmaval=0,
                            save_length=20)


            # Log the scalar values
            if args['first_gpu']:
                writer.add_scalar('loss', loss.item(), step)
            
            string = "[epoch {}][id {}/ num_of_batches {}] loss: {:1.4f} ".\
                format(epoch, i, len(loader_train), loss.item())
            if result is not None:
                psnr_train = batch_psnr(torch.clamp(result, 0., 1.), gt_train, 1.)
                ssim_train = pytorch_ssim.ssim(torch.clamp(result, 0., 1.), gt_train)
                if args['first_gpu']:
                    writer.add_scalar('PSNR on training data', psnr_train, step)
                    writer.add_scalar('SSIM on training data', ssim_train, step)
                string += "\t PSNR_train: {:1.4f}".format(psnr_train)
            if args['log_dir'] is not None:
                string = string +'\t Log: %s'%os.path.basename(args['log_dir'])
            if iter_time_st is not None:
                torch.cuda.synchronize()
                string = string +'\t forward&backward time: %f'%(time.time()-iter_time_st)
                string = string +'\t dataloader time: %f'%dataloader_time
            print(string)
            
            
            if old_time is not None:
                print('%f time for %d iterations'%(time.time()-old_time ,args['save_every']))
            old_time = time.time()
            
        # update step counter
        step+=1
        iter_end_time = time.time()
    # exit()
    return step

def profile_time_flops(model, input_shape, logger):
        print('*******************************profile the model***************************')
        from profiler import profile_best_repeat, profile_flops_ptflops
        x   =   torch.randn(input_shape).cuda()
        # if args["data_shape"] == "nf_4":
            # x_nf_4 = prepare_nf_4_from_n_f3plus1(x)
            # out, run_time   =   profile_best_repeat(model, x_nf_4, waitting=0.5)
        # else:
        out, run_time   =   profile_best_repeat(model, x, waitting=1)
        # print(' is %f\n'%run_time)
        print('input shape is'+str(input_shape)+', run time is %f\n'%run_time)
        logger.write('input shape is'+str(input_shape)+', run time is %f\n'%run_time)
        logger.write('run time is %f\n'%run_time)
        
        macs, params    =   profile_flops_ptflops(model, x, print_per_layer_stat=True)
        print('{:<30}  {:<8}\n'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}\n'.format('Number of parameters: ', params))
        logger.write('{:<30}  {:<8}\n'.format('Computational complexity: ', macs))
        logger.write('{:<30}  {:<8}\n'.format('Number of parameters: ', params))
        logger.flush()
        print('*******************************Training***************************')
#%%
def main_worker(gpu, ngpus_per_node, args):
    r"""Performs the main training loop
    """
    args.gpu = gpu
    print("args.gpu in main_worker", args.gpu)
    import os
    os.environ['OMP_NUM_THREADS']="1"
    # process argument
    args.log_dir = "./logs/"+ args.exp_name
    args.save_path = "./results/"+ args.exp_name
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    if "finetune" in args.training_mode:
        args.A = 0
        args.B = 40
    
    def integers_list(string_list):
        intergers = []
        for i in string_list:
            intergers.append(int(i))
        return intergers
    # import pdb; pdb.set_trace()
    args.channels1 = integers_list(args.channels1)
    args.channels2 = integers_list(args.channels2)
    
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    def print_info():
        print("\n### Training denoiser model ###")
        print("> Parameters:")
        for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
            print('\t{}: {}'.format(p, v))
        print('\n')
    print_info()  

    # suppress printing if not first GPU on each node


    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # import pdb; pdb.set_trace()
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            args.first_gpu = bool((not args.multiprocessing_distributed) or (args.multiprocessing_distributed and args.rank == 0))
            print("args.world_size", args.world_size, "args.dist_backend", args.dist_backend, "args.rank", args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    else:
        args.first_gpu = True
    args=vars(args)
    writer = None
    save_out_seq = None
    if args["first_gpu"]: # only the first GPU saves checkpoint
        writer, logger = init_logging(args)
    
    print('> Loading datasets ...')
    # TODO modifying following args dict to class, remove vars
    
    dataset_val_crvd = ValDataset(valsetdir="./dataset/CRVD",
                                  scene_start=args["scene_start"],
                                  scene_end=args["scene_end"],
                                  verbose=args['verbose']
                                  )
    dataset_val_crvd = DataLoader(dataset_val_crvd, 
                              batch_size=1, 
                            #   batch_size=8, 
                              shuffle=False, num_workers=4,
                              prefetch_factor=4,
                              pin_memory=True,
                              drop_last=False,
                              )
    
    data_cache = {}
    if args['training_mode'] != 'validation':
        dataset = RViDenetDataset(
                            patch_size=args['patch_size'],\
                            num_input_frames=args["sequence_length"],
                            num_train_files=args['num_train_files'],
                            A=args['A'],
                            B=args['B'],
                            fast = bool(args['training_mode'] in ["fast",]),
                            # profile = bool(args['training_mode'] == "profile",),
                            # profile = False,
                            crop= (not args['full_res']),
                            fix_noisy_index=args['fix_noisy_index'],
                            verbose=args['verbose'],
                            data_cache = data_cache
                            )
        print("Number of whole dataset: %d\n" % len(dataset))

        if args['distributed']:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = None

        loader_train = DataLoader(dataset, 
                                batch_size=args['batch_size'], 
                                shuffle=(train_sampler is None), 
                                num_workers=args['workers'],
                                pin_memory=True,
                                sampler=train_sampler,
                                drop_last= ("tswnet" in args['exp_name']),
                                )
        
        print("Number of training samples: %d\n" % int(len(loader_train)))
    # import pdb; pdb.set_trace()


    # Define GPU devices
    torch.backends.cudnn.benchmark = True # CUDNN optimization

    # Create model
    # args for transunet
    # args["image_channel"] = 4
    # args['in_channel'] =  5
    # args['out_channel'] = 4
    # args['dim'] = 64
    # args['residual_num'] = 1
    # args['norm'] = 'bn'
    if args['net_input_mode'] == 'tsm':
        # for dataset crvd noise from sigma
        args['channel_per_frame'] = 5
        args['noise_channel'] = 0
    if args['net_input_mode'] == 'fastdvd':
        args['image_channel'] = 4
    model = Model.get_model(args)
    # if not args.multiprocessing_distributed or \
    #     (args.multiprocessing_distributed and args.rank == 0): # only the first GPU saves checkpoint
    start_epoch, iteration = resume_training(args, model,)
    # Map model to device
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args['distributed']:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args['gpu'] is not None:
            torch.cuda.set_device(args['gpu'])
            model.cuda(args['gpu'])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args['batch_size'] = int(args['batch_size'] / args['world_size'])
            args['workers'] = int((args['workers'] + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args['gpu'] is not None:
        torch.cuda.set_device(args['gpu'])
        model = model.cuda(args['gpu'])
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
        model=model.cuda()
    if args['first_gpu']:
        from validate_util_crvd import validate_and_log, save_out_seq
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    print('\n learning rate %f' % args['lr'])
        
    # Resume training or start a new
    # if args['resume_optimizer']:
        # start_epoch = resume_training(args, model, optimizer=optimizer)
    # else:
    
    if args['training_mode']=='profile_time_flops':
        if args['net_input_mode'] == 'tsm':
            input_shape = (1, 25, 540, 960)
            
        elif args['net_input_mode'] == 'fastdvd':
            input_shape = (1, 25, 540, 960)
        if args["first_gpu"]: # only the first GPU saves checkpoint
            profile_time_flops(model, input_shape, logger)
        exit()
    best_psnr = -1.0
    
    if (args['training_mode'] in ['validation', 'finetune']) and args["first_gpu"]:
    # if args['training_mode'] in ['validation']:
        # import pdb; pdb.set_trace()
        psnr_val_crvd_dataset = validate_and_log(
                        model_temp=model, \
                        dataset_val=dataset_val_crvd, \
                        temp_psz = args["sequence_length"], \
                        writer_tensorboard=writer, \
                        epoch=start_epoch, \
                        logger_txt=logger, \
                        save_path=(args['save_path']),
                        lr = optimizer.param_groups[0]['lr'],
                        name="crvd",
                        mode=args['input_output_num'],
                        verbose = (args['training_mode']=='validation'),
                        downsample= (1      if args['training_mode'] == 'validation' else 2),
                        fext=       ('.png' if args['training_mode'] == 'validation' else '.jpg'),
                        )
        if args['training_mode'] == 'validation':
            exit()
    # training_params = {}
    
    step = start_epoch*len(loader_train)
    for epoch in tqdm(range(start_epoch, args['epochs'])):
        if args['distributed']:
            train_sampler.set_epoch(epoch)
        if args['cosine_annealing']:
            adjust_learning_rate(optimizer, epoch, args['lr'], args['epochs'])
            
        if args['training_mode'] == "profile":
            # step = train_one_epoch(loader_train, model, optimizer, epoch,  args, step, writer)
            print('profile the main function')
            args['epochs'] = 3
            from line_profiler import LineProfiler
            lp = LineProfiler()
            lp_wrap = lp(train_one_epoch)
            lp_wrap(loader_train, model, optimizer, epoch,  args, step, writer, save_out_seq) 
            lp.print_stats() 
            exit
        else:
            step = train_one_epoch(loader_train, model, optimizer, epoch,  args, step, writer, save_out_seq)
        # print("number of sample in data_cache",len(data_cache))
        
        # Validation and log images
        if epoch % args['save_every_epochs'] == 0:
            if args['first_gpu']: # only the first GPU saves checkpoint
                psnr_val_crvd_dataset = validate_and_log(
                                model_temp=model.module, \
                                dataset_val=dataset_val_crvd, \
                                temp_psz = args["sequence_length"], \
                                writer_tensorboard=writer, \
                                epoch=epoch, \
                                logger_txt=logger, \
                                save_path=(args['save_path']),
                                lr = optimizer.param_groups[0]['lr'],
                                name="crvd",
                                mode=args['input_output_num']
                                )
                # save model and checkpoint

                if psnr_val_crvd_dataset > best_psnr:
                    save_model_checkpoint(model, optimizer, args, epoch, best=True)
                    print('higher PSNR', psnr_val_crvd_dataset)
                    best_psnr =  psnr_val_crvd_dataset

                if epoch % (args['save_every_epochs']*4) == 0 :
                    save_model_checkpoint(model, optimizer, args, epoch, best=False)


if __name__ == "__main__":

    main()