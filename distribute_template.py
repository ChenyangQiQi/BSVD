
# Main function for training
import os
import numpy as np
import time
from datetime import date
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataloader import SfPDataset, SfPTestDataset, create_dataloader
import models as Models
from loss import get_loss, get_mae
from utils.log_util import init_logging, AverageMeter, ProgressMeter
from utils.log_util import save_model_checkpoint
from utils.log_util import resume_training
from utils.log_util import count_parameters
from utils.train_util import get_net_input_cuda, get_net_input_channel
from utils.train_util import adjust_learning_rate
from torch.utils.data import DataLoader
from utils.visualizer import markdown_visualizer, markdown_visualizer_test
from utils.visualizer import tensor2im, save_image
import random
import math
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import warnings
import builtins
import torch.distributed as dist



parser = argparse.ArgumentParser(description="Train the normal estimation network")
# Convenient Training mode
parser.add_argument("--training_mode", type=str, default="all", 
                        choices=["all", "fast", "validation", "test"],	
                        help='quick training mode switch, fast or all, or validation.\
                        this argument overwrites the following arguments')

parser.add_argument("--exp_name", type=str, default="none", \
                        help='Name of this experiments')

# Model parameters
parser.add_argument("--model", type=str, default='unet', choices=["unet", "unet_xtc", "fuseunet", 
                                                                "fuseunet_cat", "fuseunet2","resunet", 
                                                                "dorn","transunet","dialunet","FuseUnet_cat_downsample"],	
                help="model_to_use")

parser.add_argument("--dim", type=int, default=64,
                help="the dim of feature")
parser.add_argument("--residual_num", type=int, default=16, 
                help="the number of residual blocks in u-net")
parser.add_argument("--norm", type=str, default='bn',
                help="normalization method")


# Dataset parameters
parser.add_argument('--dataset', default='spwintra_deepsfp', type=str, 
                    choices= ["spwintra_deepsfp", "spwinter_deepsfp", "deepsfp", "spwintra", "spwinter"], 
                    help='path to dataset iccv2021') 

# Data Preprocessing parameters
parser.add_argument("--batch_size", type=int, default=32, 	\
                help="Training batch size")
parser.add_argument("--crop_interval", type=int, default=32, 	\
                help="Crop interval of image")

parser.add_argument("--full_resolution", action='store_true',\
                    help="use full resolution to train")
parser.add_argument("--disable_rgb_pretrained", action='store_true',\
                    help="do not use rgb pretrained models")
parser.add_argument("--interpolated_normal", action='store_true',\
                    help="use interpolated normal instead of point cloud")
# parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
parser.add_argument('--netinput', default='fiveraw_pol_posen', type=str, 
                choices= [  "onlyiun", "fiveraw",
                            "fourraw",
                            "fourraw_coord", "fiveraw_posen", 
                            "fiveraw_pol", "fiveraw_pol_coord", 
                            "fiveraw_pol_posen",
                            "fiveraw_pol_vd",
                            "fiveraw_pol_vd_withoutrho",
                            "fiveraw_pol_vd_normalprior",
                            "fiveraw_pol_rawphi_posen",
                            "fiveraw_pol_posen_withoutrho",
                            "fiveraw_pol_posen_normalprior",
                            "fourraw_normalprior",
                            "fourraw_normalprior_vd",
                            "fourraw_posen_normalprior",
                            "fiveraw_pol_rho0_posen",
                            "fiveraw_pol_rho1_posen",
                            "fiveraw_pol_rho05_posen"
                            ], 
                help='feature feed into the netowrk') 

# Training parameters
parser.add_argument("--epochs", "--e", type=int, default=1000, \
                    help="Number of total training epochs")

parser.add_argument("--cos", action='store_true',\
                    help="use cos decay learning rate")

parser.add_argument("--lr", type=float, default=1e-4, \
                    help="Initial learning rate")
parser.add_argument("--dropout", type=float, default=0., \
                    help="Initial learning rate")


# parser.add_argument("--save_every_iterations", type=int, default=20,\
                    # help="Number of training steps to log psnr and perform \
                    # orthogonalization")
parser.add_argument("--save_freq", type=int, default=50,\
                    help="Number of training epochs to save state")
parser.add_argument("--print_freq", type=int, default=10,\
                    help="Number of training epochs to save state")
parser.add_argument("--loss", type=str, default="mae", 
                    choices = ["mae","AL","TAL","perceptual","perceptual_5","perceptual_4"],
                    help='loss to train the network')
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--preload', action='store_true',
                    help="load all the data into memory")


# Dirs
parser.add_argument('--dataroot', default='./data/iccv2021', type=str, help='path to dataset iccv2021') 

# parser.add_argument("--log_dir", type=str, default="", \
#                  help='path of log files')
# parser.add_argument("--output_dir", type=str, default="", \
#                     help='path to save jpg results')

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


def test_one_set(model, args, test_txt = None):
    if test_txt:
        save_image_dir = "{}/{}".format(args.output_dir, test_txt.split("/")[-1][:-4])
    else:
        save_image_dir = "{}/{}".format(args.output_dir, "deepsfp")
    os.makedirs(save_image_dir, exist_ok=True)
    sfp_test_dataset    = SfPTestDataset(dataroot=args.dataroot, 
        txt_path=test_txt, use_deepsfp=args.use_deepsfp,crop_interval=args.crop_interval)
    loader_val = DataLoader(sfp_test_dataset, 
                              batch_size = 1, 
                              shuffle=False, num_workers=8, 
                              #prefetch_factor=4,
                              drop_last=False)

    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader_val),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        save_img_names = []
        for i, data_sample in enumerate(loader_val, 0):
            end = time.time()
            data_sample = [item.cuda() for item in data_sample]
            net_pol, net_coordinate, viewing_direction = data_sample
            # preprocessing the data using gpu
            net_in = get_net_input_cuda(net_pol, net_coordinate, viewing_direction, args)
            result = model(net_in)
            pred_camera_normal = F.normalize(result, p=2, dim=1)
            # import pdb; pdb.set_trace()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            b = pred_camera_normal.size(0)
            
            for img_idx in range(b):
                input_name = sfp_test_dataset.pol_paths[i * b + img_idx]
                base_folder = "/".join(input_name.split("/")[-3:-1])
                image_prefix = input_name.split("/")[-1][:-4]
                os.makedirs("{}/{}".format(save_image_dir, base_folder), exist_ok=True)
                save_image(tensor2im(pred_camera_normal[img_idx:img_idx+1]), "{}/{}/{}.png".format(save_image_dir, base_folder, image_prefix))
                # save_image(tensor2im(pred_camera_normal[img_idx:img_idx+1], cent=0., factor=255.), "{}/{}/{}.png".format(save_image_dir, base_folder, image_prefix))
                save_img_names.append("{}/{}.png".format(base_folder, image_prefix))
                # save_image(tensor2im(pred_camera_normal[img_idx:img_idx+1]), "{}/im{:03d}_pred.png".format(save_image_dir, i * b + img_idx))
                # print("{}/{}/{}.png".format(save_image_dir, base_folder, image_prefix))
                # np.save("{}/im{:03d}_pred.npy".format(save_image_dir, i * b + img_idx), tensor2im(pred_camera_normal[img_idx:img_idx+1]))
                # save_image(tensor2im(net_in[img_idx:img_idx+1,:3]), "{}/im{}_in.jpg".format(save_image_dir, i * b + img_idx))
            # print(save_img_names)
            if i % args.print_freq == 0:
                progress.display(i)
        markdown_visualizer_test(save_img_names, save_image_dir)


def validate(loader_val, model, args, epoch=None, writer=None):
    batch_time = AverageMeter('Data+Forward Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mae_score = AverageMeter('MAE', ':6.2f')
    other_metric = AverageMeter('other_metric', ':6.2f')
    progress = ProgressMeter(
        len(loader_val),
        [batch_time, losses, mae_score, other_metric],
        prefix='Test: ')

    # switch to evaluate mode
    save_image_dir = "{}/validation".format(args.output_dir) if args.training_mode == "validation" else "{}/{}".format(args.output_dir, epoch) 
    os.makedirs(save_image_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, data_sample in enumerate(loader_val, 0):
            # end = time.time()
            data_sample = [item.cuda() for item in data_sample]
            net_mask, net_pol, vis_camera_normal, net_gt, net_coordinate, viewing_direction = data_sample
            # preprocessing the data using gpu
            net_in = get_net_input_cuda(net_pol, net_coordinate, viewing_direction, args)
            # print("net_in.size()", net_in.size())
            result = model(net_in)
            pred_camera_normal = F.normalize(result, p=2, dim=1)
            loss = get_loss(pred_camera_normal, net_gt, args.loss)
            mae = get_mae(net_gt, pred_camera_normal, net_mask)
            other_metric_item = 0
            losses.update(loss.item(), net_pol.size(0))
            mae_score.update(mae, net_pol.size(0))
            other_metric.update(other_metric_item, net_pol.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            b = pred_camera_normal.size(0)
            for img_idx in range(b):
                save_image(tensor2im(pred_camera_normal[img_idx:img_idx+1]), "{}/im{}_pred.jpg".format(save_image_dir, i * b + img_idx))
                if net_in.shape[1] == 1:
                    # gray_input = torch.rep net_in[img_idx:img_idx+1,:3]
                    gray_input = net_in[img_idx:img_idx+1].repeat(1,3,1,1)
                else:
                    gray_input = net_in[img_idx:img_idx+1,:3]
                save_image(tensor2im(gray_input), "{}/im{}_in.jpg".format(save_image_dir, i * b + img_idx))
                save_image(tensor2im(net_gt[img_idx:img_idx+1]), "{}/im{}_gt.jpg".format(save_image_dir, i * b + img_idx))

            if i % args.print_freq == 0:
                progress.display(i)
        markdown_visualizer(save_image_dir, num=i * b + img_idx)
        print(' * MAE {mae_score.avg:.3f} other_metric {other_metric.avg:.3f}'
              .format(mae_score=mae_score, other_metric=other_metric))
    if writer is not None:
        print("writer test loss: ", losses.avg, epoch)
        writer.add_scalar('test loss', losses.avg, epoch)
        writer.add_scalar('test mae', mae_score.avg, epoch)
    return losses.avg, mae_score.avg

def train_one_epoch(loader_train, epoch, model, optimizer, scaler, writer, args):
    batch_time = AverageMeter('Forward Time', ':6.3f')
    data_time = AverageMeter('Data Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(loader_train),
        [batch_time, data_time, losses],
    prefix="Epoch: [{}]".format(epoch))        
    data_time_start = time.time()
    step = 0
    model.train()
    for i, data_sample in enumerate(loader_train, 0):
        data_time.update(time.time() - data_time_start)
        batch_time_start = time.time()
        optimizer.zero_grad()
        data_sample = [item.cuda() for item in data_sample]
        net_mask, net_pol, vis_camera_normal, net_gt, net_coordinate, viewing_direction = data_sample
        # preprocessing the data using gpu
        net_in = get_net_input_cuda(net_pol, net_coordinate, viewing_direction, args)
        if i==0: print('net_in.shape ',net_in.shape)
        with torch.cuda.amp.autocast(scaler is not None):
            result = model(net_in)
            pred_camera_normal = F.normalize(result, p=2, dim=1)
            # print(pred_camera_normal.size())
            loss = get_loss(pred_camera_normal, net_gt, args.loss)
    
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        batch_time.update(time.time() - batch_time_start)
        losses.update(loss.item(), net_pol.size(0))
        if step % args.print_freq == 0:
            progress.display(i)
        step += 1
        data_time_start = time.time()


    if writer is not None:
        # tensorboard logger
        writer.add_scalar('training loss', losses.avg, epoch)
    return model, losses.avg

def test(model, args):
    if "deepsfp" in args.dataset:
        print("Testset: DeepSfP")
        test_one_set(model, args, test_txt=None)
    if "intra" in args.dataset:
        print("Testset: sfpwild_test_intra")
        test_one_set(model, args, test_txt = "{}/sfpwild_combined.txt".format(args.dataroot))
    if "inter" in args.dataset:
        print("Testset: sfpwild_test_inter")
        test_one_set(model, args, test_txt = "{}/sfpwild_test_inter_combined.txt".format(args.dataroot))
    exit()


#%%
def main_worker(gpu, ngpus_per_node, args):
    r"""Performs the main training loop
    """

    args.gpu = gpu
    print(args.gpu)


    args.log_dir = "./logs/"+ args.exp_name 
    args.output_dir = "./results/"+ args.exp_name 
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpt_epochs'), exist_ok=True)
    
    args.use_deepsfp = True if "deepsfp" in args.dataset else False
    args.use_sfpwild = True if "spw" in args.dataset else False
    args.split = "inter" if "inter" in args.dataset else "intra"

    def print_info():
        print("\n### Training shape from polarization model ###")
        print("> Parameters:")
        for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
            print('\t{}: {}'.format(p, v))
        print('\n')
    print_info()    



    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print(args.world_size, args.dist_backend, args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()


    writer = None
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
        writer, logger = init_logging(args)
    
    args.channel = get_net_input_channel(args)

    model = Models.get_model(args)

    # print(model)
    # for name, p in model.named_parameters():
    #     print(name, p.size(),p.requires_grad)
    # model = model.cuda()
    print("args.model", args.model, args.model in ["fuseunet", "fuseunet2", "fuseunet_cat", "unet_xtc"])
    if (args.model in ["fuseunet", "fuseunet2", "fuseunet_cat", "unet_xtc", "FuseUnet_cat_downsample"]) and (not args.disable_rgb_pretrained):
        rgb_ckpt_path = "./XTConsistency/models/rgb2normal_consistency.pth"
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(rgb_ckpt_path, map_location=loc)
        print("> Resuming previous training")
        print("> Resuming previous training")

        state_dict = {}
        for key, value in checkpoint.items():
            # print(key)
            if "unet_xtc" == args.model:
                state_dict[key] = value
            else:
                state_dict['rgb_encoder.'+key] = value
            
        del checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        print(set(msg.missing_keys) )


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
        model=model.cuda()
        
    # print(model) # print model after SyncBatchNorm

    start_epoch = resume_training(args, model)
    cudnn.benchmark = True

    # model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    num_params = count_parameters(model)
    print("num_params", num_params)

    sfp_train_dataset, sfp_test_dataset = create_dataloader(args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(sfp_train_dataset)
    else:
        train_sampler = None

    loader_train = DataLoader(
        sfp_train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    loader_val = DataLoader(sfp_test_dataset, 
                              batch_size = 1, 
                              shuffle=False, num_workers=8, 
                              #prefetch_factor=4,
                              drop_last=False)
    print("\t# of training samples: %d\n" % int( len(loader_train)))


    # implement validation
    if args.training_mode == 'validation':
        test_loss_avg = validate(loader_val, model, args)
        exit()

    # implement test
    if args.training_mode == 'test':
        test(model, args)

    # Optimizer
    args.lr = args.lr * (args.batch_size / 1.0) #Linear scale the learning rate based on the batch size
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('learning rate %f' % args.lr)
    
    scaler = None 
    if args.fp16:    
        scaler = torch.cuda.amp.GradScaler()    
    
    # start from last epoch to avoid overwritting the previous result
    best_mae = 100
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.cos:
            adjust_learning_rate(optimizer, epoch, args)
        model, _ = train_one_epoch(loader_train, epoch, model, optimizer, scaler, writer, args)
        if (epoch+1) % args.save_freq == 0 or epoch==0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
                save_model_checkpoint(model, epoch, save_path=os.path.join(args.output_dir, 'ckpt_epochs/ckpt_e{}.pth'.format(epoch)))    
                _, mean_mae = validate(loader_val, model, args, epoch, writer)
                if mean_mae < best_mae:
                    best_mae = mean_mae
                    save_model_checkpoint(model, epoch, save_path=os.path.join(args.output_dir, 'ckpt.pth'))    


        
 
if __name__ == "__main__":
    main()    
