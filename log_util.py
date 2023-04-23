import os
from pickle import NONE
import time
import torch
from torch.optim.optimizer import Optimizer
import torchvision.utils as tutils
from utils.utils import batch_psnr
from tensorboardX import SummaryWriter

def init_logging(argdict):
    """This logger copy all code, don't put too many things in code folder

    Args:
        argdict ([type]): [description]

    Returns:
        writer: tensorboard SummaryWriter
        logger: open(log.txt)
    """
    
    #Initilizes the logging and the SummaryWriter modules
    
    os.makedirs(os.path.dirname(argdict['log_dir']), exist_ok=True)
    writer = SummaryWriter(argdict['log_dir'])
    logger = open(argdict['log_dir']+'/log.log', 'a')
    # file_list = ['train.py']
    # import pdb; pdb.set_trace()
    # if os.path.abspath == '/disk1/fast_video/fast_video':
    code_path = os.path.join(argdict['log_dir'],'code_copy')
    os.makedirs(code_path, exist_ok=True)
    print('cp -r *.py %s'%code_path)
    os.system('cp -r *.py %s'%code_path)
    # os.system('cp *.py %s'%code_path)
    for k, v in argdict.items():
        logger.write("\t{}: {}\n".format(k, v))
        # print('write to logger '+"\t{}: {}\n".format(k, v))
    logger.flush()
    return writer, logger


def resume_training(argdict, model, resumef=None, optimizer=None, scheduler=None):
    '''Resumes previous training or starts anew
        we assume a model use amp all through training
    '''
    if resumef is not None:
        print('resume training with ckpt %s'%resumef)
        assert os.path.isfile(resumef), "path %s is invalid"%(resumef)
    elif ('ckpt' in argdict.keys()) and (argdict['ckpt'] is not None):
        resumef = argdict['ckpt']
        print('resume training with ckpt %s'%resumef)
        assert os.path.isfile(resumef), "path %s is invalid"%(resumef)
    else:
        print('use last epoch ckpt')
        resumef = os.path.join(argdict['log_dir'], 'ckpt.pth')

    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model

    if os.path.isfile(resumef):
        # if os.path.isfile(resumef):
        loc = 'cuda:{}'.format(argdict['gpu'])
        checkpoint = torch.load(resumef, map_location=loc)
        # exit()
        print("> Resuming previous training")
        model.load_state_dict(checkpoint['state_dict'])
        if (optimizer is not None) and ('optimizer_state_dict' in checkpoint.keys()):
            print("> Resuming previous optimizer")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])        
            scheduler.last_epoch = checkpoint['scheduler_state_dict']['last_epoch']
            print(scheduler.state_dict())
        # new_epoch = argdict['epochs']
        # new_milestone = argdict['milestone']
        # current_lr = argdict['lr']
        # argdict = checkpoint['args']
        if 'training_params' in checkpoint.keys():
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            iteration = training_params['iteration']
        else:
            start_epoch = checkpoint['start_epoch']
            if 'iteration' in checkpoint.keys():
                iteration = checkpoint['iteration']
            else:
                iteration = -1
        # argdict['epochs'] = new_epoch
        # argdict['milestone'] = new_milestone
        # argdict['lr'] = current_lr
        print("=> loaded checkpoint '{}' (epoch {}) (iteration {})"\
                .format(resumef, start_epoch, iteration))
        # print("=> loaded parameters :")
        # print("==> checkpoint['optimizer']['param_groups']")
        # print("\t{}".format(checkpoint['optimizer']['param_groups']))
        # print("==> checkpoint['training_params']")
        # for k in checkpoint['training_params']:
        #     print("\t{}, {}".format(k, checkpoint['training_params'][k]))
        # argpri = checkpoint['args']
        # print("==> checkpoint['args']")
        # for k in argpri:
        #     print("\t{}, {}".format(k, argpri[k]))

        # argdict['resume_training'] = False
    # else:
    #     raise Exception("Couldn't resume training with checkpoint {}".\
    #             format(resumef))
    else:
        print("No checkpoint at {}". format(resumef))
        start_epoch = 0
        iteration = 0
        # training_params = {}
        # training_params['step'] = 0
        # training_params['current_lr'] = 0
        # training_params['no_orthog'] = argdict['no_orthog']

    # return start_epoch, training_params
    return start_epoch, iteration


def	log_train_psnr(result, imsource, loss, writer, epoch, idx_of_batch, num_minibatches, training_params, log_dir=None, iter_time_st=None):
    '''Logs trai loss.
    '''
    #Compute pnsr of the whole batch
    # psnr_train = None
    if result is not None:
        psnr_train = batch_psnr(torch.clamp(result, 0., 1.), imsource, 1.)
    else:
        psnr_train = None

    # Log the scalar values
    writer.add_scalar('loss', loss.item(), training_params['step'])
    
    string = "[epoch {}][id {}/ num_of_batches {}] loss: {:1.4f} ".\
          format(epoch+1, idx_of_batch+1, num_minibatches, loss.item())
    if psnr_train is not None:
        string += "\t PSNR_train: {:1.4f}".format(psnr_train)
        writer.add_scalar('PSNR on training data', psnr_train, \
          training_params['step'])
    if log_dir is not None:
        string = string +'\t Log: %s'%os.path.basename(log_dir)
    if iter_time_st is not None:
        torch.cuda.synchronize()
        string = string +'\t iter time: %f'%(time.time()-iter_time_st)
        
    print(string)

def save_model_checkpoint(model, optimizer, argdict, epoch, iteration=None, best=True, scheduler=None):
    """Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
    Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
    """
    if hasattr(model, "module"):
        model = model.module
    save_dict = { \
        'state_dict': model.state_dict(), \
        'start_epoch': epoch+1,
        'args': argdict,\
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        }
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict(),
    if best:
        torch.save(save_dict, os.path.join(argdict['log_dir'], 'ckpt.pth'))
    ckpt_e_path = os.path.join(argdict['log_dir'], 'ckpt_epochs/ckpt_e{:05d}.pth'.format(epoch))
    os.makedirs(os.path.dirname(ckpt_e_path), exist_ok=True)
    torch.save(save_dict, ckpt_e_path)
    del save_dict
