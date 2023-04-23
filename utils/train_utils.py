import math

def prepare_input_cuda(gt_train, imgn_train, mode="tsm"):
    
    
    if mode=="tsm":
        n, f, c, h, w = gt_train.shape
        gt_train = gt_train.cuda().reshape(-1, c, h, w)
        imgn_train = imgn_train.cuda().reshape(-1, c+1, h, w)
    elif mode == 'fastdvd':
        # gt_train_center = gt_train[:, 2, :, :, :].cuda()
        # gt_train = gt_train.reshape(n, f*c, h, w).cuda()
        # gt_train = gt_train[:, 2, :, :, :].cuda()
        n, f, c, h, w = imgn_train.shape
        
        gt_train = gt_train.cuda().reshape(n, -1, h, w)
        imgn_train = imgn_train.cuda().reshape(n, -1, h, w)
    
    return gt_train, imgn_train

def adjust_learning_rate(optimizer, epoch, lr, max_epoch):
    
    """Decay the learning rate based on schedule"""
    # lr = args.lr
    # args.warmup_epochs = 0
    # if epoch < args.warmup_epochs:
    #     lr *=  float(epoch) / float(max(1.0, args.warmup_epochs))
    #     if epoch == 0 :
    #         lr = 1e-6
    # else:
        # progress after warmup        
    # if args.cos:  # cosine lr schedule
        # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    # assert args.cos == True
    progress = float(epoch) / float(max(1, max_epoch))
    lr *= 0.5 * (1. + math.cos(math.pi * progress)) 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Epoch-{} optimizer.param_groups[0]['lr']".format(epoch), optimizer.param_groups[0]['lr'])
