from log_util import resume_training
import os
import time
import cv2
import numpy as np
import torch
import torchvision.utils as tutils
from utils.utils import batch_psnr
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_ssim
from utils.metric import compare_psnr_rvidenet
from skimage.measure import compare_psnr,compare_ssim


def validate_and_log(model_temp, dataset_val, 
                          temp_psz, writer_tensorboard, \
                            epoch, logger_txt, save_path = None, lr=None, name="", mode="5to5", verbose=False,
                            downsample = 2, fext='.jpg', vis=None):
    """Validation step after the epoch finished
    """
    # import pdb; pdb.set_trace()
    model_temp.eval()
    if hasattr(model_temp, "module"):
        model_temp = model_temp.module
    # .eval after model_temp.module
    # *** AttributeError: 'function' object has no attribute 'eval'
    t1 = time.time()
    psnr_val_dataset = 0
    ssim_val_dataset = 0
    psnr_noisy_dataset = 0
    psnr_val_srgb_dataset = 0
    ssim_val_srgb_dataset = 0
    dataset_loader = dataset_val
    data_as_batch = False
    if isinstance(dataset_loader, torch.utils.data.dataloader.DataLoader):
        dataset = dataset_loader.dataset
        data_as_batch = True
    else: 
        dataset = dataset_loader

    with torch.no_grad():
        for i, (seqn_val, seq_gt) in enumerate(tqdm(dataset_loader)):
            if data_as_batch:
                seqn_val    = seqn_val[0, ...]
                seq_gt      = seq_gt[0, ...]
            seq_start_time = time.time()
            seqn_val = seqn_val.cuda()
            seq_gt   = seq_gt.cuda()
            print('Start denoising the sequence')
            if mode == "5to1":
                out_val = denoise_seq_5to1(seq=seqn_val, \
                                    model_temporal=model_temp,
                                    temp_psz=temp_psz)
            
            elif mode == "5to5":
                # import pdb; pdb.set_trace()
                out_val = denoise_seq_5to5(seq=seqn_val, model_temporal=model_temp, temp_psz=temp_psz)
                # import pdb; pdb.set_trace()
            else: raise NotImplementedError

            psnr_val_seq        = batch_psnr(out_val.cpu(),    seq_gt.squeeze_(), 1.)
            # import pdb; pdb.set_trace()
            # psnr_rvidenet       = compare_psnr_rvidenet(out_val.cpu().numpy(),    seq_gt.squeeze_().cpu().numpy())
            # psnr_rvidenet_reshape       = compare_psnr_rvidenet(out_val.cpu().numpy().reshape(-1, 540, 960),    seq_gt.squeeze_().cpu().numpy().reshape(-1, 540, 960))
            # psnr_val_seq_uint        = batch_psnr(
            #                                 (
            #                                     np.uint16(
            #                                         out_val.cpu()*(2**12-1-240)+240)
            #                                     .astype(np.float32)-240
            #                                 )/(2**12-1-240),
            #                 seq_gt.squeeze_().cpu().numpy(),
                                            
            #                 1.)
            psnr_noisy_seq      = batch_psnr(seqn_val.cpu()[:, 0:4, ...],   seq_gt.squeeze_(), 1.)
            ssim_val_seq = pytorch_ssim.ssim(out_val,    seq_gt.squeeze_(),)
            # out_val: [0,1] (7,4,540,960)
            # np.save('out_val.npy', out_val.cpu().numpy())
            # np.save('seq_gt.npy', seq_gt.cpu().numpy())
            # gt_srgb_frame       = get_srgb_from_raw(seq_gt)
            # denoised_srgb_frame = get_srgb_from_raw(out_val)
            batch_srgb_psnr = 0
            batch_srgb_ssim = 0
            # FIXME TODO: acceleration
            # for j_batchsize in np.arange(out_val.shape[0]):
                # noisy_rgb = variable_to_cv2_image(out_val[j_batchsize].clamp(0., 1.))
                # gt_rgb = variable_to_cv2_image(seq_gt[j_batchsize].clamp(0., 1.))
                # test_srgb_psnr = compare_psnr(noisy_rgb.astype(np.float32)/255, gt_rgb.astype(np.float32)/255, data_range=1.0)
                # test_srgb_ssim = compare_ssim(noisy_rgb.astype(np.float32)/255, gt_rgb.astype(np.float32)/255, data_range=1.0, multichannel=True)            
                # batch_srgb_psnr +=  test_srgb_psnr
                # batch_srgb_ssim += test_srgb_ssim

            batch_srgb_psnr/=out_val.shape[0]
            batch_srgb_ssim/=out_val.shape[0]
            # test_srgb_psnr = compare_psnr(np.uint8(gt_srgb_frame*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame*255).astype(np.float32)/255, data_range=1.0)
            # test_srgb_ssim = compare_ssim(np.uint8(gt_srgb_frame*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame*255).astype(np.float32)/255, data_range=1.0, multichannel=True)
            n,c,h,w = out_val.shape
            test_raw_ssim = 0
            # for i in np.arange(n):
            #     # pass
            #     test_raw_ssim += compare_ssim(raw1_pack[:,:,i], raw2_pack[:,:,i], data_range=1.0)
            # test_raw_ssim /= n
            #     # todo
            psnr_val_dataset    += psnr_val_seq
            psnr_noisy_dataset  += psnr_noisy_seq
            ssim_val_dataset    += ssim_val_seq
            psnr_val_srgb_dataset    += batch_srgb_psnr
            ssim_val_srgb_dataset    += batch_srgb_ssim
            
            if verbose:
                current_path = dataset.seqs_dirs[i]
                if isinstance(current_path, list): current_path = current_path[0]
                ssim_string = "\n\tSSIM result {:.4f}".format(ssim_val_seq)
                logger_txt.write("\nFinished denoising %s"%(current_path))
                logger_txt.write("\n\tDenoised %s frames in %f s"%(str(seqn_val.shape), (time.time()-seq_start_time)))
                logger_txt.write("\n\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy_seq, psnr_val_seq))
                logger_txt.write(ssim_string)
                logger_txt.flush()
                
                print("\nFinished denoising %s"%(current_path))
                print("\n\tDenoised %s frames in %f s"%(str(seqn_val.shape), (time.time()-seq_start_time)))
                print("\n\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy_seq, psnr_val_seq))
                print(ssim_string)
                
            if save_path is not None:
                # downsample = 2
                save_out_seq(seqn_val[..., ::downsample, ::downsample], 
                             seq_gt[..., ::downsample, ::downsample], 
                             out_val[..., ::downsample, ::downsample], 
                             save_path+'/%04d'%epoch+'_%s/%04d'%(name, i, ),
                             save_length=5,
                             fext=fext,
                             vis=vis
                             )

        psnr_val_dataset    /= len(dataset)
        psnr_noisy_dataset  /= len(dataset)
        ssim_val_dataset  /= len(dataset)
        psnr_val_srgb_dataset /=len(dataset)
        ssim_val_srgb_dataset /=len(dataset)
        t2 = time.time()
        
        writer_tensorboard.add_scalar('PSNR on validation data %s'%name, psnr_val_dataset, epoch)
        writer_tensorboard.add_scalar('SSIM on validation data %s'%name, ssim_val_dataset, epoch)
        
        writer_tensorboard.add_scalar('PSNR on RGB validation data %s'%name, psnr_val_srgb_dataset, epoch)
        writer_tensorboard.add_scalar('SSIM on RGB validation data %s'%name, ssim_val_srgb_dataset, epoch)
        if lr is not None: writer_tensorboard.add_scalar('Learning rate', lr, epoch)
    
    # import pdb; pdb.set_trace()
    logger_txt.write("\n"+"*"*100)
    # if data_as_batch:
        # valsetdir = dataset.dataset.valsetdir
    # else:
    valsetdir = dataset.valsetdir
    finish_string = "\nFinished denoising the whole dataset %s"%(valsetdir)
    # if data_as_batch:
        # logger_txt.write("\nFinished denoising the whole dataset %s"%(dataset.dataset.valsetdir))
    # else:
        # logger_txt.write("\nFinished denoising the whole dataset %s"%(dataset.valsetdir))
    logger_txt.write(finish_string)
    print(finish_string)
    result_string = "\n[epoch %d] Mean PSNR noisy %.4f dB, Mean PSNR result %.4f dB, Mean SSIM result %.4f dB, validate time %.4f" \
                %( epoch, np.mean(psnr_noisy_dataset), np.mean(psnr_val_dataset), ssim_val_dataset, t2-t1)
    result_string += "\n Mean SRGB PSNR noisy %.4f dB, Mean SRGB SSIM result %.4f dB" \
                %( psnr_val_srgb_dataset, ssim_val_srgb_dataset)        
    logger_txt.write(result_string)
    print(result_string)
    # logger_txt.write("\n[epoch %d] Mean PSNR noisy %.4f dB, Mean PSNR result %.4f dB, validate time %.4f"
                # %( epoch, np.mean(psnr_noisy_dataset), np.mean(psnr_val_dataset), t2-t1))

    # if data_as_batch:
        # print("\nFinished denoising the whole dataset %s"%(dataset.dataset.valsetdir))
    # else:
        # print("\nFinished denoising the whole dataset %s"%(dataset.valsetdir))
    # print("\n[epoch %d] Mean PSNR noisy %.4f dB, Mean PSNR result %.4f dB, validate time %.4f"
        # %(epoch, np.mean(psnr_noisy_dataset), np.mean(psnr_val_dataset), (t2-t1)))
    logger_txt.flush()
    return np.mean(psnr_val_dataset)

# TODO merge with denoise_seq where noise is a separate input
def denoise_seq_5to1(seq, model_temporal, temp_psz=1):
    r"""Denoises a sequence of frames 
    Args:
        seq: Tensor. [numframes, C, H, W] array containing the noisy input frames
        model_temp: instance of the PyTorch model of the temporal denoiser
    Returns:
        denframes: Tensor, [numframes, C, H, W]
    """
    # init arrays to handle contiguous frames and related patches
    numframes, noisy_C, H, W = seq.shape
    clean_C = noisy_C-1
    # import pdb; pdb.set_trace()
    
    # 15 3 960 540 
    # (5, 3, 480, 832)
    if temp_psz !=1:
        ctrlfr_idx = int((temp_psz-1)//2)
    inframes = list()
    denframes = torch.empty((numframes, clean_C, H, W)).to(seq.device)

    # build noise map from noise std---assuming Gaussian noise
    # noise_map = noise_std.expand((1, 1, H, W))

    for fridx in range(numframes):
        if temp_psz == 1:
            inframe = seq[fridx]
            inframes_t = inframe.view((1, noisy_C, H, W)).to(seq.device)
        elif temp_psz !=1:
            inframes = []
            for idx in range(temp_psz):  #5 
                relidx = abs(fridx-ctrlfr_idx+idx) # handle border conditions, reflect
                relidx = min(relidx, -relidx + 2*(numframes-1)) # handle border conditions
                inframes.append(seq[relidx])
            inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz*noisy_C, H, W)).to(seq.device)

        output = model_temporal(inframes_t).clamp(0., 1.)
        if output.shape[0] == 5:
            output = output[ctrlfr_idx, ...]
        denframes[fridx] = output
    
    del inframes
    del inframes_t
    torch.cuda.empty_cache()

    return denframes


def denoise_seq_5to5(seq, model_temporal, temp_psz):
    r"""Denoises a sequence of frames with FastDVDnet.
    Args:
        seq: Tensor. [numframes, C, H, W] array containing the noisy input frames
        temp_psz: size of the temporal patch
        model_temp: instance of the PyTorch model of the temporal denoiser
    Returns:
        denframes: Tensor, [numframes, C, H, W]
    """
    # init arrays to handle contiguous frames and related patches
    numframes, C, H, W = seq.shape

    denframes = torch.empty((numframes, C-1, H, W)).to(seq.device)

    num_seg = numframes // temp_psz
    last_seg_len = numframes % temp_psz
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% one time loading all, out of memory
    # if last_seg_len > 0:
    # 	num_seg += 1
    # 	seq = torch.cat(seq, torch.flip(seq[-last_seg_len-1:-1]))
    # noise_map = noise_std.expand((num_seg*temp_psz, 1, H, W))
    # denframes = temp_denoise(model_temporal, seq, noise_map)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for fridx in range(num_seg):
        start, end = fridx*temp_psz, (fridx+1)*temp_psz
        inframes = seq[start : end]
        result = model_temporal(inframes).clamp(0., 1.)
        # import pdb; pdb.set_trace()
        # 
        denframes[start : end] = result

    if last_seg_len > 0:
        inframes = torch.cat((seq[num_seg*temp_psz:], 
                             torch.flip(seq[last_seg_len-temp_psz:], dims=[0])))
        if inframes.shape[0] != temp_psz:
            import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # (Pdb) inframes.shape                                                                              
        # torch.Size([5, 5, 540, 960])
        out = model_temporal(inframes).clamp(0., 1.)
        
        denframes[num_seg*temp_psz:] = out[:last_seg_len]
        del out
    del inframes
    
    torch.cuda.empty_cache()
    
    return denframes

#-------------------------------------------save and visualizer function-------------------------------

def save_out_seq(seqnoisy, seqgt, seqclean, save_dir, sigmaval=0, suffix="", save_length=5, fext='.jpg', vis=None):
    
    
    """Saves the denoised and noisy sequences under save_dir
    
    Args
    --------
    seqnoisy:   N,C,H,W
    seqclean:   N,C,H,W
    save_dir:   folder to save
    sigmaval:   int noise sigma
    suffix:     name append in path
    save_noisy: whether save noisy image
    """
    if vis==None:
        vis = vis_global
    seq_len = seqnoisy.size()[0]
    os.makedirs(save_dir, exist_ok=True)
    # fastdvd forward not work for CRVD datast
    if seqnoisy.shape[1] == 25 and len(seqnoisy.shape)==4:
        # seqnoisy
        for idx in np.arange(5):
            noisy_name = os.path.join(save_dir,\
                        ('motion_input_{}'+fext).format(idx))
            noisyimg = variable_to_cv2_image(seqnoisy[0, 5*idx: 5*(idx+1), ...].clamp(0., 1.), vis)
            cv2.imwrite(noisy_name, noisyimg)
        n, fc, h, w = seqnoisy.shape
        seqnoisy = seqnoisy[:, 10:15, :, :]
    
    
    
    for idx in range(seq_len)[0:save_length]:
        # Build Outname
        # fext = '.jpg'
        noisy_name = os.path.join(save_dir,\
                        ('n{}_input_{}').format(sigmaval, idx) + fext)
        gt_name = os.path.join(save_dir,\
                ('n{}_gt_{}_{}').format(sigmaval, suffix, idx) + fext)
        noise_map_name = os.path.join(save_dir,\
                ('n{}_noisemap_{}_{}').format(sigmaval, suffix, idx) + fext)
        if len(suffix) == 0:
            out_name = os.path.join(save_dir,\
                    ('n{}_predict_{}').format(sigmaval, idx) + fext)
        else:
            out_name = os.path.join(save_dir,\
                    ('n{}_predict_{}_{}').format(sigmaval, suffix, idx) + fext)
        
        # Save result

        noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.), vis)
        cv2.imwrite(noisy_name, noisyimg)

        # noise_map = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.)[4:5, ...])
        # cv2.imwrite(noise_map_name, noise_map)
        vis.vis_noise(seqnoisy[idx:idx+1].cpu(), noise_map_name)

        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0), vis)
        cv2.imwrite(out_name, outimg)
        
        gtimg = variable_to_cv2_image(seqgt[idx].unsqueeze(dim=0), vis)
        cv2.imwrite(gt_name, gtimg)
        
        concat_name = os.path.join(save_dir,\
                ('n{}_concat_{}_{}').format(sigmaval, suffix, idx) + fext)
        concat_img = np.concatenate([noisyimg, gtimg, outimg], axis=1)
        cv2.imwrite(concat_name, concat_img)

from utils.visualizer import Visualizer
vis_global = Visualizer()

def variable_to_cv2_image(invar, conv_rgb_to_bgr=True, vis=None):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        invar: a torch.autograd.Variable
        conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
    Returns:
        a HxWxC uint8 image
    """
    if vis is None:
        vis = vis_global
    assert torch.max(invar) <= 1.0

    size4 = len(invar.size()) == 4             
    if size4:
        # shape of invar N, C, H, W
        nchannels = invar.size()[1]
    else:
        # shape of invar C, H, W
        nchannels = invar.size()[0]

    if nchannels == 1:
        if size4:
            res = invar.data.cpu().numpy()[0, 0, :] # H, W
        else:
            res = invar.data.cpu().numpy()[0, :] # H, W
        res = (res*255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        if size4:
            res = invar.data.cpu().numpy()[0] # C, H, W
        else:
            res = invar.data.cpu().numpy() # C, H, W
        res = res.transpose(1, 2, 0) # H, W, C
        res = (res*255.).clip(0, 255).astype(np.uint8)
        if conv_rgb_to_bgr:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    elif nchannels in (4, 5): # raw data RGGB
        if len(invar.shape)==3: invar=invar[None, ...]
        # 5, 270, 480
        # RuntimeError: Sizes of tensors must match except in dimension 2. Got 135 and 134 (The offending index is 0)
        N, C, H, W = invar.shape
        invar = invar[:, :, 0:H//4*4, 0:W//4*4]
        res = vis.torch2npvis(invar)
    else:
        raise Exception('Number of color channels not supported')
    return res



if __name__ == "__main__":
    pass