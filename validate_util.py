import os
import time
import cv2
import numpy as np
import torch
import torchvision.utils as tutils
from utils import batch_psnr
import torch
import torch.nn.functional as F
from tqdm import tqdm

def validate_and_log_gt_from_loder(model_temp, dataset_val, valnoisestd, 
                          temp_psz, writer_tensorboard, \
                            epoch, logger_txt, save_path = None, lr=None, name="", mode="5to1"):
    """Validation step after the epoch finished
        dataset_val a list with length 4, each element [15, 3, 540, 960]
        epoch: int > 0 means training stage. < 0  means test stage
    """
    print('validate_and_log_gt_from_loder')
    t1 = time.time()
    psnr_val_dataset = 0
    psnr_noisy_dataset = 0
    with torch.no_grad():
        # i = 0
        for i, (seqn_val, seq_gt) in enumerate(tqdm(dataset_val)):
            seq_start_time = time.time()
            # seq_val   [20, 3, 480, 8544], (0,1)
            # noise     [20, 3, 480, 854]   (-0.65, 0.67)
            # print(seq.shape, seq.max(), seq.min())
            # print(noise.shape, noise.max(), noise.min())
            # noise = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
            # seqn_val = seq_val + noise
            seqn_val = seqn_val.cuda()
            seq_gt   = seq_gt.cuda()
            # sigma_noise = torch.cuda.FloatTensor([valnoisestd])
            # if mode == "5to1":
            #     out_val = denoise_seq(seq=seqn_val, \
            #                         noise_std=sigma_noise, \
            #                         model_temporal=model_temp,
            #                         temp_psz=temp_psz)
            if mode == "5to5":
                out_val = denoise_seq_fastdvdnet(seq=seqn_val, \
                    noise_std=sigma_noise, \
                    model_temporal=model_temp,
                    temp_psz=temp_psz)
            else: raise NotImplementedError

            psnr_val_seq        = batch_psnr(out_val.cpu(),    seq_gt.squeeze_(), 1.)
            psnr_noisy_seq      = batch_psnr(seqn_val.cpu(),   seq_gt.squeeze_(), 1.)
            psnr_val_dataset    += psnr_val_seq
            psnr_noisy_dataset  += psnr_noisy_seq
            if epoch <0:
                logger_txt.write("\nFinished denoising %s"%(os.path.basename(dataset_val.seqs_dirs[i])))
                logger_txt.write("\n\tDenoised %s frames in %f s"%(str(seqn_val.shape), (time.time()-seq_start_time)))
                logger_txt.write("\n\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy_seq, psnr_val_seq))
                logger_txt.flush()
                print("\nFinished denoising %s"%(os.path.basename(dataset_val.seqs_dirs[i])))
                print("\n\tDenoised %s frames in %f s"%(str(seqn_val.shape), (time.time()-seq_start_time)))
                print("\n\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy_seq, psnr_val_seq))
            if save_path is not None:
                downsample = 2
                save_out_seq(seqn_val[..., ::downsample, ::downsample], 
                             seq_gt[..., ::downsample, ::downsample], 
                             out_val[..., ::downsample, ::downsample], 
                             save_path+'/%04d'%epoch+'_%s/%04d_%s'%(name, i, os.path.basename(dataset_val.seqs_dirs[i])), 
                             int(valnoisestd*255)
                             )
            # i+=1
            # todo: add test time logging here
        psnr_val_dataset    /= len(dataset_val)
        psnr_noisy_dataset  /= len(dataset_val)
        t2 = time.time()
        
    if epoch >= 0:
        writer_tensorboard.add_scalar('PSNR on validation data %s'%name, psnr_val_dataset, epoch)
        if lr is not None: writer_tensorboard.add_scalar('Learning rate', lr, epoch)
        # Log validation results
        # irecon = tutils.make_grid(out_val.data[0].clamp(0., 1.),\
        #                         nrow=2, normalize=False, scale_each=False)
        # writer_tensorboard.add_image('Reconstructed validation image %s'%name, irecon, epoch)
    
    
    logger_txt.write("\n"+"*"*100)
    logger_txt.write("\nFinished denoising the whole dataset %s"%(dataset_val.valsetdir))
    logger_txt.write("\n[epoch %d] Mean PSNR noisy %.4f dB, Mean PSNR result %.4f dB, validate time %.4f"
                %( epoch, np.mean(psnr_noisy_dataset), np.mean(psnr_val_dataset), t2-t1))
    
    print("\nFinished denoising the whole dataset %s"%(dataset_val.valsetdir))
    print("\n[epoch %d] Mean PSNR noisy %.4f dB, Mean PSNR result %.4f dB, validate time %.4f"
        %(epoch, np.mean(psnr_noisy_dataset), np.mean(psnr_val_dataset), (t2-t1)))
    logger_txt.flush()
    return np.mean(psnr_val_dataset)

def validate_and_log(model_temp, dataset_val, valnoisestd, 
                          temp_psz, writer_tensorboard, \
                            epoch, logger_txt, save_path = None, lr=None, name="", mode="5to1"):
    """Validation step after the epoch finished
        dataset_val a list with length 4, each element [15, 3, 540, 960]
        epoch: int > 0 means training stage. < 0  means test stage
    """
    print('use validation version 0623 2046')
    t1 = time.time()
    psnr_val_dataset = 0
    psnr_noisy_dataset = 0
    with torch.no_grad():
        # i = 0
        for i, seq_val in enumerate(tqdm(dataset_val)):
            seq_start_time = time.time()
            # seq_val   [20, 3, 480, 8544], (0,1)
            # noise     [20, 3, 480, 854]   (-0.65, 0.67)
            # print(seq.shape, seq.max(), seq.min())
            # print(noise.shape, noise.max(), noise.min())
            noise = torch.FloatTensor(seq_val.size()).normal_(mean=0, std=valnoisestd)
            seqn_val = seq_val + noise
            seqn_val = seqn_val.cuda()
            sigma_noise = torch.cuda.FloatTensor([valnoisestd])
            if mode == "5to1":
                out_val = denoise_seq(seq=seqn_val, \
                                    noise_std=sigma_noise, \
                                    model_temporal=model_temp,
                                    temp_psz=temp_psz)
            elif mode == "5to5":
                out_val = denoise_seq_fastdvdnet(seq=seqn_val, \
                    noise_std=sigma_noise, \
                    model_temporal=model_temp,
                    temp_psz=temp_psz)
            else: raise NotImplementedError

            psnr_val_seq        = batch_psnr(out_val.cpu(),    seq_val.squeeze_(), 1.)
            psnr_noisy_seq      = batch_psnr(seqn_val.cpu(),   seq_val.squeeze_(), 1.)
            psnr_val_dataset    += psnr_val_seq
            psnr_noisy_dataset  += psnr_noisy_seq
            if epoch <0:
                logger_txt.write("\nFinished denoising %s"%(os.path.basename(dataset_val.seqs_dirs[i])))
                logger_txt.write("\n\tDenoised %s frames in %f s"%(str(seq_val.shape), (time.time()-seq_start_time)))
                logger_txt.write("\n\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy_seq, psnr_val_seq))
                logger_txt.flush()
                print("\nFinished denoising %s"%(os.path.basename(dataset_val.seqs_dirs[i])))
                print("\n\tDenoised %s frames in %f s"%(str(seq_val.shape), (time.time()-seq_start_time)))
                print("\n\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy_seq, psnr_val_seq))
            if save_path is not None:
                downsample = 2
                save_out_seq(seqn_val[..., ::downsample, ::downsample], 
                             seq_val[..., ::downsample, ::downsample], 
                             out_val[..., ::downsample, ::downsample], 
                             save_path+'/%04d'%epoch+'_%s/%04d_%s'%(name, i, os.path.basename(dataset_val.seqs_dirs[i])), 
                             int(valnoisestd*255)
                             )
            # i+=1
            # todo: add test time logging here
        psnr_val_dataset    /= len(dataset_val)
        psnr_noisy_dataset  /= len(dataset_val)
        t2 = time.time()
        
    if epoch >= 0:
        writer_tensorboard.add_scalar('PSNR on validation data %s'%name, psnr_val_dataset, epoch)
        if lr is not None: writer_tensorboard.add_scalar('Learning rate', lr, epoch)
        # Log validation results
        irecon = tutils.make_grid(out_val.data[0].clamp(0., 1.),\
                                nrow=2, normalize=False, scale_each=False)
        writer_tensorboard.add_image('Reconstructed validation image %s'%name, irecon, epoch)
    
    
    logger_txt.write("\n"+"*"*100)
    logger_txt.write("\nFinished denoising the whole dataset %s"%(dataset_val.valsetdir))
    logger_txt.write("\n[epoch %d] Mean PSNR noisy %.4f dB, Mean PSNR result %.4f dB, validate time %.4f"
                %( epoch, np.mean(psnr_noisy_dataset), np.mean(psnr_val_dataset), t2-t1))
    
    print("\nFinished denoising the whole dataset %s"%(dataset_val.valsetdir))
    print("\n[epoch %d] Mean PSNR noisy %.4f dB, Mean PSNR result %.4f dB, validate time %.4f"
        %(epoch, np.mean(psnr_noisy_dataset), np.mean(psnr_val_dataset), (t2-t1)))
    logger_txt.flush()
    return np.mean(psnr_val_dataset)
    # Log val images

def log_val_images_at_start(trainimg, writer, epoch, seq_val, seqn_val, idx):
                # Log training images
    _, _, Ht, Wt = trainimg.size()
    img = tutils.make_grid(trainimg.view(-1, 3, Ht, Wt), \
                            nrow=8, normalize=True, scale_each=True)
    writer.add_image('Training patches', img, epoch)

    # Log validation images
    img = tutils.make_grid(seq_val.data[idx].clamp(0., 1.),\
                            nrow=2, normalize=False, scale_each=False)
    imgn = tutils.make_grid(seqn_val.data[idx].clamp(0., 1.),\
                            nrow=2, normalize=False, scale_each=False)
    writer.add_image('Clean validation image {}'.format(idx), img, epoch)
    writer.add_image('Noisy validation image {}'.format(idx), imgn, epoch)
    

def denoise_seq(seq, noise_std, model_temporal, temp_psz=1):
    r"""Denoises a sequence of frames 
    Args:
        seq: Tensor. [numframes, C, H, W] array containing the noisy input frames
        noise_std: Tensor. Standard deviation of the added noise		
        model_temp: instance of the PyTorch model of the temporal denoiser
    Returns:
        denframes: Tensor, [numframes, C, H, W]
    """
    # init arrays to handle contiguous frames and related patches
    numframes, C, H, W = seq.shape
    # import pdb; pdb.set_trace()
    
    # 15 3 960 540 
    # (5, 3, 480, 832)
    if temp_psz !=1:
        ctrlfr_idx = int((temp_psz-1)//2)
    inframes = list()
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)

    # build noise map from noise std---assuming Gaussian noise
    noise_map = noise_std.expand((1, 1, H, W))

    for fridx in range(numframes):
        if temp_psz == 1:
            inframe = seq[fridx]
            inframes_t = inframe.view((1, C, H, W)).to(seq.device)
        elif temp_psz !=1:
            inframes = []
            for idx in range(temp_psz):  #5 
                relidx = abs(fridx-ctrlfr_idx+idx) # handle border conditions, reflect
                relidx = min(relidx, -relidx + 2*(numframes-1)) # handle border conditions
                inframes.append(seq[relidx])
                # print(relidx)
            # print('length ', len(inframes))
            # print('shape', inframes[0].shape)
            inframes_t = torch.stack(inframes, dim=0).contiguous().view((1, temp_psz*C, H, W)).to(seq.device)
        # append result to output list
        # print('inframes shape', inframes_t.shape)
        # print('noise_map shape', noise_map.shape)
        # import pdb; pdb.set_trace()
        output = padding_forward_clamp(model_temporal, inframes_t, noise_map)
        if output.shape[0] == 5:
            output = output[ctrlfr_idx, ...]
        denframes[fridx] = output
        
            
    # free memory up
    del inframes
    del inframes_t
    torch.cuda.empty_cache()

    # convert to appropiate type and return
    return denframes

def denoise_seq_fastdvdnet(seq, noise_std, temp_psz, model_temporal):
    r"""Denoises a sequence of frames with FastDVDnet.
    Args:
        seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
        noise_std: Tensor. Standard deviation of the added noise
        temp_psz: size of the temporal patch
        model_temp: instance of the PyTorch model of the temporal denoiser
    Returns:
        denframes: Tensor, [numframes, C, H, W]
    """
    # init arrays to handle contiguous frames and related patches
    numframes, C, H, W = seq.shape
    ctrlfr_idx = int((temp_psz-1)//2)
    # inframes = list()
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)

    # build noise map from noise std---assuming Gaussian noise
    # noise_map = noise_std.expand((1, 1, H, W))
    noise_map = noise_std.expand((temp_psz, 1, H, W))
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
        denframes[start : end] = padding_forward_clamp(model_temporal, inframes, noise_map)

    if last_seg_len > 0:
        inframes = torch.cat((seq[num_seg*temp_psz:], 
                             torch.flip(seq[last_seg_len-temp_psz:], dims=[0])))
        if inframes.shape[0] != noise_map.shape[0]:
            import pdb; pdb.set_trace()
        out = padding_forward_clamp(model_temporal, inframes, noise_map)
        denframes[num_seg*temp_psz:] = out[:last_seg_len]
    # import pdb; pdb.set_trace()

    # free memory up
    del inframes
    torch.cuda.empty_cache()
    # convert to appropiate type and return
    return denframes


def padding_forward_clamp(model, noisyframe, sigma_noise):
    '''Encapsulates call to denoising model and handles padding.
        Expects noisyframe to be normalized in [0., 1.]
        if model return a list of image, take final image as output
    '''
    # make size a multiple of four (we have two scales in the denoiser)
    # todo: Unet need another size
    sh_im = noisyframe.size()
    expanded_h = sh_im[-2]%4
    if expanded_h:
        expanded_h = 4-expanded_h
    expanded_w = sh_im[-1]%4
    if expanded_w:
        expanded_w = 4-expanded_w
    padexp = (0, expanded_w, 0, expanded_h)
    noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
    sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')
    # todo: I am afraid padding is not the best, modify it to crop
    # denoise
    input_x = torch.cat((noisyframe, sigma_noise), dim=1)
    out = model(input_x)
    if isinstance(out, list):
        out = out[-1]
    out = out.clamp(0., 1.)
    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]

    return out


OUTIMGEXT = '.jpg' # output images format
# seqn_val, seq_val, out_val
def save_out_seq(seqnoisy, seqgt, seqclean, save_dir, sigmaval, suffix="", save_noisy=True):
    
    
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
    seq_len = seqnoisy.size()[0]
    downsample_rate = 4
    os.makedirs(save_dir, exist_ok=True)
    # print('save result to %s'%(save_dir))
    for idx in range(seq_len)[0:5]:
        # Build Outname
        fext = OUTIMGEXT
        noisy_name = os.path.join(save_dir,\
                        ('n{}_input_{}').format(sigmaval, idx) + fext)
        gt_name = os.path.join(save_dir,\
                ('n{}_gt_{}_{}').format(sigmaval, suffix, idx) + fext)
        if len(suffix) == 0:
            out_name = os.path.join(save_dir,\
                    ('n{}_predict_{}').format(sigmaval, idx) + fext)
        else:
            out_name = os.path.join(save_dir,\
                    ('n{}_predict_{}_{}').format(sigmaval, suffix, idx) + fext)
        
        # Save result
        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            cv2.imwrite(noisy_name, noisyimg)

        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)
        
        gtimg = variable_to_cv2_image(seqgt[idx].unsqueeze(dim=0))
        cv2.imwrite(gt_name, gtimg)
        
        concat_name = os.path.join(save_dir,\
                ('n{}_concat_{}_{}').format(sigmaval, suffix, idx) + fext)
        concat_img = np.concatenate([noisyimg, gtimg, outimg], axis=1)
        cv2.imwrite(concat_name, concat_img)


def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        invar: a torch.autograd.Variable
        conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
    Returns:
        a HxWxC uint8 image
    """
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
    else:
        raise Exception('Number of color channels not supported')
    return res