"""
Test algorithm
"""

import torch
from Experimental_root.models import global_queue_buffer



def temp_denoise(model, noisyframe, sigma_noise, device, args=None):
    '''Encapsulates call to denoising model and handles padding.
        Expects noisyframe to be normalized in [0., 1.]
    '''

    N, C, H, W = noisyframe.size()
    
    noisyframe = noisyframe[None, ...]
    if sigma_noise != None:
        assert abs(sigma_noise.mean() - sigma_noise[0,0,0,0]) < 1e-05
        sigma_noise = torch.ones(
            (1, N, 1, H, W)).to(sigma_noise.device) * sigma_noise[0, 0, 0, 0]

    if sigma_noise != None:
        out = torch.clamp(model(noisyframe, noise_map=sigma_noise)[0, ...], 0., 1.) # Why this cannot work on multiple gpu? only one video loaded per time
    else:
        out = torch.clamp(model(noisyframe)[0, ...], 0., 1.)
    # out = reshape_for_model(out, 1, N, 3, H, W, args['model'], rev=True)
    torch.cuda.synchronize()


    return out.to(device)

def denoise_seq(seq, noise_map, temp_psz, model_temporal, future_buffer_len=0, args=None):
    r"""Denoises a sequence of frames.
    During validation, We use MIMO with global buffer for ease of usage.
    During test, we use pipeline algorithm BSVD with Buffer in each block for better memory-fidelity tradeoff
    Args:
        seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
        noise_map: Tensor. Standard deviation of the added noise
        temp_psz: size of the temporal patch
        model_temp: instance of the PyTorch model of the temporal denoiser
    Returns:
        denframes: Tensor, [numframes, C, H, W]
    """

    device = next(model_temporal.parameters()).device

    numframes, C, H, W = seq.shape
    # For BSVD, we test the video sequence in a single forward
    if temp_psz == -1: temp_psz = numframes 
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)

    
    num_seg = numframes // temp_psz
    num_last_seg_frames = numframes % temp_psz
    num_batches = num_seg  
    num_batch_frames = temp_psz 
    num_last_batch_frames = numframes % num_batch_frames

    
    global_queue_buffer._init(future_buffer_len)
    
    for fridx in range(num_batches):
        global_queue_buffer.set_batch_index(fridx)
        start, end = fridx*num_batch_frames, (fridx+1)*num_batch_frames 
        end_new = end + future_buffer_len
        if end_new > numframes:
            end_new = end
            global_queue_buffer.set_future_buffer_length(0)
        inframes = seq[start: end_new]

        denframes[start: end] = temp_denoise(model_temporal, inframes.to(device), noise_map, seq.device, args=args)[:num_batch_frames] 

    global_queue_buffer.set_future_buffer_length(0)
    if num_last_batch_frames > 0:
        if num_last_seg_frames > 0:
            last_sequence = torch.cat((seq[num_seg*temp_psz:], 
                                torch.flip(seq[-(temp_psz-num_last_seg_frames)-1:-1], dims=[0])))
            if num_last_batch_frames == num_last_seg_frames:
                out = temp_denoise(model_temporal, last_sequence.to(device), noise_map, seq.device, args=args)
                denframes[num_seg*temp_psz:] = out[:num_last_seg_frames]
            else:
                inframes = torch.cat((seq[num_batches*num_batch_frames : numframes-num_last_seg_frames],last_sequence))
                out = temp_denoise(model_temporal, inframes.to(device), noise_map, seq.device, args=args)
                denframes[num_batches*num_batch_frames:] = out[:num_last_batch_frames]
            if last_sequence is not None: del last_sequence
        else:
            out = temp_denoise(model_temporal, seq[num_batches*num_batch_frames:].to(device), noise_map, seq.device, args=args)
            denframes[num_batches*num_batch_frames:] = out
        if out is not None: del out  

    # free memory up
    if inframes is not None: del inframes
    
    if noise_map is not None: del noise_map
    
    global_queue_buffer._clean()
    torch.cuda.empty_cache()
    # convert to appropiate type and return
    return denframes
