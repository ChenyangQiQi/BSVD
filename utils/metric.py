import torch
import numpy as np
from skimage.measure import compare_psnr 
compare_psnr_rvidenet = compare_psnr
from skimage.measure import compare_ssim

# copy from fastdvd repo
def batch_psnr(img, imclean, data_range):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    if isinstance(img, torch.Tensor):
        img_cpu = img.data.cpu().numpy().astype(np.float32)
        imgclean = imclean.data.cpu().numpy().astype(np.float32)
    else:
        img_cpu = img.astype(np.float32)
        imgclean = imclean.astype(np.float32)        
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
                       data_range=data_range)
    return psnr/img_cpu.shape[0]




# SSIM metric copy from rvidenet
def pack_gbrg_raw_for_compute_ssim(raw):

    im = raw.astype(np.float32)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out

# SSIM metric copy from rvidenet
def compute_ssim_for_packed_raw(raw1, raw2):
    raw1_pack = pack_gbrg_raw_for_compute_ssim(raw1)
    raw2_pack = pack_gbrg_raw_for_compute_ssim(raw2)
    test_raw_ssim = 0
    for i in range(4):
        test_raw_ssim += compare_ssim(raw1_pack[:,:,i], raw2_pack[:,:,i], data_range=1.0)

    return test_raw_ssim/4