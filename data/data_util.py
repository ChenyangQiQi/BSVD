import numpy as np
from scipy.stats import poisson
import os
import torch


class NoiseGetter():
    def __init__(self) -> None:
        self.iso_list   =   [1600,3200,6400,12800,25600]
        self.a_list     =   [3.513262,6.955588,13.486051,26.585953,52.032536]  
        self.g_noise_var_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]
    
    def get_sigma(self, noisy_level, image):
        """
            return 1 channel sigma_sum and 4 channel sigma

        Args:
            noisy_level : [description]
            image : shape h, w, range 2**12-1, 240
                    image is clean ground truth when synthesize pseudo input
                    image is noisy image when get network input
        """
        dic = {}
        a = self.a_list[noisy_level]
        g_noise_var = self.g_noise_var_list[noisy_level]
        
        variance = a*np.maximum((image-240.0), 0.0)+g_noise_var # 300 - 13k
        sigma_sum = np.sqrt(variance)
        sigma_mean, sigma_4 =  pack_sigma(sigma_sum)

        dic['a'] = a 
        dic['g_noise_var'] = g_noise_var
        dic['variance'] = variance
        dic['sigma_sum'] = sigma_sum
        dic['sigma_mean'] = sigma_mean
        dic['sigma_4'] = sigma_4
        return dic

    def generate_noisy_raw(self, noisy_level, image):

        a = self.a_list[noisy_level]
        gaussian_noise_var = self.g_noise_var_list[noisy_level]
        poisson_noisy_img = poisson((image-240.0)/a).rvs()*a
        gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(image.shape[0], image.shape[1])
        noisy_img = poisson_noisy_img + gaussian_noise + 240.0
        noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
        
        return noisy_img
    
    def get_gaussian(self, noisy_level, image):
        noise_dict = self.get_sigma(noisy_level, image)
        sigma_sum   = noise_dict['sigma_sum']
        gaussian_noise_sum = sigma_sum[...]*\
                    np.random.randn(image.shape[0], image.shape[1])
        noisy_raw = image + gaussian_noise_sum
        noisy_raw = np.minimum(np.maximum(noisy_raw,0), 2**12-1)
        return noisy_raw


# class noise_object():
#     def __init__(self) -> None:
#         self.sigma_mean = 0
    
        
def pack_gbrg_raw(raw):
    #pack GBRG Bayer raw
    # from  shape (H,W),        range [240, (2**12-1)] 
    # to    shape (h, w, 4),    range [0, 1]
    # black_level = 240.0
    black_level = 240.0
    white_level = 2**12-1.0
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)
    
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out

def pack_sigma(im):
    #pack GBRG Bayer sigma 
    # from shape (H,W), range [240, (2**12-1)] 
    # to shape (h, w, 4), range [0, 1]
    # maximum sigma: 52*4k = 200k
    assert np.min(im) >=0
    im = im.astype(np.float32)
    # rescale the sigma like raw to keep sigma the true std of raw
    black_level = 240.0 # 240.0
    white_level = 2**12-1.0 # (2**12*52+2000)**0.5
    im = im / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out_demosaic = np.concatenate(( im[1:H:2, 0:W:2, :],
                                    im[1:H:2, 1:W:2, :],
                                    im[0:H:2, 1:W:2, :],
                                    im[0:H:2, 0:W:2, :]), axis=2)
    
    out_mean = np.mean(out_demosaic, axis=2)
    out_demosaic = out_demosaic.astype(np.float32)
    out = out_mean.astype(np.float32)
    assert np.max(out)<1.0 and np.min(out) >=0.0
    assert np.max(out_demosaic)<1.0 and np.min(out_demosaic) >=0.0
    return out, out_demosaic

def get_frame_id(gt_path):
    gt_fn = os.path.basename(gt_path)
    if gt_fn[0:5] == 'frame':
        # frame1_clean.tiff
        # frame1_clean_and_slightly_denoised
        frame_id = int(gt_fn[5])
        assert frame_id>=1 and frame_id <=7
    elif gt_fn[2]!='0':
        frame_id = int(gt_fn[2:6])
    elif gt_fn[3]!='0':
        # 000594_raw
        frame_id = int(gt_fn[3:6])
    elif gt_fn[4]!='0':
        frame_id = int(gt_fn[4:6])
    else:
        frame_id = int(gt_fn[5])
    return frame_id



# ------------------------------ Data Augmentation and Transform ------------------------

def numpy_to_torch(input_pack_nfhwc):
    """        
    swap color axis because
    numpy image: H x W x C
    torch image: C x H x W
    """
    input_nfchw = torch.from_numpy(input_pack_nfhwc.copy().astype(np.float32)).permute(0,3,1,2)
    return input_nfchw

def list_to_torch(sample_list):
    """Convert ndarrays in sample to Tensors."""
    torch_list = [numpy_to_torch(sample) for sample in sample_list]
    return torch_list

def crop_transform(sample_list, ps):
    N, C, H, W = sample_list[0].shape
    # assert H == 1080 and W == 1920, "shape is %d, %d"%(H, W)
    # H = 1080
    # W = 1920
    xx = np.random.randint(0, W - ps+1)
    # while xx%2!=0:
        # xx = np.random.randint(0, W - ps*2+1)
    yy = np.random.randint(0, H - ps+1)
    # while yy%2!=0:
        # yy = np.random.randint(0, H - ps*2+1)
    return [sample[:, :, yy:yy + ps, xx:xx + ps]  for sample in sample_list]

def crop_transform_even(sample_list, ps):
    N, C, H, W = sample_list[0].shape
    # assert H == 1080 and W == 1920, "shape is %d, %d"%(H, W)
    # H = 1080
    # W = 1920
    xx = np.random.randint(0, W - ps*2+1)
    while xx%2!=0:
        xx = np.random.randint(0, W - ps*2+1)
    yy = np.random.randint(0, H - ps*2+1)
    while yy%2!=0:
        yy = np.random.randint(0, H - ps*2+1)
    return [sample[:, :, yy:yy + ps*2, xx:xx + ps*2]  for sample in sample_list]

