"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import cv2
import torch
from skimage.measure.simple_metrics import compare_psnr


IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types

def auto_find_gpu():
    device = str(np.argmax([int(x.split()[2]) 
        for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
    print('auto find device', device)
    return device
    

def lr_scheduler(epoch, argdict):
    """Returns the learning rate value depending on the actual epoch number
    By default, the training starts with a learning rate equal to 1e-3 (--lr).
    After the number of epochs surpasses the first milestone (--milestone), the
    lr gets divided by 100. Up until this point, the orthogonalization technique
    is performed (--no_orthog to set it off).
    """
    # Learning rate value scheduling according to argdict['milestone']
    orthog = True
    
    # if len(argdict['milestone']) > 0:
    if epoch > argdict['milestone'][1]:
        current_lr = argdict['lr'] / 100.
        orthog = False
    elif epoch > argdict['milestone'][0]:
        current_lr = argdict['lr'] / 10.
    else:
        current_lr = argdict['lr']
        print('use constant learning rate %f'%current_lr)
    return current_lr, orthog

def normalize_input(img_train, aug=False, verbose=False):
    '''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
        [N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
        patch as a ground truth.
    '''
    # convert from [N, num_frames, C. H, W] in [0., 255.] to [N, num_frames*C. H, W] in  [0., 1.] 
    
    def transform(sample):
        if verbose: print('Augment the result')
        # define transformations
        do_nothing = lambda x: x
        do_nothing.__name__ = 'do_nothing'
        flipud = lambda x: torch.flip(x, dims=[2])
        flipud.__name__ = 'flipup'
        rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
        rot90.__name__ = 'rot90'
        rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
        rot90_flipud.__name__ = 'rot90_flipud'
        rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
        rot180.__name__ = 'rot180'
        rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
        rot180_flipud.__name__ = 'rot180_flipud'
        rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
        rot270.__name__ = 'rot270'
        rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
        rot270_flipud.__name__ = 'rot270_flipud'
        add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
                                 std=(5/255.)).expand_as(x).to(x.device)
        add_csnt.__name__ = 'add_csnt'

        # define transformations and their frequency, then pick one.
        aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
                    rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
        w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12] # one fourth chances to do_nothing
        transf = choices(aug_list, w_aug)

        # transform all images in array
        return transf[0](sample)
    
    img_train = img_train.reshape(img_train.shape[0], -1, \
                            img_train.shape[-2], img_train.shape[-1]) / 255.
    
    img_train = transform(img_train)

    N, C, H, W = img_train.shape
    img_train = img_train[:, :, 0:(H//16*16), 0:(W//16*16)]
    # if gt_frame == -1:
    #     assert C%3 == 0
    #     start_index = 3*(C//6)
    #     end_index   = 3*(C//6+1)
    # else:
    #     start_index = 3*gt_frame
    #     end_index   = 3*(gt_frame+1)
    # 0,1,2     3,4,5,  6,7,8   9,10,11     12,13,14
    # img_gt = img_train[:, start_index:end_index, :, :]
    return img_train

def prepare_nf_cp1_from_n_fcp1(img_input, channel=3):
    N,C, H,W = img_input.shape
    if (C-1)%channel!=0:
        import pdb; pdb.set_trace()
    assert (C-1)%channel==0, "input channel should be f3plus1"

    F = int(C / channel)
    img_train = img_input[:, 0:channel*F, :, :].reshape(-1, channel, H, W)
    noise = img_input[:, -1:, :, :] # n1hw
    noise = torch.cat([noise for _ in range(F)], axis=1).reshape((-1, 1, H, W))
    img_input = torch.cat((img_train, noise), dim=1) # n5, 4, h, w
    N1, C1, H1, W1 = img_input.shape
    assert N1 == N*F and C1 == (channel+1) and H1 == H and W1 == W, "reshape is not correct"
    return img_input

def prepare_nf_4_from_n_f3plus1(img_input):
    N,C, H,W = img_input.shape
    assert (C-1)%3==0, "input channel should be f3plus1"

    F = int(C / 3)
    img_train = img_input[:, 0:3*F, :, :].reshape(-1, 3, H, W)
    noise = img_input[:, -1:, :, :] # n1hw
    noise = torch.cat([noise for _ in range(5)], axis=1).reshape((-1, 1, H, W))
    img_input = torch.cat((img_train, noise), dim=1) # n5, 4, h, w
    N1, C1, H1, W1 = img_input.shape
    assert N1 == N*F and C1 == 4 and H1 == H and W1 == W, "reshape is not correct"
    return img_input

def prepare_tsn_input_data(img_train):
    # gt_train = copy.deepcopy(img_train)
    # img_train, gt_train = normalize_augment(data[0]['data'])
    N, FC, H, W = img_train.size()
    F = int(FC // 3) # 5
    img_train = img_train.view(-1, 3, H, W) # N5, 3, H, W
    gt_train = gt_train.view(-1, 3, H, W) # N, 3, H, W
    

    # std dev of each sequence
    stdn = torch.empty((N, 1, 1, 1)).cuda().uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
    stdn = torch.cat([stdn for _ in range(5)], axis=1).reshape((-1, 1, 1, 1)) # N5, 1, 1, 1
    # draw noise samples from std dev tensor
    noise = torch.zeros_like(img_train)
    noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

    # define noisy input
    imgn_train = img_train + noise
    gt_train = gt_train.cuda(non_blocking=True)
    imgn_train = imgn_train.cuda(non_blocking=True)
    noise = noise.cuda(non_blocking=True)
    noise_map = stdn.expand((N*F, 1, H, W)).cuda(non_blocking=True) # one channel per image
    # Send tensors to GPU
    # gt_train = gt_train.cuda(non_blocking=True)
    # imgn_train = imgn_train.cuda(non_blocking=True)
    # noise = noise.cuda(non_blocking=True)
    # noise_map = stdn.expand((N, 1, H, W)).cuda(non_blocking=True) # one channel per image
    input_x = torch.cat((imgn_train, noise_map), dim=1)
    return input_x
    

# TODO: add orthog for conv3d
def svd_orthogonalization(lyr):
    r"""Applies regularization to the training by performing the
    orthogonalization technique described in the paper "An Analysis and Implementation of
    the FFDNet Image Denoising Method." Tassano et al. (2019).
    For each Conv layer in the model, the method replaces the matrix whose columns
    are the filters of the layer by new filters which are orthogonal to each other.
    This is achieved by setting the singular values of a SVD decomposition to 1.

    This function is to be called by the torch.nn.Module.apply() method,
    which applies svd_orthogonalization() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    print('svd_orthogonalization')
    if classname.find('Conv') != -1:
        weights = lyr.weight.data.clone()
        c_out, c_in, f1, f2 = weights.size()
        dtype = lyr.weight.data.type()

        # Reshape filters to columns
        # From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
        weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

        try:
            # SVD decomposition and orthogonalization
            mat_u, _, mat_v = torch.svd(weights)
            weights = torch.mm(mat_u, mat_v.t())

            lyr.weight.data = weights.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).type(dtype)
        except:
            pass
    else:
        pass


def get_imagenames(seq_dir, pattern=None):
    """ Get ordered list of filenames
    """
    files = []
    for typ in IMAGETYPES:
        files.extend(glob.glob(os.path.join(seq_dir, typ)))

    # filter filenames
    if not pattern is None:
        ffiltered = []
        ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
        files = ffiltered
        del ffiltered

    # sort filenames alphabetically
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
    r""" Opens a sequence of images and expands it to even sizes if necesary
    Args:
        fpath: string, path to image sequence
        gray_mode: boolean, True indicating if images is to be open are in grayscale mode
        expand_if_needed: if True, the spatial dimensions will be expanded if
            size is odd
        expand_axis0: if True, output will have a fourth dimension
        max_num_fr: maximum number of frames to load
    Returns:
        seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
            The image gets normalized gets normalized to the range [0, 1].
        expanded_h: True if original dim H was odd and image got expanded in this dimension.
        expanded_w: True if original dim W was odd and image got expanded in this dimension.
    """
    # Get ordered list of filenames
    files = get_imagenames(seq_dir)

    seq_list = []
    
    if max_num_fr< 0:
        test_files = files
    else:
        test_files = files[0:max_num_fr]
    for fpath in test_files:
        # print(fpath)
        img, expanded_h, expanded_w = open_image(fpath,\
                                                   gray_mode=gray_mode,\
                                                   expand_if_needed=expand_if_needed,\
                                                   expand_axis0=False)
        seq_list.append(img)
    # print("\tOpen sequence in folder: %s, length of frames is %d"%(os.path.basename(seq_dir), len(seq_list)))
    seq = np.stack(seq_list, axis=0)
    return seq, expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
    r""" Opens an image and expands it if necesary
    Args:
        fpath: string, path of image file
        gray_mode: boolean, True indicating if image is to be open
            in grayscale mode
        expand_if_needed: if True, the spatial dimensions will be expanded if
            size is odd
        expand_axis0: if True, output will have a fourth dimension
    Returns:
        img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
            if expand_axis0=False, the output will have a shape CxHxW.
            The image gets normalized gets normalized to the range [0, 1].
        expanded_h: True if original dim H was odd and image got expanded in this dimension.
        expanded_w: True if original dim W was odd and image got expanded in this dimension.
    """
    if not gray_mode:
        # Open image as a CxHxW torch.Tensor
        img = cv2.imread(fpath)
        # from HxWxC to CxHxW, RGB image
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

    if expand_axis0:
        img = np.expand_dims(img, 0)

    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = img.shape
    if expand_if_needed:
        if sh_im[-2]%2 == 1:
            expanded_h = True
            if expand_axis0:
                img = np.concatenate((img, \
                    img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
            else:
                img = np.concatenate((img, \
                    img[:, -1, :][:, np.newaxis, :]), axis=1)


        if sh_im[-1]%2 == 1:
            expanded_w = True
            if expand_axis0:
                img = np.concatenate((img, \
                    img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
            else:
                img = np.concatenate((img, \
                    img[:, :, -1][:, :, np.newaxis]), axis=2)
    # import pdb; pdb.set_trace()
    if normalize_data:
        img = normalize(img)
    # in davis dataset the shape is not uniform, some video (e.g. skate-jump) is larger
    # larger video may cause memory issue, we crop the videos into same size
    if img.shape[-2] == 480:
        img = img[..., 0:854]
    # C,H,W =img.shape
    # img = img[:, 0:H//32*32, 0:W//32*32]
    return img, expanded_h, expanded_w

def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data/255.)
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
