#%%
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data.dataset import Dataset
import cv2
from .data_util import pack_gbrg_raw, NoiseGetter

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types



class ToyValDataset(Dataset):
    #  CRVD scenes 7–11 for objective validation.
    # /home/chenyangqi/disk1/fast_video/data/CRVD/indoor_raw_noisy
    def __init__(self):
        super().__init__()

        self.seqs_dirs_noisy = []
        self.seqs_dirs_gt = []
        self.seqs_iso = []
        print("\n Loading Topy dataset")

    def __getitem__(self, index):
        # 3, 5, 2, 2, 
        # 1,2,3
        noisy_seq = [1,2,3]
        gt_seq = [-1,-2,-3]
        base_tensor = torch.ones(1, 5, 4, 4)
        # input_seq = [i*base_tensor for i in input_seq]
        seq_noisy = torch.cat([i*base_tensor for i in noisy_seq], dim=0)
        seq_gt = torch.cat([i*base_tensor for i in gt_seq], dim=0)
        print("seq_noisy", seq_noisy)
        print("seq_gt", seq_gt)
        return seq_noisy, seq_gt

    def __len__(self):
        return 1

class ValDataset(Dataset):
    #  CRVD scenes 7–11 for objective validation.
    # /home/chenyangqi/disk1/fast_video/data/CRVD/indoor_raw_noisy
    def __init__(self, 
                valsetdir="./dataset/CRVD",
                num_input_scenes=-1,
                num_input_frames=15,
                scene_start = 7,
                scene_end = 11,
                verbose=False
                ):
        super().__init__()

        self.seqs_dirs_noisy = []
        self.seqs_dirs_gt = []
        self.seqs_iso = []
        iso_list = [1600,3200,6400,12800,25600]
        self.iso_list   = [1600,3200,6400,12800,25600]
        self.a_list     = [3.513262,6.955588,13.486051,26.585953,52.032536]
        self.g_noise_var_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]
        self.noise_getter = NoiseGetter()
        self.verbose=verbose
        print("\n Loading validation dataset")
        for iso in iso_list:
            print('read iso={}'.format(iso))
            for scene_id in range(scene_start, scene_end+1):
                print(valsetdir+'/scene{}/ISO{}/frame*_noisy0.tiff'.format(scene_id, iso))
                # print("valsetdir+'/scene{}/ISO{}/frame*_noisy0.tiff'.format(scene_id, iso))")
                noisy_paths     = sorted(glob(valsetdir+'/indoor_raw_noisy/scene{}/ISO{}/frame*_noisy0.tiff'.format(scene_id, iso)))
                gt_paths        = sorted(glob(valsetdir+'/indoor_raw_gt/scene{}/ISO{}/frame*_clean_and_slightly_denoised.tiff'.format(scene_id, iso)))
                assert len(noisy_paths) == len(gt_paths), "noisy and gt paths num should be equal"
                # test_gt = cv2.imread(valsetdir+'indoor_raw_test_gt_slightly_denoised/scene{}/ISO{}/frame*_clean_and_slightly_denoised.tiff'.format(scene_id, iso, i),-1).astype(np.float32)
                
                # test_gt = (test_gt-240)/(2**12-1-240)
                self.seqs_dirs_noisy.append(noisy_paths)
                self.seqs_dirs_gt.append(gt_paths)
                self.seqs_iso.append(iso)
        if num_input_scenes > 0:
            self.seqs_dirs_noisy    = self.seqs_dirs_noisy[0:num_input_scenes]
            self.seqs_dirs_gt       = self.seqs_dirs_gt[0:num_input_scenes]
            self.seqs_iso           = self.seqs_iso[0:num_input_scenes]
        self.seqs_dirs = self.seqs_dirs_noisy
        print("number of validation sequence",                 len(self.seqs_dirs_noisy))
        print("number of paths in 1st validation sequence",    len(self.seqs_dirs_noisy[0]))
        self.valsetdir = valsetdir
        self.num_input_frames = num_input_frames
        # Look for subdirs with individual sequences
        # self.seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))


    def __getitem__(self, index):
        # return shape is in [0, 1]
        iso = self.seqs_iso[index]
        noisy_level = self.iso_list.index(iso)
        # a = self.a_list[noisy_level-1]
        # g_noise_var = self.g_noise_var_list[noisy_level-1]

        # 7, 4, 540, 960, 0-1
        seq_noisy = self.open_sequence(self.seqs_dirs_noisy[index], expand_if_needed=False, \
                    max_num_fr=self.num_input_frames, sigma=True, noisy_level=noisy_level)
        # sigma_sum = a*(gt_patch-240)+g_noise_var #TODO 256, 256 very large  300 - 13k, input_pack is small 0-1
        # sigma_mean =  pack_sigma(sigma_sum)
        if self.verbose:
            print(self.seqs_dirs_noisy[index][0])
        
        seq_gt = self.open_sequence(self.seqs_dirs_gt[index], expand_if_needed=False, \
                    max_num_fr=self.num_input_frames, sigma=False)
        return torch.from_numpy(seq_noisy), torch.from_numpy(seq_gt)

    def __len__(self):
        return len(self.seqs_dirs_noisy)
    
        
    def open_sequence(self, seq_dir, expand_if_needed=False, max_num_fr=100, sigma=False, noisy_level=None):
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
        files = seq_dir

        seq_list = []
        
        if max_num_fr< 0:
            test_files = files
        else:
            test_files = files[0:max_num_fr]
        for fpath in test_files:
            # print(fpath)
            img = self.open_image(fpath,\
                                                    expand_axis0=True, 
                                                    sigma=sigma, 
                                                    noisy_level=noisy_level)
            seq_list.append(img)
        # print("\tOpen sequence in folder: %s, length of frames is %d"%(os.path.basename(seq_dir), len(seq_list)))
        # 1 540, 960, 28
        seq = np.concatenate(seq_list, axis=0)
        seq = seq.transpose(0, 3, 1, 2)
        return seq

    def open_image(self, fpath, expand_axis0=True, sigma=False, noisy_level=None):
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

        raw = cv2.imread(fpath, -1)
        # 1080, 1920, 215-4k

        if expand_axis0:
            img = np.expand_dims(pack_gbrg_raw(raw), 0)
        # 1, 540, 960, 4
        # Handle odd sizes
        if sigma:
            dic = self.noise_getter.get_sigma(noisy_level, raw)
            sigma_mean = dic['sigma_mean']
            img = np.concatenate([img, sigma_mean[None, ..., None]], axis=3)
            
        return img


    
if __name__=="__main__":
    loader = ValDataset(valsetdir="/home/chenyangqi/disk1/fast_video/data/CRVD",
                                )

#%%
    i=0
    (seq_noisy, seq_gt) = loader[i]
    noise = seq_noisy[:, 0:4, :, :] - seq_gt
    noise_std = torch.std(noise[1, ...])
    estimate_std = torch.mean(seq_noisy[1, 4, :, :])
    print(seq_gt[:, :, 0, 0])
    print(seq_noisy[:, :, 0, 0])
    print(noise[:, :, 0,0])
    print(noise_std, estimate_std)


#%%
    i=0
    (seq_noisy, seq_gt) = loader[i]
    # crop a path
    start   = 32
    end     = 64
    seq_noisy   =   seq_noisy[0:1, :, start:end,    start:end]
    seq_gt      =   seq_gt[0:1, :,    start:end,    start:end]
    noise = seq_noisy[:, 0:4, :, :] - seq_gt
    noise_std = torch.std(noise[0, ...])
    estimate_std = torch.mean(seq_noisy[0, 4, :, :])
# seq_noisy[0, :, ::300, ::300]
#%%

    vis = Visualizer()
    # vis.save(seq_noisy, './trash/root_sigma/noisy_crop.png')
    # vis.save(seq_gt,    './trash/root_sigma/gt_crop.png')
#%
    # noise_map = vis.vis_noise(seq_noisy, save_path='./trash/root_sigma/noise_map_crop.png')
    from torch.utils.data import DataLoader
    import time
    st = time.time()
    loader_validation = DataLoader(loader, 
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=32,
                                    prefetch_factor=2,
                                    )
    for i, sample in enumerate(loader_validation):
        (seq_noisy, seq_gt) = sample
        seq_noisy = seq_noisy[0]
        seq_gt = seq_gt[0]
        vis.save(seq_noisy, './trash/validate_loader/%d_noisy.png'%i)
        vis.save(seq_gt,    './trash/validate_loader/%d_gt.png'%i)
        noise_map = vis.vis_noise(seq_noisy, save_path='./trash/validate_loader/%d_noise_map.png'%i)
    print('use time', time.time()-st)


#%%

# %% crop
    def crop(start, end):
        i=24
        (seq_noisy, seq_gt) = loader[i]
        # crop a path
        seq_noisy   =   seq_noisy[0:1, :, start:end,    start:end]
        seq_gt      =   seq_gt[0:1, :,    start:end,    start:end]
        noise = seq_noisy[:, 0:4, :, :] - seq_gt
        noise_std = torch.std(noise[0, ...])
        estimate_std = torch.mean(seq_noisy[0, 4, :, :])
        # seq_noisy[0, :, ::300, ::300]

        vis = Visualizer()
        vis.save(seq_noisy, './trash/root_sigma/noisy_crop%d_%d.png'%(start, end))
        vis.save(seq_gt,    './trash/root_sigma/gt_crop%d_%d.png'%(start, end))
        noise_map = vis.vis_noise(seq_noisy, save_path='./trash/root_sigma/noise_map_crop%d_%d.png'%(start, end))

    crop(33, 65)
# %%
