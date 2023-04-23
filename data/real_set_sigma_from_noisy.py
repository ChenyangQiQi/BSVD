# TODO 5:20pm clean code and generate noise from noisy image
# Following RViDeNet, we use SRVD videos plus the scenes 1–6 from CRVD for training, 
#%%

import os, time
# os.environ['CUDA_VISIBLE_DEVICES']="8"

import torch
import torch.nn as nn
import numpy as np
import glob
import cv2
from torch.utils.data.dataset import Dataset
from data.data_util import pack_gbrg_raw, NoiseGetter, get_frame_id
from data.data_util import list_to_torch, crop_transform
from tqdm import tqdm
import copy

def get_noisy_level(gt_path):
    gt_path_split_list = gt_path.split('/')
    assert "ISO" in gt_path_split_list[-2]
    iso_list   = [1600,3200,6400,12800,25600]
    noisy_level = iso_list.index(int(gt_path_split_list[-2][3:]))
    return noisy_level

def generate_path(gt_path, shift, noisy_frame_index_for_other):
    shifted_name = os.path.basename(gt_path)
    frame_id = get_frame_id(shifted_name)
    shifted_name = list(shifted_name)
    shifted_name[5] = str(frame_id+shift)
    shifted_name = ''.join(shifted_name)
    gt_frame_path = os.path.join(os.path.dirname(gt_path), shifted_name)
    
    noisy_raw_path = gt_path.replace('indoor_raw_gt', 'indoor_raw_noisy')
                            # .replace('clean_and_slightly_denoised', 'noisy%d'%noisy_frame_index_for_other)
    noisy_base_name = 'frame{}_noisy{}.tiff'.format(frame_id+shift, noisy_frame_index_for_other)
    noisy_raw_path = noisy_raw_path.replace(os.path.basename(noisy_raw_path), noisy_base_name)
    return gt_frame_path, noisy_raw_path

def test_generate_path():
    gt_path = '/home/chenyangqi/disk1/fast_video/data/CRVD/indoor_raw_gt/scene6/ISO3200/frame5_clean_and_slightly_denoised.tiff'
    # noisy_raw = cv2.imread(self.folder+'/indoor_raw_noisy/scene{}/ISO{}/frame{}_noisy{}.tiff'. \
    #                         format(scene_ind, self.iso_list[noisy_level-1], frame_ind+shift, noisy_frame_index_for_other),-1)

    # frame_id = get_frame_id('/home/chenyangqi/disk1/fast_video/data/CRVD/indoor_raw_gt/scene6/ISO3200/frame5_clean_and_slightly_denoised.tiff')
    # gt_path = '/home/chenyangqi/disk1/fast_video/data/CRVD/indoor_raw_gt/scene6/ISO3200/frame5_clean_and_slightly_denoised.tiff'
    gt_frame_path, noisy_raw_path = generate_path(gt_path, -1, 6)

def imread_npy_cache(gt_frame_path):
    assert gt_frame_path[-5:] == ".tiff", "image path type"
    gt_npy_path = gt_frame_path.replace(".tiff", ".npy")
    if os.path.isfile(gt_npy_path):
        # print("npy file hit"+gt_npy_path)
        gt_raw = np.load(gt_npy_path)
    else:
        print("npy file miss"+gt_npy_path)
        gt_raw = cv2.imread(gt_frame_path, -1).astype(np.uint16)
        np.save(gt_npy_path, gt_raw)
        print("save "+gt_npy_path)
    return gt_raw

class TrainCRVDPairedDataset(Dataset):
    # Number of possible netin 6*5*3 = 90
    def __init__(self, 
                 patch_size=128,
                 num_input_scenes=6,
                 num_input_frames=5,
                 folder = "./dataset/CRVD",
                 debug=False,
                 crop=True,
                 fix_noisy_index= False,
                 verbose=False,
                 data_cache=None
                 ):
        super().__init__()
        self.debug = debug
        self.folder = folder
        self.ps = patch_size
        self.num_input_frames = num_input_frames
        self.fix_noisy_index = fix_noisy_index
        self.noise_getter = NoiseGetter()
        self.crop = crop
        self.verbose=verbose
        margin = int(self.num_input_frames/2.0) # 2
        
        self.gt_paths = []
        self.video_list = sorted(glob.glob(self.folder+'/indoor_raw_gt/scene[1-6]/ISO*'))
        if num_input_scenes>0:
            self.video_list = sorted(glob.glob(self.folder+'/indoor_raw_gt/scene[1-%d]/ISO*'%num_input_scenes))
            
        for i in np.arange(len(self.video_list)):
            self.gt_paths += sorted(glob.glob(self.video_list[i]+'/frame*_clean_and_slightly_denoised.tiff'))[margin:-margin]
        print("Initialize data cache")
        self.data_cache = {}
        # for index in tqdm(np.arange(len(self))):
        #     gt_path = self.gt_paths[index]

        #     for shift in range(int(-1*(self.num_input_frames-1)/2.0),
        #                         int((self.num_input_frames+1)/2.0)
        #                         ):
        #         # noisy_frame_index_for_other = np.random.randint(0,9+1)
        #         # if self.fix_noisy_index:
        #             # noisy_frame_index_for_other = 0
        #         gt_frame_path, noisy_raw_path = generate_path(gt_path, shift, 0)
        #         gt_raw = self.imread_memory_cache(gt_frame_path)
                # for noisy_frame_index_for_other in np.arange(0, 10):
                #     gt_frame_path, noisy_raw_path = generate_path(gt_path, shift, noisy_frame_index_for_other)
                #     noisy_raw = self.imread_memory_cache(noisy_raw_path)
                # noisy_raw = imread_npy_cache(noisy_raw_path)
    def __getitem__(self, index):

        gt_path = self.gt_paths[index]
        noisy_level = get_noisy_level(gt_path)
        if self.verbose:
            print(gt_path)
        input_pack_list = []
        gt_raw_pack_list = []

        xx = None
        yy = None
        for shift in range(int(-1*(self.num_input_frames-1)/2.0),
                            int((self.num_input_frames+1)/2.0)
                            ):
            noisy_frame_index_for_other = np.random.randint(0,9+1)
            if self.fix_noisy_index:
                noisy_frame_index_for_other = 0
            gt_frame_path, noisy_raw_path = generate_path(gt_path, shift, noisy_frame_index_for_other)
            # print(noisy_raw_path)
            # gt_npy_path = gt_frame_path.replace(".tiff", ".npy")
            # if os.path.isfile(gt_npy_path):
            #     gt_raw = np.load(gt_npy_path)
            # else:
            #     gt_raw = cv2.imread(gt_frame_path, -1).astype(np.uint16)
            #     np.save(gt_npy_path, gt_raw)
            # gt_raw = cv2.imread(gt_frame_path, -1)
            gt_raw = imread_npy_cache(gt_frame_path)
            # gt_raw = self.imread_memory_cache(gt_frame_path, warn=True)
            if self.crop:
                if xx is None:
                    H, W = gt_raw.shape
                    xx = np.random.randint(0, W - self.ps*2+1)
                    while xx%2!=0:
                        xx = np.random.randint(0, W - self.ps*2+1)
                    yy = np.random.randint(0, H - self.ps*2+1)
                    while yy%2!=0:
                        yy = np.random.randint(0, H - self.ps*2+1)
                gt_raw = gt_raw[yy:yy + self.ps*2, xx:xx + self.ps*2]
            gt_raw_pack = pack_gbrg_raw(gt_raw)

            # noisy_raw = cv2.imread(noisy_raw_path, -1)
            noisy_raw = imread_npy_cache(noisy_raw_path)
            # noisy_raw = self.imread_memory_cache(noisy_raw_path)
            if self.crop:
                noisy_raw = noisy_raw[yy:yy + self.ps*2, xx:xx + self.ps*2]
                
            # noisy_raw_full = noisy_raw

            # noisy_patch = noisy_raw_full
            input_pack = pack_gbrg_raw(noisy_raw)
            # from line_profiler import LineProfiler
            # lp = LineProfiler()
            # lp_wrap = lp(self.noise_getter.get_sigma)
            # lp_wrap(noisy_level, noisy_raw)
            # lp.print_stats() 
            # import pdb; pdb.set_trace()
            noise_dict = self.noise_getter.get_sigma(noisy_level, noisy_raw)
            sigma_mean = noise_dict['sigma_mean']
            input_pack = np.concatenate([input_pack, sigma_mean[..., None]], axis=-1)
            
            input_pack_list.append(input_pack)
            gt_raw_pack_list.append(gt_raw_pack)
            if self.debug and shift == 2:
                a, g_noise_var = noise_dict['a'], noise_dict['g_noise_var']
                variance, sigma_sum, sigma_mean = noise_dict['variance'], noise_dict['sigma_sum'], noise_dict['sigma_mean']
                print('print from train_real_set')
                print('get shift %d gt path \t %s'%(shift, gt_frame_path))
                print('get shift %d noisy path \t %s'%(shift, noisy_raw_path))
                print('noisy_level',noisy_level)
                print('a',a)
                print('g_noise_var',g_noise_var)
                print('gt_raw',gt_raw[0:2, 0:2])
                print('noisy_raw',noisy_raw[0:2, 0:2])
                print('variance',variance[0:2, 0:2])
                print(sigma_sum[0:2, 0:2])
                print(sigma_mean[0:2, 0:2])

        input_pack_nfhwc = np.stack(input_pack_list, axis=0)
        gt_pack_nfhwc = np.stack(gt_raw_pack_list, axis=0)
        sample = (gt_pack_nfhwc, input_pack_nfhwc)

        # sample = self.transform(sample)
        sample = list_to_torch(sample)
        # import pdb; pdb.set_trace()
        # print(self.ps)
        # if not crop_early: the below line is too inefficient
        # sample = crop_transform(sample, self.ps)
        return sample
    
    def __len__(self):
        return len(self.gt_paths)
    
    def imread_memory_cache(self, gt_frame_path, warn=False):
        assert gt_frame_path[-5:] == ".tiff", "image path type"
        assert os.path.isfile(gt_frame_path)
        # gt_npy_path = gt_frame_path.replace(".tiff", ".npy")
        if gt_frame_path in self.data_cache:
            # print("npy file hit"+gt_frame_path)
            gt_raw = self.data_cache[gt_frame_path]
        else:
            if warn: print("npy file miss"+gt_frame_path)
            gt_raw = cv2.imread(gt_frame_path, -1).astype(np.uint16)
            # np.save(gt_npy_path, gt_raw)
            self.data_cache[gt_frame_path] = copy.deepcopy(gt_raw)
            if warn: 
                print("save "+gt_frame_path)
                print("number of sample in data_cache in loader imread_memory_cache",len(self.data_cache))
        
        return gt_raw



#%%
if __name__=="__main__":
    crvd_dataset = TrainCRVDPairedDataset(
                            patch_size=256,
                            num_input_frames=5,
                            debug=True
                            )
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrap = lp(crvd_dataset.__getitem__)
    # lp_wrap(0)
    # lp.print_stats() 
    _ = crvd_dataset[0]

#%%
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp_wrap = lp(crvd_dataset.__getitem__)
    lp_wrap(0, crop_early=False)
    lp.print_stats() 
#%%

    #%%
    i=0
    (seq_gt, seq_noisy) = crvd_dataset[i]
    #%%
    noise = seq_noisy[:, 0:4, :, :] - seq_gt
    noise_std = torch.std(noise[1, ...])
    estimate_std = torch.mean(seq_noisy[1, 4, :, :])
    print(noise_std, estimate_std)
    #%%
    print(len(crvd_dataset))
    for i in np.arange(start=0, stop=90, step=10):
        # print(i)
        sample = crvd_dataset[i]
        print(sample[0].shape)
        print(sample[1].shape)
        # break

    #%%
    from utils.visualizer import Visualizer
    # import utils

    vis = Visualizer()
    for i in np.arange(0, 90, 10):
        data_tuple = crvd_dataset[i]
        f, c, h, w = data_tuple[0].shape
        gt_train, imgn_train = data_tuple
        gt_train = gt_train.reshape(-1, c, h, w)
        imgn_train = imgn_train.reshape(-1, c+1, h, w)
        vis.save(imgn_train, './trash/sigma_from_noisy/%d_noisy_deleteme.png'%i)
        vis.save(gt_train, './trash/sigma_from_noisy/%d_gt_deleteme.png'%i)            
        # print(imgn_train[0, :, ::50, ::50])
        # vis.save(result, './result_deleteme.png')   
        print(i)
        # break
    # break
#%%
    from torch.utils.data import DataLoader
    # from torch.utils.data import Data
    dataloader = DataLoader(crvd_dataset, batch_size=16, 
                              shuffle=True, num_workers=8, 
                              prefetch_factor=20)
    import time
    st = time.time()
    for i_batch, sample_batched in enumerate(dataloader):
        # worker_info = torch.utils.data.get_worker_info()
        # print(worker_info)
        gt_img      = sample_batched[0].cuda()
        input_img   = sample_batched[1].cuda()
        print(i_batch, sample_batched[0].size(),
            sample_batched[1].size())
        if i_batch == 64: break
    print('use time %f'%(time.time()-st))
#%%
    n, f, c, h, w = sample_batched[0].size()
    for i in np.arange(n):
        for j in np.arange(f):
            print('./trash/dataset_batch/%d_%d_noisy.png'%(i, j))
            vis.save(sample_batched[0][i, j:j+1, ...], './trash/dataset_batch/%d_%d_clean.png'%(i, j))
            vis.save(sample_batched[1][i, j:j+1, ...], './trash/dataset_batch/%d_%d_noisy.png'%(i, j))
            

