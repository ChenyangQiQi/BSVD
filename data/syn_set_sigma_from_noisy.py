#%%
# from __future__ import division
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import numpy as np
import glob
import cv2
from torch.utils.data.dataset import Dataset

if __name__ == '__main__':
    os.system("cd ..")  
#     from data_util import pack_gbrg_raw, NoiseGetter, get_frame_id
#     from data_util import list_to_torch, crop_transform
# else:
from data.data_util import pack_gbrg_raw, NoiseGetter, get_frame_id
from data.data_util import list_to_torch, crop_transform

from utils.visualizer import Visualizer

SEED = 2021
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def generate_name(number):
    name = list('000000_raw.tiff')
    num_str = str(number)
    for i in range(len(num_str)):
        name[5-i] = num_str[-(i+1)]
    name = ''.join(name)
    return name

def imread_npy_cache(gt_frame_path):
    assert gt_frame_path[-5:] == ".tiff", "image path type"
    gt_npy_path = gt_frame_path.replace(".tiff", ".npy")
    if os.path.isfile(gt_npy_path):
    # if False:
        # print("npy file hit"+gt_npy_path)
        try:
            gt_raw = np.load(gt_npy_path)
        except:
            print("cannot load", gt_npy_path)
    else:
        print("npy file miss"+gt_npy_path)
        gt_raw = cv2.imread(gt_frame_path, -1).astype(np.uint16)
        np.save(gt_npy_path, gt_raw)
        # if np.random.rand() < 0.01: 
            # print("npy file miss"+gt_npy_path)
        print("save "+gt_npy_path)
    return gt_raw

class TrainSRVDPairedDataset(Dataset):
    """Validation dataset. Loads all the images in the video folder on memory.
       Memory comsumption in all mode: 62/63 -> 65
       
    """
    def __init__(self, 
                 patch_size=128,
                 num_train_files=-1,
                 num_input_frames=5, 
                 folder = './dataset/SRVD_data/raw_clean',
                 noise_type="gaussian_sum",
                 
                 ):
        
        super().__init__()
        self.folder = folder
        self.ps = patch_size  # patch size for training
        self.num_input_frames = num_input_frames
        self.noise_type = noise_type
        
        self.noise_getter = NoiseGetter()
        margin = int(self.num_input_frames/2.0) # 2
        
        self.gt_paths1 = sorted(glob.glob(self.folder+'/MOT17-02_raw/*.tiff'))[margin:-margin]
        self.gt_paths2 = sorted(glob.glob(self.folder+'/MOT17-09_raw/*.tiff'))[margin:-margin]
        self.gt_paths3 = sorted(glob.glob(self.folder+'/MOT17-10_raw/*.tiff'))[margin:-margin]
        self.gt_paths4 = sorted(glob.glob(self.folder+'/MOT17-11_raw/*.tiff'))[margin:-margin] # data order stange len355, 0-750

        self.gt_pathsall = [self.gt_paths1, self.gt_paths2, self.gt_paths3, self.gt_paths4]
        

        self.gt_paths = []
        if num_train_files == -1: num_train_files=4

        for i in np.arange(num_train_files):
            self.gt_paths += self.gt_pathsall[i]
        self.data_cache = {}

    def __getitem__(self, index, crop_early=True):
        '''
        step1:  generate clean and noisy image pair
        step2:  generate sigma map from noisy image as net input

        '''
        gt_path = self.gt_paths[index]

        #select center frame
        frame_id = get_frame_id(gt_path)
        noisy_level = np.random.randint(0,5)
        # noisy_level = np.random.randint(1,1+1)

        input_pack_list = []
        gt_pack_list = []

        xx = None
        yy = None
        for shift in range(int(-1*(self.num_input_frames-1)/2.0),
                            int((self.num_input_frames+1)/2.0)
                            ):
            # ------------------------------------step 1 -------------------------------------------
            
            # step1:  generate clean and noisy image pair
            
            gt_frame_name = generate_name(frame_id+shift)
            gt_frame_path = list(gt_path)
            gt_frame_path[-len(os.path.basename(gt_path)):] = gt_frame_name
            gt_frame_path = ''.join(gt_frame_path)


            # gt_raw_full = cv2.imread(gt_frame_path,-1)
            # gt_raw_full = self.imread_memory_cache(gt_frame_path)
            gt_raw_full = imread_npy_cache(gt_frame_path)
            if crop_early:
                if xx is None:
                    H, W = gt_raw_full.shape
                    xx = np.random.randint(0, W - self.ps*2+1)
                    while xx%2!=0:
                        xx = np.random.randint(0, W - self.ps*2+1)
                    yy = np.random.randint(0, H - self.ps*2+1)
                    while yy%2!=0:
                        yy = np.random.randint(0, H - self.ps*2+1)
                gt_raw_full = gt_raw_full[yy:yy + self.ps*2, xx:xx + self.ps*2]

                
            gt_pack = pack_gbrg_raw(gt_raw_full)
            if self.noise_type=="gaussian_sum":
                
                noisy_raw = self.noise_getter.get_gaussian(noisy_level, gt_raw_full)
            
            
            if self.noise_type=="poisson_gaussian":
                noisy_raw = self.noise_getter.generate_noisy_raw(gt_raw_full.astype(np.float32), noisy_level)
            
            input_pack = pack_gbrg_raw(noisy_raw)
            
            # ------------------------------------step 2 -------------------------------------------
            # step2:  generate sigma map from noisy image as net input
            noise_dict = self.noise_getter.get_sigma(noisy_level, noisy_raw)
            sigma_mean = noise_dict['sigma_mean']
            sigma_4 = noise_dict['sigma_4']
            
            input_pack = np.concatenate([input_pack, sigma_mean[..., None]], axis=-1)
            input_pack = np.minimum(input_pack, 1.0)
                            
            input_pack_list.append(input_pack)
            
            # 1, h, w, 4
            gt_pack_list.append(gt_pack)
        
        input_pack_nfhwc = np.stack(input_pack_list, axis=0)
        gt_pack_nfhwc = np.stack(gt_pack_list, axis=0)
        
        sample = (gt_pack_nfhwc, input_pack_nfhwc)
        sample = list_to_torch(sample)
        if not crop_early:
            sample = crop_transform(sample, self.ps)
        return sample
    
    def __len__(self):
        # return int(len(self.gt_paths)/self.batch_size)
        return len(self.gt_paths)
    
    def imread_memory_cache(self, gt_frame_path):
        assert gt_frame_path[-5:] == ".tiff", "image path type"
        assert os.path.isfile(gt_frame_path)
        # gt_npy_path = gt_frame_path.replace(".tiff", ".npy")
        if gt_frame_path in self.data_cache:
            # print("npy file hit"+gt_frame_path)
            gt_raw = self.data_cache[gt_frame_path]
        else:
            # print("npy file miss"+gt_frame_path)
            gt_raw = cv2.imread(gt_frame_path, -1).astype(np.uint16)
            # np.save(gt_npy_path, gt_raw)
            self.data_cache[gt_frame_path] = gt_raw
            # print("save "+gt_npy_path)
        return gt_raw


#%%
if __name__ == "__main__":
    # loader_train = TrainSRVDPairedLoader(batch_size=64, num_input_frames=5, num_train_files=1)
    vis = Visualizer()
    srvd_dataset = TrainSRVDPairedDataset(
                                patch_size=128,
                                num_train_files=-1,
                                num_input_frames=5,
                                noise_type="gaussian_sum",
                                
                                )
#%%
    def get_batch():
        print(len(srvd_dataset))
        for i in np.arange(16):
            sample = srvd_dataset[i]
            print(sample[0].shape)
            print(sample[1].shape)
            break
    #%%
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp_wrap = lp(srvd_dataset.__getitem__)
    lp_wrap(0, crop_early=True)
    lp.print_stats() 
#%%
    
    (seq_gt, seq_noisy) = sample
    noise = seq_noisy[:, 0:4, :, :] - seq_gt
    noise_std = torch.std(noise[1, ...])
    estimate_std = torch.mean(seq_noisy[1, 4, :, :])
#%%

    for i in np.arange(0, 2000, 100):
        data_tuple = srvd_dataset[i]
        f, c, h, w = data_tuple[0].shape
        gt_train, imgn_train = data_tuple
        gt_train = gt_train.reshape(-1, c, h, w)
        imgn_train = imgn_train.reshape(-1, c+1, h, w)
        vis.save(imgn_train, './trash/syn_set_sigma_from_noisy/%d_noisy_deleteme.png'%i)
        vis.save(gt_train, './trash/syn_set_sigma_from_noisy/%d_gt_deleteme.png'%i)
        vis.vis_noise(imgn_train, './trash/syn_set_sigma_from_noisy/%d_noisemap_deleteme.png'%i)
        # vis.save(result, './result_deleteme.png')   
        print(i)

#%%
    # len(self)
    # 222
    from torch.utils.data import DataLoader
    # from torch.utils.data import Data
    dataloader = DataLoader(srvd_dataset, batch_size=32, 
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
                
#%%
    img_shape = gaussian_noise_sum.shape
    H = img_shape[0]
    W = img_shape[1]
    out = np.concatenate((gaussian_noise_sum[1:H:2, 0:W:2, :],
                            gaussian_noise_sum[1:H:2, 1:W:2, :],
                            gaussian_noise_sum[0:H:2, 1:W:2, :],
                            gaussian_noise_sum[0:H:2, 0:W:2, :]), axis=2)
    noise_std = np.std(out, axis=2)/(2**12-1-240.0)
    estimate_std = np.mean(sigma_mean)
    
    # vis = Visualizer()
    # for i, data_tuple in enumerate(loader_train):
    #     n, f, c, h, w = data_tuple[0].shape
    #     gt_train, imgn_train = data_tuple
    #     gt_train = gt_train.reshape(-1, c, h, w)
    #     imgn_train = imgn_train.reshape(-1, c+1, h, w)
    #     vis.save(imgn_train, './trash/1_%d_noisy_deleteme.png'%i)
    #     vis.save(gt_train, './trash/1_%d_gt_deleteme.png'%i)            
    #     # vis.save(result, './result_deleteme.png')   
    #     print(i)
    # gt_data, in_data = data_tuple
    # model_gt_data = gt_data.reshape(-1, 4, 128, 128)
    # model_in_data = in_data.reshape(-1, 5, 128, 128)
    # nf, c, h, w = model_gt_data.shape
    # for i in np.arange(nf):
    #     visualize(model_gt_data[i:i+1, ...], "./trash/image2/gt_%d.png"%i)
    #     visualize(model_in_data[i:i+1, 0:4, ...], "./trash/image2/in_%d.png"%i)
    #     print(model_in_data[i:i+1, 4, ::100, ::100])

    
# %%
