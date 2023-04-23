#%%
'''Implements a sequence dataloader using NVIDIA's DALI library.

The dataloader is based on the VideoReader DALI's module, which is a 'GPU' operator that loads
and decodes H264 video codec with FFmpeg.

Based on
https://github.com/NVIDIA/DALI/blob/master/docs/examples/video/superres_pytorch/dataloading/dataloaders.py
'''
import os
# os.environ['CUDA_VISIBLE_DEVICES']='8'
import numpy as np
import subprocess
# from nvidia.dali.pipeline import Pipeline
# from nvidia.dali.plugin import pytorch
# import nvidia.dali.ops as ops
# import nvidia.dali.types as types

import glob
import torch
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
from utils.utils import open_sequence
from utils.visualizer import Visualizer
from tqdm import tqdm

NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

# from train_finetune_loader import TrainCRVDPairedLoader
# from train_pretrain_loader import TrainSRVDPairedLoader

# from train_finetune_loader_multithreads import TrainCRVDPairedDataset
# from train_pretrain_loader_multithreads import TrainSRVDPairedDataset
# from train_finetune_loader_0715 import TrainCRVDPairedDataset as TrainCRVDPairedDataset_0715
# from train_pretrain_loader_0715 import TrainSRVDPairedDataset as TrainSRVDPairedDataset_0715
# from .train_finetune_loader import TrainCRVDPairedDataset
# from .train_pretrain_loader import TrainSRVDPairedDataset
SEED = 2021
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



class TupleTensorRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p =0.5) -> None:
        super().__init__()
        self.p = p
    def forward(self, sample_tuple):
        if torch.rand(1) < self.p:
            gt_pack_nfhwc, input_pack_nfhwc = sample_tuple
            return F.hflip(gt_pack_nfhwc), F.hflip(input_pack_nfhwc)
        else:
            return sample_tuple
        
# class RViDenetDataset(Dataset):
#     def __init__(self,patch_size=128,
#                  num_input_frames=5,
#                  num_train_files=-1,
#                  transform= None
#                 )-> None:
#         super().__init__()

#         self.srvd_dataset = TrainSRVDPairedDataset(
#                 patch_size=patch_size, 
#                 num_input_frames=num_input_frames, 
#                 noise_type="gaussian_sum",
#                 num_train_files=num_train_files,
#                 transform=transform
#                 )
#         self.crvd_dataset = TrainCRVDPairedDataset(
#                 patch_size=patch_size, 
#                 num_input_frames=num_input_frames,
#                 transform=transform
#                 )
#     def __getitem__(self, index):
#         if index < len(self.srvd_dataset):
#             return self.srvd_dataset.__getitem__(index)
#         else:
#             return self.crvd_dataset. \
#                     __getitem__((index - len(self.srvd_dataset))%len(self.crvd_dataset))
#     # 10, 10  
#     def __len__(self):
#         # return int(len(self.gt_paths)/self.batch_size)
#         #      2500 / 100   20-> 160
#         return len(self.srvd_dataset)+ len(self.crvd_dataset)

# transforms.RandomHorizontalFlip()
# 0715 is input concat with sigma not variance
# class RViDenetDatasetAB(Dataset):
#     def __init__(self,patch_size=128,
#                  num_input_frames=5,
#                  num_train_files=-1,
#                  A = 1,
#                  B = 1,
#                 augmentation = None,
#                 )-> None:
#         super().__init__()
#         self.A = A; self.B = B
#         if augmentation == None:
#             transform = ToTensor()
#             print('use no augmentation')
#         elif augmentation == 'flip':
#             transform= transforms.Compose([
#                     ToTensor(),
#                     TupleTensorRandomHorizontalFlip()
#                     ])
#         self.srvd_dataset = TrainSRVDPairedDataset(
#                 patch_size=patch_size, 
#                 num_input_frames=num_input_frames, 
#                 noise_type="gaussian_sum",
#                 num_train_files=num_train_files,
#                 transform=transform
#                 )
        
#         self.crvd_dataset = TrainCRVDPairedDataset(
#                 patch_size=patch_size, 
#                 num_input_frames=num_input_frames,
#                 transform=transform
#                 )
#     def __getitem__(self, index):
#         if index < self.A*len(self.srvd_dataset):
#             return self.srvd_dataset.__getitem__(index%(len(self.srvd_dataset)))
#         else:
#             return self.crvd_dataset. \
#                     __getitem__((index - self.A*len(self.srvd_dataset))%len(self.crvd_dataset))
#     # 10, 10  
#     def __len__(self):
#         # return int(len(self.gt_paths)/self.batch_size)
#         #      2500 / 100   20-> 160
#         return self.A*len(self.srvd_dataset)+ self.B*len(self.crvd_dataset)

class RViDenetDatasetAB_sigma_from_noisy(Dataset):
    def __init__(self,patch_size=128,
                 num_input_frames=5,
                 num_train_files= -1,
                 A = 1,
                 B = 1,
                 fast = False,
                 profile=False,
                 crop=True,
                 fix_noisy_index=False,
                 verbose=False,
                 data_cache = None
                )-> None:
        
        super().__init__()
        self.A = A; self.B = B
        self.profile=profile
        from data.syn_set_sigma_from_noisy  import TrainSRVDPairedDataset
        from data.real_set_sigma_from_noisy import TrainCRVDPairedDataset
        if fast:
            num_train_files = 1
            
        self.srvd_dataset = TrainSRVDPairedDataset(
                patch_size=patch_size, 
                num_input_frames=num_input_frames, 
                noise_type="gaussian_sum",
                num_train_files=num_train_files,
                # data_cache=data_cache
                )
        print("len(self.srvd_dataset) ", len(self.srvd_dataset))
        for i in np.arange(10)*(len(self.srvd_dataset)//10):
            print(self.srvd_dataset.gt_paths[i])
        self.crvd_dataset = TrainCRVDPairedDataset(
                patch_size=patch_size, 
                num_input_scenes=num_train_files,
                num_input_frames=num_input_frames,
                crop=crop,
                fix_noisy_index=fix_noisy_index,
                verbose=verbose,
                data_cache=data_cache
                )
        print("len(self.crvd_dataset) ", len(self.crvd_dataset))
        for i in np.arange(10)*(len(self.crvd_dataset)//10):
            print(self.crvd_dataset.gt_paths[i])
    def __getitem__(self, index, profile=False):
        if index < int(self.A*len(self.srvd_dataset)):
            if self.profile:
                print('profile the datalaoder getitem')
                from line_profiler import LineProfiler
                lp = LineProfiler()
                lp_wrap = lp(self.srvd_dataset.__getitem__)
                lp_wrap(index%len(self.srvd_dataset))
                lp.print_stats() 
                exit
            return self.srvd_dataset.__getitem__( index% len(self.srvd_dataset) ) 
        else:
            if self.profile:
                print('profile the datalaoder getitem')
                from line_profiler import LineProfiler
                lp = LineProfiler()
                lp_wrap = lp(self.crvd_dataset.__getitem__)
                lp_wrap((index - int( self.A*len(self.srvd_dataset) ))%len(self.crvd_dataset)) 
                lp.print_stats() 
                exit
            return self.crvd_dataset. \
                    __getitem__((index - int( self.A*len(self.srvd_dataset) ) )%len(self.crvd_dataset))
    # 10, 10  
    def __len__(self):
        # import pdb; pdb.set_trace()
        # return int(len(self.gt_paths)/self.batch_size)
        #      2500 / 100   20-> 160
        return int(self.A*len(self.srvd_dataset))+ int(self.B*len(self.crvd_dataset))


class ValDataset(Dataset):
    """Validation dataset. Loads all the images in the video folder on memory.
       Memory comsumption in all mode: 62/63 -> 65
       
    """
    def __init__(self, valsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
        super(ValDataset, self).__init__()
        self.gray_mode = gray_mode
        self.valsetdir = valsetdir
        self.num_input_frames = num_input_frames
        # Look for subdirs with individual sequences
        self.seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))


    def __getitem__(self, index):
        # return shape is in [0, 1]
        seq, _, _ = open_sequence(self.seqs_dirs[index], self.gray_mode, expand_if_needed=False, \
                    max_num_fr=self.num_input_frames)
        return torch.from_numpy(seq)

    def __len__(self):
        return len(self.seqs_dirs)

#%% 
def ValDataset_test():
    valsetdir = '../data/fastdvd_data/test_sequences'
    dataset_val = ValDataset(valsetdir=valsetdir, gray_mode=False)
    # list of length 4, element of shape (15, 3, 540, 960)

    #%%
    loader_train = train_dali_loader(batch_size=64,\
                                file_root='../data/fastdvd_data/training/mp4',\
                                crop_size=96,\
                                epoch_size=256000,\
                                random_shuffle=False,\
                                sequence_length=5,
                                temp_stride= 3)

    #%%
    for i, data in enumerate(loader_train, 0):
        out = data[0]['data']
        print(out.shape)
        # [64, 5, 3, 96, 96]
        if i == 2: break
    # %%
    valsetdir = '../data/fastdvd_data/test_sequences'
    dataset_val = ValDataset(valsetdir=valsetdir, gray_mode=False)

#%%
def learn_torch_DataLoader():
    from torch.utils.data import DataLoader
    # class ValDataset(Dataset):
    #     """Validation dataset. Loads all the images in the dataset folder on memory.
    #     """
    #     def __init__(self, valsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):

    path = '/home/chenyangqi/disk1/fast_video/data/DAVIS-2017-test-dev-480p/JPEGImages/480p'
    val_dataset = ValDataset(valsetdir=path, gray_mode=False, num_input_frames=10)
    # def test_dataloader():
    train_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
        
        
    for train_batch in train_dataloader:
        print(train_batch.shape)

#%%    
if __name__ == "__main__":
    
    

    # vis = Visualizer()
    # tuple_transforms = transforms.Compose([
    #                                     ToTensor(),
    #                                     TupleTensorRandomHorizontalFlip()
    #                                     ])
    srvd_dataset = RViDenetDatasetAB_sigma_from_noisy(
                                patch_size=256,
                                num_train_files=-1,
                                num_input_frames=5,
                                A=1, B=1
                                )
                                
    #%%
    print(len(srvd_dataset))
    for i in np.arange(10):
        sample = srvd_dataset[i]
        print(sample[0].shape)
        print(sample[1].shape)
        break

#%%

    for i in np.arange(0, int(len(srvd_dataset)), int(len(srvd_dataset)/20)):
        data_tuple = srvd_dataset[i]
        f, c, h, w = data_tuple[0].shape
        gt_train, imgn_train = data_tuple
        gt_train = gt_train.reshape(-1, c, h, w)
        imgn_train = imgn_train.reshape(-1, c+1, h, w)
        vis.save(imgn_train, './trash/dataset_sigma_from_noisy/%d_noisy_deleteme.png'%i)
        vis.save(gt_train, './trash/dataset_sigma_from_noisy/%d_gt_deleteme.png'%i)            
        vis.vis_noise(imgn_train, './trash/dataset_sigma_from_noisy/%d_noisemap_deleteme.png'%i)
        # vis.(gt_train, './trash/dataset_all_AB10/%d_noise_map_deleteme.png'%i)            
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
        if i_batch == 16: break
    print('use time %f'%(time.time()-st))
#%%
    n, f, c, h, w = sample_batched[0].size()
    for i in np.arange(n):
        for j in np.arange(f):
            print('./trash/dataset_batch/%d_%d_noisy.png'%(i, j))
            vis.save(sample_batched[0][i, j:j+1, ...], './trash/dataset_batch/%d_%d_clean.png'%(i, j))
            vis.vis_noise(sample_batched[1][i, j:j+1, ...], './trash/dataset_batch/%d_%d_noise_map.png'%(i, j))
            vis.save(sample_batched[1][i, j:j+1, ...], './trash/dataset_batch/%d_%d_noisy.png'%(i, j))
                    
# %%
