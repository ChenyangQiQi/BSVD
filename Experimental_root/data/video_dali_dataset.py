'''Implements a sequence dataloader using NVIDIA's DALI library.

The dataloader is based on the VideoReader DALI's module, which is a 'GPU' operator that loads
and decodes H264 video codec with FFmpeg.

Based on
https://github.com/NVIDIA/DALI/blob/master/docs/examples/video/superres_pytorch/dataloading/dataloaders.py
'''
import os
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import glob
import torch
from torch.utils.data.dataset import Dataset
from Experimental_root.data.utils_common import open_sequence, get_imagenames
from Experimental_root.data.utils_common import normalize_augment
from basicsr.utils.registry import DATASET_REGISTRY
import os

class VideoReaderPipeline(Pipeline):
    ''' Pipeline for reading H264 videos based on NVIDIA DALI.
    Returns a batch of sequences of `sequence_length` frames of shape [N, F, C, H, W]
    (N being the batch size and F the number of frames). Frames are RGB uint8.
    Args:
        batch_size: (int)
                Size of the batches
        sequence_length: (int)
                Frames to load per sequence.
        num_threads: (int)
                Number of threads.
        device_id: (int)
                GPU device ID where to load the sequences.
        files: (str or list of str)
                File names of the video files to load.
        crop_size: (int)
                Size of the crops. The crops are in the same location in all frames in the sequence
        random_shuffle: (bool, optional, default=True)
                Whether to randomly shuffle data.
        step: (int, optional, default=-1)
                Frame interval between each sequence (if `step` < 0, `step` is set to `sequence_length`).
    '''
    def __init__(self, batch_size, sequence_length, num_threads, device_id, files, \
                 crop_size, random_shuffle=True, step=-1, prefetch_size = 16):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12,
                                                  prefetch_queue_depth =prefetch_size,
                                                  py_num_workers=32)
        #Define VideoReader
        self.reader = ops.VideoReader(device="gpu", \
                                        filenames=files, \
                                        sequence_length=sequence_length, \
                                        normalized=False, \
                                        random_shuffle=random_shuffle, \
                                        image_type=types.RGB, \
                                        dtype=types.UINT8, \
                                        step=step, \
                                        initial_fill=prefetch_size)

        # Define crop and permute operations to apply to every sequence
        self.crop = ops.CropMirrorNormalize(device="gpu",
                                      crop=crop_size,
                                      output_layout='FCHW')
        self.uniform = ops.random.Uniform(range=(0.0, 1.0))  # used for random crop
# 		self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])

    def define_graph(self):
        '''Definition of the graph--events that will take place at every sampling of the dataloader.
        The random crop and permute operations will be applied to the sampled sequence.
        '''
        input = self.reader(name="Reader")
        cropped = self.crop(input, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        return cropped

@DATASET_REGISTRY.register()
class train_dali_loader():
    '''Sequence dataloader.
    Args:
        batch_size: (int)
            Size of the batches
        file_root: (str)
            Path to directory with video sequences
        sequence_length: (int)
            Frames to load per sequence
        crop_size: (int)
            Size of the crops. The crops are in the same location in all frames in the sequence
        epoch_size: (int, optional, default=-1)
            Size of the epoch. If epoch_size <= 0, epoch_size will default to the size of VideoReaderPipeline
        random_shuffle (bool, optional, default=True)
            Whether to randomly shuffle data.
        temp_stride: (int, optional, default=-1)
            Frame interval between each sequence
            (if `temp_stride` < 0, `temp_stride` is set to `sequence_length`).
    '''
    # def __init__(self, batch_size, file_root, sequence_length, \
    #              crop_size, epoch_size=-1, random_shuffle=True, temp_stride=-1):
    def __init__(self, opt):
        # Builds list of sequence filenames
        # batch_size
        # file_root
        # sequence_length,
        # crop_size
        # epoch_size=-1
        # random_shuffle=True
        # temp_stride=
        self.opt = opt
        batch_size=     opt['batch_size_per_gpu']
        file_root=      opt['trainset_dir']
        sequence_length=opt['temp_patch_size']
        crop_size=      opt['patch_size']
        epoch_size=     opt['max_number_patches']
        # blind =         opt['blind']
        random_shuffle= opt.get('use_shuffle', True)
        prefetch_size= opt.get('prefetch_size', 16)
        if 'noise_shape' not in self.opt:
            self.opt['noise_shape'] = 'NF'
        if self.opt['noise_shape'] == 'N':
            pass
        temp_stride=-1
        
        container_files = os.listdir(file_root)
        container_files = [file_root + '/' + f for f in container_files]
        # Define and build pipeline
        self.pipeline = VideoReaderPipeline(batch_size=batch_size, \
                                            sequence_length=sequence_length, \
                                            num_threads=32, \
                                            device_id=0, \
                                            files=container_files, \
                                            crop_size=crop_size, \
                                            random_shuffle=random_shuffle,\
                                            step=temp_stride,
                                            prefetch_size=prefetch_size)
        self.pipeline.build()

        # Define size of epoch
        if epoch_size <= 0:
            self.epoch_size = self.pipeline.epoch_size("Reader")
        else:
            self.epoch_size = epoch_size
        self.dali_iterator = pytorch.DALIGenericIterator(pipelines=self.pipeline, \
                                                        output_map=["data"], \
                                                        size=self.epoch_size, \
                                                        auto_reset=True)

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        return self
        # return self.dali_iterator.__iter__() # [0]['data']
        # put training time data augmentation and preprocessing here
        # gt = self.dali_iterator.__iter__()[0]['data']
        # return gt
    def __next__(self):
        # TODO: validate it can run over whole dataset
        next_item =  self.dali_iterator.__next__() # [0]['data']
        data = next_item[0]['data'] # 
        img_train, gt_train = normalize_augment(data)
        N, F, C, H, W = img_train.size()
        # F = int(FC / 3)
        # img_train = img_train.view(-1, 3, H, W)
        # gt_train = gt_train.view(-1, 3, H, W)
        # FIXME different from fastdvd the noise can be assumed to be identical
        # https://github.com/m-tassano/fastdvdnet/blob/master/train_fastdvdnet.py
        if self.opt['noise_shape'] == 'NF':
            stdn = torch.empty((N, F, 1, 1, 1)).cuda().uniform_(self.opt['noise_ival'][0]/255.0, to=self.opt['noise_ival'][1]/255.0)
        elif self.opt['noise_shape'] == 'N':
            stdn = torch.empty((N, 1, 1, 1, 1)).cuda().uniform_(self.opt['noise_ival'][0]/255.0, to=self.opt['noise_ival'][1]/255.0)
        # stdn = torch.empty((N, 1, 1, 1, 1)).cuda().uniform_(self.opt['noise_ival'][0]/255.0, to=self.opt['noise_ival'][1]/255.0)
        # draw noise samples from std dev tensor
        noise = torch.zeros_like(img_train)
        noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

        # define noisy input
        imgn_train = img_train + noise
        noise_map = stdn.expand((N, F, 1, H, W))

        gt_train = gt_train.cuda(non_blocking=True)
        imgn_train = imgn_train.cuda(non_blocking=True)
        noise = noise.cuda(non_blocking=True)
        noise_map = stdn.expand((N, F, 1, H, W)).cuda(non_blocking=True) # one channel per image

        # Evaluate model and optimize it
        # imgn_train = torch.cat((imgn_train, noise_map), dim=1) #(N*F, 4, H, W)
        return_dict = {}
        return_dict['gt'] = gt_train
        return_dict['lq'] = imgn_train
        return_dict['noise_map'] = noise_map

        if self.opt.get('blind', False):
            return_dict.pop('noise_map')
        return return_dict
        

# NUMFRXSEQ_VAL = 85	# number of frames of each sequence to include in validation dataset
# VALSEQPATT = '*' # pattern for name of validation sequence

@DATASET_REGISTRY.register()
class ValFolderDataset(Dataset):
    """Validation dataset. Loads all the images in the video folder on memory.
       output range [0,1]
       
    """
    def __init__(self, opt):
        super(ValFolderDataset, self).__init__()
        self.opt = opt
        self.valsetdir=opt['valsetdir']
        self.gray_mode=opt.get('gray_mode', False)
        self.num_input_frames=opt['num_validation_frames']
        self.valnoisestd = opt['valnoisestd']
        self.scene_name = opt.get('scene_name', None)
        # Look for subdirs with individual sequences
        self.seqs_dirs = sorted([pth for pth in glob.glob(os.path.join(self.valsetdir, '*'))
                                 if os.path.isdir(pth)])
        self.base_folder = sorted([os.path.basename(pth) for pth in glob.glob(os.path.join(self.valsetdir, '*'))
                                 if os.path.isdir(pth)])
        if self.scene_name is not None:
            self.seqs_dirs = [d for d in self.seqs_dirs if self.scene_name in d]
            self.base_folder = [d for d in self.base_folder if self.scene_name in d]
        self.num_frames  = [ min(len(get_imagenames(seqs_dir)), self.num_input_frames) for seqs_dir in self.seqs_dirs]
    def __getitem__(self, index):
        # return shape is in [0, 1]
        seq, _, _ = open_sequence(self.seqs_dirs[index], self.gray_mode, expand_if_needed=False, \
                    max_num_fr=self.num_input_frames)
        gt = torch.from_numpy(seq)[None, ...]
        N, F, C, H, W = gt.size()
        
        
        # TODO gaussian noise generator can be merged with DAVIS daliloader
        noise = torch.FloatTensor(gt.size()).normal_(mean=0, std=self.opt['valnoisestd']/255.0)
        seqn_val = gt + noise
        seqn_val = seqn_val.cuda()
        sigma_noise = torch.cuda.FloatTensor([self.opt['valnoisestd']/255.0])
        noise_map = sigma_noise.expand((N, F, 1, H, W)).cuda()

        return_dict =  {
            'gt': gt, # shape 1, N, C, H, W
            'lq': seqn_val, # shape 1, N, C, H, W
            'noise_map': noise_map, # shape 1, N, 1, H, W
            'folder': self.base_folder[index],
            "index": index
        }
        if self.opt.get('blind', False):
            return_dict.pop('noise_map')
        return return_dict

    def __len__(self):
        return len(self.seqs_dirs)
