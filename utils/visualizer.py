import torch
import numpy as np
import os
import cv2

class Visualizer():
    """[summary]

    Args:
        input_data (torch.tensor): n, 4, h, w 0-1
        save_path ([type]): '../results/0705/gt_sRGB_deleteme.png'
    """
    def __init__(self) -> None:
        super().__init__()
        # import pdb; pdb.set_trace()
        self.isp = torch.load('isp/ISP_CNN.pth').cuda()
    
    def vis_noise(self, input_data, save_path=None):
        assert len(input_data.shape)==4, "shape of tensor should be 4d"
        assert input_data.shape[1]==5, "5 channel for noise data"
    
        noise_map = np.uint8(255.0*(input_data[0, 4:5, ...])**0.5).transpose(1,2,0)
        # print(noise_map.shape)
        if save_path is not None:
            cv2.imwrite(save_path, noise_map)
        return noise_map
    
    def torch2npvis(self, input_data):
        assert len(input_data.shape)==4, "shape of tensor should be 4d"
        # 7,5,540, 960
        input_data = input_data[0:1, 0:4, ...]
        input_data = input_data.cuda()
        gt_srgb_frame = self.postprocess(self.isp(input_data))[0]
        return np.uint8(gt_srgb_frame*255)
        # cv2.imwrite(save_path, np.uint8(gt_srgb_frame*255))
        
    def save(self, input_data, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image = self.torch2npvis(input_data)
        cv2.imwrite(save_path, image)

    def postprocess(self, output):
        output = output.cpu()
        output = output.detach().numpy().astype(np.float32)
        output = np.transpose(output, (0, 2, 3, 1))
        output = np.clip(output,0,1)
        return output