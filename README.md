# Fast Video Processing on CRVD dataset

# Dependencies

## Environment

Create your environment
```bash
conda env create -f environment.yaml -n py37torch170
# for 3090GPU
# conda env create -f environment_3090.yaml -n py37torch170
# pip install -r requirements.txt
```
<!-- Note: this project needs the [NVIDIA DALI](https://github.com/NVIDIA/DALI) package for training. The tested version of DALI is 0.10.0. If you prefer to install it yourself (supposing you have CUDA 10.0), you need to run
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali==0.10.0 
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110


``` -->
i
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
## other utils

```bash
pip install opencv-python
pip install scipy
pip install scikit_image==0.16.2
pip install tqdm
pip install tensorboardX
```


# Data

Keep the folder structure as (to polish structure)
```
--EfficientVideoDenoising
  |--dataset
   |--CRVD
   |--SRVD_data
  |--configs
  |--data
  |--model
  |--train_crvd_ddp.py
  |--logs
  |--results

```
# Train
```bash
# train tswnet_16
./configs/train_crvd_tswnet/0830_train_crvd_ptswnet_l1_seq7.sh
# train tswnet_24
./configs/train_crvd_tswnet/0907_train_crvd_ptswnet_l1_lr_same_inference_time_1e-5.sh
```

# Validation

```bash

bash ./configs/memory_conv/1113_wnet_memconv_none_as_end.sh 

# validate tswnet_24

```
