# BSVD for Fast Video Processing on CRVD dataset

Note that the code in this branch is quite dirty. And this is only for those who just want to reproduce the performance reported in BSVD paper.

Currently, I have moved to research on video diffusion model, which is the state-of-the-art in image and video generation.

If you are looking for the best model for video denoising, it is strongly recommended to train a video denoising model, which naturally support video denoising.

Another cheap and fast option is using a pretrained diffusion model for zero-shot video enhancement like that in [FateZero](https://github.com/ChenyangQiQi/FateZero).
We have present a initl experiment of zero-sho video restoration at 3:00 minute of thie [video](https://github.com/ChenyangQiQi/FateZero#-demo-video).
# Dependencies

## Environment

Create your environment
```bash
conda env create -f environment.yaml -n py37torch170
# for 3090GPU
# conda env create -f environment_3090.yaml -n py37torch170
# pip install -r requirements.txt

```
To install pytorch for cuda11
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## other utils

```bash
pip install opencv-python
pip install scipy
pip install scikit_image==0.16.2
pip install tqdm
pip install tensorboardX
```


# Data
Download the pretrained checkpoint as training logs:

```bash
wget https://github.com/ChenyangQiQi/BSVD/releases/download/v0.0.1/logs.tar.gz
tar -xvf logs.tar.gz
```

Download the CRVD testset from [onedrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cqiaa_connect_ust_hk/EhMFewzVZrRFvrobZ3Z7JCsBYsQD-iNKoLLalad0uc4RCg?e=0CL69m).

Keep the folder structure as
```
--EfficientVideoDenoising
  |--configs
  |--data
  |--model
  |--train_crvd_ddp.py
  |--logs
    |--1106_1008_crvd_tswnet16_batch16_lr_1e-4_seq7_nonorm_10_300
      |--ckpt_epochs
    |--1113_wnet_memconv_none_as_end
  |--results
  |--dataset
    |--CRVD
    |--SRVD_data

```


# Validation

```bash

bash ./configs/memory_conv/1113_wnet_memconv_none_as_end.sh 

# validate tswnet_24

```

Check the result at 'logs/1113_wnet_memconv_none_as_end'
# Train (in progress)
<!-- ```bash
# train tswnet_16
./configs/train_crvd_tswnet/0830_train_crvd_ptswnet_l1_seq7.sh
# train tswnet_24
./configs/train_crvd_tswnet/0907_train_crvd_ptswnet_l1_lr_same_inference_time_1e-5.sh
``` -->


