CUDA_VISIBLE_DEVICES=4 python train_crvd_ddp.py \
    --exp_name      1113_wnet_memconv_none_as_end \
    --training_mode validation \
    --model         mem_wnet_nonorm_none_as_end\
    --workers 4 \
    --shift shift \
    --sequence_length 7 \
    --gpu 0 \

# [9] GeForce RTX 2080 Ti | 41'C,   7 % |  3768 / 11019 MB | chenyangqi:python/35754(3761M)
# 0.86/7 = 0.12s
