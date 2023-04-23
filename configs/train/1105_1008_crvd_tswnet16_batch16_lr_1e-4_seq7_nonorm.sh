OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,3,4,5 python train_crvd_ddp.py \
    --exp_name      1106_1008_crvd_tswnet16_batch16_lr_1e-4_seq7_nonorm \
    --training_mode all \
    --model=wnet_no_norm \
    --lr            1e-4 \
    --batch_size    4 \
    --patch_size    128 \
    --save_every_epochs 200 \
    --save_every    1000 \
    --epochs 12000 \
    --workers 8 \
    --channels1 16 32 64 \
    --channels2 16 32 64 \
    --shift shift \
    --loss l1 \
    --A 1 \
    --B 30 \
    --sequence_length 7 \
    --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 \

