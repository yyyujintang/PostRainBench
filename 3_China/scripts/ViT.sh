export CUDA_VISIBLE_DEVICES=1
nohup python train_Shandong.py --model="SViT_CAM_Two" --device=1 --seed=10 --input_data="gdaps_kim" \
                --num_epochs=100 \
                --rain_thresholds 0.1 2.0 \
                --log_dir logs/logs_0925_China \
                --batch_size 8 \
                --tsvit_patch_size 8 \
                --tsvit_time_emd_dim 6 \
                --temporal_depth 4 \
                --spatial_depth 4 \
                --lr 0.0001 \
                --dropout 0.0 \
                --use_two \
                --loss ce+mse \
                --alpha 100 \
                --kernel_size 3 \
                --weight_version 4 \
                --wd_ep 100 \
                --custom_name="China_SViT_CAMT_bs8_ep100_v4_seed_10" &