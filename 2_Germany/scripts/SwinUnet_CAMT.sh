export CUDA_VISIBLE_DEVICES=2
nohup python train_Germany.py --model="SwinUnet_CAM_Two" --device=2 --seed=0 --input_data="gdaps_kim" \
                --num_epochs=30 \
                --rain_thresholds 0.00001 2.0 \
                --log_dir logs/logs_0925_Germany \
                --batch_size 20 \
                --dataset_dir /GermanyData/ \
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
                --weight_version 2 \
                --wd_ep 100 \
                --custom_name="Germany_SwinUnet_CAMT_bs24_ep30_seed_0" &