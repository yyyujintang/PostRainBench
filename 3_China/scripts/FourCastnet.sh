export CUDA_VISIBLE_DEVICES=2
python train_Shandong.py --model="FourCastNet" --device=2 --seed=10 --input_data="gdaps_kim" \
                --num_epochs=100  \
                --rain_thresholds 0.1 2.0 \
                --start_lead_time 6 --end_lead_time 88 \
                --log_dir logs/logs_1121_China \
                --batch_size 8 \
                --window_size 1 \
                --lr 0.0001 \
                --dropout 0.0 \
                --loss ce \
                --wd_ep 100 \
                --custom_name="China_fourcastnet_bs8_ep100_seed_10"

