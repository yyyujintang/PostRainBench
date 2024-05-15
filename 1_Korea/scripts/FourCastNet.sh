export CUDA_VISIBLE_DEVICES=0
python train_Korea.py --model="FourCastNet" --device=0 --seed=0 --input_data="gdaps_kim" \
                --num_epochs=50  --normalization \
                --rain_thresholds 0.1 10.0 \
                --start_lead_time 6 --end_lead_time 88 \
                --interpolate_aws \
                --intermediate_test \
                --log_dir logs/logs_1121_Korea \
                --batch_size 1 \
                --window_size 6 \
                --dataset_dir /KoMet/nims/ \
                --loss ce \
                --custom_name="Korea_forecastnet_ce_50ep_seed_0" 