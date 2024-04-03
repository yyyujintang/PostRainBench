export CUDA_VISIBLE_DEVICES=0
python train.py --model="convlstm" --device=0 --seed=0 --input_data="gdaps_kim" \
                --num_epochs=20  --normalization \
                --rain_thresholds 0.1 10.0 \
                --start_lead_time 6 --end_lead_time 88 \
                --interpolate_aws \
                --intermediate_test \
                --log_dir logs/logs_1019_Korea \
                --batch_size 1 \
                --window_size 6 \
                --dataset_dir /KoMet/nims/ \
                --loss smile \
                --custom_name="Korea_convlstm_smile_20ep_seed_0_SMILE" 