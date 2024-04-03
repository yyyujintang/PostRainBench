export CUDA_VISIBLE_DEVICES=0
nohup python train_Germany.py --model="convlstm" --device=0 --seed=10 --input_data="gdaps_kim" \
                --num_epochs=50 \
                --rain_thresholds 0.1 2.0 \
                --log_dir logs/logs_0925_Germany \
                --batch_size 20 \
                --dataset_dir /GermanyData/ \
                --window_size 1 \
                --lr 0.001 \
                --dropout 0.0 \
                --loss ce \
                --wd_ep 100 \
                --custom_name="Germany_convlstm_bs24_ep50_seed_10" &