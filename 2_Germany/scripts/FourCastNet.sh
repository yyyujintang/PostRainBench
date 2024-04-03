export CUDA_VISIBLE_DEVICES=0
python train_Germany.py --model="FourCastNet" --device=0 --seed=0 --input_data="gdaps_kim" \
                --num_epochs=30 \
                --rain_thresholds 0.00001 2.0 \
                --log_dir logs/logs_0925_Germany \
                --batch_size 20 \
                --dataset_dir /GermanyData/ \
                --window_size 1 \
                --lr 0.001 \
                --dropout 0.0 \
                --loss ce \
                --wd_ep 100 \
                --custom_name="Germany_FourCastNet_bs24_ep30_seed_0" 