#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o log/bid_penntreebank_bid
cd ..
python main.py --model inv --train_dir bidirectional --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell lstm --name bid_lstm 2>&1 >> log/bid_Plstm &
python main.py --model inv --train_dir bidirectional --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --dropout 0.5 --name bid_lstm_drop 2>&1 >> log/bid_Plstm_drop &
python main.py --model inv --train_dir bidirectional --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell lstm --tie --name bid_lstm_tie 2>&1 >> log/bid_Plstm_tie &
python main.py --model inv --train_dir bidirectional --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --tie --dropout 0.5 --name bid_lstm_drop_tie 2>&1 >> log/bid_Plstm_drop_tie &
wait
