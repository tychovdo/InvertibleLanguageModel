#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o log/inv_penntreebank_inv
cd ..
python main.py --model inv --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell lstm --name inv_lstm 2>&1 >> log/inv_Plstm &
python main.py --model inv --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --dropout 0.5 --name inv_lstm_drop 2>&1 >> log/inv_Plstm_drop &
python main.py --model inv --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell lstm --tie --name inv_lstm_tie 2>&1 >> log/inv_Plstm_tie &
python main.py --model inv --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --tie --dropout 0.5 --name inv_lstm_drop_tie 2>&1 >> log/inv_Plstm_drop_tie &
wait
