#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o log/penntreebank
cd ..
python main.py --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell gru --name gru 2>&1 >> log/Pgru &
python main.py --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --name lstm 2>&1 >> log/Plstm &
python main.py --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell gru --dropout 0.5 --name gru_drop 2>&1 >> log/Pgru_drop &
python main.py --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --dropout 0.5 --name lstm_drop 2>&1 >> log/Plstm_drop &
wait
python main.py --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell gru --tie --name gru_tie 2>&1 >> log/Pgru_tie &
python main.py --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --tie --name lstm_tie 2>&1 >> log/Plstm_tie &
python main.py --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell gru --tie --dropout 0.5 --name gru_drop_tie 2>&1 >> log/Pgru_drop_tie &
python main.py --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --tie --dropout 0.5 --name lstm_drop_tie 2>&1 >> log/Plstm_drop_tie &
wait
