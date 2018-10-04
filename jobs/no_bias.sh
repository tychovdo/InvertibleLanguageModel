#!/bin/bash
#SBATCH -t 00:45:00
#SBATCH -p gpu
#SBATCH -o log/no_bias
cd ..
python main.py --no_bias --model inv --train_dir bidirectional --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell lstm --tie --name bid_lstm_tie 2>&1 >> log/bid_Plstm_tie_nb &
python main.py --no_bias --model inv --train_dir bidirectional --dataroot ./datasets/penntreebank --gpu_id 1 --rnn_cell lstm --tie --dropout 0.5 --name bid_lstm_drop_tie 2>&1 >> log/bid_Plstm_drop_tie_nb &
wait
python main.py --no_bias --model inv --train_dir bidirectional --gpu_id 0 --rnn_cell lstm --tie --name bid_lstm_tie 2>&1 >> log/bid_Slstm_tie_nb &
python main.py --no_bias --model inv --train_dir bidirectional --gpu_id 1 --rnn_cell lstm --tie --dropout 0.5 --name bid_lstm_drop_tie 2>&1 >> log/bid_Slstm_drop_tie_nb &
wait

