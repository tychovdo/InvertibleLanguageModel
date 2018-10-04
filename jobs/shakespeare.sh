#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -p gpu
#SBATCH -o log/shakespeare
cd ..
python main.py --gpu_id 0 --rnn_cell gru --name gru 2>&1 >> log/Sgru &
python main.py --gpu_id 1 --rnn_cell lstm --name lstm 2>&1 >> log/Slstm &
python main.py --gpu_id 0 --rnn_cell gru --dropout 0.5 --name gru_drop 2>&1 >> log/Sgru_drop &
python main.py --gpu_id 1 --rnn_cell lstm --dropout 0.5 --name lstm_drop 2>&1 >> log/Slstm_drop &
wait
python main.py --gpu_id 0 --rnn_cell gru --tie --name gru_tie 2>&1 >> log/Sgru_tie &
python main.py --gpu_id 1 --rnn_cell lstm --tie --name lstm_tie 2>&1 >> log/Slstm_tie &
python main.py --gpu_id 0 --rnn_cell gru --tie --dropout 0.5 --name gru_drop_tie 2>&1 >> log/Sgru_drop_tie &
python main.py --gpu_id 1 --rnn_cell lstm --tie --dropout 0.5 --name lstm_drop_tie 2>&1 >> log/Slstm_drop_tie &
wait
