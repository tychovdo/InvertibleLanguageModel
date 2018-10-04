#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -p gpu
#SBATCH -o log/wikitext2
cd ..
python main.py --dataroot ./datasets/wikitext-2 --gpu_id 0 --lr 10 --rnn_cell gru --name gru 2>&1 >> log/Wgru &
python main.py --dataroot ./datasets/wikitext-2 --gpu_id 1 --lr 10 --rnn_cell lstm --name lstm 2>&1 >> log/Wlstm &
wait
python main.py --dataroot ./datasets/wikitext-2 --gpu_id 0 --lr 10 --rnn_cell gru --dropout 0.5 --name gru_drop 2>&1 >> log/Wgru_drop &
python main.py --dataroot ./datasets/wikitext-2 --gpu_id 1 --lr 10 --rnn_cell lstm --dropout 0.5 --name lstm_drop 2>&1 >> log/Wlstm_drop &
wait
python main.py --dataroot ./datasets/wikitext-2 --gpu_id 0 --lr 10 --rnn_cell gru --tie --name gru_tie 2>&1 >> log/Wgru_tie &
python main.py --dataroot ./datasets/wikitext-2 --gpu_id 1 --lr 10 --rnn_cell lstm --tie --name lstm_tie 2>&1 >> log/Wlstm_tie &
wait
python main.py --dataroot ./datasets/wikitext-2 --gpu_id 0 --lr 10 --rnn_cell gru --tie --dropout 0.5 --name gru_drop_tie 2>&1 >> log/Wgru_drop_tie &
python main.py --dataroot ./datasets/wikitext-2 --gpu_id 1 --lr 10 --rnn_cell lstm --tie --dropout 0.5 --name lstm_drop_tie 2>&1 >> log/Wlstm_drop_tie &
wait
