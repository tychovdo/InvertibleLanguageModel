#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -p gpu
#SBATCH -o log/bid_wikitext2_bid
cd ..
python main.py --lr 10 --model inv --train_dir bidirectional --dataroot ./datasets/wikitext-2 --gpu_id 0 --rnn_cell lstm --name lstm 2>&1 >> log/bid_Wlstm &
python main.py --lr 10 --model inv --train_dir bidirectional --dataroot ./datasets/wikitext-2 --gpu_id 1 --rnn_cell lstm --dropout 0.5 --name lstm_drop 2>&1 >> log/bid_Wlstm_drop &
wait
python main.py --lr 10 --model inv --train_dir bidirectional --dataroot ./datasets/wikitext-2 --gpu_id 0 --rnn_cell lstm --tie --name lstm_tie 2>&1 >> log/bid_Wlstm_tie &
python main.py --lr 10 --model inv --train_dir bidirectional --dataroot ./datasets/wikitext-2 --gpu_id 1 --rnn_cell lstm --tie --dropout 0.5 --name lstm_drop_tie 2>&1 >> log/bid_Wlstm_drop_tie &
wait
python main.py --lr 10 --model inv --train_dir bidirectional --dataroot ./datasets/wikitext-2 --gpu_id 0 --rnn_cell lstm --no_bias --tie --name lstm_tie 2>&1 >> log/bid_Wlstm_tie_nb &
python main.py --lr 10 --model inv --train_dir bidirectional --dataroot ./datasets/wikitext-2 --gpu_id 1 --rnn_cell lstm --no_bias --tie --dropout 0.5 --name lstm_drop_tie 2>&1 >> log/bid_Wlstm_drop_tie_nb &
wait

