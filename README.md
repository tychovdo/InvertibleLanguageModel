# Invertible Language Model

## Example usage

```
$ python main.py --model inv --train_dir bidirectional --dataroot ./datasets/penntreebank --gpu_id 0 --rnn_cell lstm --tie --dropout 0.5 --name any_name
```

## Acknowledgements

MemCNN: PyTorch Framework for Developing Memory Efficient Deep Invertible Networks
* [memcnn](https://github.com/silvandeleemput/memcnn) 
Reference: Sil C. van de Leemput, Jonas Teuwen, Rashindra Manniesing. MemCNN: a Framework for Developing Memory Efficient Deep Invertible Networks. International Conference on Learning Representations (ICLR) 2018 Workshop Track. (https://iclr.cc/)
