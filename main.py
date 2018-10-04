import argparse
import time
import math
import os
import torch
import torch.nn as nn
import numpy as np

import os
import os.path
import torch
import torch.optim
from models.rnn_model import Rnn_Model
from models.inv_model import Inv_Model
from models.dub_model import Dub_Model

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (-1 = cpu)')

# Data
parser.add_argument('--dataroot', type=str, default='./datasets/shakespeare', help='Dataset location')
parser.add_argument('--max_tokens', type=int, default=1000000, help='number of tokens')

# Model
parser.add_argument('--model', type=str, default='rnn', help='Model [rnn, inv, conv]')
parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--n_hidden', type=int, default=512, help='Size of layers')
parser.add_argument('--embed_size', type=int, default=512, help='Size of embeddings')
parser.add_argument('--rnn_cell', type=str, default='lstm', help='Type of RNN cell [gru, lstm, rnn]')

# Training
parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
# parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--lr', type=float, default=20, help='Learning rate')
parser.add_argument('--optim', type=str, default='clip', help='Optimizer [sgd, adam]')
parser.add_argument('--seq_length', type=int, default=30, help='sequence length')
parser.add_argument('--epoch', type=int, default=40, help='Amount of epochs')
parser.add_argument('--dropout', type=float, default=0.0, help='Amount of dropout (0 = no dropout)')
parser.add_argument('--tie', action='store_true', help='Word embedding tie')
parser.add_argument('--train_dir', type=str, default='forward', help='Train direction [forward, backward, bidirectional]')
parser.add_argument('--no_bias', action='store_true', help='No bias in word embedding')

# Logging and saving
parser.add_argument('--generate', action='store_true', help='Also generate some test sentences during training')
parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
parser.add_argument('--name', type=str, default='model', help='model name')
args = parser.parse_args()

# CUDA
if args.gpu_id == -1:
    device = torch.device("cpu")
else:
    assert torch.cuda.is_available()
    device = args.gpu_id
    torch.cuda.set_device(args.gpu_id)
    print('CUDA Device: %s (%s)' % (torch.cuda.current_device(), torch.cuda.get_device_name(args.gpu_id)))



def read_file(path, endtoken='\n'):
    ''' Read files and return list of words '''
    lines = []
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            lines += line.split() + [endtoken]
    return lines
            
def build_dict(text, max_tokens=100):
    ''' Build dictionary from list of words '''
    words, counts = np.unique(text, return_counts=True)
    _, sorted_words = zip(*reversed(sorted(zip(counts, words))))
    sorted_words = list(sorted_words[:min(len(sorted_words), max_tokens) - 1])
    sorted_words = ['<unk>'] + sorted_words
    
    print('From %s to %s tokens' % (len(words), len(sorted_words)))
    
    ids = range(len(sorted_words))

    id2word = dict(zip(ids, sorted_words))
    word2id = dict(zip(sorted_words, ids))
    
    return word2id, id2word

def build_data(text, word2id, batch_size=10):
    text = torch.LongTensor([word2id.get(x, 0) for x in text])
    batched = text[:-(len(text) % batch_size)].view(batch_size, -1).t().contiguous().to(device)
    return batched

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_batch_bwd(source, i):
    seq_len = min(args.seq_length, len(source) - 1 - i)
    data = source[i+1:i+1+seq_len].flip(0, 1)
    target = source[i:i+seq_len].flip(0, 1)
    target = target.view(-1)
    return data, target

def evaluate(model, data_source):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.
    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.seq_length):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, args.n_tokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)

def generate(model, id2word, n_words=25, temperature=1.0, n_warmup=100):
    with torch.no_grad():
        gen_hidden = model.init_hidden(1)
        input = torch.randint(args.n_tokens, (1, 1), dtype=torch.long).to(device)

        text = []
        for i in range(n_warmup + n_words):
            output, gen_hidden = model(input, gen_hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_id = torch.multinomial(word_weights, 1)[0]

            input.fill_(word_id)
            
            if i >= n_warmup:
                text.append(id2word[int(word_id.cpu().numpy())])
    return ' '.join(text).encode('utf-8')
    

def main():
    print('Invertible Language Model.')
    # Read files
    train_text = read_file(os.path.join(args.dataroot, 'train.txt'))
    valid_text = read_file(os.path.join(args.dataroot, 'valid.txt'))
    test_text = read_file(os.path.join(args.dataroot, 'test.txt'))
    print('len(train)=', len(train_text))
    print('len(valid)=', len(valid_text))
    print('len(test)=', len(test_text))

    # Build dictionary
    word2id, id2word = build_dict(train_text, max_tokens=args.max_tokens)
    args.n_tokens = len(word2id) + 1

    # Create batches
    train_data = build_data(train_text, word2id, args.batch_size)
    valid_data = build_data(valid_text, word2id, args.batch_size)
    test_data = build_data(test_text, word2id, args.batch_size)

    # Load model
    if args.model == 'rnn':
        model = Rnn_Model(args.n_layers, args.n_hidden, args.embed_size, args.rnn_cell, args.n_tokens, args.dropout, args.tie, args.no_bias).to(device)
    elif args.model == 'inv':
        model = Inv_Model(args.n_layers, args.n_hidden, args.embed_size, args.rnn_cell, args.n_tokens, args.dropout, args.tie, args.no_bias).to(device)
    elif args.model == 'dub':
        model = Dub_Model(args.n_layers, args.n_hidden, args.embed_size, args.rnn_cell, args.n_tokens, args.dropout, args.tie, args.no_bias).to(device)
    else:
        raise NotImplemented('Unknown model: %s' % args.model)
    print('Number of parameters:', len(torch.cat([x.view(-1) for x in model.parameters()])))

    # Initialize optimizer
    lr = args.lr
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.99))
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optim == 'clip':
        pass

    criterion = nn.CrossEntropyLoss()

    best_val_loss = None

    for epoch in range(1, args.epoch+1):
        epoch_start_time = time.time()

        # Train
        model.train()
        total_loss = 0.
        total_loss_fwd = 0.
        total_loss_bwd = 0.
        start_time = time.time()
        
        if args.train_dir == 'forward' or args.train_dir == 'bidirectional':
            hidden_fwd = model.init_hidden(args.batch_size)
        if args.train_dir == 'backward' or args.train_dir == 'bidirectional':
            hidden_bwd = model.init_hidden(args.batch_size)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.seq_length)):
            model.zero_grad()

            if args.train_dir == 'forward' or args.train_dir == 'bidirectional':
                data_fwd, targets_fwd = get_batch(train_data, i)
                hidden_fwd = repackage_hidden(hidden_fwd)
                output_fwd, hidden_fwd = model(data_fwd, hidden_fwd)
                loss_fwd = criterion(output_fwd.view(-1, args.n_tokens), targets_fwd)
                total_loss_fwd += loss_fwd.item()

            if args.train_dir == 'backward' or args.train_dir == 'bidirectional':
                data_bwd, targets_bwd = get_batch_bwd(train_data, i)
                hidden_bwd = repackage_hidden(hidden_bwd)
                output_bwd, hidden_bwd = model.inverse(data_bwd, hidden_bwd)
                loss_bwd = criterion(output_bwd.view(-1, args.n_tokens), targets_bwd)
                total_loss_bwd += loss_bwd.item()

            if args.train_dir == 'forward':
                loss = loss_fwd
            elif args.train_dir == 'backward':
                loss = loss_bwd
            elif args.train_dir == 'bidirectional':
                loss = (loss_fwd + loss_bwd) / 2
            else:
                raise NotImplemented('Unknown train direction: [%s]' % args.train_dir)

            loss.backward()
            
            if args.optim == 'clip':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                for p in model.parameters():
                    p.data.add_(-lr, p.grad.data)
            else:
                optimizer.step()

            total_loss += loss.item()

            if batch % args.print_freq == 0 and batch > 0:
                cur_loss = total_loss / args.print_freq

                elapsed = time.time() - start_time

                log_str = '| epoch {:3d}/{:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '.format(
                    epoch, args.epoch, batch, len(train_data) // args.seq_length, lr,
                    elapsed * 1000 / args.print_freq)

                log_total = False
                if log_total:
                    log_str += 'loss {:5.2f} | ppl{:8.2f} | '.format(cur_loss, math.exp(cur_loss))

                if args.train_dir == 'forward' or args.train_dir == 'bidirectional':
                    cur_loss_fwd = total_loss_fwd / args.print_freq
                    log_str += 'loss_fwd {:5.2f} | ppl_fwd {:8.2f} |'.format(cur_loss_fwd, math.exp(cur_loss_fwd))
                if args.train_dir == 'backward' or args.train_dir == 'bidirectional':
                    cur_loss_bwd = total_loss_bwd / args.print_freq
                    log_str += 'loss_bwd {:5.2f} | ppl_bwd {:8.2f} |'.format(cur_loss_bwd, math.exp(cur_loss_bwd))
                print(log_str)
                
                total_loss = 0
                total_loss_fwd = 0
                total_loss_bwd = 0
                start_time = time.time()

        # Evaluate on validation set after epoch
        val_loss = evaluate(model, valid_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        
        if args.generate:
            print('---------- Generated 0.5 ------------')
            print(generate(model, id2word, temperature=0.5))
            print('---------- Generated 1.0 ------------')
            print(generate(model, id2word, temperature=1.0))
            print('---------- Generated 2.0 ------------')
            print(generate(model, id2word, temperature=2.0))
            print('---------- Generate complete --------')
        
        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            lr /= 4.0

    # Save model
    model_dir = 'checkpoints/%s' % args.name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model, 'checkpoints/%s/final.pt' % args.name)
        
    # Evaluate on test set after training
    test_loss = evaluate(model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


if __name__ == '__main__':
    main()
