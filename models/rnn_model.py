import torch
from torch import nn
from torch.nn import init

class Rnn_Model(nn.Module):
    def __init__(self, n_layers, n_hidden, embed_size, rnn_cell, n_tokens, dropout=0, tie_weights=False, no_bias=False):
        super(Rnn_Model, self).__init__()
        self.encoder = nn.Embedding(n_tokens, embed_size)
        self.decoder = nn.Linear(n_hidden, n_tokens, bias=not no_bias)
        self.dropout = nn.Dropout(dropout)

        # Cell
        if rnn_cell == 'lstm':
            self.rnn = nn.LSTM(embed_size, n_hidden, n_layers, dropout=dropout)
        elif rnn_cell == 'gru':
            self.rnn = nn.GRU(embed_size, n_hidden, n_layers, dropout=dropout)
        elif rnn_cell == 'rnn':
            self.rnn = nn.RNN(embed_size, n_hidden, n_layers, nonlinearity='tanh', dropout=dropout)
        elif rnn_cell == 'custom':
            self.rnn = nn.LSTM(embed_size, n_hidden, n_layers, dropout=dropout)
        else:
            raise('Unknown RNN_CELL: %s' % rnn_cell)

        if tie_weights:
            assert n_hidden == embed_size:
            self.decoder.weight = self.encoder.weight

        self.no_bias = no_bias
        self.rnn_cell = rnn_cell
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        self.init_weights()


    def init_weights(self, gain=0.02):
        init.normal_(self.encoder.weight.data, 0.0, gain)
        init.normal_(self.decoder.weight.data, 0.0, gain)
        if not self.no_bias:
            init.normal_(self.decoder.bias.data, 0.0, gain)

    def forward(self, input, hidden):
        out = self.dropout(self.encoder(input))
        out, hidden = self.rnn(out, hidden)
        out = self.dropout(out)
        decoded = self.decoder(out.view(-1, out.size(2)))
        return decoded.view(out.size(0), out.size(1), -1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_cell == 'lstm':
            return (weight.new_zeros(self.n_layers, bsz, self.n_hidden),
                    weight.new_zeros(self.n_layers, bsz, self.n_hidden))
        else:
            return weight.new_zeros(self.n_layers, bsz, self.n_hidden)
