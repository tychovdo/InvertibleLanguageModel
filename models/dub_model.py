import torch
from torch import nn
from torch.nn import init
from memcnn.models.revop import ReversibleBlock

class Dub_Model(nn.Module):
    ''' Experimental affine coupling... '''
    def __init__(self, n_layers, n_hidden, embed_size, rnn_cell, n_tokens, dropout=0, tie_weights=False, no_bias=False):
        super(Dub_Model, self).__init__()
        self.encoder = nn.Embedding(n_tokens, embed_size)
        self.decoder = nn.Linear(n_hidden, n_tokens, bias=not no_bias)
        self.dropout = nn.Dropout(dropout)

        self.rnn_Fs_mult = []
        self.rnn_Fs_add = []
        self.rnn_Gs_mult = []
        self.rnn_Gs_add = []
        if rnn_cell == 'lstm':
            self.rnn_Fs_mult.append(nn.LSTM(embed_size // 2, n_hidden // 2, n_layers, dropout=dropout, bias=True))
            self.rnn_Fs_add.append(nn.LSTM(embed_size // 2, n_hidden // 2, n_layers, dropout=dropout, bias=True))
            self.rnn_Gs_mult.append(nn.LSTM(embed_size // 2, n_hidden // 2, n_layers, dropout=dropout, bias=True))
            self.rnn_Gs_add.append(nn.LSTM(embed_size // 2, n_hidden // 2, n_layers, dropout=dropout, bias=True))
        elif rnn_cell == 'gru':
            self.rnn_Fs_mult.append(nn.GRU(embed_size // 2, n_hidden // 2, n_layers, dropout=dropout))
            self.rnn_Fs_add.append(nn.GRU(embed_size // 2, n_hidden // 2, n_layers, dropout=dropout))
            self.rnn_Gs_mult.append(nn.GRU(embed_size // 2, n_hidden // 2, n_layers, dropout=dropout))
            self.rnn_Gs_add.append(nn.GRU(embed_size // 2, n_hidden // 2, n_layers, dropout=dropout))
        elif rnn_cell == 'rnn':
            self.rnn_Fs_mult.append(nn.RNN(embed_size // 2, n_hidden // 2, n_layers, nonlinearity='tanh', dropout=dropout))
            self.rnn_Fs_add.append(nn.RNN(embed_size // 2, n_hidden // 2, n_layers, nonlinearity='tanh', dropout=dropout))
            self.rnn_Gs_mult.append(nn.RNN(embed_size // 2, n_hidden // 2, n_layers, nonlinearity='tanh', dropout=dropout))
            self.rnn_Gs_add.append(nn.RNN(embed_size // 2, n_hidden // 2, n_layers, nonlinearity='tanh', dropout=dropout))
        else:
            raise NotImplemented('Unknown RNN_CELL: %s' % rnn_cell)

        if tie_weights:
            assert n_hidden == embed_size
            self.decoder.weight = self.encoder.weight

        self.mods = nn.ModuleList([*self.rnn_Fs_mult, *self.rnn_Fs_add, *self.rnn_Gs_mult, *self.rnn_Gs_add])
        self.rnn_cell = rnn_cell
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.no_bias = no_bias

        self.init_weights()
        
        print('Number of RNN parameters:', len(torch.cat([x.view(-1) for x in self.mods.parameters()])))

    def fun(self, input):
        # print(input)
        # return torch.exp(input)
        # return torch.log(input)
        # print(input.shape)
        out = 1 + torch.relu(input)
        # out = 0.5 + torch.sigmoid(input)
        # print(out.view(-1).mean().cpu().detach().numpy(), out.view(-1).std().cpu().detach().numpy())
        return out
        # return torch.sigmoid(input) * 2 - 1

    def init_weights(self, gain=0.02):
        init.normal_(self.encoder.weight.data, 0.0, gain)
        init.normal_(self.decoder.weight.data, 0.0, gain)
        if not self.no_bias:
            init.normal_(self.decoder.bias.data, 0.0, gain)

    def forward(self, input, hidden):
        out = self.dropout(self.encoder(input))
        halfsize = out.shape[2] // 2

        for rnn_F_add, rnn_F_mult, rnn_G_add, rnn_G_mult in zip(self.rnn_Fs_add, self.rnn_Fs_mult,
                                                                self.rnn_Gs_add, self.rnn_Gs_mult):
            # Split x
            out_x1 = out[:, :, :halfsize]
            out_x2 = out[:, :, halfsize:]
            print(1, out_x1.contiguous().view(-1).mean().cpu().detach().numpy(), out_x1.contiguous().view(-1).std().cpu().detach().numpy())
            print(2, out_x2.contiguous().view(-1).mean().cpu().detach().numpy(), out_x2.contiguous().view(-1).std().cpu().detach().numpy())

            # Split h
            if self.rnn_cell == 'lstm':
                h_x1 = [x[:, :, :halfsize].contiguous() for x in hidden]
                h_x2 = [x[:, :, halfsize:].contiguous() for x in hidden]
            else:
                h_x1 = hidden[:, :, :halfsize]
                h_x2 = hidden[:, :, halfsize:]

            # F(x2)
            if self.rnn_cell == 'lstm':
                out_F_add, h_F_add = rnn_F_add(out_x2.contiguous(), h_x2)
                out_F_mult, h_F_mult = rnn_F_mult(out_x2.contiguous(), h_x2)
            else:
                out_F_add, h_F_add = rnn_F_add(out_x2.contiguous(), h_x2.contiguous())
                out_F_mult, h_F_mult = rnn_F_mult(out_x2.contiguous(), h_x2.contiguous())

            # y1 = x1 + F(x2)
            out_y1 = out_x1 * self.fun(out_F_mult) + out_F_add
            if self.rnn_cell == 'lstm':
                h_y1 = list([h_x1_p * self.fun(h_F_p_mult) + h_F_p_add for h_x1_p, h_F_p_mult, h_F_p_add in zip(h_x1, h_F_mult, h_F_add)])
            else:
                h_y1 = h_x1 * self.fun(h_F_mult) + h_F_add

            # G(y1)
            out_G_add, h_G_add = rnn_G_add(out_y1, h_y1)
            out_G_mult, h_G_mult = rnn_G_mult(out_y1, h_y1)

            # y2 = x2 + G(y1)
            out_y2 = out_x2 * self.fun(out_G_mult) + out_G_add
            if self.rnn_cell == 'lstm':
                h_y2 = list([h_x2_p * self.fun(h_G_p_mult) + h_G_p_add for h_x2_p, h_G_p_mult, h_G_p_add in zip(h_x2, h_G_mult, h_G_add)])
            else:
                h_y2 = h_x2 * self.fun(h_G_mult) + h_G_add
                
            # Concatenate
            out = torch.cat([out_y1, out_y2], 2)
            if self.rnn_cell == 'lstm':
                hidden = list([torch.cat([h_y1_p, h_y2_p], 2) for h_y1_p, h_y2_p in zip(h_y1, h_y2)])
            else:
                hidden = torch.cat([h_y1, h_y2], 2)

        out = self.dropout(out)
        decoded = self.decoder(out.view(-1, out.size(2)))
        return decoded.view(out.size(0), out.size(1), -1), hidden


    def inverse(self, input, hidden):
        out = self.dropout(self.encoder(input))
        halfsize = out.shape[2] // 2

        for rnn_F_mult, rnn_F_add, rnn_G_mult, rnn_G_add in zip(reversed(self.rnn_Fs_mult), reversed(self.rnn_Fs_add),
                                                                reversed(self.rnn_Gs_mult), reversed(self.rnn_Gs_add)):
            # Split y
            out_y1 = out[:, :, :halfsize]
            out_y2 = out[:, :, halfsize:]

            # Split h
            if self.rnn_cell == 'lstm':
                h_y1 = [x[:, :, :halfsize].contiguous() for x in hidden]
                h_y2 = [x[:, :, halfsize:].contiguous() for x in hidden]
            else:
                h_y1 = hidden[:, :, :halfsize]
                h_y2 = hidden[:, :, halfsize:]

            # G(y1)
            out_G_mult, h_G_mult = rnn_G_mult(out_y1, h_y1)
            out_G_add, h_G_add = rnn_G_add(out_y1, h_y1)

            # x2 = y2 - G(y1)
            out_x2 = (out_y2 - out_G_add) / self.fun(out_G_mult)
            if self.rnn_cell == 'lstm':
                h_x2 = list([(h_y2_p - h_G_p_add) / self.fun(h_G_p_mult) for h_y2_p, h_G_p_mult, h_G_p_add in zip(h_y2, h_G_mult, h_G_add)])
            else:
                h_x2 = (h_y2 - h_G_add) / self.fun(h_G_mult)

            # F(x2)
            if self.rnn_cell == 'lstm':
                out_F_mult, h_F_mult = rnn_F_mult(out_x2.contiguous(), h_x2)
                out_F_add, h_F_add = rnn_F_add(out_x2.contiguous(), h_x2)
            else:
                out_F_mult, h_F_mult = rnn_F_mult(out_x2.contiguous(), h_x2.contiguous())
                out_F_add, h_F_add = rnn_F_add(out_x2.contiguous(), h_x2.contiguous())

            # x1 = y1 - F(x2)
            out_x1 = (out_y1 - out_F_add) / self.fun(out_F_mult)
            if self.rnn_cell == 'lstm':
                h_x1 = list([(h_y1_p - h_F_p_add) / self.fun(h_F_p_mult) for h_y1_p, h_F_p_mult, h_F_p_add in zip(h_y1, h_F_mult, h_F_add)])
            else:
                h_x1 = (h_y1 - h_F_add) / self.fun(h_F_mult)
                
            # Concatenate x
            out = torch.cat([out_x1, out_x2], 2)
            if self.rnn_cell == 'lstm':
                hidden = list([torch.cat([h_x1_p, h_x2_p], 2) for h_x1_p, h_x2_p in zip(h_x1, h_x2)])
            else:
                hidden = torch.cat([h_x1, h_x2], 2)

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

