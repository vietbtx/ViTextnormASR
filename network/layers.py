import torch
import torch.nn as nn

import numpy as np

def init_xavier_normal(tensor):
    param = nn.Parameter(tensor)
    nn.init.xavier_normal_(param)
    return param

class BiLSTMLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, rnn_layers=1, dropout=0.5):
        super(BiLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, rnn_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.output_dim = hidden_dim

    def forward(self, input_, input_mask):
        length = input_mask.sum(-1)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_ = input_[sorted_idx]
        packed_input = nn.utils.rnn.pack_padded_sequence(input_, sorted_lengths.data.tolist(), batch_first=True)
        output, (hidden, _) = self.lstm(packed_input)
        padded_outputs = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        return padded_outputs[reversed_idx], hidden[:, reversed_idx]

class BiGRULayer(nn.Module):

    def __init__(self, input_dim, hidden_dim=512, rnn_layers=1, dropout=0.5):
        super(BiGRULayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim//2, rnn_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.output_dim = hidden_dim

    def forward(self, input_, input_mask):
        length = input_mask.sum(-1)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_ = input_[sorted_idx]
        packed_input = nn.utils.rnn.pack_padded_sequence(input_, sorted_lengths.data.tolist(), batch_first=True)
        output, (hidden, _) = self.gru(packed_input)
        padded_outputs = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]
        _, reversed_idx = torch.sort(sorted_idx)
        return padded_outputs[reversed_idx], hidden[:, reversed_idx]

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_heads=3, dropout=0.5):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.weight = init_xavier_normal(torch.FloatTensor(n_heads, input_dim, hidden_dim))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(n_heads*hidden_dim, input_dim)        
        self.norm = nn.LayerNorm(input_dim)
        self.output_dim = input_dim
    
    def forward(self, input_):
        input_size = input_.size(0)
        logits = input_.repeat(self.n_heads, 1, 1).view(self.n_heads, -1, self.input_dim)
        logits = torch.bmm(logits, self.weight).view(input_size * self.n_heads, -1, self.hidden_dim)
        attn = torch.bmm(logits, logits.transpose(1, 2)) / np.sqrt(self.hidden_dim)
        attn = self.softmax(attn)
        outputs = torch.bmm(attn, logits)
        outputs = torch.split(outputs, input_size, dim=0)
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.linear(outputs)
        outputs = self.dropout(outputs)
        return self.norm(input_ + outputs), attn

class BiaffineAttention(nn.Module):

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2*in_features, out_features, bias=True)
        self.reset_parameters()

    def forward(self, x1, x2):
        return self.bilinear(x1, x2) + self.linear(torch.cat((x1, x2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()