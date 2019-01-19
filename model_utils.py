# coding=UTF-8
from __future__ import division
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
import torch.nn as nn

INI = 1e-2

def get_pretrained_weights(weights_file):
    with open(weights_file) as f:
        weights = [[float(w) for w in line.strip().split()[1:]] for line in f.read().decode('utf-8').strip().split('\n')[1:]]
    weight_size = len(weights[0])
    # Add two zero vectors for UNK and PAD
    weights = [[0]*weight_size]*2 + weights
    return torch.tensor(weights)

def build_rnn(rnn_cell, input_size, hidden_size, num_layers, dropout=0, bidirectional=False):
    if rnn_cell == 'lstm':
        rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
    else:
        rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
    return rnn

def init_rnn_states(rnn_cell, hidden_size, batch_size, num_layers, bidirectional=False, cuda=False):
    if rnn_cell == 'lstm':
        states = (
            Variable(torch.empty(num_layers*(2 if bidirectional else 1), batch_size, hidden_size).uniform_(-INI, INI)),
            Variable(torch.empty(num_layers*(2 if bidirectional else 1), batch_size, hidden_size).uniform_(-INI, INI))
        )
        states = tuple(s.cuda() for s in states) if cuda else states
    else:
        states = Variable(torch.empty(num_layers*(2 if bidirectional else 1), batch_size, hidden_size).uniform_(-INI, INI))
        states = states.cuda() if cuda else states
    return states

def extract_sentences_from_logits(source, logit, max_length):
    extracted = source.new_zeros(source.size())
    pass

def reward_weighted_cross_entropy_loss_multisample():
    pass

def maximal_marginal_relevance():
    pass
