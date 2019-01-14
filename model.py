# coding=UTF-8
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model_utils import *
from my_args import args


class WordEmbedding(nn.Module):
    """Word Embedding
    """
    def __init__(self,
        embedding_size,
        weights,
        trainable=False):
        """Initial function
        """
        super(WordEmbedding, self).__init__()
        self._weights = weights
        self._embedding = nn.Embedding.from_pretrained(weights, freeze=trainable)
        print(self._embedding)
        self._aux_embedding = nn.Embedding(2, embedding_size, padding_idx=0)

    def forward(self, input_):
        mask = input_.eq(1)
        #print(new)
        embedded = self._embedding(input_)
        embedded += self._aux_embedding(mask.long()) # for training UNK embedding
        return embedded

class ConvSentEncoder(nn.Module):
    """Convolutional Sentence Encoder
    """
    def __init__(self,
        embedding_size,
        sentembed_size,
        max_filter_length,
        dropout=0.5):
        """Initial function
        """
        super(ConvSentEncoder, self).__init__()
        self._embedding_size = embedding_size
        self._sentembed_size = sentembed_size
        self._max_filter_length = max_filter_length
        self._dropout = args.dropout

        self._convs = nn.ModuleList(
            [nn.Conv1d(self._embedding_size, self._sentembed_size//self._max_filter_length, i+1) for i in range(self._max_filter_length)])


    def forward(self, input_):
        conv_in = F.dropout(input_.transpose(1, 2), self._dropout, training=self.training)
        output = torch.cat(
            [F.relu(conv(conv_in)).max(dim=2)[0]
            for conv in self._convs], dim=1)

        return output

class DocEncoder(nn.Module):
    """Document Encoder
    """
    def __init__(self,
        sentembed_size,
        size,
        num_layers,
        rnn_cell='gru',
        dropout=0,
        bidirectional=False):
        """Initial function
        """
        super(DocEncoder, self).__init__()
        self._sentembed_size = sentembed_size
        self._size = size
        self._num_layers = num_layers
        self._rnn_cell = rnn_cell
        self._dropout = dropout
        self._bidirectional = bidirectional
        self._rnn = build_rnn(
            self._rnn_cell,
            self._sentembed_size,
            self._size,
            self._num_layers,
            self._dropout,
            self._bidirectional)

    def forward(self, input_, hidden):
        return self._rnn(input_, hidden)

    def init_hidden(self, batch_size):
        return init_rnn_states(self._rnn_cell, self._size, batch_size, self._num_layers, self._bidirectional)

class Extractor(nn.Module):
    """Sentence Extractor
    """
    def __init__(self,
        sentembed_size,
        size,
        num_layers,
        rnn_cell='gru',
        target_label_size=2):
        """Initial function
        """
        super(Extractor, self).__init__()
        self._sentembed_size = sentembed_size
        self._size = size
        self._num_layers = num_layers
        self._rnn_cell = rnn_cell
        self._target_label_size = target_label_size
        self._rnn = build_rnn(
            self._rnn_cell,
            self._sentembed_size,
            self._size,
            self._num_layers,
        )
        self._linear = nn.Linear(self._size, self._target_label_size)

    def forward(self, input_, hidden):
        output, hidden = self._rnn(input_, hidden)
        prob = F.softmax(self._linear(output), dim=2)
        return prob, prob.max(dim=2)[1]

class RewardWeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super(RewardWeightedCrossEntropyLoss, self).__init__()

    def forward(self, input_, target, rewards, weights):
        loss = F.cross_entropy(input_, target, reduction='none')
        if args.weighted_loss:
            loss = weights * loss
        loss = loss.view(-1, 1, args.max_doc_length)

        rewards = rewards.repeat(args.max_doc_length, 1).transpose(0, 1).view(-1, 1, args.max_doc_length)

        reward_weighted_loss = rewards * loss
        reward_weighted_loss = torch.sum(reward_weighted_loss, 2)
        reward_weighted_loss = torch.mean(reward_weighted_loss)
        return reward_weighted_loss


if __name__ == '__main__':
    """import random
    input_ = Variable(torch.tensor([random.randint(0, 1) for _ in range(10)]))
    mask = input_.eq(0)
    print(input_, mask)
    weights = torch.randn(10000, 200)
    embedding = WordEmbedding(args.embedding_size, weights)
    embedded = embedding(input_)
    print(embedded)
    print(input_)
    encoder = ConvSentEncoder(args.embedding_size, args.sentembed_size, args.max_filter_length, args.dropout)
    inp = torch.tensor([[[
        [random.random() for _ in range(args.embedding_size)]
        for i in range(args.max_sent_length)] for _ in range(args.max_doc_length)] for _ in range(5)])
    print(inp.size())
    inp = inp.view(-1, args.max_sent_length, args.embedding_size)
    senenc = encoder(inp)
    print(senenc.size())
    senenc = senenc.view(5, -1, args.sentembed_size).transpose(0, 1)
    docencoder = DocEncoder(args.sentembed_size, args.size, args.num_layers)
    hidden = docencoder.init_hidden(5)
    docenc, hidden = docencoder(senenc, hidden)
    print(hidden.size())
    extractor = Extractor(args.sentembed_size, args.size, args.num_layers, args.rnn_cell)
    prob, logits = extractor(senenc, hidden)
    print(logits)"""
    import random
    lossfn = RewardWeightedCrossEntropyLoss()
    a = torch.randn(300, 2)
    b = torch.LongTensor([random.randint(0, 1) for _ in range(300)])
    w = torch.randn(300)
    r = torch.randn(6)
    loss = lossfn(a, b, r, w)
    print(loss)
