# coding=UTF-8
from __future__ import division
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_args import args
from model import MultiHeadAttnExtractor
from model import AttnExtractor
from model import ConvSentEncoder
from model import DocEncoder
from model import Extractor
from model import WordEmbedding
from model_utils import get_pretrained_weights


class Generator(nn.Module):
    """Generator
    """
    def __init__(self):
        super(Generator, self).__init__()

        weights = get_pretrained_weights(args.pretrained_wordembedding)
        self._embedding = WordEmbedding(args.embedding_size, weights, trainable=args.trainable_wordembed)

        self._senenc = ConvSentEncoder(args.embedding_size,
            args.sentembed_size,
            args.max_filter_length,
            args.dropout)

        self._docenc = DocEncoder(args.sentembed_size,
            args.size,
            args.num_layers,
            args.rnn_cell,
            args.dropout,
            args.bidirectional)

        self._docext = AttnExtractor(args.sentembed_size,
            args.size,
            args.num_layers,
            args.rnn_cell) if args.attn else Extractor(args.sentembed_size,
            args.size,
            args.num_layers,
            args.rnn_cell)

        if args.aux_embedding:
            weights = get_pretrained_weights(args.pretrained_aux_wordembedding)
            self._aux_embedding = WordEmbedding(args.embedding_size, weights, trainable=args.trainable_wordembed)
            self._aux_senenc = ConvSentEncoder(args.embedding_size,
                args.sentembed_size,
                args.max_filter_length,
                args.dropout)
            self._linear = nn.Linear(args.sentembed_size*2, args.sentembed_size)

    def forward(self, input_):
        batch_size = input_.size(0)
        input_, aux_input_ = input_[:, :args.max_doc_length], input_[:, args.max_doc_length:]

        # [batch_size, max_doc_length, max_sent_length] -> [batch_size, (max_doc_length*max_sent_length)]
        input_ = input_.view(batch_size, -1)
        embedded = self._embedding(input_)
        # [batch_size, (max_doc_length*max_sent_length)] -> [(batch_size*max_doc_length), max_sent_length, embedding_dim]
        embedded = embedded.view(-1, args.max_sent_length, args.embedding_size)

        sent_embedding = self._senenc(embedded)

        if args.aux_embedding:
            # [batch_size, max_doc_length, max_sent_length] -> [batch_size, (max_doc_length*max_sent_length)]
            aux_input_ = aux_input_.view(batch_size, -1)
            aux_embedded = self._aux_embedding(aux_input_)
            # [batch_size, (max_doc_length*max_sent_length)] -> [(batch_size*max_doc_length), max_sent_length, embedding_dim]
            aux_embedded = aux_embedded.view(-1, args.max_sent_length, args.embedding_size)

            aux_sent_embedding = self._aux_senenc(aux_embedded)

            #[(batch_size*max_doc_length), sentembed_size] -> [(batch_size*max_doc_length), sentembed_size*2]
            new_sent_embedding = torch.cat((sent_embedding, aux_sent_embedding), dim=1)
            sent_embedding = self._linear(new_sent_embedding)

        #[(batch_size*max_doc_length), sentembed_size] -> [batch_size, max_doc_length, sentembed_size]
        sent_embedding = sent_embedding.view(-1, args.max_doc_length, args.sentembed_size)

        #[batch_size, max_doc_length, sentembed_size] -> [max_doc_length, batch_size, sentembed_size]
        sent_enc = sent_embedding.transpose(0, 1)
        if args.doc_encoder_reverse:
            sent_enc = sent_enc.flip([2])
        sent_ext = sent_embedding.transpose(0, 1)

        # Initialize hidden state of encoder
        enchid = self._docenc.init_hidden(batch_size, (torch.cuda.is_available() and args.use_gpu))
        # Encode document
        encout, enchid = self._docenc(sent_enc, enchid)
        # Extract sentences
        if args.bidirectional:
            enchid = tuple(hid.view(args.num_layers, 2, -1, args.size) for hid in enchid) if args.rnn_cell=='lstm' else enchid.view(args.num_layers, 2, -1, args.size)
            enchid = tuple(hid[:, 0] + hid[:, 1] for hid in enchid) if args.rnn_cell=='lstm' else enchid[:, 0] + enchid[:, 1]
        probs, logits = self._docext(sent_ext, enchid, encout) if args.attn else self._docext(sent_ext, enchid)

        if args.attn:
            self.all_attn_weights = self._docext.all_attn_weights
            
        return probs.transpose(0, 1), logits.transpose(0, 1)

    def _encode(self, input_):
        self._senenc(input_)

    def _decode(self):
        pass

    def sample(self, rollout):
        pass
