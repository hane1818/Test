# coding=UTF-8
from __future__ import division
from __future__ import unicode_literals

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from my_args import args

PAD_ID=0
UNK_ID=1

class Data(Dataset):
    """My Data
    """
    def __init__(self, data_type='training'):
        self.data_type = data_type
        self.filenames = []
        self._docs = []
        self._auxdocs = []
        self._singlelabels = []
        self._multiplelabels = []
        self._highlights = []
        self._rewards = []
        self._weights = []

        self._read_data(data_type)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        def padding(source, max_size):
            if len(source) >= max_size:
                return source[:max_size]
            padids = [PAD_ID] * (max_size - len(source))
            return source + padids

        docname = self.filenames[idx]
        # Document
        doc_tensor = []
        for i in range(args.max_doc_length):
            sent = []
            if i < len(self._docs[idx]):
                sent = self._docs[idx][i][:]
            sent = padding(sent, args.max_sent_length)
            doc_tensor.append(sent)

        # Auxiliary Document
        if args.aux_embedding:
            for i in range(args.max_doc_length):
                sent = []
                if i < len(self._auxdocs[idx]):
                    sent = self._auxdocs[idx][i][:]
                sent = padding(sent, args.max_sent_length)
                doc_tensor.append(sent)

        doc_tensor = torch.LongTensor(doc_tensor)

        # Labels
        slabel_tensor = self._singlelabels[idx][:]
        slabel_tensor = padding(slabel_tensor, args.max_doc_length)
        slabel_tensor = torch.LongTensor(slabel_tensor)

        mlabel_tensor = []
        reward_tensor = []
        for i in range(args.num_sample_rollout):
            if i < len(self._multiplelabels[idx]):
                thislabels = [1 if item in self._multiplelabels[idx][i] else 0 for item in range(args.max_doc_length)]
                reward_tensor.append(self._rewards[idx][i])
            else:
                thislabels = [1 if item in self._multiplelabels[idx][0] else 0 for item in range(args.max_doc_length)]
                reward_tensor.append(self._rewards[idx][0])
            mlabel_tensor.append(thislabels)
        random_idx = random.randint(0, args.num_sample_rollout-1)
        mlabel_tensor = torch.LongTensor(mlabel_tensor[random_idx])
        reward_tensor = torch.FloatTensor(reward_tensor[random_idx])

        # Weights
        weights_tensor = padding(self._weights[idx][:], args.max_doc_length)
        weights_tensor = torch.FloatTensor(weights_tensor)

        # Highlights
        highlights_tensor = []
        for i in range(args.max_doc_length):
            sent = []
            if i < len(self._highlights[idx]):
                sent = self._highlights[idx][i][:]
            sent = padding(sent, args.max_sent_length)
            highlights_tensor.append(sent)
        highlights_tensor = torch.LongTensor(highlights_tensor)

        if torch.cuda.is_available() and args.use_gpu:
            return docname, doc_tensor.cuda(), slabel_tensor.cuda(), mlabel_tensor.cuda(), reward_tensor.cuda(), weights_tensor.cuda(), highlights_tensor.cuda()
        return docname, doc_tensor, slabel_tensor, mlabel_tensor, reward_tensor, weights_tensor, highlights_tensor

    def process_negative_samples(self, docnames, logits, probs):
        #Find docs
        #use probs to sample
        #extract summary sentences per sample
        pass

    def write_prediction_summary(self, docnames, logits, path):
        sum_dir = os.path.join(args.train_dir, '.'.join([path, self.data_type, 'summary']))
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        for docname, logit in zip(docnames, logits):
            docname = docname.split('-')[-1]
            with open(os.path.join(args.doc_sentence_directory, self.data_type, 'mainbody', docname)+'.mainbody') as f:
                fulldoc = np.array(f.read().decode('utf-8').strip().split('\n'))
                doc_len = fulldoc.shape[0]
                summary = fulldoc[logit.cpu()[:doc_len].nonzero().squeeze()]

            with open(os.path.join(sum_dir, docname)+'.summary', 'w') as f:
                if not summary.shape:
                    summary = np.expand_dims(summary, 0)
                assert type(summary.tolist()) is list
                f.write('\n'.join(summary.tolist()[:args.summary_ratio*doc_len]))

    def _read_data(self, data_type):
        file_prefix = os.path.join(args.preprocessed_data_directory, '.'.join([args.data_mode, data_type]))
        with open(file_prefix+'.doc') as doc, \
             open(file_prefix+'.auxdoc') as auxdoc, \
             open(file_prefix+'.label.singleoracle') as singleoracle, \
             open(file_prefix+'.label.multipleoracle') as multipleoracle, \
             open(file_prefix+'.highlights') as highlights:   # UNDO: Highlight file
            doc_list = doc.read().strip().split('\n\n')
            auxdoc_list = auxdoc.read().strip().split('\n\n')
            singleoracle_list = singleoracle.read().strip().split('\n\n')
            multipleoracle_list = multipleoracle.read().strip().split('\n\n')
            highlights_list = highlights.read().strip().split('\n\n')
            assert len(doc_list) == len(auxdoc_list) == len(singleoracle_list) == len(multipleoracle_list) == len(highlights_list)

        print("Reading data......")
        doccount = 0
        for doc, auxdoc, soracle, moracle, highlights in zip(doc_list, auxdoc_list, singleoracle_list, multipleoracle_list, highlights_list)[:30000]:
            doc_lines = doc.strip().split('\n')
            auxdoc_lines = auxdoc.strip().split('\n')
            soracle_lines = soracle.strip().split('\n')
            moracle_lines = moracle.strip().split('\n')
            highlights_lines = highlights.strip().split('\n')

            filename = doc_lines[0].strip()

            if (filename == auxdoc_lines[0].strip()) and \
               (filename == soracle_lines[0].strip()) and \
               (filename == moracle_lines[0].strip()) and \
               (filename == highlights_lines[0].strip()):
                self.filenames.append(filename)

                # Document
                thisdoc = []
                for line in doc_lines[1:args.max_doc_length+1]:
                    thissent = [int(item) for item in line.strip().split()]
                    thisdoc.append(thissent)
                self._docs.append(thisdoc)

                # Auxiliary Document
                thisdoc = []
                for line in auxdoc_lines[1:args.max_doc_length+1]:
                    thissent = [int(item) for item in line.strip().split()]
                    thisdoc.append(thissent)
                self._auxdocs.append(thisdoc)

                # Single Oracle
                thissoracle = []
                for line in soracle_lines[1:args.max_doc_length+1]:
                    thissoracle.append(int(line.strip()))
                self._singlelabels.append(thissoracle)

                # Multiple Oracle
                thismoracle = []
                thisreward = []
                for line in moracle_lines[2:args.num_sample_rollout+2]:
                    thismoracle.append([int(item) for item in line.strip().split()[:-1]])
                    thisreward.append([float(line.strip().split()[-1])])
                self._multiplelabels.append(thismoracle)
                self._rewards.append(thisreward)

                # Highlights
                thishighlights = []
                for line in highlights_lines[1:args.max_doc_length+1]:
                    thishighlights.append([int(item) for item in line.strip().split()])
                self._highlights.append(thishighlights)

                # Weight
                originaldoclen = int(moracle_lines[1].strip())
                thisweight = [1 for _ in range(originaldoclen)][:args.max_doc_length]
                self._weights.append(thisweight)

            else:
                print("Some problem with %s.* files. Exiting!" % file_prefix)
                exit(0)

            if doccount%10000==0:
                print("%d ..."%doccount)
            doccount += 1

        self.fileindices = [i for i in range(len(self.filenames))]


if __name__ == '__main__':
    data = Data()
    data[5]
