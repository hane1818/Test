# coding=UTF-8
import argparse

parser = argparse.ArgumentParser()

## Global settings
parser.add_argument('--exp_mode', type=str, default='train')
parser.add_argument('--model_to_load', type=int)
parser.add_argument('--data_mode', type=str, default='cnn')
parser.add_argument('--use_gpu', type=bool, default=True)

## Word level features
parser.add_argument('--embedding_size', type=int, default=200)
parser.add_argument('--trainable_wordembed', type=bool, default=False)

## Sentence level features
parser.add_argument('--max_sent_length', type=int, default=30)
parser.add_argument('--sentembed_size', type=int, default=175)
parser.add_argument('--dropout', type=float, default=0.5)

## Document level features
parser.add_argument('--max_doc_length', type=int, default=50)
parser.add_argument('--target_label_size', type=int, default=2)

## CNN features
parser.add_argument('--max_filter_length', type=int, default=7)

## RNN features
parser.add_argument('--size', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--rnn_cell', type=str, default='gru')
parser.add_argument('--bidirectional', type=bool, default=False)

## Encoder Layer features
# Sentence Encoder
parser.add_argument('--aux_embedding', type=bool, default=False)

# Document Encoder
parser.add_argument('--doc_encoder_reverse', type=bool, default=True)

# Extractor

## Reinforcement Learning
parser.add_argument('--rl', type=bool, default=False)
parser.add_argument('--num_sample_rollout', type=int, default=10)

## Maximal Marginal Relevance
parser.add_argument('--mmr', type=bool, default=False)
parser.add_argument('--lambda', type=float, default=0.7)
parser.add_argument('--max_sum_len', type=int, default=10)

## Pretrain features
parser.add_argument('--pretrain_dir', type=str, default='/address/to/pretrain/directory')
parser.add_argument('--pretrain_lr_G', type=float, default=0.001)
parser.add_argument('--pretrain_lr_D', type=float, default=0.001)

## Training features
parser.add_argument('--train_dir', type=str, default='/address/to/training/directory')
parser.add_argument('--train_epoches', type=int, default=30)
parser.add_argument('--learning_rate_G', type=float, default=0.001)
parser.add_argument('--learning_rate_D', type=float, default=0.001)
parser.add_argument('--weighted_loss', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--training_checkpoint', type=int, default=1)
parser.add_argument('--summary_ratio', type=float, default=1)

#### Input file addresses

# Pretrained word embedding
parser.add_argument('--pretrained_wordembedding', type=str, default='/address/to/pretrained-wordembedding')
parser.add_argument('--pretrained_aux_wordembedding', type=str, default='/address/to/pretrained-aux-wordembedding')

# Data directory address
parser.add_argument('--preprocessed_data_directory', type=str, default='/address/to/preprocessed-data-directory')
parser.add_argument('--gold_summary_directory', type=str, default='/address/to/gold-summary-directory')
parser.add_argument('--doc_sentence_directory', type=str, default='/address/to/doc-sentence-directory')

#### ROUGE file addresses
parser.add_argument('--rouge_path', type=str, default='/address/to/rouge-file-directory')
parser.add_argument('--rouge_data_directory', type=str, default='/address/to/rouge-data-directory')

args = parser.parse_args()
