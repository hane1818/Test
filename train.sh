rm -r $1/*;
mkdir $1;
python train.py --pretrained_wordembedding cnn-dailymail/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec \
--pretrained_aux_wordembedding cnn-dailymail/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec \
--preprocessed_data_directory cnn-dailymail/preprocessed-input-directory \
--gold_summary_directory cnn-dailymail/Baseline-Gold-Models \
--doc_sentence_directory cnn-dailymail/CNN-DM-Filtered-TokenizedSegmented \
--data_mode dailymail \
--rouge_path /share/homes/hane/ROUGE-RELEASE-1.5.5 \
--rouge_data_directory /share/homes/hane/ROUGE-RELEASE-1.5.5/data \
--train_epoches 200 \
--aux_embedding false \
--bidirectional false \
--doc_encoder_reverse true \
--rl true \
--num_sample_rollout 10 \
--size 600 \
--sentembed_size 350 \
--max_doc_length 110 \
--max_sent_length 100 \
--rnn_cell lstm \
--train_dir $1 > $1/train.log;
