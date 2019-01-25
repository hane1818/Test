rm -r $1/*;
mkdir $1;
python train.py --pretrained_wordembedding ~/Desktop/Corpus/CNN-DailyMail/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec \
--pretrained_aux_wordembedding ~/Desktop/Corpus/CNN-DailyMail/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec \
--preprocessed_data_directory ~/Desktop/Corpus/CNN-DailyMail/preprocessed-input-directory \
--gold_summary_directory ~/Desktop/Corpus/CNN-DailyMail/Baseline-Gold-Models \
--doc_sentence_directory ~/Desktop/Corpus/CNN-DailyMail/CNN-DM-Filtered-TokenizedSegmented \
--data_mode cnn \
--rouge_path ~/ROUGE-RELEASE-1.5.5 \
--rouge_data_directory ~/ROUGE-RELEASE-1.5.5/data \
--train_epoches 200 \
--aux_embedding false \
--bidirectional false \
--doc_encoder_reverse true \
--rl true \
--num_sample_rollout 5 \
--size 600 \
--sentembed_size 350 \
--max_doc_length 110 \
--max_sent_length 100 \
--rnn_cell lstm \
--dropout 0 \
--train_dir $1 #> $1/train.log;
