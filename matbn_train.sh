rm -r $1;
mkdir $1;
python train.py --pretrained_wordembedding data/CNA_Skip_vectors.txt \
--pretrained_aux_wordembedding data/CNA_Skip_vectors_char_200.txt \
--preprocessed_data_directory data/preprocessed \
--gold_summary_directory data/gold-summary \
--doc_sentence_directory data/Human_TD \
--data_mode matbn-human \
--rouge_path /Users/hane/ROUGE-RELEASE-1.5.5 \
--rouge_data_directory /Users/hane/ROUGE-RELEASE-1.5.5/data \
--train_epoches 200 \
--doc_encoder_reverse \
--attn \
--coverage \
--aux_embedding \
--num_sample_rollout 5 \
--train_dir $1 > $1/train.log;
#--aux_embedding \
#--attn \
#--coverage \
#--rl \
#--bidirectional \
