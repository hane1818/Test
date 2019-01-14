import os
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.parse import CoreNLPParser

#os.environ["CLASSPATH"] = "/share/homes/hane/stanford/jars"
#os.environ["STANFORD_MODELS"] = "/share/homes/hane/stanford/models"

sent = "Everyone is entitled to all the rights and freedoms set forth in this Declaration, without distinction of any kind, such as race, colour, sex, language, religion, political or other opinion, national or social origin, property, birth or other status. From UN"

ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
print(list(ner_tagger.tag(sent.split())))

top_dir = 'Dataset'
directory = ['cnn', 'dailymail']
preprocessed_data = ['data/preprocessed-input-directory/cnn-dailymail.{}.doc'.format(i) for i in ['training', 'test', 'validation']]
feature_dict = {}
feature_data = ['data/preprocessed-input-directory/cnn-dailymail.{}.acou'.format(i) for i in ['training', 'test', 'validation']]

with open('data/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec') as f:
    vocab = [line.strip().split()[0] for line in f.read().strip().split('\n')]
    vocab = ['_PAD', '_UNK'] + vocab

for pd, fd in zip(preprocessed_data, feature_data):
    fea_file = open(fd, "w")
    with open(pd) as f:
        docs = f.read().strip().split('\n\n')
        for doc in docs:
            sents = doc.strip().split('\n')
            name = sents[0].strip()
            sents = sents[1:]
            feature_dict[name] = []
            fea_file.write(name)
            fea_file.write('\n')
            """with open(os.path.join(top_dir, '/'.join(name.strip().split('-')))+'.story') as d:
                main_doc = d.read().strip().split('@highlight')[0]
                main_doc = sent_tokenize(' '.join(main_doc.split()))
                assert len(sents) == len(main_doc)"""
            for i, sen in enumerate(sents):
                feature_dict[name].append([])
                feature_dict[name][-1].append(str(i)) # absolute position
                feature_dict[name][-1].append(str(i/len(sents))) # relative position
                words = sen.split()
                feature_dict[name][-1].append(str(len(words)))
                words = [vocab[int(w)] for w in words]
                #named_entities = [ne for _, ne in list(ner_tagger.tag(words)) if ne != 'O']
                #feature_dict[name][-1].append(str(len(named_entities)))
                fea_file.write(' '.join(feature_dict[name][-1]))
                fea_file.write('\n')
                fea_file.write('\n')
            print(name)
    fea_file.close()

with open("lexical_feature.json", "w") as f:
    json.dump(feature_dict, f)

