#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 04:59:05 2018

@author: wujs
"""
from utils import is_stop_or_number
import cPickle
import time
import gensim
import numpy as np
from tqdm import tqdm

stime = time.time()
print('load Google News word entity...')
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
vocabs = model.vocab
print('utilize time:',time.time()-stime,'to finish load all words embeddings...')
unig_power = 0.6
id2word = {}
word2id = {}
word_embedding=[]
w_f_start = {}
w_f_end = {}
total_freq=0.0

w_f_start_at_unig_power = {}
w_f_end_at_unig_power = {}
total_freq_at_unig_power=0.0

word2id['UNK_PAD'] = 0
id2word[0] = 'UNK_PAD'
word_embedding.append(np.zeros((300,)))

word2id['UNK_DEL'] = 1
id2word[1] = 'UNK_DEL'
word_embedding.append(np.random.normal(0,1.0,(300,)))

word_embed_file=open('data/intermediate/word_embedding.txt','wb')

w_id = 1
with open('data/deep-ep-data/word_wiki_freq.txt') as file_:
  for line in tqdm(file_):
    line = line[:-1]
    
    word,freq = line.split('\t')
    freq = float(freq)
    if is_stop_or_number(word)==False:
      if word in vocabs:
        w_id += 1
        id2word[w_id] = word
        word2id[word] = w_id
        word_embedding.append(model[word])
        if freq < 100: 
          freq = 100
        w_f_start[w_id] = total_freq
        total_freq = total_freq + freq
        w_f_end[w_id] = total_freq
        
        
        w_f_start_at_unig_power[w_id] = total_freq_at_unig_power
        total_freq_at_unig_power = total_freq_at_unig_power + np.power(freq,unig_power)
        w_f_end_at_unig_power[w_id] = total_freq_at_unig_power
      
total_num_words = w_id
word_embedding = np.asarray(word_embedding,np.float32)

ret_param = {'word2id':word2id,'w_f_start':w_f_start,'w_f_end':w_f_end,
             'w_f_start_at_unig_power':w_f_start_at_unig_power,
             'w_f_end_at_unig_power':w_f_end_at_unig_power,'total_freq_at_unig_power':total_freq_at_unig_power,
             'word_embedding':word_embedding,'total_num_words':total_num_words,'total_freq':total_freq
             }
print('start to save:',len(word2id))
stime = time.time()
cPickle.dump(ret_param,open('data/deep-ep-data/word_info.p','w'),True)
print('cost time:',time.time()-stime)

for w_id in range(len(id2word)):
  word_embed_file.write(id2word[w_id]+' '+' '.join(map(str,word_embedding[w_id]))+'\n')
  word_embed_file.flush()

word_embed_file.close()
#stime = time.time()
#ret_param = cPickle.load(open('data/deep-ep-data/word_info.p'))
#print('cost time:',time.time()-stime)