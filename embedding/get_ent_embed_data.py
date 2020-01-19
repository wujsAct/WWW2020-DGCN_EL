#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sys import version_info
version = version_info.major
if version==2:
  from get_word_info_utils import WordFreqVectorLoader
else:
  from .get_word_info_utils import WordFreqVectorLoader

import pickle
import numpy as np
import argparse
import random
from tqdm import tqdm
import time

def gen_pos_neg_wds(ent_2_wiki_canonical,pos_num,word_freq_loader):
  if ent_id in ent_2_wiki_canonical:
    wds_list = ent_2_wiki_canonical[ent_id]
    pos_wds=np.random.choice(wds_list,pos_num)
    neg_wds = word_freq_loader.random_w_id_unig_power(set(pos_wds))
  else:
    pos_wds = word_freq_loader.random_w_id_unig_power_pos()
    neg_wds = word_freq_loader.random_w_id_unig_power(set(pos_wds))
  return [ent_id,pos_num,neg_wds]


class EntEmbedUtils(object):
  def __init__(self,args):
    self.args = args
    self.word_freq_loader=WordFreqVectorLoader(args)
    self.ent2id = pickle.load(open('data/intermediate/ent2id.p'))
    self.ent_size = len(self.ent2id)
    #self.ent_2_wiki_hyperlink = self.get_ent_2_wiki_hyperlink_info()
    self.ent_2_wiki_canonical = self.get_ent_2_wiki_canonical_info()

  def get_ent_2_wiki_canonical_info(self):
    print('load canonical infos..')
    ent_2_wiki_canonical = {}
    with open('data/intermediate/wiki_canonical_words_RLTD.txt') as file_:
      for line in tqdm(file_):
        items = line.split(',')
        ent_id = items[0]

        wds_list = map(int,items[1].split(' '))
        ent_2_wiki_canonical[ent_id] = wds_list

    return ent_2_wiki_canonical

  def gen_train_iter_data(self,iter_epoch):
    if iter_epoch > 200:
      for ent_id_batch,pos_words_batch,neg_words_batch in self.gen_train_iter_data_hyperlink():
        yield ent_id_batch,pos_words_batch,neg_words_batch
    else:
      for ent_id_batch,pos_words_batch,neg_words_batch in self.gen_train_iter_data_canonical():
        yield ent_id_batch,pos_words_batch,neg_words_batch

  def gen_train_iter_data_canonical(self):
    sample_no = 0
    ent_id_batch = []
    pos_words_batch = []
    neg_words_batch = []

    ent_id_list = self.ent2id.keys()
    random.shuffle(ent_id_list)


    for ent_id in ent_id_list:
      if ent_id in self.ent_2_wiki_canonical:
        #print('1 ent_id in ent_2_wiki_canonical:',ent_id)
        wds_list = self.ent_2_wiki_canonical[ent_id]
        pos_wds=np.random.choice(wds_list,self.args.pos_num)
        neg_wds = self.word_freq_loader.random_w_id_unig_power(set(pos_wds))

      else:
        #print('2 ent_id not in ent_2_wiki_canonical:',ent_id)
        pos_wds = self.word_freq_loader.random_w_id_unig_power_pos()
        neg_wds = self.word_freq_loader.random_w_id_unig_power(set(pos_wds))

      ent_id_batch.append(self.ent2id[ent_id])
      pos_words_batch.append(pos_wds)
      neg_words_batch.append(neg_wds)

      sample_no += 1
      if sample_no % self.args.batch_size==0:
        yield ent_id_batch,pos_words_batch,neg_words_batch

        ent_id_batch = []
        pos_words_batch=[]
        neg_words_batch=[]

  def gen_train_iter_data_hyperlink(self):
    sample_no = 0
    ent_id_batch = []
    pos_words_batch = []
    neg_words_batch = []

    fname= 'data/intermediate/wiki_hyperlink_words_RLTD.txt'

    with open(fname) as file_:
      for line in tqdm(file_):

        line = line.strip()
        items = line.split(',')
        ent_id = items[0]

        wds_list = items[1].split(' ')

        if len(wds_list)==0:
          continue
        else:
          try:
            wds_list = map(int,wds_list)
          except:
            print(wds_list)
            continue
        pos_wds=np.random.choice(wds_list,self.args.pos_num)
        neg_wds = self.word_freq_loader.random_w_id_unig_power(set(wds_list))

        if self.ent2id[ent_id]==None:
          continue

        ent_id_batch.append(self.ent2id[ent_id])
        pos_words_batch.append(pos_wds)
        neg_words_batch.append(neg_wds)


        sample_no += 1
        if sample_no % self.args.batch_size==0:
          yield ent_id_batch,pos_words_batch,neg_words_batch

          ent_id_batch = []
          pos_words_batch=[]
          neg_words_batch=[]
          sample_no=0
  '''
  def gen_train_iter_data_hyperlink(self):
    sample_no = 0
    ent_id_batch = []
    pos_words_batch = []
    neg_words_batch = []

    ent_id_list = self.ent2id.keys()
    random.shuffle(ent_id_list)

    for ent_id in ent_id_list:
      if ent_id in self.ent_2_wiki_hyperlink:
        #print('1 ent_id in ent_2_wiki_canonical:',ent_id)
        wds_list = self.ent_2_wiki_hyperlink[ent_id]
        pos_wds=np.random.choice(wds_list,self.args.pos_num)
        neg_wds = self.word_freq_loader.random_w_id_unig_power(set(pos_wds))
      else:
        #print('2 ent_id not in ent_2_wiki_canonical:',ent_id)
        pos_wds = self.word_freq_loader.random_w_id_unig_power_pos()
        neg_wds = self.word_freq_loader.random_w_id_unig_power(set(pos_wds))

      ent_id_batch.append(self.ent2id[ent_id])
      pos_words_batch.append(pos_wds)
      neg_words_batch.append(neg_wds)


      sample_no += 1
      if sample_no % self.args.batch_size==0:
        yield ent_id_batch,pos_words_batch,neg_words_batch

        ent_id_batch = []
        pos_words_batch=[]
        neg_words_batch=[]'''

if __name__=='__main__':
    #test the module...
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',type=float,default=0.1)
    parser.add_argument('--wd_embed_dims',type=int,default=300)
    parser.add_argument('--entity_embed_dims',type=int,default=300)
    parser.add_argument('--pos_num',type=int,default=20)
    parser.add_argument('--neg_num',type=int,default=5)
    parser.add_argument('--batch_size',type=int,default=500)
    parser.add_argument('--is_test',action='store_true') #store is true, and no store is False

    args=parser.parse_args()
    dataUtils = EntEmbedUtils(args)
    stime = time.time()
    for ent_id,pos_words,neg_words in dataUtils.gen_train_iter_data(1):
      print(ent_id)
    print('cost time:',time.time()-stime)
