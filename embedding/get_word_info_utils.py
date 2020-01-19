#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sys import version_info
version = version_info.major

if version ==2:
  from is_stop_or_number import is_stop_or_number
else:
  from .is_stop_or_number import is_stop_or_number


import random
import numpy as np
import pickle
import argparse

class WordFreqVectorLoader(object):
  def __init__(self,args):
    self.args = args
    print('load word infos...')
    if version==2:
      ret_param = pickle.load(open('data/deep-ed-data/word_info.p','rb'))
    else:
      ret_param = pickle.load(open('data/deep-ed-data/word_info.p','rb'),
                              encoding='iso-8859-1')

    self.word2id =ret_param['word2id']
    self.id2word = {self.word2id[wd]:wd for wd in self.word2id}
    self.word_embedding=ret_param['word_embedding']

    self.w_f_start = ret_param['w_f_start']
    self.w_f_end = ret_param['w_f_end']
    self.total_freq=ret_param['total_freq']

    self.w_f_start_at_unig_power = ret_param['w_f_start_at_unig_power']
    self.w_f_end_at_unig_power = ret_param['w_f_end_at_unig_power']
    self.total_freq_at_unig_power=ret_param['total_freq_at_unig_power']

    self.total_num_words = ret_param['total_num_words']
    print('finish build WordFreqVectorLoader')

  def is_in_vocabs(self,w_id):
    if w_id >=0 and w_id <= self.total_num_words:
      return w_id
    else:
      return 0

  def get_wd_in_wdict(self,word):
    rel_wd =None
    #detail about data process...
    if is_stop_or_number(word)==False and word!='':
      if word.isupper():   #this may be efficient~
        word = word.capitalize()

      if word in self.word2id:
        rel_wd = self.word2id[word]

    return rel_wd

  # word frequency
  def get_word_freq(self,w_id):
    assert (w_id==self.is_in_vocabs)
    return self.w_f_end[w_id]-self.w_f_start[w_id] + 1

  #p(w) prior
  def get_word_prior(self,w_id):
    return self.get_word_freq(w_id)/self.total_freq

  #Generates an random word sampled from the word unigram frequency.
  def random_w_id(self,total_freq,w_f_start,w_f_end):
    i_start = 1
    i_end = self.total_num_words

    j = random.random()*total_freq

    #get the entity ids by using binary search
    while i_start <= i_end:
      i_mid = (i_start+i_end)/2

      if w_f_start[i_mid] <= j and j <= w_f_end[i_mid]:
        return i_mid
      elif w_f_start[i_mid] >j:
        i_end = i_mid - 1
      elif w_f_end[i_mid] < j:
        i_start = i_mid +1
    return -1

  def random_w_id_unig_power_pos(self):
    ret_w = []
    word_num = 0
    while word_num<self.args.pos_num:
      rand_w = self.random_w_id(self.total_freq_at_unig_power,self.w_f_start_at_unig_power,
                                self.w_f_end_at_unig_power)

      if rand_w!=-1:
        ret_w.append(rand_w)
        word_num += 1
    return ret_w


  def random_w_id_unig_power(self,pos_wds):
    ret_w = []
    neg_word_num = 0
    while neg_word_num<self.args.neg_num*self.args.pos_num:
      rand_w = self.random_w_id(self.total_freq_at_unig_power,self.w_f_start_at_unig_power,self.w_f_end_at_unig_power)

      if rand_w!=-1 and rand_w not in pos_wds:
        ret_w.append(rand_w)
        neg_word_num += 1

    return np.reshape(ret_w,[self.args.pos_num,self.args.neg_num])

if __name__=='__main__':
    #test the module...
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',type=float,default=0.01)
    parser.add_argument('--pos_num',type=int,default=20)
    parser.add_argument('--neg_num',type=int,default=5)

    args=parser.parse_args()

    w_freq_loader = WordFreqVectorLoader(args)
    pos_w = [10]
    print(w_freq_loader.random_w_id_unig_power(pos_w))
    print(w_freq_loader.random_w_id_unig_power(pos_w))
    print(w_freq_loader.random_w_id_unig_power(pos_w) )

    print(w_freq_loader.get_wd_in_wdict('American'))
