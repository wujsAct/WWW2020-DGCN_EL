# -*- coding: utf-8 -*-

from tqdm import tqdm
import pymongo
from pymongo import MongoClient
from collections import defaultdict
import re
import numpy as np
import codecs

def get_float_decimal(float_number):
  strs = str(float_number)

  strs = '{:.3f}'.format(Decimal(strs))
  return float(strs)

def preprocess_mention(mention):
  mention = mention
  pattern = re.compile(r'\b\w+\b')
  match = pattern.finditer(mention)

  new_line = ''
  for m in match:
    s= m.start(0)

    e = m.end(0)

    new_line +=( mention[s] + mention[s+1:e]+' ')

  return new_line.strip().lower()

def get_lower_2_normal_dict():
  lower2normal={}
  mention_total_freq={}
  with codecs.open('data/deep-ed-data/crosswikis_wikipedia_p_e_m.txt') as file_:
    for line in tqdm(file_):
      line = line.strip()
      items = line.split('\t')
      try:
        ment_name  = unicode(items[0],'utf-8')
        lower2normal[ment_name.lower()]=ment_name
        freq = int(items[1])

        if ment_name not in mention_total_freq:
          mention_total_freq[ment_name]=0

        mention_total_freq[ment_name] += freq
      except:
        print(items, 'is wrong')

      #mention_total_freq[ment_name]+=freq

  with codecs.open('data/deep-ed-data/yago_p_e_m.txt') as file_:
      for line in tqdm(file_):
        line = line.strip()
        items = line.split('\t')
        try:
          ment_name  = unicode(items[0],'utf-8')
          freq = int(items[1])
          lower2normal[ment_name.lower()]=ment_name
          if ment_name not in mention_total_freq:
            mention_total_freq[ment_name]=0
          mention_total_freq[ment_name]+=freq
        except:
          print(items, 'is wrong')

  return lower2normal,mention_total_freq

class mongoUtils(object):
  def __init__(self,db_name):
    client = MongoClient('mongodb://192.168.3.196:27017')
    #print client
    self.db = client[db_name] # database name
    self.collection_wiki = self.db["crosswikis_wikipedia_p_e_m"]  # collection wiki
    self.collection_yago = self.db["yago_p_e_m"]  #collection yago

    self.lower2normal, self.mention_total_freq= get_lower_2_normal_dict()

  def insert_cross_wiki_record(self):
    with codecs.open('data/deep-ed-data/crosswikis_wikipedia_p_e_m.txt') as file_:
      for line in tqdm(file_):
        line = line.strip()

        items = line.split('\t')

        ment_name  = unicode(items[0],'utf-8')
        freq = items[1]
        linking_wikis_name = '\t'.join(items[2:])

        insert_data = {'mention':ment_name,'total_freq':freq,'linking_ents':linking_wikis_name}

        self.collection_wiki.insert_one(insert_data)

  def insert_yago_record(self):
    with open('data/deep-ed-data/yago_p_e_m.txt') as file_:
      for line in tqdm(file_):
        line = line.strip()

        items = line.split('\t')

        ment_name  = items[0]
        freq = items[1]
        linking_wikis_name = '\t'.join(items[2:])
        insert_data = {'mention':ment_name,'total_freq':freq,'linking_ents':linking_wikis_name}
        self.collection_yago.insert_one(insert_data)
  def add_index(self):

    self.collection_wiki.create_index([("mention",pymongo.HASHED)])
    self.collection_yago.create_index([("mention", pymongo.HASHED )])


  '''
  @funtion: utilize mention string to search candidate entities
  @input: mention
  @output:candidate entities [wiki_title_1,...,]
  '''
  def search_mention(self,ment_str):
    cand_ents_freq = defaultdict(int)

    cand_ents = defaultdict(float)
    total_freq = 0.0
    for wiki_items in self.collection_wiki.find({"mention":ment_str}):
      if wiki_items!=None:
        wiki_total_freq = float(wiki_items['total_freq'])
        total_freq += wiki_total_freq
        wiki_linking_ents = wiki_items['linking_ents']
        for enti in wiki_linking_ents.split('\t'):
          enti_item = enti.split(',')
          if len(enti_item)>=3:
            key = enti_item[0]
            freq= enti_item[1]

            cand_ents_freq[key] += int(freq)

    for key in cand_ents_freq:
      freq = cand_ents_freq[key]
      cand_ents[key]=freq/total_freq

    for yago_items in self.collection_yago.find({"mention": ment_str}):
      if yago_items!=None:
        yago_total_freq = float(yago_items['total_freq'])
        total_freq += yago_total_freq
        ygao_linking_ents = yago_items['linking_ents']
        for enti in ygao_linking_ents.split('\t'):
          enti_item = enti.split(',')
          if len(enti_item)>=2:
            key = enti_item[0]
            cand_ents[key] = min(1.0,cand_ents[key]+1/yago_total_freq)

    for key in cand_ents:
      cand_ents[key]=get_float_decimal(cand_ents[key])

    return cand_ents,total_freq

if __name__=="__main__":
  mongo_utils = mongoUtils('wiki')
  #mongo_utils.insert_yago_record()
  #mongo_utils.insert_cross_wiki_record()
  
  # # create index after inserted the dataset...
  mongo_utils.add_index() 
