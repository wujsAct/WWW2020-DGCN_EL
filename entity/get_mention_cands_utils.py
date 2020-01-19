# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
import codecs
import re
from collections import defaultdict
def get_wiki_redirect():
  wiki_2_redir_title={}

  with codecs.open('data/deep-ed-data/wiki_redirects.txt','r','utf-8') as file_:
    for line in tqdm(file_):
      line = line[:-1]
      items = line.split('\t')
      if len(items)>=2:
        title=items[0]
        redir_title = items[1]
        wiki_2_redir_title[title] = redir_title

  return wiki_2_redir_title

def get_disam_wiki_id_name():
  wiki_name_2_id = {}
  wiki_id_2_name = {}
  with codecs.open('data/deep-ed-data/wiki_disambiguation_pages.txt','r','utf-8') as file_:
    for line in tqdm(file_):
      line = line[:-1].replace('(disambiguation)','')
      items = line.split('\t')
      ids = items[0]
      name = items[1]
      wiki_name_2_id[name] = ids
      wiki_id_2_name[ids]=name

  with codecs.open('data/deep-ed-data/wiki_name_id_map_from_corpus.txt','r','utf-8') as file_:
    for line in tqdm(file_):
      line = line[:-1]
      items = line.split('\t')
      name = items[0]
      ids = items[1]
      wiki_name_2_id[name] = ids
      wiki_id_2_name[ids]=name

  with codecs.open('data/deep-ed-data/wiki_name_id_map.txt','r','utf-8') as file_:
    for line in tqdm(file_):
      line = line[:-1]
      items = line.split('\t')
      name = items[0]; ids = items[1]
      wiki_name_2_id[name] = ids
      wiki_id_2_name[ids]=name

  return wiki_name_2_id,wiki_id_2_name


def get_sid_aNosNo(dir_path):
  sid = 0
  sid2aNosNo= {}
  aNosNo2id = {}
  with open(dir_path+'process/sentid2aNosNoid.txt') as file_:
    for line in file_:
      aNosNo = line.strip()
      sid2aNosNo[sid] = aNosNo
      aNosNo2id[aNosNo] = str(sid)
      sid += 1
  return sid2aNosNo,aNosNo2id


def get_aid2ents(dir_path):
  aid2ents={}
  with codecs.open(dir_path+'process/'+'entMen2aNosNoid.txt','r','utf8') as file_:
    for line in file_:
      line = line.strip()
      items = line.split('\t')
      aNosNo = items[1]
      aNo = int(aNosNo.split('_')[0])
      if aNo not in aid2ents:
        aid2ents[aNo] = []
      aid2ents[aNo].append(line)
  return aid2ents

def preprocess_gold_mention(mention):
  mention = mention
  pattern = re.compile(r'\b\w+\b')
  match = pattern.finditer(mention)

  new_line = ''
  for m in match:
    s= m.start(0)
    e = m.end(0)
    new_line +=( mention[s] + mention[s+1:e]+' ')

  return new_line.strip().lower()

def modify_uppercase_phrase(mention):
  if mention==mention.upper():
    mention = mention.lower()
    pattern = re.compile(r'\b\w+\b')
    match = pattern.finditer(mention)

    new_line =  []
    for m in match:
      s= m.start(0)

      e = m.end(0)
      new_line.append(mention[s].upper() + mention[s+1:e])

    return ' '.join(new_line)
  else:
    return mention

def get_mention_cands(mongo_utils,mention):
  #attention to the logistic inherent...
  cur_m = modify_uppercase_phrase(mention)

  cur_ret_cands,cur_freq = mongo_utils.search_mention(cur_m)

  if len(cur_ret_cands)==0:
    cur_m=mention

  if mention in mongo_utils.mention_total_freq and mongo_utils.mention_total_freq[mention] > mongo_utils.mention_total_freq[cur_m]:
    cur_m= mention

  cur_ret_cands,cur_freq = mongo_utils.search_mention(cur_m)

  low_m = mention.lower()
  if len(cur_ret_cands)==0 and low_m in mongo_utils.lower2normal:
    cur_m = mongo_utils.lower2normal[low_m]

  cur_ret_cands,cur_freq = mongo_utils.search_mention(cur_m)

  return cur_m,cur_ret_cands
