# -*- coding: utf-8 -*-
"""
#get coref person...
"""

import codecs
from tqdm import tqdm
import re
from decimal import Decimal

def get_float_decimal(float_number):
  strs = str(float_number)

  strs = '{:.3f}'.format(Decimal(strs))
  return float(strs)

def gen_ent_person(wiki_2_redir_title,wiki_name_2_id):
  ent_person = {}

  with codecs.open('data/deep-ed-data/persons.txt','r','utf-8') as file_:
    for line in tqdm(file_):
      line = line.strip()
      ent = line
      if ent in wiki_2_redir_title:
        ent = wiki_2_redir_title[ent]

      if ent in wiki_name_2_id:
        ent_person[wiki_name_2_id[ent]]=1
      else:
        continue
  return ent_person


def get_cands_from_relateness(tag,ent2id):
  ids = len(ent2id)

  with open('data/deep-ed-data/relatedness/'+tag+'.svm') as file_:
    for line in file_:
      line = line.strip()
      items = line.split(' ')

      ents = items[-2]
      e1,e2 = ents.split('-')
      if e1 not in ent2id:
        ent2id[e1] = ids
        ids += 1
      if e2 not in ent2id:
        ent2id[e2] = ids
        ids += 1
  print(tag,ids)
  return ent2id

def is_match(mention,strs):
  mention=mention.lower()
  strs = strs.lower()
  try:
    mention = mention.replace('.','\.').replace('*','\*')
    mention = mention.replace('(','\(')
    mention = mention.replace(')','\)')
    pattern1 = re.compile('\s'+mention)
    pattern2 = re.compile(mention+'\s')
  except:
    print(mention,' string change is wrong....')
  if len(re.findall(pattern1,strs))!=0 or len(re.findall(pattern2,strs))!=0:
    return True
  else:
    return False

def mention_refers_to_person(ent_person,ment_cands):
  top_p=0.0
  top_ent=-1

  for cand_wiki_id in ment_cands:
    items = ment_cands[cand_wiki_id]

    p_e_m=float(items[0])

    if p_e_m > top_p:
      top_p = p_e_m
      top_ent = cand_wiki_id

  if top_ent in ent_person:
    return True
  else:
    return False

def get_coref_person(mention,ment_cands,ent_person,mentions_aNo):
  num_added_mentions = 0
  coref_ment = mention
  for ment in mentions_aNo:
    if is_match(mention,ment) and mention_refers_to_person(ent_person,ment_cands):
#      print(mention)
#      print(ment)
#      print(mention_refers_to_person(ment))
#      print('----------------')
      coref_ment_cands = mentions_aNo[ment].split('\t')

      num_added_mentions += 1
      coref_ment=ment
      for cand_item in coref_ment_cands:
        cand_item_list =cand_item.split(',')
        key=cand_item_list[0]
        coref_key_score = float(cand_item_list[1])
        name = cand_item_list[2]

        if key in ment_cands:
          ment_cands[key][0] += coref_key_score
        else:
          ment_cands[key]=[]
          ment_cands[key].append(coref_key_score)
          ment_cands[key].append(name)

  if num_added_mentions >1:
    for key in ment_cands:
      float_number=ment_cands[key][0]/num_added_mentions
      ment_cands[key][0]=get_float_decimal(float_number)


  return coref_ment,ment_cands
