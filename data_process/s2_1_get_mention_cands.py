# -*- coding: utf-8 -*-
import sys
sys.path.append('Project Absolute Path')
from entity import mongoUtils,get_disam_wiki_id_name,get_sid_aNosNo,get_mention_cands,get_wiki_redirect
import numpy as np
from tqdm import tqdm
import codecs
import cPickle
import re
import os

def get_entMents(mongo_utils,dataset,dir_path,aNosNo2id,wiki_name_2_id,wiki_2_redir_title):
  cand_ent_nums=[]
  gold_nil_kg = 0
  all_ents = 0
  gold_not_in_wiki =0
  no_candidate_ents=0
  entMents = {}
  with codecs.open(dir_path+'process/'+'entMen2aNosNoid.txt','r','utf-8') as file_:
    for line in tqdm(file_):
      line = line.strip().split('\t\t')[0]

      items = line.split('\t')
      mention = items[0]
      aNosNo = items[1]
      start = int(items[2])
      end = int(items[3])
      gold_ent = ""
      sent_id = aNosNo2id[aNosNo]
      '''
      @we need to transfer gold wiki entity into wiki ids
      '''
      gold_wiki_entity = None
      gold_ent = None
      #gold_ent is the wiki title
      if dataset == 'aida':
        gold_ent = items[4].replace('_',' ')
      else:
        gold_ent = items[4]

      #KBP NIL pattern
      pattern = re.compile('NIL\d+')

      if gold_ent.lower() == 'nil' or gold_ent.lower()=='' or len(re.findall(pattern,gold_ent))==1:
        gold_nil_kg += 1
        gold_ent = 'NIL'
      else:
        all_ents += 1
        if gold_ent in wiki_2_redir_title:
          gold_ent = wiki_2_redir_title[gold_ent]
          gold_wiki_entity='NIL'

        if gold_ent in wiki_name_2_id:
          gold_wiki_entity = wiki_name_2_id[gold_ent]
        else:
          gold_not_in_wiki+=1
          print('gold_not_in_wiki:',gold_ent)
          gold_wiki_entity='NIL'

      mention,candidate_wiki_entity = get_mention_cands(mongo_utils,mention)
      '''
      @revise: we delay filter when feed in the model ...
      '''
      org_candidate_wiki_entity_sorted = sorted(candidate_wiki_entity.iteritems(), key=lambda d:d[1], reverse = True)
      #we reserve the top 30~~
      #candidate_wiki_entity_sorted = get_coref_person(mention,candidate_wiki_entity,mentions_aNo)

      has_gold_cand = False
      for item_ents in org_candidate_wiki_entity_sorted[:30]:
        if gold_wiki_entity == item_ents[0]:
          has_gold_cand = True
          break

      if has_gold_cand==False and gold_ent!='NIL':
        no_candidate_ents += 1

      #Entity mentions with candidates are utilized to train the model
      cand_ent_nums.append(len(org_candidate_wiki_entity_sorted[:30]))
      mention_items = {'mention':mention,
                       'start':start,
                       'end':end,
                       'org_cands':org_candidate_wiki_entity_sorted,
                       'coref_cands':None,
                       'gold_ent':gold_wiki_entity}
      if sent_id not in entMents:
        entMents[sent_id]=[]

      entMents[sent_id].append(mention_items)

#      if has_gold_cand==True and gold_ent!='NIL':
#        print(mention)
#        print(gold_ent,gold_wiki_entity)
#        print(candidate_wiki_entity_sorted)
#        print('---------------------')
  recall_ents=all_ents-gold_not_in_wiki-no_candidate_ents
  params={'average_cand_nums':np.average(cand_ent_nums),
          'gold_nil':gold_nil_kg,
          'all_ents_not_nil':all_ents,
          'gold_not_in_wiki':gold_not_in_wiki,
          'no_candidate_ents':no_candidate_ents,
          'recall_ents':recall_ents,
          'recall ratio':recall_ents*1.0/all_ents,
          }
  print(params)
  cPickle.dump(params,open(feature_path+'cand_params.p','wb'))
  return entMents

if __name__ == "__main__":
  mongo_utils = mongoUtils('wiki')
  '''
  for mention in ['distress']:#['U.S.', 'PKK','BOJ','NCAA','ILO','NYC','UN','SAB']:
    ment,candidate_wiki_entity=get_mention_cands(mongo_utils,mention)
    sorted_candidate_wiki_entity = sorted(candidate_wiki_entity.iteritems(), key=lambda d:d[1], reverse = True)

    print('new ment:',ment)
    print(sorted_candidate_wiki_entity)
    print(len(sorted_candidate_wiki_entity))
    feature_path=''
    print('-----------------------')'''


  dataset_list=['aida','aida','aida','ace2004','msnbc','aquaint','wikipedia','clueweb']
  tag_list=['testa','testb','train','ace2004','msnbc','aquaint','wikipedia','clueweb']
  emnlp_file_list =['aida_testA.csv','aida_testB.csv','aida_train.csv',
                    'wned-ace2004.csv','wned-msnbc.csv','wned-aquaint.csv',
                    'wned-wikipedia.csv','wned-clueweb.csv']

  wiki_2_redir_title=get_wiki_redirect()
  wiki_name_2_id,wiki_id_2_name= get_disam_wiki_id_name()

  for i in range(3):
    dataset = dataset_list[i]
    tag=tag_list[i]

    print('dataset:',dataset)
    print('tag:',tag)

    #if dataset == 'aida' or dataset == 'KBP2014':
    #  dir_path = 'data/'+dataset+'/'+tag+'/'
    #else:
    #  dir_path ='data/'+dataset+'/'

    feature_path = dir_path+"features/"

    sid2aNosNo,aNosNo2id =get_sid_aNosNo(dir_path)

    entMents = get_entMents(mongo_utils,dataset,dir_path,aNosNo2id,wiki_name_2_id,wiki_2_redir_title)
    if not os.path.exists(feature_path):
      os.mkdir(feature_path)
    print('start to save the entMents:')
    cPickle.dump(entMents,open(feature_path+'entMents.p','wb'))
    print('-----------------------')
