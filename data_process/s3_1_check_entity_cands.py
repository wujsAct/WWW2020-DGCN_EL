# -*- coding: utf-8 -*-
import sys
sys.path.append('Project Absolute Path')
from entity import get_sid_aNosNo
from tqdm import tqdm
import cPickle


def check_cands(dataset,tag):
  if dataset == 'ace2004' or dataset=='msnbc' or dataset=='aquaint' or dataset=='wikipedia' or dataset=='clueweb':
    dir_path = 'data/'+dataset+'/'
  else:
    dir_path = 'data/'+dataset+'/'+tag+'/'

  feature_path = dir_path+'features/'

  entMents_file = dir_path+'features/entMents.p'

  entMents = cPickle.load(open(entMents_file,'rb'))


  #aid2ents = get_aid2ents(dir_path)
  sid2aNosNo,aNosNo2id =get_sid_aNosNo(dir_path)
  emnlp_doc2ents=cPickle.load(open(feature_path+'emnlp_doc2ents.p','r'))
  right_ment=0.0
  all_ment=0.0
  for key,val in tqdm(entMents.items()):
    sent_id = int(key)
    aNosNo=sid2aNosNo[sent_id]
    aNo = int(aNosNo.split('_')[0])
    if aNo not in emnlp_doc2ents:
      continue
    emnlp_ment_dict = emnlp_doc2ents[aNo]


    ent_items = val
    for ent_item in ent_items:
      gold_ent = ent_item['gold_ent']
      mention =ent_item['mention']
      candList=ent_item['org_cands']
      '''
      mention_items = {'mention':mention,
                         'start':start,
                         'end':end,
                         'org_cands':org_candidate_wiki_entity_sorted,
                         'coref_cands':candidate_wiki_entity_sorted,
                         'gold_ent':gold_wiki_entity}'''
      if gold_ent == None:
        continue

      emnlp_cand_set = set()
      if mention in emnlp_ment_dict:
        emnlp_cands_list = emnlp_ment_dict[mention].split('\t')

        for emnlp_cand_i in emnlp_cands_list:
          ent = emnlp_cand_i.split(',')[0]
          if ent!=u'EMPTYCAND':
            continue

      new_emnlp_cand_set=set()
      idx=0
      for idx in range(len(candList)):
        cand_ent = candList[idx]
        ent = cand_ent[0]
        new_emnlp_cand_set.add(ent)


      all_ment+=1
      if emnlp_cand_set & new_emnlp_cand_set == emnlp_cand_set:
        right_ment += 1
  print('all ments:',all_ment)
  print('right ments:',right_ment)


if __name__ == '__main__':
  dataset_list=['aida','aida','aida','ace2004','msnbc','aquaint','wikipedia','clueweb']
  tag_list=['testa','testb','train','ace2004','msnbc','aquaint','wikipedia','clueweb']

  for i in range(len(dataset_list)):
    dataset = dataset_list[i]
    tag=tag_list[i]
    print('dataset:',dataset)
    print('tag:',tag)

    check_cands(dataset,tag)
