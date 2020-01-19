# -*- coding: utf-8 -*-
import sys
sys.path.append('Project Absolute Path/DGCN_EL')
import os
from entity import get_disam_wiki_id_name,get_sid_aNosNo,get_wiki_redirect
from gen_coref_person import get_coref_person,gen_ent_person,get_cands_from_relateness
from tqdm import tqdm
import cPickle

def get_cands(wiki_id_2_name,dataset,tag,ent2id):
  ent_id = len(ent2id)
  if dataset == 'ace2004' or dataset=='msnbc' or dataset=='aquaint' or dataset=='wikipedia' or dataset=='clueweb':
    dir_path = 'data/'+dataset+'/'
  else:
    dir_path = 'data/'+dataset+'/'+tag+'/'

  entMents_file = dir_path+'features/entMents.p'
  entMents_file_new = dir_path+'features/entMents_new.p'

  entMents = cPickle.load(open(entMents_file,'rb'))

  entMents_new = {}

  #aid2ents = get_aid2ents(dir_path)
  sid2aNosNo,aNosNo2id =get_sid_aNosNo(dir_path)
  emnlp_doc2ents=cPickle.load(open(feature_path+'emnlp_doc2ents.p','r'))
  right_ment=0.0
  all_ment=0.0
  recall_ment=0.0
  mention_wrong = 0.0
  for key,val in tqdm(entMents.items()):
    sent_id = int(key)
    aNosNo=sid2aNosNo[sent_id]
    aNo = int(aNosNo.split('_')[0])
    if aNo not in emnlp_doc2ents:
      #print('doc',aNo,'has not ent mentions',len(val))
      continue
    emnlp_ment_dict = emnlp_doc2ents[aNo]

    ent_items = val
    '''
    mention_items = {'mention':mention,
                       'start':start,
                       'end':end,
                       'org_cands':org_candidate_wiki_entity_sorted,
                       'coref_cands':candidate_wiki_entity_sorted,
                       'gold_ent':gold_wiki_entity}'''
    new_ent_item_list=[]

    for ent_item in ent_items:

      gold_ent = ent_item['gold_ent']
      mention =ent_item['mention']

      if gold_ent==None:
        continue
      all_ment += 1
      if mention not in emnlp_ment_dict:
        #print('mention:',mention,' deal wrong')
        mention_wrong+=1
        continue

      right_ment+=1
      emnlp_cands_list = emnlp_ment_dict[mention].split('\t')
      new_ent_item = dict(ent_item)
      new_cand_dict={}

      org_cand_dict={}
      for cand_str in emnlp_cands_list:
        cand_item = cand_str.split(',')
        if len(cand_item)<3:
          #print(cand_item)
          org_cand_dict[cand_item[0]]=None  #EMPTYCAND
        else:
          org_cand_dict[cand_item[0]]=[float(cand_item[1]),','.join(cand_item[2:])]

      if tag=='train':
        last_cand_idx = None

        for cand_str in emnlp_cands_list[:30]:
          if cand_str == u'EMPTYCAND':  #add the gold ent during the training.
            new_cand_dict[gold_ent]=[0.0,wiki_id_2_name[gold_ent]]
          else:
            cand_item = cand_str.split(',')
            new_cand_dict[cand_item[0]]=[float(cand_item[1]),cand_item[2]]
            last_cand_idx=cand_item[0]

        if gold_ent not in new_cand_dict:
          if last_cand_idx!=None and len(new_cand_dict)==30:
            new_cand_dict.pop(last_cand_idx)

          if gold_ent in org_cand_dict:
            new_cand_dict[gold_ent]=org_cand_dict[gold_ent]
          else:
            new_cand_dict[gold_ent]=[0.0,wiki_id_2_name[gold_ent]]
        else:
          recall_ment+=1
      else:
        if len(org_cand_dict)==1:
          cand_str = org_cand_dict.keys()[0]
          if cand_str==u'EMPTYCAND':  #do not consider EMPTYCAND during test.
            continue

        mention,coref_cand_dict = get_coref_person(mention,org_cand_dict,ent_person,emnlp_ment_dict)
        #top 30 cands
        coref_cands = sorted(coref_cand_dict.items(),key=lambda x:x[1][0],reverse=True)[:30]
        for cand_item in coref_cands:
          new_cand_dict[cand_item[0]]=cand_item[1]
        if gold_ent not in new_cand_dict:  #do not consider 'gold not in cands' during test.
          continue
        else:
          recall_ment += 1

      new_ent_item['coref_cands']=new_cand_dict
      new_ent_item['coref_mention']=mention
      if new_ent_item['coref_mention'] != new_ent_item['mention']:
        print(new_ent_item['mention'],new_ent_item['coref_mention'])


      for cand_wikiId in new_cand_dict:
        if cand_wikiId not in ent2id:
          ent2id[cand_wikiId]=ent_id
          ent_id += 1
      if gold_ent not in ent2id:
        ent2id[gold_ent]=ent_id
        ent_id += 1

      new_ent_item_list.append(new_ent_item)

    entMents_new[sent_id]=list(new_ent_item_list)

  cPickle.dump(entMents_new,open(entMents_file_new,'wb'))
  print(tag,len(ent2id))

  print('all ments:',all_ment)
  print('right ment:',right_ment)
  print('wrong ment:',mention_wrong)
  print('recall ment:',recall_ment,recall_ment*1.0/all_ment)
  print('-----------------')
  return ent2id

if __name__ == '__main__':
  ent2id = {}
  ent2id=get_cands_from_relateness("validate",ent2id)
  ent2id=get_cands_from_relateness('test',ent2id)

  dataset_list=['aida','aida','aida','ace2004','msnbc','aquaint','wikipedia','clueweb']
  tag_list=['testa','testb','train','ace2004','msnbc','aquaint','wikipedia','clueweb']
  wiki_name_2_id,wiki_id_2_name= get_disam_wiki_id_name()
  ent_person={}

  if os.path.isfile('data/intermediate/ent_person.p') ==False:
    wiki_name_2_id,wiki_id_2_name= get_disam_wiki_id_name()
    wiki_2_redir_title=get_wiki_redirect()

    ent_person=gen_ent_person(wiki_2_redir_title,wiki_name_2_id)
    cPickle.dump(ent_person,open('data/intermediate/ent_person.p','wb'))
  else:
    ent_person = cPickle.load(open('data/intermediate/ent_person.p'))

  for i in range(len(dataset_list)):
    dataset = dataset_list[i]
    tag=tag_list[i]
    print('dataset:',dataset)
    print('tag:',tag)

    if dataset == 'aida' or dataset == 'KBP2014':
      dir_path = 'data/'+dataset+'/'+tag+'/'
    else:
      dir_path ='data/'+dataset+'/'

    feature_path = dir_path+"features/"


    ent2id=get_cands(wiki_id_2_name,dataset,tag,ent2id)

  print(len(ent2id))

  #cPickle.dump(ent2id,open('data/intermediate/ent2id_no_padding.p','wb'))
