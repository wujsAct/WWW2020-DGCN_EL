# -*- coding: utf-8 -*-
"""generate the entity linking dataset
"""
import numpy as np
from sys import version_info
version = version_info.major
if version==2:
  from get_word_info_utils import WordFreqVectorLoader
  from entity import get_sid_aNosNo,load_entity_vector
else:
  from .get_word_info_utils import WordFreqVectorLoader
  from .entity import get_sid_aNosNo,load_entity_vector

import pickle
from tqdm import tqdm
from bson.binary import Binary
from pymongo import MongoClient
import string

class LinkingDataReader(object):
  def __init__(self,args,word_freq_loader,dataset,tag):
    self.dataset = dataset
    self.tag = tag
    self.ent2id = args['ent2id']
    self.word_freq_loader = word_freq_loader
    self.punct = string.punctuation
    if dataset == 'aida' or dataset=='KBP2014':
      self.dir_path = 'data/'+self.dataset+'/'+self.tag+'/'
    else:
      self.dir_path ='data/'+self.dataset+'/'

    self.data_fname = self.dir_path+'process/'+self.tag+'Data.txt'
    self.sid2aNosNo,self.aNosNo2id =get_sid_aNosNo(self.dir_path)

    '''
    @load candidate entities
    '''
    entMents_file = self.dir_path+'features/entMents_new.p'

    #[mention,start,end,candidate_wiki_entity_sorted,gold_wiki_entity]   瀵逛簬姣忎釜entity item閮芥槸杩欐牱鐨勫搱锛?
    self.entMents = pickle.load(open(entMents_file,'rb'))

    print(len(self.entMents))

  def get_ctx_LR(self,tag,ent_ment_s,ent_ment_e,aNo_words,fix_lent):
    ctx_lent = 0
    ctx=[]
    ctx_wds=[]

    for i in range(200):
      if tag =='left':
        rel_i = ent_ment_s-i-1
      else:
        rel_i = ent_ment_e+i

      if rel_i == len(aNo_words) or rel_i < 0:
        break

      wd_item = aNo_words[rel_i]
      if '&' in wd_item:
        wd_item = wd_item.replace('&','')

      if '-' in wd_item:
        wd_list = wd_item.split('-')
      else:
        wd_list=[wd_item]

      for wd in wd_list:
        if wd in self.punct or wd in ['-LRB-','-RRB-', "``","''"]:
          continue
        else:
          ctx_lent += 1
          ctx_wds.append(wd)
          if ctx_lent == fix_lent*2:
            break

        rel_wd = self.word_freq_loader.get_wd_in_wdict(wd)

        if rel_wd!=None:
          ctx.append(rel_wd)

      if ctx_lent == fix_lent*2:
        break
    if len(ctx) > fix_lent:
      ctx = ctx[:fix_lent]

    if tag =='left':
      ctx.reverse()
      ctx_wds.reverse()

    return ctx_wds,ctx

  def extract_ment_words_clueweb(self,ent_ment_strs,aNo_words,ment_size):
    ret_wds=[]
    ent_ment_wd_list = ent_ment_strs.split()
    print(ent_ment_strs)
    for i in range(len(ent_ment_wd_list)):
      wd  = ent_ment_wd_list[i]
      rel_wd = self.word_freq_loader.get_wd_in_wdict(wd)
      if rel_wd!=None:
        ret_wds.append(rel_wd)

        if len(ret_wds)==ment_size:
          break
    if len(ret_wds)< ment_size:
      ret_wds+=[0]*(ment_size-len(ret_wds))
    return ret_wds

  def extract_ment_words(self,ent_ment_s,ent_ment_e,aNo_words,ment_size):
    ret_wds=[]
    for i in range(ent_ment_s,ent_ment_e):
      wd  = aNo_words[i]
      rel_wd = self.word_freq_loader.get_wd_in_wdict(wd)
      if rel_wd!=None:
        ret_wds.append(rel_wd)

        if len(ret_wds)==ment_size:
          break
    if len(ret_wds)< ment_size:
      ret_wds+=[0]*(ment_size-len(ret_wds))
    return ret_wds

  def extract_ctx_words(self,ent_ment_s,ent_ment_e,aNo_words,ctx_size):

    non_wd_1,ctx_left = self.get_ctx_LR('left',ent_ment_s,ent_ment_e,aNo_words,ctx_size)
    non_wd_2,ctx_right = self.get_ctx_LR('right',ent_ment_s,ent_ment_e,aNo_words,ctx_size)

    ctx = ctx_left + ctx_right

    #print('ctx lent:',len(ctx))
    if len(ctx) < ctx_size*2:
      #print ent_ment_s,ent_ment_e,len(ctx)
      ctx += [0]*(ctx_size*2-len(ctx))

    assert(len(ctx)==ctx_size*2)
    return ctx,non_wd_1,non_wd_2

  def extract_sent_words(self,ent_ment_s,ent_ment_e,sent_s,sent_e,aNo_words,ctx_size):
    left = max(sent_s,ent_ment_e-ctx_size)
    right = min(sent_e,ent_ment_s+ctx_size)

    left_wds = self.extract_ment_words(left,ent_ment_e,aNo_words,ctx_size)

    right_wds = self.extract_ment_words(ent_ment_s,right,aNo_words,ctx_size)

    assert(len(left_wds)==len(right_wds)==ctx_size)
    '''
    if left < ent_ment_e:
      print(left,ent_ment_e)
      print(ent_ment_s,right)
      print(aNo_words[left:ent_ment_e],left_wds)
      print(aNo_words[ent_ment_s:right],right_wds)
      print('--------------------')'''
    return left_wds,right_wds

  def get_cands_id(self,ent_ment_cand_ents,gold_linking_ent):
    ment_cands_tag = [0]*30

    ment_cands_id = []
    ment_cands_prob = []

    idx=-1
    for cand_ids in ent_ment_cand_ents:
      idx+=1
      prob = ent_ment_cand_ents[cand_ids][0]
      if cand_ids not in self.ent2id:
        print('cand not in ent2id ...',cand_ids)
      else:
        if cand_ids == gold_linking_ent:
          ment_cands_tag[idx] = 1

        ment_cands_id.append(self.ent2id[cand_ids])
        ment_cands_prob.append(prob)

    lent = len(ment_cands_prob)
    if lent<30:
      ment_cands_id += [0]*(30-lent)
      ment_cands_prob += [0]*(30-lent)

    if sum(ment_cands_tag)!=1:
      print(gold_linking_ent)
      exit(0)

    assert(sum(ment_cands_tag)==1)
    assert(len(ment_cands_id)==len(ment_cands_tag))
    assert(len(ment_cands_id)==len(ment_cands_prob))
    return ment_cands_id,ment_cands_prob,ment_cands_tag

  def get_coref_w(self,ment_cands_id_batch):
    train_size = len(ment_cands_id_batch)
    has_coref_ents=0.0

    has_coref_dict=dict()
    for i in range(train_size):
      tag_i_list = list(ment_cands_id_batch[i])

      for j in range(i+1,train_size):
        tag_j_list = list(ment_cands_id_batch[j])

        if sorted(tag_i_list) == sorted(tag_j_list):
          if i not in has_coref_dict:
            has_coref_dict[i]=0
          if j not in has_coref_dict:
            has_coref_dict[j]=0
          has_coref_dict[i] += 1
          has_coref_dict[j] += 1

    for key in has_coref_dict:
      if has_coref_dict[key]>=1:
        has_coref_ents += 1
    coref_w = 0.0
    if train_size!=0:
      coref_w=has_coref_ents/train_size

    return coref_w

  def get_alpha_mention2word(self,aNo_words,ment_surface_id_batch):
    ment_no = len(ment_surface_id_batch)
    aNo_words_lent = 0.0
    for i in range(len(aNo_words)):
      wd  = aNo_words[i]
      rel_wd = self.word_freq_loader.get_wd_in_wdict(wd)
      if rel_wd!=None:
        aNo_words_lent+=1

    ment_lent = 0.0
    #[str(aNo)+'_'+str(i),str(ent_ment_item['start']),str(ent_ment_item['end']),ent_ment_item['mention']])
    #ment_non_wd.append(ent_ment_item['mention']

    for i in range(len(ment_surface_id_batch)):
      for j in range(len(ment_surface_id_batch[0])):
        if ment_surface_id_batch[i][j]!=0:
          ment_lent+= 1

    m2w = ment_no/aNo_words_lent

    return m2w
  def gen_feature_for_doc(self,aNo,aNo_words,sid_list,sid_lent_list):
    word_rel_id = 0
    assert(len(sid_lent_list) == len(sid_list))

    ment_info = []
    ment_cands_id_batch=[]  #golden
    ment_cands_prob_batch=[] #candidate prior probability
    ment_cands_lent_batch=[]
    ment_cands_tag_batch=[]
    ment_doc_ctx_id_batch=[]
    ment_sent_ctx_id_batch=[]
    ment_sent_left_ctx_id_batch = []
    ment_sent_right_ctx_id_batch = []
    ment_surface_id_batch=[]
    ment_non_wd =[]

    #we add mention sid and aNo
    ment_sid_batch=[]
    sent_index_batch = []

    for i in range(len(sid_list)):
      sent_id = sid_list[i]
      '''
      mention_items = {'mention':mention,
                       'start':start,
                       'end':end,
                       'org_cands':org_candidate_wiki_entity_sorted,
                       'coref_cands':candidate_wiki_entity_sorted,
                       'gold_ent':gold_wiki_entity}'''
      sent_index_batch.append([word_rel_id,word_rel_id+sid_lent_list[i]])

      if sent_id in self.entMents:
        for ent_ment_item in self.entMents[sent_id]:
          gold_linking_ent = ent_ment_item['gold_ent']

          if gold_linking_ent =='nil' or  \
          gold_linking_ent not in self.ent2id or gold_linking_ent == None:
            continue
          sent_s = word_rel_id
          sent_e = word_rel_id + sid_lent_list[i]
          ent_ment_s = word_rel_id+ent_ment_item['start']
          ent_ment_e = word_rel_id+ent_ment_item['end']

          ent_ment_cand_ents=ent_ment_item['coref_cands']

          ment_cands_id,ment_cands_prob,ment_cands_tag = self.get_cands_id(ent_ment_cand_ents,gold_linking_ent)
          #ment_infos = aNo_words[ent_ment_s:ent_ment_e]
#          print('ment_infos:', ment_infos, ent_ment_item['mention'])
#          print(ment_cands_id)
#          print(ment_cands_prob)
#          print(ment_cands_tag)

          #print ent_ment_s,ent_ment_e
          ment_doc_ctx_id,non_wd1,non_wd2 = self.extract_ctx_words(ent_ment_s,ent_ment_e,aNo_words,50)
#          print(non_wd1)
#          print(non_wd2)
          ment_sent_ctx_id,non_wd1,non_wd2 = self.extract_ctx_words(ent_ment_s,ent_ment_e,aNo_words,10)

          ment_left,ment_right=self.extract_sent_words(ent_ment_s,ent_ment_e,sent_s,sent_e,aNo_words,10)  #fixed 5 words as its relation indicator
          '''
          if self.dataset=='clueweb':
            ment_surface_id = self.extract_ment_words_clueweb(ent_ment_item['coref_mention'],aNo_words,5)
          else:
            ment_surface_id = self.extract_ment_words(ent_ment_s,ent_ment_e,aNo_words,5)'''

          ment_surface_id = self.extract_ment_words(ent_ment_s,ent_ment_e,aNo_words,5)
          ment_cands_lent_batch.append(len(ent_ment_cand_ents))
          ment_cands_id_batch.append(ment_cands_id)
          ment_cands_prob_batch.append(ment_cands_prob)
          ment_cands_tag_batch.append(ment_cands_tag)

          ment_doc_ctx_id_batch.append(ment_doc_ctx_id)
          ment_sent_left_ctx_id_batch.append(ment_left)
          ment_sent_right_ctx_id_batch.append(ment_right)
          ment_sent_ctx_id_batch.append(ment_sent_ctx_id)
          ment_surface_id_batch.append(ment_surface_id)

          ment_sid_batch.append(i)

          ment_info.append([str(aNo)+'_'+str(i),str(ent_ment_item['start']),str(ent_ment_item['end']),ent_ment_item['mention']])
          ment_non_wd.append(ent_ment_item['mention'])
      word_rel_id += sid_lent_list[i]

    m2w = self.get_alpha_mention2word(aNo_words,ment_info)
    #coref_w = self.get_coref_w(ment_cands_id_batch)
    assert(len(sent_index_batch)==len(sid_list))
    assert(len(ment_sid_batch)==len(ment_info))
    return  [aNo,m2w,aNo_words,sent_index_batch,ment_sid_batch,ment_non_wd,ment_info,np.asarray(ment_cands_id_batch,dtype=np.int32),\
                    np.asarray(ment_cands_lent_batch,dtype=np.int32),\
                    np.asarray(ment_cands_prob_batch,dtype=np.float32),\
                    np.asarray(ment_cands_tag_batch,dtype=np.float32),\
                    np.asarray(ment_doc_ctx_id_batch,np.int32),\
                    np.asarray(ment_sent_ctx_id_batch,np.int32),\
                    np.asarray(ment_sent_left_ctx_id_batch,np.int32),\
                    np.asarray(ment_sent_right_ctx_id_batch,np.int32),\
                    np.asarray(ment_surface_id_batch,np.int32)]

  def get_input_train(self):
    sid = -1
    '''
    #generate context words around entity mention in a document or sentence?
    variable size mini-batches are used during training
    '''
    prev_aNo = 0 #current document no.
    s_words = []
    aNo_words = [] #words in a document
    sid_list = [] #we utilize this parameter to rewrite the entity mention index, to becoming into the document index
    sid_lent_list=[]
    dataset_num_mentions = 0.0
    non_empty_candiates = 0.0
    with open(self.data_fname) as file_:
      for line in file_:
        if line not in ['\n','\r\n']:
          line = line[:-1]
          word = line.split('\t')[0]
          s_words.append(word)
        else:
          sid += 1
          aNosNo = self.sid2aNosNo[sid]
          aNo = int(aNosNo.split('_')[0])
          if aNo == prev_aNo:
            aNo_words += s_words
            sid_list.append(sid)
            sid_lent_list.append(len(s_words))
          else:
            doc_features = self.gen_feature_for_doc(prev_aNo,aNo_words,sid_list,sid_lent_list)
            ment_cands_id_batch=doc_features[2]
            if len(ment_cands_id_batch)!=0:
              yield prev_aNo,doc_features
            else:
              print(prev_aNo,'has no mention....!!!')
            #start a new batch
            prev_aNo = aNo
            sid_list=[]
            sid_lent_list=[]
            aNo_words=[]
            aNo_words += list(s_words)
            sid_list.append(sid)
            sid_lent_list.append(len(s_words))
          s_words = []

    doc_features = self.gen_feature_for_doc(prev_aNo,aNo_words,sid_list,sid_lent_list)
    #print(aNo_words)
    if len(ment_cands_id_batch)!=0:
      yield prev_aNo,doc_features
    else:
      print(prev_aNo,' has no candidates...')

    print('dataset_num_mentions:',dataset_num_mentions)
    print('non_empty_candiates:',non_empty_candiates)

if __name__=='__main__':
  args = {}

  word_freq_loader = WordFreqVectorLoader(args)
  word_embedding = word_freq_loader.word_embedding
  data_loader_args={}

  data_loader_args['word2id'] = word_freq_loader.word2id
  ent2id,ent2type,entity_embedding,ent_2_mid = load_entity_vector(data_loader_args['word2id'],word_embedding)

  client = MongoClient('mongodb://192.168.3.202:27017')
  args['ent2id'] = ent2id

  for data_info in [
                 ['aida','testa','aida_testa'],
                    ['aida','testb','aida_testb'],
                    ['aida','train','aida_train'],
                    ['ace2004','ace2004','ace2004'],
                    ['msnbc','msnbc','msnbc'],
                    ['aquaint','aquaint','aquaint'],
                    ['wikipedia','wikipedia','wikipedia'],
                    ['clueweb','clueweb','clueweb']
                    ]:
    dataset = data_info[0]
    tag = data_info[1]
    col_name= data_info[2]
    liking_data_reader = LinkingDataReader(args,word_freq_loader,dataset,tag)

    db = client['EntityLinking'] # database name
    collection = db[col_name]  # collection wiki
    liking_data_reader.get_input_train()

    record_list = list(collection.find({}))


    for aNo,doc_features in  liking_data_reader.get_input_train():
      record={}

      aNo,m2w,aNo_words,sent_index_batch,ment_sid_batch,ment_non_wd,ment_info,ment_cands_id_batch,ment_cands_lent_batch,ment_cands_prob_batch,\
        ment_cands_tag_batch,ment_doc_ctx_id_batch,ment_sent_ctx_id_batch,\
        ment_sent_left_ctx_id_batch,ment_sent_right_ctx_id_batch,\
              ment_surface_id_batch = doc_features

      record['ment_info'] = ment_info
      record['ment_cands_id_batch'] = Binary(pickle.dumps(ment_cands_id_batch,protocol=2))
      record['ment_cands_lent_batch'] = Binary(pickle.dumps(ment_cands_lent_batch,protocol=2))
      record['ment_cands_prob_batch'] = Binary(pickle.dumps(ment_cands_prob_batch,protocol=2))
      record['ment_cands_tag_batch'] = Binary(pickle.dumps(ment_cands_tag_batch,protocol=2))
      record['ment_doc_ctx_id_batch'] = Binary(pickle.dumps(ment_doc_ctx_id_batch,protocol=2))
      record['ment_sent_ctx_id_batch'] = Binary(pickle.dumps(ment_sent_ctx_id_batch,protocol=2))
      record['ment_sent_left_ctx_id_batch'] = Binary(pickle.dumps(ment_sent_left_ctx_id_batch,protocol=2))
      record['ment_sent_right_ctx_id_batch'] = Binary(pickle.dumps(ment_sent_right_ctx_id_batch,protocol=2))
      record['ment_surface_id_batch'] = Binary(pickle.dumps(ment_surface_id_batch,protocol=2))

      record['aNo']=aNo
      record['aNo_words']=aNo_words
      record['sent_index_batch']=sent_index_batch
      record['ment_sid_batch']=ment_sid_batch
      record['m2w']=aNo_words
      print('--------------------------------')
      collection.insert_one(record)
