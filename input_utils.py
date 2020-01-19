# -*- coding: utf-8 -*-
from sys import version_info
version=version_info.major
import numpy as np
import pickle
import codecs
from tqdm import tqdm


def get_doc_sent_for_ments(aNo_words,sent_index_batch,ment_sid_batch):
  sent_list=[]
  sent_lent_list=[]

  for idx in ment_sid_batch:
    sent_idx = sent_index_batch[idx]

    sent=aNo_words[sent_idx[0]:sent_idx[1]]
    sent_list.append(sent)
    sent_lent_list.append(len(sent))

  max_lent = max(sent_lent_list)
  new_sent_list = []
  for sent_i in sent_list:
    if len(sent_i) < max_lent:
      new_sent_list.append(sent_i+['' for i in range(max_lent-len(sent_i))])
    else:
      new_sent_list.append(sent_i)

  return new_sent_list,sent_lent_list

def gen_ent_person():
  ent_person = {}

  with codecs.open('data/deep-ep-data/persons.txt','r','utf-8') as file_:
    for line in tqdm(file_):
      line = line.strip()
      ent = line
      ent_person[ent]=1

  return ent_person

def get_mini_batch(wiki_id_2_name,id2ent,ent_person,model_type,tag,ent_nums,cand_nums,word_freq_loader,record,self_other_mask,mask_type):
  if version==2:
    ment_info_old = record['ment_info']
    aNo = record['aNo']
    m2w = record['m2w']

    aNo_words = record['aNo_words']
    sent_index_batch = record['sent_index_batch']
    ment_sid_batch_org = record['ment_sid_batch']

    ment_cands_id_batch_org=pickle.loads( record['ment_cands_id_batch'] )
    ment_cands_lent_batch_org = pickle.loads( record['ment_cands_lent_batch'] )
    ment_cands_prob_batch_org = pickle.loads( record['ment_cands_prob_batch'] )
    ment_cands_tag_batch_org = pickle.loads( record['ment_cands_tag_batch'] )

    ment_sent_ctx_id_batch_org = pickle.loads( record['ment_sent_ctx_id_batch'] )
    ment_sent_left_ctx_id_batch_org = pickle.loads( record['ment_sent_left_ctx_id_batch'] )
    ment_sent_right_ctx_id_batch_org = pickle.loads( record['ment_sent_right_ctx_id_batch'] )
    ment_surface_id_batch_org = pickle.loads( record['ment_surface_id_batch'] )
    ment_doc_ctx_id_batch_org = pickle.loads(record['ment_doc_ctx_id_batch'] )

  else:
    ment_info_old = record['ment_info']
    aNo = record['aNo']
    m2w = record['m2w']
    aNo_words = record['aNo_words']
    sent_index_batch = record['sent_index_batch']
    ment_sid_batch_org = record['ment_sid_batch']

    ment_cands_id_batch_org=pickle.loads(record['ment_cands_id_batch'],
                                         encoding='iso-8859-1')
    ment_cands_lent_batch_org = pickle.loads(record['ment_cands_lent_batch'],
                                             encoding='iso-8859-1' )
    ment_cands_prob_batch_org = pickle.loads(record['ment_cands_prob_batch'],
                                             encoding='iso-8859-1' )
    ment_cands_tag_batch_org = pickle.loads(record['ment_cands_tag_batch'],
                                            encoding='iso-8859-1')
    ment_sent_ctx_id_batch_org = pickle.loads(record['ment_sent_ctx_id_batch'],
                                              encoding='iso-8859-1')
    ment_sent_left_ctx_id_batch_org = pickle.loads(record['ment_sent_left_ctx_id_batch'],
                                                   encoding='iso-8859-1')
    ment_sent_right_ctx_id_batch_org = pickle.loads(record['ment_sent_right_ctx_id_batch'],
                                                    encoding='iso-8859-1')
    ment_surface_id_batch_org = pickle.loads(record['ment_surface_id_batch'],
                                             encoding='iso-8859-1')
    ment_doc_ctx_id_batch_org = pickle.loads(record['ment_doc_ctx_id_batch'],
                                             encoding='iso-8859-1')

  old_sample_size=len(ment_cands_tag_batch_org)

  ment_idx_list = range(old_sample_size)

  ment_info = np.array(ment_info_old)[ment_idx_list]
  ment_cands_id_batch = ment_cands_id_batch_org[ment_idx_list]
  ment_sent_ctx_id_batch = ment_sent_ctx_id_batch_org[ment_idx_list]
  ment_cands_prob_batch=ment_cands_prob_batch_org[ment_idx_list]
  ment_cands_tag_batch  =ment_cands_tag_batch_org[ment_idx_list]
  ment_surface_id_batch = ment_surface_id_batch_org[ment_idx_list]
  ment_doc_ctx_id_batch = ment_doc_ctx_id_batch_org[ment_idx_list]
  ment_sent_left_ctx_id_batch = ment_sent_left_ctx_id_batch_org[ment_idx_list]
  ment_sent_right_ctx_id_batch = ment_sent_right_ctx_id_batch_org[ment_idx_list]
  ment_cands_lent_batch = ment_cands_lent_batch_org[ment_idx_list]
  ment_sid_batch = ment_sid_batch_org

  sample_size=len(ment_cands_id_batch)

  S2_cand_mask_4=gen_S2_mask(sample_size,cand_nums)

  #generate the candidates names
  cands_reshape  = np.reshape(ment_cands_id_batch,[-1])
  all_cands=len(cands_reshape)
  cand_mask_pad=cands_reshape.astype(np.bool).astype(np.float)
  cand_mask_pad = np.array(np.tile(cand_mask_pad,[all_cands,1]),np.float32)



  if mask_type!='NIL':
    #gen_cand_mask(sample_size,cand_nums)
    cand_mask_4,cand_mask_2 = gen_doc_idx_mask(sample_size,cand_nums,
                                               self_other_mask,mask_type)
    cand_adj_mask = get_ajacent_matrix_train(ment_cands_tag_batch,sample_size,cand_nums)
  else:
    cand_mask_4=None#np.ones((sample_size,cand_nums,sample_size,cand_nums))
    cand_mask_2=None#np.ones((sample_size*cand_nums,sample_size*cand_nums))
    cand_adj_mask = None#np.ones((sample_size*cand_nums,sample_size*cand_nums))


  return [aNo,m2w,aNo_words,sent_index_batch,ment_sid_batch,\
          ment_idx_list,ment_info,ment_cands_id_batch,ment_cands_lent_batch,\
        ment_cands_prob_batch, ment_cands_tag_batch,\
        ment_doc_ctx_id_batch,ment_sent_ctx_id_batch,ment_sent_left_ctx_id_batch,\
        ment_sent_right_ctx_id_batch,ment_surface_id_batch,\
        S2_cand_mask_4,cand_mask_pad,cand_mask_4,cand_mask_2,cand_adj_mask,\
        sample_size]

def get_ajacent_matrix_train(tag,ment_no,cand_no):
    tag = np.reshape(tag,-1)

    S2_mask = np.zeros((ment_no*cand_no,ment_no*cand_no),dtype=np.float32)

    for i in range(ment_no*cand_no):
        tag_i = tag[i]
        if tag_i==1:
            for j in range(i,ment_no*cand_no):
                tag_j = tag[j]

                if tag_j==1:
                    S2_mask[i,j]=1
                    S2_mask[j,i]=1
    return  S2_mask


def gen_S2_mask(ment_no,cand_no):
  t = np.ones((ment_no,cand_no,ment_no,cand_no),dtype=np.float32)
  for i in range(ment_no):
    t[i,:,i,:] = np.zeros((cand_no,cand_no))  #only constraint to their own~

  return t

def gen_cand_mask(ment_no,cand_no):
  t = np.ones((ment_no,cand_no,ment_no,cand_no),dtype=np.float32)
  for i in range(ment_no):
    t[i,:,i,:] = np.diag(np.ones([cand_no]))  #only constraint to their own~

  return t,np.reshape(t,[ment_no*cand_no,ment_no*cand_no])

def gen_doc_idx_mask(ment_no,cand_no,self_other_mask,mask_type):
  sim_mask = []
  if mask_type=='dist_count':
    for i in range(ment_no):
      val_list =[]
      for j in range(0,i):
        val_list.append(1/(i-j+1.0))

      for j in range(i,ment_no):
        val_list.append(1/(j-i+1.0))
      sim_mask.append(val_list)
  elif mask_type =='window':
    #2019-4-26, window size is 3
    window_size=20
    empty = [0 for i in range(ment_no)]

    for i in range(ment_no):
      val_list=list(empty)
      for k  in range(max(0,i-window_size),min(i+window_size,ment_no)):
        val_list[k]=1
      sim_mask.append(val_list)

  sim_mask = np.array(sim_mask)
  sim_mask = np.expand_dims(sim_mask,1)
  sim_mask = np.expand_dims(sim_mask,-1)
  sim_mask=np.tile(sim_mask,[1,cand_no,1,cand_no])

  if self_other_mask==True:
    for i in range(ment_no):
      sim_mask[i,:,i,:]=np.diag([1 for idx in range(cand_no)])


  sim_mask_2 = np.reshape(sim_mask,[ment_no*cand_no,ment_no*cand_no])

  return sim_mask,sim_mask_2

def gen_verb_ctx(word_freq_loader,ment_surface_id_batch_org,ment_doc_ctx_id_batch_org):
  ment_surface_name=dict()
  for ment in ment_surface_id_batch_org:
    for wd_id in ment:
      wd = word_freq_loader.id2word[wd_id]
      ment_surface_name[wd.lower()]=1

  ment_no = len(ment_surface_id_batch_org)
  ment_doc_ctx_id_batch_new = []
  for j in range(ment_no):
    sent_list=[]
    for wd_id in ment_doc_ctx_id_batch_org[j]:
      if wd_id == 0:
        break
      wd = word_freq_loader.id2word[wd_id]
      if wd.lower() not in ment_surface_name and wd[0].isupper()== False:
        #word_freq_loader.id2word[wd_id]
        sent_list.append(wd_id)
    sent_lent = len(sent_list)
    if len(sent_list)>=60:
      sent_list = sent_list[:60]
    else:
      sent_list = sent_list +[0 for i in range(60-sent_lent)]
    assert(len(sent_list)==60)

    ment_doc_ctx_id_batch_new.append(sent_list)
  assert(len(ment_doc_ctx_id_batch_new)==ment_no)

  return np.array(ment_doc_ctx_id_batch_new,np.int32)

def insert_random_pad(ent_nums,ment_cands_id_batch_org,ment_cands_tag_batch_org):
  ment_no = len(ment_cands_id_batch_org)
  new_ment_cands_id_batch_org=[]
  single_cand=0
  for i in range(ment_no):
    ment_cand_list = list(ment_cands_id_batch_org[i])
    if len(np.nonzero(ment_cand_list)[0])==1:
      rand_cand = ent_nums

      ment_cand_list[1]=rand_cand
      single_cand+=1

    new_ment_cands_id_batch_org.append(ment_cand_list)
  assert(len(new_ment_cands_id_batch_org)==ment_no)

  return new_ment_cands_id_batch_org
