# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:42:24 2018

@author: DELL
"""
import os
import pickle
import numpy as np
from bson.binary import Binary
from pymongo import MongoClient
from input_utils import get_mini_batch,get_doc_sent_for_ments
import random
from sys import version_info
version = version_info.major
import codecs
from tqdm import tqdm

def get_elmo_embedding(sess,elmo,ment_info,tf_tokens_input,tf_tokens_length):
  tf_embeddings=embeddings = elmo(
    inputs={
    "tokens": tf_tokens_input,
    "sequence_len": tf_tokens_length
    },
    signature="tokens",
    as_dict=True)["elmo"]
  embeddings = sess.run(tf_embeddings)
  #print('embeddings shape:',embeddings)
  ment_elmo_embed=[]
  
  for i in range(len(ment_info)):
    ment_item=ment_info[i]
    ment_s,ment_e =int(ment_item[1]),int(ment_item[2])
    
    ment_elmo = embeddings[i,ment_s:ment_e,:]
    ment_elmo_rep=np.sum(ment_elmo,0)
    #print(np.shape(ment_elmo))
    #print(np.shape(ment_elmo_rep))
    #print('-----------')
    ment_elmo_embed.append(ment_elmo_rep)
  
  ment_elmo_embed=np.array(ment_elmo_embed,np.float32)
  return ment_elmo_embed

def revise_prior_score(ment_info,ment_cands_id_batch,ment_cands_prob_batch,wiki_id_2_name,id2ent,ent_person,
                       ment_cands_tag_batch,final_score,prior_score):
  new_final_score = np.array(final_score)
  new_ment_cands_prob_batch=np.array(ment_cands_prob_batch)
  
  gold_no = np.argmax(ment_cands_tag_batch,1)
  pred_no = np.argmax(final_score,1)
  wrong=0.0
  right=0.0
  
  for i in range(len(ment_info)):
    pred_right=False
    if gold_no[i] == pred_no[i]:
      pred_right = True
      
    ment_has_people_cand = False
    cand_margin_dict={}
    for idx in range(len(ment_cands_id_batch[i])):
      ids=ment_cands_id_batch[i,idx]
      wiid = id2ent[ids]
      if wiid in wiki_id_2_name:
        cand_name = wiki_id_2_name[wiid]
        
        margin = 0
        total_in_flag=False
        for j in range(i)[::-1]:
          if cand_name.lower() == ment_info[j][3].lower():
            margin = i-j
            total_in_flag=True
            break
        
        if cand_name in ent_person and (margin!=0 or total_in_flag):
          cand_margin_dict[str(idx)+'_'+str(j)]=margin
          #print(total_in_flag,margin)
          ment_has_people_cand=True
    
    #we revise the prior score...
    if ment_has_people_cand:
      if len(cand_margin_dict)!=0.0:
        cand_infos = sorted(cand_margin_dict.items(),key=lambda d:d[1])[0][0]
        idx = int(cand_infos.split('_')[0])
        ref_ment_idx = int(cand_infos.split('_')[1])
        
        #print(ment_cands_prob_batch[i][gold_no[i]],ment_cands_prob_batch[ref_ment_idx][gold_no[ref_ment_idx]])
        new_ment_cands_prob_batch[i][gold_no[i]]=ment_cands_prob_batch[ref_ment_idx][gold_no[ref_ment_idx]]
        
        #we regard this one as idx
        if ment_cands_tag_batch[ref_ment_idx,pred_no[ref_ment_idx]]==ment_cands_tag_batch[i][idx]:
          gold_no_ref_i=idx
          
          new_final_score[i][gold_no_ref_i]=max(np.max(final_score[i])+0.1,
                                             0.5*(final_score[ref_ment_idx,pred_no[ref_ment_idx]]+
                                             final_score[i][gold_no_ref_i]))
          

          if pred_right==True:
            if gold_no[i]!=gold_no_ref_i:
              wrong+=1
              
          else:
            if gold_no[i]==gold_no_ref_i:
              right+=1
              '''
              print(prior_score[i][gold_no[i]],prior_score[i][pred_no[i]],prior_score[ref_ment_idx,pred_no[ref_ment_idx]])
              print(final_score[i][gold_no[i]],final_score[i][pred_no[i]],final_score[ref_ment_idx,pred_no[ref_ment_idx]])
              
              print(new_final_score[i][gold_no[i]],
                    new_final_score[i][pred_no[i]],
                    new_final_score[ref_ment_idx,pred_no[ref_ment_idx]])
              print('--------------')'''
  return wrong,right,new_final_score,new_ment_cands_prob_batch

def get_training_stop_flag(train_loss_list):
  loss_list_lent = len(train_loss_list)
  loss_increase_time = 0
  
  if loss_list_lent > 100:
    for i in range(100,loss_list_lent)[::-1]:
      if i>=2:
        if train_loss_list[i] > train_loss_list[i-1] and train_loss_list[i]>train_loss_list[i-2]:
          loss_increase_time +=1
  
  if loss_increase_time > 20:
    return True
  else:
    return False
  
def get_test_ret(sess,args,tag,data_dict,ent_linking_model,test_collection,
                 ent_2_mid,mid_2_ent,word_freq_loader,wiki_id_2_name,id2ent,ent_person):
  
  link_loss_list=[]
  loss_list=[]
  total_ment_list=[]
  
  right_ment_list=[]
  
  total_ents=0
  local_score_path = ''
  if tag in ['train','testa','testb']:
    col_name = 'aida'+'_'+tag
    local_score_path=col_name
  else:
    col_name = tag
    local_score_path = col_name
  ent_level_no=5
  ent_level_score_list=[0.3,0.1,0.03,0.01,0.005]
  total_wrong_ent = np.array([0.0 for i in range(ent_level_no)])
  total_high_ent=np.array([0.0 for i in range(ent_level_no)])
  total_wrong_high_max_sum=np.array([0.0 for i in range(ent_level_no)])
  total_wrong_high_max_num =np.array([0.0 for i in range(ent_level_no)])
  
  if args.model_type == 'GraphCNNGlobalEntLinkModel' or args.model_type=='CNNGlobalEntLinkModel':
    col_name = col_name + '_sub'
  global_step=0
  data_2_mentNums = {'train':18541,'testa':4791,'testb':4485,'ace2004':257,'msnbc':656,'aquaint':727,'wikipedia':6821,'clueweb':11154,'kbp_evaluation':3064}
  total_ref_wrong=0.0
  total_ref_right=0.0
  for record in test_collection:
    fid=str(record['_id'])
    
    if 'Global' in args.model_type:
      if args.data_source=='EntityLinking':
        pre_train_local_score = np.load('data/local_score/'+local_score_path+'/'+str(fid)+'.npy')
      else:
        '''
        if version==2:
          pre_train_local_score = pickle.loads(record['ment_local_p_score_batch'])
        else:
          pre_train_local_score = pickle.loads(record['ment_local_p_score_batch'],
                                                 encoding='iso-8859-1')'''
        
        pre_train_local_score = np.load('data/'+args.data_source+'/'+local_score_path+'/'+str(fid)+'.npy')
        '''
        rel_data_source = 'data/'+args.data_source+'/'+'mg_0.5'+'_lr_0.0001'
        pre_train_local_score = np.load(rel_data_source+'/'+local_score_path+'/'+str(fid)+'.npy')'''
        '''
        if version==2:
          pre_train_p_local_score = pickle.loads(record['ment_local_p_score_batch'])
        else:
          pre_train_p_local_score = pickle.loads(record['ment_local_p_score_batch'],
                                                 encoding='iso-8859-1')'''
    
    item = data_dict[fid]
    
    aNo,m2w,aNo_words,sent_index_batch,ment_sid_batch,\
    ment_idx_list,ment_info,ment_cands_id_batch,ment_cands_lent_batch,\
        ment_cands_prob_batch, ment_cands_tag_batch,\
        ment_doc_ctx_id_batch,ment_sent_ctx_id_batch,ment_sent_left_ctx_id_batch,\
        ment_sent_right_ctx_id_batch,ment_surface_id_batch,\
        S2_cand_mask_4,cand_mask_pad,cand_mask_4,cand_mask_2,cand_adj_mask,\
        sample_size= item
    
    
    total_ents += sample_size
    
    feed_dict={
        ent_linking_model.ment_cand_id:ment_cands_id_batch,
        ent_linking_model.ment_cand_prob:ment_cands_prob_batch,
        ent_linking_model.linking_tag:ment_cands_tag_batch,
        ent_linking_model.ment_surface_ids:ment_surface_id_batch,
        ent_linking_model.ment_sent_ctx:ment_sent_ctx_id_batch,
        ent_linking_model.S2_cand_mask_4:S2_cand_mask_4,
        ent_linking_model.ment_sent_left_ctx:ment_sent_left_ctx_id_batch,
        ent_linking_model.ment_sent_right_ctx:ment_sent_right_ctx_id_batch,
        ent_linking_model.ment_doc_ctx:ment_doc_ctx_id_batch,
        ent_linking_model.cand_mask_pad:cand_mask_pad,
        ent_linking_model.ment_cands_lent:ment_cands_lent_batch,
        ent_linking_model.keep_prob:1.0,
        ent_linking_model.keep_prob_V:1.0,
        ent_linking_model.keep_prob_D:1.0,
        ent_linking_model.lr:1.0,
        ent_linking_model.sample_size:sample_size,
        ent_linking_model.is_training:False
        }
    
    if 'Global' in args.model_type:
      feed_dict[ent_linking_model.alpha]=m2w
      feed_dict[ent_linking_model.local_ment_cand_prob] = pre_train_local_score[ment_idx_list]
    
    if True:
      feed_dict[ent_linking_model.cand_mask_2d]=cand_mask_2
      feed_dict[ent_linking_model.cand_mask_4d]=cand_mask_4
    
    
    if args.data_source != 'EntityLinking':
      if args.date=='6_24':
        file_dir= 'data/'+args.data_source+'/elmo/'+local_score_path+'/'+str(fid)+'.npy'
        ment_surface_id_batch_org=np.load(file_dir)
      else:
        ment_surface_id_batch_org = pickle.loads( record['ment_surface_elmo_embed'] )
      feed_dict[ent_linking_model.ment_surface_elmo_embed]=ment_surface_id_batch_org
    else:
      file_dir= 'data/'+'EntityLinking'+'/elmo/'+local_score_path+'/'+str(fid)+'.npy'
      feed_dict[ent_linking_model.ment_surface_elmo_embed]=np.load(file_dir)
      
    norm_ent_adj=[]
    
    link_loss,right_ment_num,prior,final_score,global_step = sess.run([
        ent_linking_model.link_loss,
        ent_linking_model.right_ment_num,
        ent_linking_model.p_e_m,
        ent_linking_model.l_final_score,
        ent_linking_model.global_step
        ],feed_dict)
    
    link_loss_list.append(link_loss)
    loss_list.append(link_loss)
    
    wrong,right,new_final_score,new_ment_cands_prob_batch=revise_prior_score(ment_info,ment_cands_id_batch,ment_cands_prob_batch,
                                   wiki_id_2_name,id2ent,ent_person,
                                   ment_cands_tag_batch,final_score,ment_cands_prob_batch)
    total_ref_right+=right
    total_ref_wrong+=wrong
    
    right_ment_num-=wrong
    right_ment_num+=right
    
    right_ment_list.append(right_ment_num*1.0) 
    total_ment_list.append(sample_size*1.0)
    
    if 'Simple' not in args.model_type:
      #to analysis different level connection...
      #first level
      #model_adj = ent_linking_model.gcn_layers['norm_ent_adj_0_'+str(0)+'_'+ str(args.gcn_kernel_size-1)]
      
      model_adj = ent_linking_model.gcn_layers['norm_ent_adj_'+str(args.lbp_iter_num-1)+'_'+ str(args.gcn_kernel_size-1)]
      
      norm_ent_adj,gcn_h = sess.run([model_adj,ent_linking_model.gcn_h],feed_dict)
      if aNo==216:
        np.save('demo_216_'+str(args.date),gcn_h)
      
      if args.test==True and tag=='testb':
        #np.save('data/local_score/'+local_score_path+'/'+str(record['_id']),final_score)
        if aNo==216:
          print('[*************]',aNo,m2w)
        wrong_ent,high_ent,wrong_high_max_sum,wrong_high_max_num = get_ret(aNo,args,tag,word_freq_loader,
            id2ent,wiki_id_2_name,ent_2_mid,prior,
            ment_info,sample_size,ment_surface_id_batch,
            ment_cands_id_batch,ment_cands_tag_batch,
            new_ment_cands_prob_batch,
            ment_doc_ctx_id_batch,
            new_final_score,norm_ent_adj)
        total_wrong_ent+= np.array(wrong_ent)
        total_high_ent +=np.array(high_ent)
        total_wrong_high_max_num+= np.array(wrong_high_max_num)
        total_wrong_high_max_sum +=np.array(wrong_high_max_sum)
        #print('---------------------------------------')
        #print('---------------------------------------')
    else:
      if args.model_type == 'SimpleCtxCNNLocalEntLinkModel' and args.test==True:
        for i in range(sample_size):
#          #if len(np.nonzero(ment_cands_id_batch[i])[0])==1:
          print(tag,final_score[i])
        
        if args.save_local_score:
          #np.save('data/local_np_score/'+local_score_path+'/'+str(record['_id']),final_score)
          if not os.path.exists('data/'+args.data_source+'/'):
            os.makedirs('data/'+args.data_source+'/')
          
          rel_data_source = 'data/'+args.data_source+'/'+'mg_'+str(args.margin_param)+'_lr_'+str(args.learning_rate_start)
          
          if not os.path.exists(rel_data_source):
            os.makedirs(rel_data_source)
            
          if not os.path.exists(rel_data_source+'/'+local_score_path+'/'):
            os.makedirs(rel_data_source+'/'+local_score_path+'/')
          np.save(rel_data_source+'/'+local_score_path+'/'+str(record['_id']),final_score)
  
  link_loss = np.sum(link_loss_list)
  loss = np.sum(loss_list)
  accuracy = np.sum(right_ment_list)/data_2_mentNums[tag]
  
  precision = 1.0*np.sum(right_ment_list)/total_ents
  recall = 1.0*np.sum(right_ment_list)/data_2_mentNums[tag]
  f1 = 2*precision*recall/(precision+recall)
  if  args.test==True and tag =='testb' and 'Simple' not in args.model_type:
    print(np.sum(total_wrong_ent))
    for i in range(ent_level_no):
      #print('%d, %.4f' %(total_wrong_high_max_num[i],total_wrong_high_max_sum[i]/total_wrong_high_max_num[i]))
      print('%f,%d,%d' %(ent_level_score_list[i],total_high_ent[i],total_wrong_ent[i]))
  doc_avg_acc=np.average(np.array(right_ment_list)/np.array(total_ment_list))
  print('----------------------------')
  print('steps: %d, tag: %s: link_loss %.4f, loss: %.4f,avg acc:%.2f ,ACC: %.2f, f1: %.2f, %d, %d, %d' 
          %(global_step,
          tag,
          link_loss, 
          loss,
          doc_avg_acc*100,
          accuracy*100,
          f1*100,
          data_2_mentNums[tag],
          np.sum(right_ment_list),
          total_ents)
        )
  print('total_ref_right: ',total_ref_right,' total_ref_wrong:',total_ref_wrong)
  
  
  return link_loss,loss,accuracy,f1

#get_sub_cands(sess,'train',ent_linking_model,train_collection,learning_rate)
def get_sub_cands(sess,args,word_freq_loader,tag,ent_linking_model,db,learning_rate,filter_cands,
                  wiki_id_2_name,id2ent,ent_person):
  
  local_score_path = ''
  if tag in ['train','testa','testb']:
    col_name = 'aida'+'_'+tag
    local_score_path=col_name
  else:
    col_name = tag
    local_score_path = col_name
    
  client = MongoClient('mongodb://192.168.3.196:27017')
  #client = MongoClient('mongodb://localhost:27017')
  
  #db_new = client['EntityLinking']
  
  #db_new = client['entity_linking']
  
  db_new=client[args.data_source]
  if tag in ['train','testa','testb']:
    test_collection = db['aida_'+tag]
    test_collection_sub =  db_new['aida_'+tag+'_sub']
  else:
    test_collection = db[tag]
    test_collection_sub =  db_new[tag+'_sub']
  
  nil_ment = 0
  all_ment = 0
  for i,record in enumerate(test_collection.find()):
    item = get_mini_batch(wiki_id_2_name,id2ent,ent_person,args.model_type,tag,10,
                                 args.cand_nums,word_freq_loader,record,
                                 args.self_other_mask,args.mask_type)
      
    aNo,m2w,aNo_words,sent_index_batch,ment_sid_batch,\
    ment_idx_list,ment_info_batch,ment_cands_id_batch,ment_cands_lent_batch,\
        ment_cands_prob_batch, ment_cands_tag_batch,\
        ment_doc_ctx_id_batch,ment_sent_ctx_id_batch,ment_sent_left_ctx_id_batch,\
        ment_sent_right_ctx_id_batch,ment_surface_id_batch,\
        S2_cand_mask_4,cand_mask_pad,cand_mask_4,cand_mask_2,cand_adj_mask,\
        sample_size= item
        
    feed_dict = {
                  ent_linking_model.ment_cand_id:ment_cands_id_batch,
                  ent_linking_model.ment_cand_prob:ment_cands_prob_batch,
                  ent_linking_model.linking_tag:ment_cands_tag_batch,
                  ent_linking_model.ment_surface_ids:ment_surface_id_batch,
                  ent_linking_model.ment_sent_ctx:ment_sent_ctx_id_batch,
                  ent_linking_model.ment_sent_left_ctx:ment_sent_left_ctx_id_batch,
                  ent_linking_model.ment_sent_right_ctx:ment_sent_right_ctx_id_batch,
                  ent_linking_model.ment_doc_ctx:ment_doc_ctx_id_batch,
                  ent_linking_model.ment_cands_lent:ment_cands_lent_batch,
                  ent_linking_model.cand_mask_pad:cand_mask_pad,
                  ent_linking_model.cand_mask_2d:cand_mask_2,
                  ent_linking_model.cand_mask_4d:cand_mask_4,
                  ent_linking_model.S2_cand_mask_4:S2_cand_mask_4,
                  ent_linking_model.keep_prob:1.0,
                  ent_linking_model.keep_prob_V:1.0,
                  ent_linking_model.keep_prob_D:1.0,
                  ent_linking_model.lr:args.learning_rate_start,
                  ent_linking_model.is_training:False,
                  ent_linking_model.sample_size:sample_size
              }
    
    doc_fid = record['_id']
    file_dir= 'data/'+'EntityLinking'+'/elmo/'+local_score_path+'/'+str(doc_fid)+'.npy'
    elmo_embed = np.load(file_dir)
    feed_dict[ent_linking_model.ment_surface_elmo_embed]=elmo_embed
    
    prior_score,final_score,mask= sess.run([ent_linking_model.p_e_m,ent_linking_model.l_final_score,
                                 ent_linking_model.mask],feed_dict
                      )
    
    wrong,right,new_final_score=revise_prior_score(ment_info_batch,ment_cands_id_batch,ment_cands_prob_batch,
                                   wiki_id_2_name,id2ent,ent_person,
                                   ment_cands_tag_batch,final_score,ment_cands_prob_batch)
    
    #p_e_ctx = p_e_ctx * mask
    #we extract top 5 based on prior and top 5 based on ctx score
    ment_cands_id_batch_sub = []
    ment_cands_prob_batch_sub=[]
    ment_cands_lent_batch_sub=[]
    ment_cands_tag_batch_sub = []
    ment_doc_ctx_id_batch_sub = []
    ment_sent_ctx_id_batch_sub=[]
    ment_surface_id_batch_sub=[]
    ment_surface_elmo_embed_sub = []
    ment_local_p_score_batch_sub=[]
    ment_info_batch_sub = []
    ment_sent_left_ctx_id_batch_sub=[]
    ment_sent_right_ctx_id_batch_sub =[]
    
    ment_sid_batch_sub=[]
    for j in range(sample_size):
      ment_no = j
      gold_id = np.argsort(ment_cands_tag_batch[ment_no]*(-1))[0]
      
      ment_cands_local_p_score_sub_i=[]
      ment_cands_id_sub_i = []
      ment_cands_prob_batch_sub_i = []
      ment_cands_tag_batch_sub_i = [0]*filter_cands
      ment_cands_lent_batch_sub_i = 0
      
      
      
      prior_score_top4_id = list(np.argsort(ment_cands_prob_batch[ment_no]*(-1))[:3])
      random.shuffle(prior_score_top4_id)
      for ids in prior_score_top4_id:
        cand_id = ment_cands_id_batch[ment_no][ids]
        if cand_id==0:
          continue
        prob = ment_cands_prob_batch[ment_no][ids]
        if cand_id not in ment_cands_id_sub_i:
          ment_cands_id_sub_i.append(cand_id)
          ment_cands_prob_batch_sub_i.append(prob)
          ment_cands_local_p_score_sub_i.append(new_final_score[ment_no,ids])
          
      ment_pred_score=new_final_score[ment_no]  #we just top 7 from the local model...
      
      ctx_score_top3_id = list(np.argsort(ment_pred_score*(-1)))
      #random.shuffle(ctx_score_top3_id)
      
      for ids in ctx_score_top3_id:
        if len(ment_cands_id_sub_i)==filter_cands:
          break
        
        
        cand_id = ment_cands_id_batch[ment_no,ids] 
        if cand_id==0:
          continue
        
        prob = ment_cands_prob_batch[ment_no,ids]
        if cand_id not in ment_cands_id_sub_i:
          ment_cands_id_sub_i.append(cand_id)
          ment_cands_prob_batch_sub_i.append(prob)
          ment_cands_local_p_score_sub_i.append(new_final_score[ment_no,ids])

      
      cand_num = len(ment_cands_id_sub_i)
      #print cand_num
      if cand_num < filter_cands:
        ment_cands_id_sub_i +=[0]*(filter_cands-cand_num)
        ment_cands_prob_batch_sub_i +=[0]*(filter_cands-cand_num)
        ment_cands_local_p_score_sub_i+=[0]*(filter_cands-cand_num)
      
      #杩欓噷鎴戜滑闇€瑕佺‘瀹氫竴涓媑old id鏄惁鍐嶅唴鍛紵
      gold_cand_id = ment_cands_id_batch[ment_no,gold_id]
      
      if tag=='train':
        if gold_cand_id not in ment_cands_id_sub_i:
          ment_cands_id_sub_i[-1]=gold_cand_id
          ment_cands_prob_batch_sub_i[-1]=ment_cands_prob_batch[ment_no,gold_id]
          ment_cands_local_p_score_sub_i[-1]=ment_pred_score[gold_id]
      
      if gold_cand_id in ment_cands_id_sub_i:
        for tag_id,cand_id in enumerate(ment_cands_id_sub_i):
          if gold_cand_id == cand_id:
            ment_cands_tag_batch_sub_i[tag_id] = 1
            
      assert(len(ment_cands_prob_batch_sub_i)==filter_cands)    
      
      all_ment+= 1
      
      #assert(sum(ment_cands_tag_batch_sub_i)!=0)   #杩欎釜灏卞緢闅捐В閲婂暒锛?
      if sum(ment_cands_tag_batch_sub_i)==0:
        nil_ment += 1
        #print 'do not recall the right entity:',nil_ment
      else:
        ment_sid_batch_sub.append(ment_sid_batch[j])
        ment_cands_id_batch_sub.append(ment_cands_id_sub_i)
        ment_cands_prob_batch_sub.append(ment_cands_prob_batch_sub_i)
        ment_cands_lent_batch_sub.append(ment_cands_lent_batch_sub_i)
        ment_cands_tag_batch_sub.append(ment_cands_tag_batch_sub_i) 
        ment_doc_ctx_id_batch_sub.append(ment_doc_ctx_id_batch[ment_no])
        ment_sent_ctx_id_batch_sub.append(ment_sent_ctx_id_batch[ment_no])
        ment_surface_id_batch_sub.append(ment_surface_id_batch[ment_no])
        ment_surface_elmo_embed_sub.append(elmo_embed[ment_no])
        ment_info_batch_sub.append(list(ment_info_batch[ment_no]))
        ment_sent_left_ctx_id_batch_sub.append(ment_sent_left_ctx_id_batch[ment_no])
        ment_sent_right_ctx_id_batch_sub.append(ment_sent_right_ctx_id_batch[ment_no])
        ment_local_p_score_batch_sub.append(ment_cands_local_p_score_sub_i)
    
    ment_cands_id_batch_sub = np.asarray(ment_cands_id_batch_sub,dtype=np.int32)
    ment_cands_lent_batch_sub = np.asarray(ment_cands_lent_batch_sub,dtype=np.int32)
    ment_cands_prob_batch_sub = np.asarray(ment_cands_prob_batch_sub,dtype=np.float32)
    ment_cands_tag_batch_sub = np.asarray(ment_cands_tag_batch_sub,dtype=np.float32)
    ment_doc_ctx_id_batch_sub = np.asarray(ment_doc_ctx_id_batch_sub,dtype=np.int32)
    ment_sent_ctx_id_batch_sub = np.asarray(ment_sent_ctx_id_batch_sub,dtype=np.int32)
    ment_surface_id_batch_sub = np.asarray(ment_surface_id_batch_sub,dtype=np.int32)
    ment_surface_elmo_embed_sub = np.asarray(ment_surface_elmo_embed_sub,dtype=np.float32)
    ment_sent_left_ctx_id_batch_sub = np.asarray(ment_sent_left_ctx_id_batch_sub,dtype=np.int32)
    ment_sent_right_ctx_id_batch_sub = np.asarray(ment_sent_right_ctx_id_batch_sub,dtype=np.int32)
    ment_local_p_score_batch_sub = np.asarray(ment_local_p_score_batch_sub,dtype=np.float32)
    
    
#    print np.shape(ment_cands_id_batch_sub)
#    print np.shape(ment_cands_lent_batch_sub)
#    print np.shape(ment_cands_prob_batch_sub)
#    print np.shape(ment_cands_tag_batch_sub)
#    print np.shape(ment_ctx_id_batch_sub)
    
    record={}
    record['aNo']=aNo
    record['m2w']=m2w
    record['aNo_words']=aNo_words
    record['sent_index_batch']=sent_index_batch
    record['ment_sid_batch'] = ment_sid_batch_sub
    
    record['ment_cands_id_batch'] = Binary(pickle.dumps(ment_cands_id_batch_sub,protocol=2))
    record['ment_cands_lent_batch'] = Binary(pickle.dumps(ment_cands_lent_batch_sub,protocol=2))
    record['ment_cands_prob_batch'] = Binary(pickle.dumps(ment_cands_prob_batch_sub,protocol=2))
    record['ment_cands_tag_batch'] = Binary(pickle.dumps(ment_cands_tag_batch_sub,protocol=2))
    record['ment_doc_ctx_id_batch'] = Binary(pickle.dumps(ment_doc_ctx_id_batch_sub,protocol=2))
    record['ment_sent_ctx_id_batch'] = Binary(pickle.dumps(ment_sent_ctx_id_batch_sub,protocol=2))
    record['ment_sent_left_ctx_id_batch'] = Binary(pickle.dumps(ment_sent_left_ctx_id_batch_sub,protocol=2))
    record['ment_sent_right_ctx_id_batch'] = Binary(pickle.dumps(ment_sent_right_ctx_id_batch_sub,protocol=2))
    record['ment_surface_id_batch'] = Binary(pickle.dumps(ment_surface_id_batch_sub,protocol=2))
    record['ment_info'] = ment_info_batch_sub
    record['ment_local_p_score_batch']=Binary(pickle.dumps(ment_local_p_score_batch_sub,protocol=2))
    record['ment_surface_elmo_embed']=Binary(pickle.dumps(ment_surface_elmo_embed_sub,protocol=2))
    
    test_collection_sub.insert_one(record)
  print(tag, all_ment, nil_ment, all_ment-nil_ment)
  
def get_ret(aNo,args,tag,word_freq_loader,
            id2ent,wiki_id_2_name,ent_2_mid,prior,
            ment_info,test_size,ment_surface_id_batch,
            ment_cands_id_batch,ment_cands_tag_batch,
            ment_cands_prob_batch,
            ment_doc_ctx_id_batch,
            final_score,norm_ent_adj):
  
  aNo_pred= {}
  right_tag = np.argmax(ment_cands_tag_batch,-1)
  pred_tag = np.argmax(final_score,-1)
  
  ent_level_no=5
  wrong_ent = [0 for i in range(ent_level_no)]
  high_ent=[0 for i in range(ent_level_no)]
  wrong_high_max_sum=[0 for i in range(ent_level_no)]
  wrong_high_max_num=[0 for i in range(ent_level_no)]
  
  
  ment_name_list=[]
  
  for j in range(test_size):
    ment_name=''
    for wd_id in ment_surface_id_batch[j]:
      if wd_id == 0:
        break
      ment_name = ment_name +' '+word_freq_loader.id2word[wd_id].lower()
    ment_name_list.append(ment_name.strip())
  
  for j in range(test_size):
    pred_j = pred_tag[j]
    pred_j_ent = ment_cands_id_batch[j][pred_j]
    right_j = right_tag[j]
    right_j_ent=ment_cands_id_batch[j][right_j]
    
    sub_norm_ent_adj=norm_ent_adj[j*args.cand_nums:(j+1)*args.cand_nums,:]
    if pred_j_ent!=right_j_ent:
      flag='Incorrect'
    else:
      flag='Correct'
    
    max_pred_score_j = final_score[j][pred_tag[j]] 
    right_entid = ment_cands_tag_batch[j][right_tag[j]]
    pred_entid =  ment_cands_tag_batch[j][pred_tag[j]]
    aNo_pred['\t'.join(ment_info[j])] = list([flag,max_pred_score_j,right_entid,pred_entid])
    
    #print(j*args.cand_nums,(j+1)*args.cand_nums)
    ment_name=ment_name_list[j]
    
    sent_list=[]
    for wd_id in ment_doc_ctx_id_batch[j]:
      if wd_id == 0:
        break
      sent_list.append(word_freq_loader.id2word[wd_id])
    
    gold_no = np.argmax(ment_cands_tag_batch[j])
    pred_no = np.argmax(final_score[j])
    #gold_no_idx = get_index( ment_cands_prob_batch[j][gold_no])
    gold_no_idx = get_score_index( ment_cands_prob_batch[j][gold_no])
    
    
    high_ent[gold_no_idx] += 1
    
    if True:
      max_cand_prob = max(list(ment_cands_prob_batch[j]))
      
      if flag=='Incorrect':
        wrong_ent[gold_no_idx] += 1
        wrong_high_max_sum[gold_no_idx] += max_cand_prob
        wrong_high_max_num[gold_no_idx] +=1
        
        if gold_no_idx==9:
          
          max_cand_prob = max(list(ment_cands_prob_batch[j]))
          max_cand_prob_idx = get_index_minor(max_cand_prob)
          
          
          wrong_high_max_sum[max_cand_prob_idx] += max_cand_prob
          wrong_high_max_num[max_cand_prob_idx] +=1
          
          max_cand_prob = ment_cands_prob_batch[j][gold_no]
          max_cand_prob_idx = get_index_minor(max_cand_prob)
          
          wrong_high_max_sum[max_cand_prob_idx] += max_cand_prob
          wrong_high_max_num[max_cand_prob_idx] +=1
      
      if aNo==216:
        print(test_size,j,'ment:',ment_name,'gold no:',gold_no,'pred_no:',pred_no,flag)
        print(' '.join(sent_list))
      
      for ids_i in range(args.cand_nums):
      #for ids_i in [gold_no,pred_no]:
        norm_adj = np.reshape(sub_norm_ent_adj[ids_i],[test_size,args.cand_nums])
        ids = ment_cands_id_batch[j][ids_i]
        wiid = id2ent[ids]
        if wiid in wiki_id_2_name:
          wiki_name=wiki_id_2_name[wiid]
        else:
          wiki_name=None
        
        if aNo==216:
          print(ids_i,ent_2_mid[ids],wiid,wiki_name, 
              np.round(prior[j][ids_i],4),
              np.round(ment_cands_prob_batch[j][ids_i],4)
              ,np.round(final_score[j][ids_i],4))
        
        link_2_gold_num=0.0
        
        for idx_j in range(test_size):
          ids_j = np.argmax(norm_adj[idx_j,:])
          score_j = np.max(norm_adj[idx_j,:])
          #wujs
          wiid_j = id2ent[ment_cands_id_batch[idx_j,ids_j]]
          if wiid_j in wiki_id_2_name:
            wiki_name_j=wiki_id_2_name[wiid_j]
          else:
            wiki_name_j=None
            
          if ment_cands_tag_batch[idx_j,ids_j]==1.0:
            link_2_gold_num += 1
          if aNo==216:
            print([
                 idx_j,score_j,wiid_j,wiki_name_j,
                 ment_cands_tag_batch[idx_j,ids_j],
                 ment_cands_prob_batch[idx_j,ids_j],
                 final_score[idx_j,ids_j]
                 ])
        if aNo==216:
          print('link 2 gold num:',link_2_gold_num)
          print('----------------------------------')
      if aNo==216:
        print('------------------------------')
        print('------------------------------')
        print('------------------------------')
      
  return wrong_ent,high_ent,wrong_high_max_sum,wrong_high_max_num

def get_index(score):
  if score>= 0.9:
    return 0
  elif score>= 0.8:
    return 1
  elif score>= 0.7:
    return 2
  elif score>= 0.6:
    return 3
  elif score>= 0.5:
    return 4
  elif score>= 0.4:
    return 5
  elif score>= 0.3:
    return 6
  elif score>= 0.2:
    return 7
  elif score>= 0.1:
    return 8
  else:
    return 9

def get_score_index(score):
  if score>= 0.3:
    return 0
  elif score>= 0.1:
    return 1
  elif score>= 0.03:
    return 2
  elif score>= 0.01:
    return 3
  else:
    return 4
    
def get_index_minor(score):
  score_dict={i:0.01*(9-i) for i in range(10)}
   
  if score >= score_dict[0]:
    return 0
  
  for i in range(1,10):
    if score<score_dict[i-1] and score>=score_dict[i]:
      return i
  
  return 9