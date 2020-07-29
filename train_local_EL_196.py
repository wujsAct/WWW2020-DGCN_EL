# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:56:32 2017

@author: wujs
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from train_utils import get_test_ret,get_mini_batch
import random
import sys
import pickle
from tqdm import tqdm
from model import SimpleCNNLocalEntLinkModel,SimpleCtxCNNLocalEntLinkModel
from embeddings import WordFreqVectorLoader
from entity import load_entity_vector,load_entity_vector_unk,get_disam_wiki_id_name
from pymongo import MongoClient
from train_utils import get_sub_cands
from input_utils import gen_ent_person

flags = tf.app.flags
flags.DEFINE_integer("epoch",30,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",128,"Epoch to train[25]")
flags.DEFINE_string("model_type",'CNNGlobalEntLinkModel'," loss_type")
flags.DEFINE_integer("entity_embedding_dims",300,"entity_embedding_dims")
flags.DEFINE_integer("ent_ctx_lent",100,"ent_ctx_lent")
flags.DEFINE_integer("re_train_num",0,"re_train_num")
flags.DEFINE_integer("start_num",0,"start_num")
flags.DEFINE_integer("cand_nums",30,"candidate entity nums")
flags.DEFINE_integer("prior_num",1,"prior_num [0,1,2,3,4,5]")
flags.DEFINE_integer("s2_width_0",5," s2_witdh_0")
flags.DEFINE_integer("s2_width_0_1",5," s2_witdh_0")
flags.DEFINE_integer("s2_width_0_elmo",20," s2_witdh_0")
flags.DEFINE_integer("s2_width_1",5," s2_witdh_1")
flags.DEFINE_integer("s2_width_2",5," s2_witdh_2")
flags.DEFINE_integer("word_dim",300,"word embedding")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_boolean("test",False,"apply dropout during training")
flags.DEFINE_float("max_norm",0.0,"[0.001,0.0001] max_norm")
flags.DEFINE_float("keep_prob",1.0,"[0.001,0.0001]")
flags.DEFINE_float("keep_prob_D",1.0,"[0.001,0.0001]")
flags.DEFINE_float("keep_prob_V",1.0,"[0.001,0.0001]")
flags.DEFINE_float("epsilon",1e-08,"[0.001,0.0001]")
flags.DEFINE_string("loss_type",'softmax'," loss_type")
flags.DEFINE_string("date",'4_5',"model date")
flags.DEFINE_float("l2_w",0.0," l2_w")
flags.DEFINE_float("margin_param",0.2,"0.5, we need to try other methods margin_param")
flags.DEFINE_float("learning_rate_start",1e-4,"start lr")
flags.DEFINE_float("learning_rate_end",1e-5,"end lr")
flags.DEFINE_string("logs_path","logs","path of saved model")
flags.DEFINE_string("restore","checkpoint","path of saved model")
flags.DEFINE_integer("ent_nums",200,"ent_nums")
flags.DEFINE_integer("word_nums",300,"word_nums")
flags.DEFINE_integer("rel_nums",300,"rel_nums")
flags.DEFINE_string("has_type",'T',"wether utilize the type information")
flags.DEFINE_string("feature",'S2',"S0,S1,S2")
flags.DEFINE_string("use_unk",'unk',"S0,S1,S2")
flags.DEFINE_integer("seed",0,"[10,20,30]")

flags.DEFINE_boolean("insert_sub",False,"apply dropout during training")
flags.DEFINE_integer("filter_cand_num",7,"7,10")
flags.DEFINE_boolean("self_other_mask",False,"False or True")
flags.DEFINE_string("mask_type",'dist_count',"[dist_count,window]")
flags.DEFINE_string("data_source",'EntityLinking',"entity_linking, EntityLinking")
flags.DEFINE_boolean("save_local_score",False,"apply dropout during training")


args = flags.FLAGS

def main(_):
  word_freq_loader = WordFreqVectorLoader(args)
  word_embedding = word_freq_loader.word_embedding
  
  args.word_nums = np.shape(word_embedding)[0]
  
  data_loader_args={}
  
  data_loader_args['word2id'] = word_freq_loader.word2id
  
  if args.use_unk =='none':
    ent2id,ent2type,entity_embedding,ent_2_mid = load_entity_vector(data_loader_args['word2id'],word_embedding)
  else:
    ent2id,ent2type,entity_embedding,ent_2_mid = load_entity_vector_unk(data_loader_args['word2id'],word_embedding)
    
  id2ent={ent2id[key]:key for key in ent2id}
  wiki_name_2_id,wiki_id_2_name=get_disam_wiki_id_name()
    
  mid_2_ent = {ent_2_mid[key]:key for key in ent_2_mid}
  
  args.ent_nums = np.shape(entity_embedding)[0]
  data_loader_args['ent2id'] = ent2id
  ent_person=gen_ent_person()
  
  if args.seed ==0:
    args.seed = np.random.randint(low=0,high=2**31-2)
  tf.set_random_seed(args.seed)
  
  
  if args.model_type == 'SimpleCtxCNNLocalEntLinkModel':
    ent_linking_model = SimpleCtxCNNLocalEntLinkModel(args)
  elif args.model_type == 'SimpleCNNLocalEntLinkModel':
    ent_linking_model = SimpleCNNLocalEntLinkModel(args)
  else:
    print('wrong model types '+args.model_type)
    exit(0)
    
  client = MongoClient('mongodb://192.168.3.196:27017')
  #client = MongoClient('mongodb://localhost:27017')
  db = client['EntityLinking']
  
  learning_rate=args.learning_rate_start
  
  
  train_collection_records= list(db['aida_train'].find({}))
  train_doc_id_lists = range(len(train_collection_records))
  
  testa_collection = list(db["aida_testa"].find({}))
  testb_collection = list(db["aida_testb"].find({}))
  ace2004_collection = list(db["ace2004"].find({}))    
  msnbc_collection = list(db["msnbc"].find({}))
  aquaint_collection = list(db["aquaint"].find({}))
  train_collection_wikipedia =list(db['wikipedia'].find({}))
  train_collection_clueweb = list(db['clueweb'].find({}))
  
  tag_list = ['train','testa','testb','msnbc','aquaint','ace2004','clueweb','wikipedia']
  collection_list = [train_collection_records,testa_collection,testb_collection,
                         msnbc_collection,aquaint_collection,ace2004_collection,
                         train_collection_clueweb,train_collection_wikipedia]
  
  data_dict={}
  for i in range(len(tag_list)):
    tag = tag_list[i]
    collection=collection_list[i]
    
    data_dict[tag]={}
    for record in tqdm(collection):
      doc_fid = record['_id']
      item=get_mini_batch(wiki_id_2_name,id2ent,ent_person,args.model_type,tag,7,
                                 args.cand_nums,word_freq_loader,record,
                                 args.self_other_mask,args.mask_type)
      
      data_dict[tag][str(doc_fid)]=item
  
  
  init_feed_dict={ent_linking_model.ent_embed_pl:entity_embedding,
                          ent_linking_model.word_embed_pl:word_embedding,
                          ent_linking_model.ent_type_embed_pl:np.asarray(ent2type,np.float32)
                  }
  
  re_init_flag=True
  
  #we need to choose initilization several times ....
  for re_init in range(1):
    if re_init_flag==False:
      print('%s' %('we have find a good initilizer for this parameters...'))
      break
    print('initilization:',re_init)
    model_save_str = '_'.join([
                               str(args.feature),
                               str(args.has_type),
                               str(args.loss_type),
                               str(args.learning_rate_start),
                               str(args.margin_param),
                               str(args.s2_width_0),
                               str(args.s2_width_0_1),
                               'kp'+str(args.keep_prob),
                               'l2'+str(args.l2_w),
                               'mn'+str(args.max_norm),
                               'sd'+str(args.seed),
                               'st'+str(args.start_num),
                             ])
    model_save_str=model_save_str+'_re:'+str(re_init)
    
    if args.self_other_mask ==True:
      model_save_str +='_som'  #'som':self other mask...
    
    if args.mask_type=='window':
      model_save_str=model_save_str+"_"+"window"
    
    model_save_str+='_elmo:'+str(args.s2_width_0_elmo)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter('logs/train', graph=sess.graph)
    init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    sess.run(init_op,init_feed_dict)
    
    print(model_save_str)
    if args.test==True:
      if args.insert_sub==False:
        if ent_linking_model.load(sess,args.restore,model_save_str+'_max'):
          print( "[*] ent_linking_model is loaded...")
        else:
          print("[*] There is no checkpoint for ent_linking_model_"+model_save_str)
          continue
      else:
        if ent_linking_model.load(sess,args.restore,model_save_str+'_max'):
          print( "[*] ent_linking_model_990 is loaded...")
        else:
          print("[*] There is no checkpoint for ent_linking_model_"+model_save_str)
          continue
      
      S0_type_w,S0_cand_w,\
      S1_type_diag_w,S1_cand_diag_w,\
      S2_type_diag_w,S2_cand_diag_w=sess.run([ent_linking_model.S0_type_w,ent_linking_model.S0_cand_w,
                                     ent_linking_model.score_diag_w_dict['S1_type_diag_w'],
                                     ent_linking_model.score_diag_w_dict['S1_cand_diag_w'],
                                     ent_linking_model.score_diag_w_dict['S2_type_diag_w'],
                                     ent_linking_model.score_diag_w_dict['S2_cand_diag_w']])
      print(np.linalg.norm(S0_type_w,ord=2))
      print(np.linalg.norm(S0_cand_w,ord=2))
      print(np.linalg.norm(S1_type_diag_w,ord=2))
      print(np.linalg.norm(S1_cand_diag_w,ord=2))
      print(np.linalg.norm(S2_type_diag_w,ord=2))
      print(np.linalg.norm(S2_cand_diag_w,ord=2))
      
      pr_list = []
      for i in range(len(tag_list)):
        tag = tag_list[i]
        
        link_loss,loss,accuracy,f1=get_test_ret(sess,args,tag,data_dict[tag],ent_linking_model,
                                                collection_list[i],ent_2_mid,mid_2_ent,word_freq_loader,
                                                wiki_id_2_name,id2ent,ent_person)
        if tag!='train' and tag!='testa':
          if tag=='testb':
            pr_list.append(str(accuracy))
          else:
            pr_list.append(str(f1))
      
      print(','.join(pr_list))
      
      if args.insert_sub:
        for filter_cand_num in [5]:
          for tag in tag_list:
            get_sub_cands(sess,args,word_freq_loader,tag,ent_linking_model,db,learning_rate,filter_cand_num,
                          wiki_id_2_name,id2ent,ent_person)
      continue
        
    elif args.re_train_num!=0:
      if ent_linking_model.load(sess,args.restore,model_save_str+'_max'):
        print( "[*] ent_linking_model"+ model_save_str+'_init'+" is loaded...")
      else:
        print("[*] There is no checkpoint for ent_linking_model_"+model_save_str)
    
    
    min_link_loss = sys.float_info.max
    max_testa_f1 = sys.float_info.min
    max_testb_f1 = sys.float_info.min
    max_testa_acc,max_testb_acc = sys.float_info.min,sys.float_info.min
    
    lr_flag = True
    
    testa_accuracy_list=[]
    testa_link_loss,testa_loss,\
    testa_accuracy,testa_f1 = get_test_ret(sess,args,tag_list[1],data_dict[tag_list[1]],ent_linking_model,
                                           collection_list[1],ent_2_mid,mid_2_ent,word_freq_loader,
                                           wiki_id_2_name,id2ent,ent_person)
    
    if args.re_train_num!=0:
      testa_accuracy_list=[0,0]
      for tt in range(4):
        testa_accuracy_list.append(testa_accuracy)
    
    if testa_f1>=max_testa_f1 and testa_link_loss <= min_link_loss: 
      max_testa_f1 = testa_f1
      min_link_loss=testa_link_loss
      max_testa_acc=testa_accuracy
      
      ent_linking_model.save(sess,args.restore,model_save_str)
    
    testb_link_loss,testb_loss,\
    testb_acc,testb_f1=get_test_ret(sess,args,tag_list[2],data_dict[tag_list[2]],ent_linking_model,
                                           collection_list[2],ent_2_mid,mid_2_ent,word_freq_loader,
                                           wiki_id_2_name,id2ent,ent_person)
    if max_testb_f1<=testb_f1:
      max_testb_f1=testb_f1
      max_testb_acc=testb_acc
      ent_linking_model.save(sess,args.restore,model_save_str+'_max')
    
    print(model_save_str)
    #coref_num_list = []
    #key: step
    #val:[aida_A,aida_B,ace2004,msnbc,aquaint,wiki,clueweb]
    result_list=[]
    #key: epoch
    #val: train_loss
    train_loss_list=[]
    stime = time.time()
    
    train_flag = True
    for i in range(args.epoch):
      local_score_path='aida_train'
      doc_num = 0
      
      random.shuffle(train_doc_id_lists)
      random.shuffle(train_doc_id_lists)
      all_loss = 0.0
      if train_flag==False:
        break
      
      doc_num+=1
      for rnd_idx in train_doc_id_lists:
        record = train_collection_records[rnd_idx]
        doc_fid = record['_id']
        item = data_dict['train'][str(doc_fid)]
        
        if train_flag==False:
          break
      
          
        aNo,m2w,aNo_words,sent_index_batch,ment_sid_batch,\
        ment_idx_list,ment_info,ment_cands_id_batch,ment_cands_lent_batch,\
          ment_cands_prob_batch, ment_cands_tag_batch,\
          ment_doc_ctx_id_batch,ment_sent_ctx_id_batch,ment_sent_left_ctx_id_batch,\
          ment_sent_right_ctx_id_batch,ment_surface_id_batch,\
          S2_cand_mask_4,cand_mask_pad,cand_mask_4,cand_mask_2,cand_adj_mask,\
          sample_size = item
        
        feed_dict={
            ent_linking_model.ment_cand_id:ment_cands_id_batch,
            ent_linking_model.ment_cand_prob:ment_cands_prob_batch,
            ent_linking_model.linking_tag:ment_cands_tag_batch,
            ent_linking_model.ment_surface_ids:ment_surface_id_batch,
            ent_linking_model.ment_sent_ctx:ment_sent_ctx_id_batch,
            ent_linking_model.ment_sent_left_ctx:ment_sent_left_ctx_id_batch,
            ent_linking_model.ment_sent_right_ctx:ment_sent_right_ctx_id_batch,
            ent_linking_model.ment_doc_ctx:ment_doc_ctx_id_batch,
            ent_linking_model.ment_cands_lent:ment_cands_lent_batch,
            ent_linking_model.cand_mask_2d:cand_mask_2,
            ent_linking_model.cand_mask_4d:cand_mask_4,
            ent_linking_model.cand_mask_pad:cand_mask_pad,
            ent_linking_model.S2_cand_mask_4:S2_cand_mask_4,
            ent_linking_model.keep_prob:args.keep_prob,
            ent_linking_model.sample_size:sample_size,
            ent_linking_model.is_training:True,
            ent_linking_model.lr:learning_rate,
            ent_linking_model.keep_prob_V:args.keep_prob_V,
            ent_linking_model.keep_prob_D:args.keep_prob_D
            }
        file_dir= 'data/'+'EntityLinking'+'/elmo/'+local_score_path+'/'+str(doc_fid)+'.npy'
        feed_dict[ent_linking_model.ment_surface_elmo_embed]=np.load(file_dir)
          
        _,loss,link_loss,l2_loss,global_step,summary = sess.run([ent_linking_model.trainer,
                                               ent_linking_model.loss,
                                               ent_linking_model.link_loss,
                                               ent_linking_model.l2_loss,
                                               ent_linking_model.global_step,
                                               ent_linking_model.merged_summary_op
                                                ],feed_dict
                          )
        all_loss += loss
        summary_writer.add_summary(summary, global_step)
        sess.run(ent_linking_model.clip_all_weights) #we do not constrain..
        if loss <0.0:
          exit(0)
        
        '''
        if  global_step % 1000 == 0 or global_step==1:
          accuracy,rel_lr,final_score,prior_score,p_e_ctx_feature\
          =sess.run([ent_linking_model.accuracy,
                              ent_linking_model.rel_lr,
                              ent_linking_model.final_score,
                              ent_linking_model.p_e_m,
                              ent_linking_model.local_p_e_ctx_score,
                              ], feed_dict)
          print ('[Time:%f, Epoch: %d, steps:%d] train: link_loss %.6f, loss: %.6f,l2_loss %.6f, Acc: %.2f, lr:%.6f' \
               %(time.time()-stime,i,global_step,link_loss, loss,l2_loss, accuracy*100,rel_lr))
        '''
        interval = 1000
        
#        #testa_accuracy is very low or testa_accuracy is stable...
#        #testa
#        if (global_step>30000 and
#            (
#             (testa_accuracy_list[-1]==testa_accuracy_list[-2]
#                   and testa_accuracy_list[-2]==testa_accuracy_list[-3]
#                   and testa_accuracy_list[-3]==testa_accuracy_list[-4]
#                   and testa_accuracy_list[-4]==testa_accuracy_list[-5]
#                   )
#             )
#             ):  #the performance of validation is bad 
#          train_flag=False
#          sess.close()
#          break #back the training process...
#            
        if global_step %interval==0 or global_step==1:
          print('--------------------------------------')
          print('--------------------------------------')
          testa_link_loss,testa_loss,\
          testa_accuracy,testa_f1 = get_test_ret(sess,args,tag_list[1],data_dict[tag_list[1]],
                                                 ent_linking_model,collection_list[1],
                                                 ent_2_mid,mid_2_ent,word_freq_loader,
                                                 wiki_id_2_name,id2ent,ent_person)
          testa_accuracy_list.append(testa_accuracy)
          
          #revise: 2018-5-20
          if min_link_loss >=testa_loss:
            min_link_loss = testa_loss
            ent_linking_model.save(sess,args.restore,model_save_str+'_MinLoss')
          
          if testa_f1>=max_testa_f1:
            max_testa_f1 = testa_f1
            max_testa_acc = testa_accuracy
            ent_linking_model.save(sess,args.restore,model_save_str)
          
          testb_link_loss,testb_loss,\
          testb_acc,testb_f1=get_test_ret(sess,args,tag_list[2],data_dict[tag_list[2]],
                                                 ent_linking_model,collection_list[2],
                                                 ent_2_mid,mid_2_ent,word_freq_loader,
                                                 wiki_id_2_name,id2ent,ent_person)
          
          if max_testb_f1<=testb_f1:
            max_testb_f1=testb_f1
            max_testb_acc = testb_acc
            
            ent_linking_model.save(sess,args.restore,model_save_str+'_max')
            
          ret_list=[global_step,testa_accuracy,testa_f1,
                    testb_acc,testb_f1]
          print('-------------------------------------------------------------------------')
          print('-------------------------------------------------------------------------')
          for tag_id in range(3,len(tag_list)-2):
            tag_link_loss,tag_loss,\
            tag_acc,tag_f1=get_test_ret(sess,args,tag_list[tag_id],data_dict[tag_list[tag_id]],
                                                 ent_linking_model,collection_list[tag_id],
                                                 ent_2_mid,mid_2_ent,word_freq_loader,
                                                 wiki_id_2_name,id2ent,ent_person)
            ret_list.append(tag_acc)
            ret_list.append(tag_f1)
          result_list.append(ret_list)
      
      print('[Epoch:',i,'] ', all_loss)
      train_loss_list.append([i,all_loss])
      '''
      #2019-4-9
      #early stop, training loss do not decrease...
      
      if get_training_stop_flag(train_loss_list):
        train_flag=False
        sess.close()
        break'''
      
      log_file='logs/EntityLinking/'+args.model_type
      if not os.path.exists(log_file):
        os.makedirs(log_file)
      date_log_file=log_file+'/'+args.date
      if not os.path.exists(date_log_file):
        os.makedirs(date_log_file)
      
      ret_loss_performance={'result_list':result_list,
                            'train_loss_list':train_loss_list,
                            'max_testa':max_testa_acc,
                            'max_testb':max_testb_acc}
      pickle.dump(ret_loss_performance,open(date_log_file+'/'+model_save_str+'_ret.pkl','wb'))
      
      if global_step/len(train_doc_id_lists)== 990:
        print('epoch is 990')
        print('break down...')
        ent_linking_model.save(sess,args.restore,model_save_str+'_990')
        exit(0)
    #if max(testa_accuracy_list) >=0.91:
    #  re_init_flag=False
    sess.close()
      
if __name__=="__main__":
  tf.app.run()