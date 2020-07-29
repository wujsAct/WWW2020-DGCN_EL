# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:56:32 2017

@author: wujs
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sys import version_info
version = version_info.major
import tensorflow as tf
import numpy as np
import time
from train_utils import get_test_ret,get_mini_batch,get_training_stop_flag
from input_utils import gen_ent_person,get_doc_sent_for_ments
import random
import sys
import pickle
from model import GraphCNNGlobalEntLinkModel,GraphCNNGlobalRandomEntLinkModel,\
                  GraphCNNCtxGlobalEntLinkModel,GraphCNNCtxGlobalMCEntLinkModel
from embeddings import WordFreqVectorLoader
from entity import load_entity_vector,load_ent_w2v_embed,\
                   load_entity_vector_unk,get_disam_wiki_id_name,\
                   load_ent_transE_embed
from pymongo import MongoClient
import json
from tqdm import tqdm

flags = tf.app.flags
flags.DEFINE_integer("epoch",30,"Epoch to train[25]")
flags.DEFINE_integer("batch_size",128,"Epoch to train[25]")
flags.DEFINE_string("model_type",'CNNGlobalEntLinkModel'," loss_type")
flags.DEFINE_integer("entity_embedding_dims",300,"entity_embedding_dims")
flags.DEFINE_integer("ent_ctx_lent",100,"ent_ctx_lent")
flags.DEFINE_integer("re_train_num",0,"re_train_num")
flags.DEFINE_integer("start_num",0,"start_num")
flags.DEFINE_integer("cand_nums",10,"candidate entity nums")
flags.DEFINE_integer("prior_num",1,"prior_num [0,1,2,3,4,5]")
flags.DEFINE_integer("gcn_kernel_size",1," gcn_kernel_size")
flags.DEFINE_integer("s2_width_0",5," s2_witdh_0")
flags.DEFINE_integer("s2_width_0_elmo",5," s2_witdh_0")
flags.DEFINE_integer("s2_width_0_1",5," s2_witdh_0")
flags.DEFINE_integer("s2_width_1",5," s2_witdh_1")
flags.DEFINE_integer("s2_width_2",5," s2_witdh_2")
flags.DEFINE_integer("word_dim",300,"word embedding")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_boolean("test",False,"apply dropout during training")
flags.DEFINE_float("max_norm",0.0,"[0.001,0.0001] max_norm")
flags.DEFINE_float("keep_prob",0.8,"[0.001,0.0001]")
flags.DEFINE_float("keep_prob_D",1.0,"[0.001,0.0001]")
flags.DEFINE_float("keep_prob_V",1.0,"[0.001,0.0001]")
flags.DEFINE_float("epsilon",1e-08,"[0.001,0.0001]")
flags.DEFINE_string("loss_type",'margin'," loss_type")
flags.DEFINE_string("date",'4-5',"model date")
flags.DEFINE_float("reg_w",0.0," reg_w")
flags.DEFINE_float("l2_w",0.0," l2_w")
flags.DEFINE_float("link_w",1.0," link_w")
flags.DEFINE_float("margin_param",0.5,"0.5, we need to try other methods margin_param")
flags.DEFINE_float("learning_rate_start",1e-4,"start lr")
flags.DEFINE_float("learning_rate_end",1e-5,"end lr")
flags.DEFINE_string("logs_path","logs/global/","path of saved model")
flags.DEFINE_string("restore","checkpoint_global","path of saved model")
flags.DEFINE_integer("ent_nums",200,"ent_nums")
flags.DEFINE_integer("word_nums",300,"word_nums")
flags.DEFINE_integer("rel_nums",300,"rel_nums")
flags.DEFINE_string("use_unk",'none',"S0,S1,S2")
#2019-2-2
flags.DEFINE_integer("lbp_iter_num",5,"lbp_iter_num")
flags.DEFINE_float("diag_self",1.0,"diag_self")
flags.DEFINE_boolean("diag_self_train",False,"Flase")
flags.DEFINE_string("A_adj_mask",'mask',"use mask")
flags.DEFINE_string("S12_score_mask",'unmask',"use mask")
flags.DEFINE_string("gcn_activation",'relu',"[relu, tanh]")
flags.DEFINE_boolean("gcn_hidden_V",True,"gcn_hidden_V")
flags.DEFINE_string("gcn_weighted_sum",'ZT',"ZT or WS")
flags.DEFINE_string("A_diag",'diag',"diag or dense")
flags.DEFINE_float("residual_w",1.0,"residual weights")
flags.DEFINE_float("sim_scale",1.0,"sim_scale")
flags.DEFINE_integer("A_diag_dim",0,"[10,20,30]")
#721818695
flags.DEFINE_integer("seed",721818695,"[10,20,30]")
flags.DEFINE_string("optimizer",'normal',"norm or clip")
flags.DEFINE_string("data_source",'entity_linking',"entity_linking, EntityLinking")

#2019-4-26
flags.DEFINE_string("message_opt",'sum',"sum or max")
flags.DEFINE_boolean("self_other_mask",False,"False or True")
flags.DEFINE_string("mask_type",'dist_count',"[dist_count,window]")

flags.DEFINE_integer("s2_width_1_elmo",1,"[1,5]")
flags.DEFINE_string("score_merge_type",'average','average, attention, MLP')
flags.DEFINE_string("score_activation",'linear','tanh, relu')

args = flags.FLAGS


                                   
def main(_):
  if args.seed ==0:
    args.seed = np.random.randint(low=0,high=2**31-2)
    
  tf.set_random_seed(args.seed)
  
  
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
  
  if os.path.exists('data/ent_transE_embed.npy'):
    transE_entity_embedding=np.load('data/ent_transE_embed.npy')
  else:
    transE_entity_embedding=load_ent_transE_embed(ent2id)
    
  if os.path.exists('data/ent_w2v_embed.npy'):
    w2v_entity_embedding=np.load('data/ent_w2v_embed.npy')
  else:
    w2v_entity_embedding=load_ent_w2v_embed(ent2id)
  
  ent_person=gen_ent_person()
  wiki_name_2_id,wiki_id_2_name=get_disam_wiki_id_name()
  
  
  mid_2_ent = {ent_2_mid[key]:key for key in ent_2_mid}
  
  args.ent_nums = np.shape(entity_embedding)[0]
  data_loader_args['ent2id'] = ent2id
  
  if args.model_type == 'GraphCNNGlobalEntLinkModel':
    ent_linking_model = GraphCNNGlobalEntLinkModel(args)
  elif args.model_type == 'GraphCNNGlobalRandomEntLinkModel':
    ent_linking_model = GraphCNNGlobalRandomEntLinkModel(args)
  elif args.model_type == 'GraphCNNCtxGlobalEntLinkModel':
    ent_linking_model = GraphCNNCtxGlobalEntLinkModel(args)
  elif args.model_type == 'GraphCNNCtxGlobalMCEntLinkModel':
    ent_linking_model = GraphCNNCtxGlobalMCEntLinkModel(args)
  else:
    print('wrong model types '+args.model_type)
  
  
  client = MongoClient('mongodb://192.168.3.196:27017')
  #client = MongoClient('mongodb://localhost:27017')
  
  db = client[args.data_source]
  
  learning_rate=args.learning_rate_start
  train_collection_wikipedia =list(db['wikipedia_sub'].find({}))
  train_collection_clueweb = list(db['clueweb_sub'].find({}))
  train_collection_aida = list(db['aida_train_sub'].find({}))
  
  train_collection_records= train_collection_aida
  train_doc_id_lists = list(range(len(train_collection_records)))
  
  
  testa_collection = list(db["aida_testa_sub"].find({}))
  testb_collection = list(db["aida_testb_sub"].find({}))
  ace2004_collection = list(db["ace2004_sub"].find({}))    
  msnbc_collection = list(db["msnbc_sub"].find({}))
  aquaint_collection = list(db["aquaint_sub"].find({}))
  
  tag_list = ['train','testa','testb','msnbc','aquaint','ace2004','clueweb','wikipedia']
  
  collection_list = [train_collection_aida,testa_collection,testb_collection,
                       msnbc_collection,aquaint_collection,ace2004_collection,
                       train_collection_clueweb,train_collection_wikipedia]
  
  data_dict={}
  for i in range(len(tag_list)):
    tag = tag_list[i]
    collection=collection_list[i]
    
    data_dict[tag]={}
    for record in tqdm(collection):
      doc_fid = record['_id']
      item=get_mini_batch(wiki_id_2_name,id2ent,ent_person,args.model_type,tag,10,
                                 args.cand_nums,word_freq_loader,record,
                                 args.self_other_mask,args.mask_type)
      
      data_dict[tag][str(doc_fid)]=item
      
  '''
  @2019-5-3
  #we revise the entity without type to zeros for its type embedding
  #we revise the entity without mid w2v embedding to zeros for its type embedding
  '''
  print('w2v shape:',np.shape(np.asarray(w2v_entity_embedding,np.float32)))
  init_feed_dict={ent_linking_model.ent_embed_pl:entity_embedding,
                          ent_linking_model.word_embed_pl:word_embedding,
                          ent_linking_model.ent_type_embed_pl:np.asarray(ent2type,np.float32),
                          ent_linking_model.w2v_ent_embed_pl:np.asarray(w2v_entity_embedding,np.float32),
                          ent_linking_model.transE_ent_embed_pl:np.asarray(transE_entity_embedding,np.float32)
                  }
  re_init_flag=True
  clueweb_acc,clueweb_f1, wiki_acc,wiki_f1=0.0,0.0,0.0,0.0
  #we need to choose initilization several times ....
  for re_init in range(1):
    if re_init_flag==False:
      
      print('%s' %('we have find a good initilizer for this parameters...'))
      break
    print('initilization:',re_init)
    '''
    model_save_str = '_'.join([str(args.loss_type),str(args.cand_nums),str(args.learning_rate_start),
                             str(args.margin_param),str(args.s2_width_0),str(args.s2_width_0_1),
                             'lp'+str(args.lbp_iter_num),args.use_unk,
                             'l2'+str(args.l2_w),'regl2'+str(args.reg_w),
                             'mn'+str(args.max_norm),'kp'+str(args.keep_prob),
                             'vkp'+str(args.keep_prob_V),'rw'+str(args.residual_w),'sc'+str(args.sim_scale),
                             args.A_adj_mask,args.gcn_activation,str(args.A_diag_dim),str(args.gcn_kernel_size),                             'pr'+str(args.prior_num),'sd'+str(args.seed),'st'+str(args.start_num),
                             'ep'+str(args.epsilon)])'''
    
    model_save_str = '_'.join([str(args.margin_param),str(args.s2_width_0),
                               str(args.learning_rate_start),
                             'lp'+str(args.lbp_iter_num),'kp'+str(args.keep_prob),
                              str(args.A_diag_dim),str(args.gcn_kernel_size),
                             'sd'+str(args.seed),args.A_adj_mask,args.A_diag
                             ])
    '''
    model_save_str = '_'.join([str(args.margin_param),str(args.s2_width_0_1),
                               str(args.learning_rate_start),
                             'lp'+str(args.lbp_iter_num),'kp'+str(args.keep_prob),
                              str(args.A_diag_dim),str(args.gcn_kernel_size),
                             'sd'+str(args.seed),args.A_adj_mask,args.A_diag
                             ])'''
    
    model_save_str+='_'+'diagS:'+str(args.diag_self)
    model_save_str+='_'+'diagST:'+str(args.diag_self_train)
    
    if args.data_source!='EntityLinking':
      model_save_str +='_'+args.data_source
    
    if args.message_opt=='max':
      model_save_str +='_max'
  
    if args.mask_type!='dist_count':
      model_save_str +="_"+args.mask_type 
    
    model_save_str+='_elmo:'+str(args.s2_width_0_elmo)
    
    if args.s2_width_1_elmo!=1:
      model_save_str+='_elmo1:'+str(args.s2_width_1_elmo)
    
    if args.l2_w!=0.0:
      model_save_str+='_l2:'+str(args.l2_w)
      
    if args.reg_w!=0.0:
      model_save_str+='_regl2:'+str(args.reg_w)
    
    if args.date!='6_24':
      model_save_str+='_st:'+str(args.start_num)
      
      model_save_str+='_rst:'+str(re_init)
    
    if args.loss_type!='margin':
      model_save_str+='_lt:sum'
    
    if args.residual_w==0.0:
      model_save_str+='_nresi'
      
    if args.sim_scale!=1.0:
      model_save_str+='_sc:'+str(args.sim_scale)
    
    if args.score_merge_type!='MLP':
      model_save_str+= '_'+args.score_merge_type
    
    if args.score_activation!='linear':
      model_save_str+= '_'+args.score_activation
      
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    summary_writer = tf.summary.FileWriter('logs/train', graph=sess.graph)
    init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    sess.run(init_op,init_feed_dict)
    
    print(model_save_str)
    if args.test==True:
      if ent_linking_model.load(sess,args.restore,model_save_str+'_max'):
        print( "[*] ent_linking_model is loaded...")
      else:
        print("[*] There is no checkpoint for ent_linking_model_"+model_save_str+'_max')
        continue
    elif args.re_train_num!=0:
      if ent_linking_model.load(sess,args.restore,model_save_str+'_max'):
        print( "[*] ent_linking_model"+ model_save_str+'_init'+" is loaded...")
      else:
        print("[*] There is no checkpoint for ent_linking_model_"+model_save_str)
    
    S0_type_w,S0_cand_w,\
    S1_type_diag_w,S1_cand_diag_w,\
    S2_type_diag_w,S2_cand_diag_w=sess.run([ent_linking_model.S0_type_w,ent_linking_model.S0_cand_w,
                                   ent_linking_model.score_diag_w_dict['S1_type_diag_w'],
                                   ent_linking_model.score_diag_w_dict['S1_cand_diag_w'],
                                   ent_linking_model.score_diag_w_dict['S2_type_diag_w'],
                                   ent_linking_model.score_diag_w_dict['S2_cand_diag_w']]
                                   )
    print(S0_type_w[1,:10])
    print(np.linalg.norm(S0_type_w,ord=2))
    print(np.linalg.norm(S0_cand_w,ord=2))
    print(np.linalg.norm(S2_type_diag_w,ord=2))
    print(np.linalg.norm(S2_cand_diag_w,ord=2))
    
    
    if args.test == True:
      pr_list = []
      #for i in range(len(tag_list)):
      for i in [2]:
        tag = tag_list[i]
        
        link_loss,loss,accuracy,f1=get_test_ret(sess,args,tag,data_dict[tag],ent_linking_model
                                                ,collection_list[i],
                                                ent_2_mid,mid_2_ent,word_freq_loader,
                                                wiki_id_2_name,id2ent,ent_person)
        print(tag,link_loss,loss,accuracy,f1)
        if tag!='train' and tag!='testa':
          if tag=='testb':
            pr_list.append(accuracy)
          else:
            pr_list.append(f1)
        
      
      print(','.join(map(str,pr_list)))
      continue
    
    min_link_loss = sys.float_info.max
    max_testa_f1 = sys.float_info.min
    max_testb_f1 = sys.float_info.min
    max_testa_acc,max_testb_acc = sys.float_info.min, sys.float_info.min
    
    lr_flag = True
    testa_accuracy_list=[]
    testa_link_loss,testa_loss,\
    testa_accuracy,testa_f1 = get_test_ret(sess,args,'testa',data_dict['testa'],ent_linking_model,testa_collection,
             ent_2_mid,mid_2_ent,word_freq_loader,wiki_id_2_name,id2ent,ent_person)
    
    if args.re_train_num!=0:
      testa_accuracy_list=[0,0]
      for tt in range(4):
        testa_accuracy_list.append(testa_accuracy)
    
    if testa_f1>=max_testa_f1 and testa_link_loss <= min_link_loss: 
      max_testa_f1 = testa_f1
      max_testa_acc = testa_accuracy
      min_link_loss=testa_link_loss
      ent_linking_model.save(sess,args.restore,model_save_str)
    
    testb_link_loss,testb_loss,testb_acc,testb_f1=get_test_ret(sess,args,'testb',data_dict['testb'],
                                                               ent_linking_model,
                                                               testb_collection,
                                                               ent_2_mid,mid_2_ent,word_freq_loader,
                                                               wiki_id_2_name,id2ent,ent_person)
    if max_testb_f1<=testb_f1:
      max_testb_f1=testb_f1
      max_testb_acc = testb_acc
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
      doc_num = 0
      local_score_path='aida_train'
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
        
        if args.data_source=='EntityLinking':
          pre_train_local_score =np.load('data/local_score/'+\
                       local_score_path+'/'+str(doc_fid)+'.npy')
        else:
          #pre_train_local_score =np.load('data/gcn_local_np_score/'+\
          #             local_score_path+'/'+str(doc_fid)+'.npy')
          
          pre_train_local_score=np.load('data/'+args.data_source+'/'+
                                        local_score_path+'/'+str(record['_id'])+'.npy')
          '''
          rel_data_source = 'data/'+args.data_source+'/'+'mg_0.5'+'_lr_0.0001'
          pre_train_local_score = np.load(rel_data_source+'/'+local_score_path+'/'+str(record['_id'])+'.npy')'''
        
          '''
          if 'Global' in args.model_type:
            if version==2:
              pre_train_local_score = pickle.loads(record['ment_local_p_score_batch'])
            else:
              pre_train_local_score = pickle.loads(record['ment_local_p_score_batch'],
                                                           encoding='iso-8859-1')
            
            if version==2:
              pre_train_p_local_score = pickle.loads(record['ment_local_p_score_batch'])
            else:
              pre_train_p_local_score = pickle.loads(record['ment_local_p_score_batch'],
                                                           encoding='iso-8859-1')'''
        
        aNo,m2w,aNo_words,sent_index_batch,ment_sid_batch,\
        ment_idx_list,ment_info,ment_cands_id_batch,ment_cands_lent_batch,\
        ment_cands_prob_batch, ment_cands_tag_batch,\
        ment_doc_ctx_id_batch,ment_sent_ctx_id_batch,ment_sent_left_ctx_id_batch,\
        ment_sent_right_ctx_id_batch,ment_surface_id_batch,\
        S2_cand_mask_4,cand_mask_pad,cand_mask_4,cand_mask_2,cand_adj_mask,\
        sample_size = item
        
        #tt = random.choice(range(len(ment_cands_id_batch)))
        #print(sample_size,tt)
        #print(ment_sent_left_ctx_id_batch[tt])
        #print(ment_sent_right_ctx_id_batch[tt])
        #continue
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
            ent_linking_model.cand_adj_mask:cand_adj_mask,
            ent_linking_model.sample_size:sample_size,
            ent_linking_model.is_training:True,
            ent_linking_model.lr:learning_rate,
            ent_linking_model.keep_prob:args.keep_prob,
            ent_linking_model.keep_prob_V:args.keep_prob_V,
            ent_linking_model.keep_prob_D:args.keep_prob_D,
            }
        
        if 'Global' in args.model_type:
          feed_dict[ent_linking_model.local_ment_cand_prob] = pre_train_local_score[ment_idx_list]
          feed_dict[ent_linking_model.cand_mask_2d]=cand_mask_2
          feed_dict[ent_linking_model.cand_mask_4d]=cand_mask_4
          
          
          feed_dict[ent_linking_model.alpha]=m2w
          #file_dir= 'data/'+args.data_source+'/elmo/'+local_score_path+'/'+str(doc_fid)+'.npy'
          #feed_dict[ent_linking_model.ment_surface_elmo_embed]=np.load(file_dir)
          ment_surface_id_batch_org = pickle.loads( record['ment_surface_elmo_embed'] )
          feed_dict[ent_linking_model.ment_surface_elmo_embed]=ment_surface_id_batch_org
        
        _,loss,link_loss,reg_loss,global_step,summary,right_feature = sess.run([ent_linking_model.trainer,
                                               ent_linking_model.loss,
                                               ent_linking_model.link_loss,
                                               ent_linking_model.reg_loss,
                                               ent_linking_model.global_step,
                                               ent_linking_model.merged_summary_op,
                                               ent_linking_model.right_feature
                                                ],feed_dict
                          )
        
#        print('right-----------------------')
#        tt = random.choice(range(len(ment_cands_id_batch)))
#        print(ment_sent_left_ctx_id_batch[0])
#        print(ment_sent_right_ctx_id_batch[0])
#        print(right_feature[0][-1])
#        print('-----------------')
#        print(ment_sent_left_ctx_id_batch[tt])
#        print(ment_sent_right_ctx_id_batch[tt])
#        print(right_feature[tt][-1])
#        print('------------------------------')
#        print('------------------------------')
#        print('------------------------------')
        all_loss += loss
        summary_writer.add_summary(summary, global_step)
        sess.run(ent_linking_model.clip_all_weights) #we do not constrain..
        
        '''
        A_adj = sess.run(ent_linking_model.gcn_layers['norm_ent_adj_0_'+
                                                    str(args.lbp_iter_num-1)+'_'+\
                                                    str(args.gcn_kernel_size-1)],feed_dict)'''
        
        A_adj = sess.run(ent_linking_model.gcn_layers['norm_ent_adj_'+
                                                    str(args.lbp_iter_num-1)+'_'+\
                                                    str(args.gcn_kernel_size-1)],feed_dict)
        
        
        if  global_step % 3000 == 0 or global_step==1:
          accuracy,rel_lr,final_score,prior_score,pt_local_score,\
          p_e_ctx_feature,l2_loss,diag_val_weight=sess.run([ent_linking_model.accuracy,
                              ent_linking_model.rel_lr,
                              ent_linking_model.final_score,
                              ent_linking_model.p_e_m,
                              ent_linking_model.local_np_score,
                              ent_linking_model.local_p_e_ctx_score,
                              ent_linking_model.l2_loss,
                              ent_linking_model.diag_val_weight,
                              ], feed_dict)
          print ('[Time:%f, Epoch: %d, steps:%d] train: link_loss %.4f, loss: %.4f,reg_w %.4f, l2_w: %.4f, Acc: %.2f, diag_val_weight:%.3f, lr:%.5f' \
               %(time.time()-stime,i,global_step,link_loss,loss,reg_loss,l2_loss,accuracy*100,diag_val_weight,learning_rate))
          
          
          tt = random.choice(range(len(ment_cands_id_batch)))
          print('A_adj:',list(A_adj[tt*args.cand_nums,max(0,tt-6)*args.cand_nums:min(tt+6,sample_size)*args.cand_nums]))

          print('prior',list(ment_cands_prob_batch[tt]),prior_score[tt])
          print('pre-train local score',list(pre_train_local_score[tt]),pt_local_score[tt])
          print('pred score',list(final_score[tt]))
          
          if args.score_merge_type=='attention':
            ms_att_w=sess.run(ent_linking_model.merge_score_att_w)
            print('merge_score_att_w:',ms_att_w)
          '''
          if final_score[tt][0]<-0.5:
            train_flag=False
            sess.close()
            print('initilizaer is wrong...')
            break #back the training process...'''
            
          print('cand_id:',list(ment_cands_id_batch[tt]))
          print('glod_list:',list(ment_cands_tag_batch[tt]))
          print('--------------------------------')
          print('--------------------------------')
        
        interval = 1000
        
        #testa_accuracy is very low or testa_accuracy is stable...
        #testa
        #
        if args.model_type =='GraphCNNGlobalRandomEntLinkModel':
          stop_flag= i>5
        else:
          stop_flag = i>30
          
        if stop_flag and max_testa_acc<0.85:  #the performance of validation is bad 
          train_flag=False
          sess.close()
          print('initilizaer is wrong...')
          break #back the training process...
            
        if global_step %interval==0 or global_step==1:
          print('--------------------------------------')
          print('--------------------------------------')
          testa_link_loss,testa_loss,\
          testa_accuracy,testa_f1 = get_test_ret(sess,args,'testa',data_dict['testa'], ent_linking_model,testa_collection,
                                                 ent_2_mid,mid_2_ent,word_freq_loader,
                                                 wiki_id_2_name,id2ent,ent_person)
          
          testa_accuracy_list.append(testa_accuracy)
          #testa_f1>=max_testa_f1 and 
          if min_link_loss >=testa_loss:
            min_link_loss = testa_loss
            max_testa_f1 = testa_f1
            max_testa_acc = testa_accuracy
            ent_linking_model.save(sess,args.restore,model_save_str)
          
          testb_link_loss,testb_loss,testb_acc,testb_f1=get_test_ret(sess,args,'testb',data_dict['testb'],ent_linking_model,
                                                                     testb_collection,
                                                                     ent_2_mid,mid_2_ent,word_freq_loader,
                                                                     wiki_id_2_name,id2ent,ent_person)
          if max_testb_f1<=testb_f1:
            max_testb_f1=testb_f1
            max_testb_acc = testb_acc
            ent_linking_model.save(sess,args.restore,model_save_str+'_max')
          
          _,_,msnbc_acc,msnbc_f1=get_test_ret(sess,args,'msnbc',data_dict['msnbc'],ent_linking_model,msnbc_collection,
                                       ent_2_mid,mid_2_ent,word_freq_loader,wiki_id_2_name,id2ent,ent_person)
          _,_,aquaint_acc,aquaint_f1=get_test_ret(sess,args,'aquaint',data_dict['aquaint'],ent_linking_model,aquaint_collection,
                                         ent_2_mid,mid_2_ent,word_freq_loader,wiki_id_2_name,id2ent,ent_person)
          _,_,ace_acc,ace_f1=get_test_ret(sess,args,'ace2004',data_dict['ace2004'],ent_linking_model,ace2004_collection,
                                     ent_2_mid,mid_2_ent,word_freq_loader,wiki_id_2_name,id2ent,ent_person)
          
          if max_testb_f1<=testb_f1:
            _,_,clueweb_acc,clueweb_f1=get_test_ret(sess,args,'clueweb',data_dict['clueweb'],ent_linking_model,train_collection_clueweb,
                                           ent_2_mid,mid_2_ent,word_freq_loader,wiki_id_2_name,id2ent,ent_person)
            _,_,wiki_acc,wiki_f1=get_test_ret(sess,args,'wikipedia',data_dict['wikipedia'],ent_linking_model,train_collection_wikipedia,
                                           ent_2_mid,mid_2_ent,word_freq_loader,wiki_id_2_name,id2ent,ent_person)
          print('-------------------------------------------------------------------------')
          print('-------------------------------------------------------------------------')
          result_list.append([global_step,loss,
                              testa_link_loss,
                                   testa_accuracy,testa_f1,
                                   testb_acc,testb_f1,
                                   msnbc_acc,msnbc_f1,
                                   aquaint_acc,aquaint_f1,
                                   ace_acc,ace_f1,
                                   clueweb_acc,clueweb_f1,
                                   wiki_acc,wiki_f1
                                   ])
          
              
      print('[Epoch:',i,'] ', all_loss)
      train_loss_list.append([i,all_loss])
      '''
      #2019-4-9
      #early stop, training loss do not decrease...
      
      if get_training_stop_flag(train_loss_list):
        train_flag=False
        sess.close()
        break'''
      log_file='logs/'+args.data_source+'/'+args.model_type
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
      
    sess.close()
      
if __name__=="__main__":
  tf.app.run()