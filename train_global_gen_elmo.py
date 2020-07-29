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
from train_utils import get_test_ret,get_mini_batch,get_training_stop_flag,get_elmo_embedding
from input_utils import gen_ent_person,get_doc_sent_for_ments
import random
import sys
import pickle
from model import GraphCNNGlobalEntLinkModel,GraphCNNGlobalRandomEntLinkModel,\
                  GraphCNNCtxGlobalEntLinkModel,GraphCNNCtxNSGlobalEntLinkModel
from embeddings import WordFreqVectorLoader
from entity import load_entity_vector,load_ent_w2v_embed,\
                   load_entity_vector_unk,get_disam_wiki_id_name,\
                   load_ent_transE_embed
from pymongo import MongoClient
import json
from tqdm import tqdm
import tensorflow_hub as hub

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
flags.DEFINE_integer("s2_width_0_1",5," s2_witdh_0")
flags.DEFINE_integer("s2_width_1",5," s2_witdh_1")
flags.DEFINE_integer("s2_width_2",5," s2_witdh_2")
flags.DEFINE_integer("word_dim",300,"word embedding")
flags.DEFINE_boolean("dropout",True,"apply dropout during training")
flags.DEFINE_boolean("test",False,"apply dropout during training")
flags.DEFINE_float("max_norm",0.0,"[0.001,0.0001] max_norm")
flags.DEFINE_float("keep_prob",0.8,"[0.001,0.0001]")
flags.DEFINE_float("keep_prob_D",1.0,"[0.001,0.0001]")
flags.DEFINE_float("keep_prob_V",0.9,"[0.001,0.0001]")
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
flags.DEFINE_boolean("diag_self_train",True,"Flase")
flags.DEFINE_string("A_adj_mask",'mask',"use mask")
flags.DEFINE_string("S12_score_mask",'unmask',"use mask")
flags.DEFINE_string("gcn_activation",'relu',"[relu, tanh]")
flags.DEFINE_boolean("gcn_hidden_V",True,"gcn_hidden_V")
flags.DEFINE_string("gcn_weighted_sum",'ZT',"ZT or WS")
flags.DEFINE_string("A_diag",'diag',"diag or dense")
flags.DEFINE_float("residual_w",0.0,"residual weights")
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

flags.DEFINE_integer("data_tag_number",0,"[0,1,2,3,4,5,6,7,8]")
flags.DEFINE_integer("data_item_number",0,"[0,1,2,3,4,5,6,7,8]")
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
  
  elmo = hub.Module("model/elmo2/", trainable=False)
  client = MongoClient('mongodb://192.168.3.196:27017')
  
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
  sess.run(init_op)
    
  db = client[args.data_source]
  if args.data_source !='EntityLinking':
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
  else:
    train_collection_wikipedia =list(db['wikipedia'].find({}))
    train_collection_clueweb = list(db['clueweb'].find({}))
    train_collection_aida = list(db['aida_train'].find({}))
    train_collection_records= train_collection_aida
    train_doc_id_lists = list(range(len(train_collection_records)))
    
    
    testa_collection = list(db["aida_testa"].find({}))
    testb_collection = list(db["aida_testb"].find({}))
    ace2004_collection = list(db["ace2004"].find({}))    
    msnbc_collection = list(db["msnbc"].find({}))
    aquaint_collection = list(db["aquaint"].find({}))
  
  
  tag_list = ['train','testa','testb','msnbc','aquaint','ace2004','clueweb','wikipedia']
    
  collection_list = [train_collection_aida,testa_collection,testb_collection,
                         msnbc_collection,aquaint_collection,ace2004_collection,
                         train_collection_clueweb,train_collection_wikipedia]
  
  data_dict={}
  
  for i in [args.data_tag_number]:
    
    tag = tag_list[i]
    
    if tag in ['train','testa','testb']:
      col_name = 'aida'+'_'+tag
    else:
      col_name = tag
    
    base_path = 'data/'+args.data_source+'/'
    if not os.path.exists(base_path):
      os.makedirs(base_path)
    elmo_path=base_path+'elmo/'
    if not os.path.exists(elmo_path):
      os.makedirs(elmo_path)
      
    tag_elmo_path=base_path+'elmo/'+col_name+'/'
    if not os.path.exists(tag_elmo_path):
      os.makedirs(tag_elmo_path)
    
    
    collection=collection_list[i][args.data_item_number*100:min((args.data_item_number+1)*100,len(collection_list[i]))]
      
    data_dict[tag]={}
    for record in tqdm(collection):
      doc_fid = record['_id']
      item=get_mini_batch(wiki_id_2_name,id2ent,ent_person,args.model_type,tag,10,
                                   args.cand_nums,word_freq_loader,record,
                                   args.self_other_mask,args.mask_type)
        
      aNo,m2w,aNo_words,sent_index_batch,ment_sid_batch,\
          ment_idx_list,ment_info,ment_cands_id_batch,ment_cands_lent_batch,\
          ment_cands_prob_batch, ment_cands_tag_batch,\
          ment_doc_ctx_id_batch,ment_sent_ctx_id_batch,ment_sent_left_ctx_id_batch,\
          ment_sent_right_ctx_id_batch,ment_surface_id_batch,\
          S2_cand_mask_4,cand_mask_pad,cand_mask_4,cand_mask_2,cand_adj_mask,\
          sample_size = item

      sent_list,sent_lent_list=get_doc_sent_for_ments(aNo_words,sent_index_batch,ment_sid_batch)
      ment_elmo_embed=get_elmo_embedding(sess,elmo,ment_info,sent_list,sent_lent_list)
      np.save(tag_elmo_path+'/'+str(doc_fid),ment_elmo_embed)
    
  sess.close()
      
if __name__=="__main__":
  tf.compat.v1.app.run()