# -*- coding: utf-8 -*-

import tensorflow as tf
import math
from sys import version_info
version = version_info.major
if version==2:
  from base_model import Model
  from layers import margin_loss,BiLSTM
else:
  from .base_model import Model
  from .layers import margin_loss,BiLSTM

def get_top_k(a,mask,num_top,s2_width_0_1,cand_nums):
  a=tf.reshape(a,(-1,100))

  # Find top elements
  a_top, a_top_idx = tf.nn.top_k(a, num_top, sorted=False)
  # Apply softmax
  a_top_sm = tf.nn.softmax(a_top)
  # Reconstruct into original shape
  a_shape = tf.shape(a)
  a_row_idx = tf.tile(tf.range(a_shape[0])[:, tf.newaxis], (1, num_top))
  scatter_idx = tf.stack([a_row_idx, a_top_idx], axis=-1)  # generate scatter_index
  result = tf.scatter_nd(scatter_idx, a_top_sm, a_shape)  # generate matrix

  result=tf.reshape(result,(-1,cand_nums,s2_width_0_1,100))

  mask = tf.tile(tf.expand_dims(tf.expand_dims(mask,1),1),[1,cand_nums,s2_width_0_1,1])

  #we need to delete the mask word
  result = tf.multiply(result,mask)
  result = tf.div(result,tf.reduce_sum(result,-1,keepdims=True)+1e-8)

  return result

def get_seq_lent(x):
  used = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
  seq_length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)
  return seq_length

def max_norm_regularizer(threshold, axes=1, name="max_norm",collection="max_norm"):
  def max_norm(weights):
    clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
    clipped_weights = tf.assign(weights, clipped, name=name)
    tf.add_to_collection(collection,clipped_weights)
    return None
  return max_norm

class SimpleCtxCNNLocalEntLinkModel(Model):
  def __init__(self,args):
    super(SimpleCtxCNNLocalEntLinkModel, self).__init__()
    self.global_step = tf.Variable(0, trainable=False)
    self.args = args
    self.batch_size = args.batch_size
    #to enlarge the maximum constrain...
    self.max_norm_reg_1 = max_norm_regularizer(threshold=self.args.max_norm,axes=None)
    self.keep_prob = tf.placeholder(tf.float32,name='keep_prob_entity_embd')
    self.keep_prob_D = tf.placeholder(tf.float32,name='keep_prob_D')
    self.keep_prob_V = tf.placeholder(
          tf.float32,
          name='keep_prob_V'
        )
    self._init_placeholder()
    self._init_embeddings()

    self._init_main()
    self._init_optimizer()

    self.clip_all_weights = tf.get_collection("max_norm")
    print('clip_all_weights:',self.clip_all_weights)

  def _init_optimizer(self):
    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print('tvars:',tvars)

    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name])
    self.loss = self.link_loss + self.args.l2_w*self.l2_loss
    tf.summary.scalar("loss",self.loss)

    self.merged_summary_op = tf.summary.merge_all()

    self.rel_lr = self.lr
    self.optimizer =tf.train.AdamOptimizer(learning_rate=self.rel_lr)

    self.trainer = self.optimizer.minimize(self.loss,global_step=self.global_step)

  def _init_main(self):
    self._local_score()
    final_score=self.global_p_e_ctx[:,:,0]

    if self.args.loss_type=='softmax':
      self.l_final_score = final_score

      final_score_softmax = tf.nn.softmax(final_score)
      final_score_softmax = tf.multiply(tf.cast(self.mask,tf.float32),final_score_softmax)
      self.final_score =final_score_softmax/tf.reduce_sum(final_score_softmax,-1,keepdims=True)
      self.link_loss =  tf.reduce_mean(-tf.reduce_sum(self.linking_tag * tf.log(self.final_score+1e-8), axis=1))
    else:
      self.l_final_score = final_score
      self.final_score = final_score

      self.neg_mask = tf.subtract(tf.cast(tf.cast(self.ment_cand_id,tf.bool),tf.float32),
                                      tf.cast(self.linking_tag,tf.float32))
      self.link_loss =margin_loss(labels=self.linking_tag,logits=self.final_score,neg_mask=self.neg_mask,
                                                  margin_param=self.args.margin_param)

    correct_prediction = tf.equal(tf.argmax(self.linking_tag, 1), tf.argmax(self.final_score, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy",self.accuracy)
    self.right_ment_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

  def _gen_left_right_ctx(self):
    self.layers={}
    self.attention_dims=50
    self.rnn_size=150
    self.layers['BiLSTM'] = BiLSTM(self.rnn_size)
    self.layers['att_weights'] = {
    'h_m':tf.Variable(tf.truncated_normal([self.args.word_dim,self.attention_dims],stddev=0.01)),
    'h1': tf.Variable(tf.truncated_normal([2*self.rnn_size,self.attention_dims],stddev=0.01)),
    'h2': tf.Variable(tf.truncated_normal([self.attention_dims,1],stddev=0.01)),
    }


    self.right_feature,_,_=self.layers['BiLSTM'](self.ment_sent_right_ctx_embed)
    self.left_feature,_,_=self.layers['BiLSTM'](self.ment_sent_left_ctx_embed)


    lstm_feature = tf.concat([self.right_feature,self.left_feature],1)

    att_w_m = tf.einsum('aij,jk->aik',tf.expand_dims(self.ment_surface_feature,1),self.layers['att_weights']['h_m'])

    att_w1 = tf.nn.tanh(tf.einsum('aij,jk->aik',lstm_feature,self.layers['att_weights']['h1'])+att_w_m)
    self.att_w2 = tf.nn.softmax(tf.einsum('aij,jk->aik',att_w1,self.layers['att_weights']['h2'])[:,:,0],-1)

    att_w = tf.tile(tf.expand_dims(self.att_w2,-1),[1,1,2*self.rnn_size])

    lstm_feature = tf.reduce_sum(tf.multiply(lstm_feature , att_w),1)

    lstm_feature = tf.nn.dropout(lstm_feature,self.keep_prob)
    print('lstm_feature:',lstm_feature)
    return lstm_feature

  def _S0_score(self):
    self.lstm_feature = self._gen_left_right_ctx()
    self.S0_ctx_w = tf.get_variable(
                  name='S0_ctx_w',
                  dtype=tf.float32,
                  shape=(self.args.word_dim,self.args.s2_width_0),
                  initializer=tf.contrib.layers.xavier_initializer()
            )
    self.S0_ctx_bias = tf.get_variable(
                  name='S0_ctx_bias',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0),
                  initializer=tf.initializers.zeros()
            )
    S0_ctx_r = tf.nn.relu(tf.add(tf.matmul(self.lstm_feature,self.S0_ctx_w),
                                 self.S0_ctx_bias))
    self.S0_ctx = tf.tile(tf.expand_dims(S0_ctx_r,1),[1,self.args.cand_nums,1])

    #S^(0)
    #entity meniton prior, mention reduction, type reduction, mention reduction, type reduction
    #to reinforence the training process...
    self.prior_bag_num = self.args.prior_num
    self.rel_ment_cand_prob = tf.tile(tf.expand_dims(self.ment_cand_prob/(self.prior_bag_num*1.0),
                                                     -1),[1,1,self.prior_bag_num])

    #revise:3-6
    #we may break down this model...
    #to avoid the zeros..
    self.p_e_m= tf.log(tf.minimum(0.98,tf.maximum(1e-3,self.rel_ment_cand_prob)))*(-1.0)

    #revise:3-8
    #we seperate the cand and type embedding matrix
    #we increase the type and cand feature dimension...
    self.S0_cand_w = tf.get_variable(
                  name='S0_cand_w',
                  dtype=tf.float32,
                  shape=(self.args.word_dim,self.args.s2_width_0),
                  initializer=tf.contrib.layers.xavier_initializer()
                  #initializer=tf.initializers.random_normal(stddev=1.0/math.sqrt(self.args.s2_width_0))
            )
    self.S0_cand_bias = tf.get_variable(
                  name='S0_cand_bias',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0),
                  initializer=tf.initializers.zeros()
            )
    self.S0_cand = tf.nn.relu(tf.add(tf.einsum('aij,jk->aik',self.ment_cand_id_embed,self.S0_cand_w),
                                     self.S0_cand_bias))

    self.S0_cand_w_1 = tf.get_variable(
                  name='S0_cand_w_1',
                  dtype=tf.float32,
                  shape=(self.args.word_dim,self.args.s2_width_0),
                  initializer=tf.contrib.layers.xavier_initializer()
                  #initializer=tf.initializers.random_normal(stddev=1.0/math.sqrt(self.args.s2_width_0))
              )
    self.S0_cand_bias_1 = tf.get_variable(
                  name='S0_cand_bias_1',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0),
                  initializer=tf.initializers.zeros()
            )

    self.S0_cand_1 = tf.nn.relu(tf.add(tf.einsum('aij,jk->aik',self.semantic_entity,self.S0_cand_w_1),
                                       self.S0_cand_bias_1))

    self.S0_ment_w = tf.get_variable(
                  name='S0_ment_w',
                  dtype=tf.float32,
                  shape=(self.args.word_dim,self.args.s2_width_0),
                  initializer=tf.contrib.layers.xavier_initializer()
                  #initializer=tf.initializers.random_normal(stddev=1.0/math.sqrt(self.args.s2_width_0))
            )
    self.S0_ment_bias = tf.get_variable(
                  name='S0_ment_bias',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0),
                  initializer=tf.initializers.zeros()
            )
    S0_ment_r = tf.nn.relu(tf.add(tf.matmul(self.ment_surface_feature,self.S0_ment_w),
                                  self.S0_ment_bias))

    self.S0_ment = tf.tile(tf.expand_dims(S0_ment_r,1),[1,self.args.cand_nums,1])

    self.S0_ment_elmo_w = tf.get_variable(
                  name='S0_ment_elmo_w',
                  dtype=tf.float32,
                  shape=(1024,self.args.s2_width_0_elmo),
                  initializer=tf.contrib.layers.xavier_initializer()
            )

    self.S0_ment_elmo_bias = tf.get_variable(
                  name='S0_ment_elmo_bias',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0_elmo),
                  initializer=tf.initializers.zeros()
            )
    self.ment_surface_elmo_embed = tf.nn.l2_normalize(self.ment_surface_elmo_embed,-1)
    S0_ment_elmo_r = tf.nn.relu(tf.add(tf.matmul(self.ment_surface_elmo_embed,self.S0_ment_elmo_w),
                                  self.S0_ment_elmo_bias))

    self.S0_ment_elmo = tf.tile(tf.expand_dims(S0_ment_elmo_r,1),[1,self.args.cand_nums,1])

    self.S0_type_w = tf.get_variable(
                  name='S0_type_w',
                  dtype=tf.float32,
                  shape=(self.args.word_dim,self.args.s2_width_0),
                  initializer=tf.contrib.layers.xavier_initializer()
                  #initializer=tf.initializers.random_normal(stddev=1.0/math.sqrt(self.args.s2_width_0))
              )
    self.S0_type_bias = tf.get_variable(
                  name='S0_type_bias',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0),
                  initializer=tf.initializers.zeros()
            )
    self.S0_type = tf.nn.relu(tf.add(tf.einsum('aij,jk->aik',self.cand_type_embed,self.S0_type_w),
                                     self.S0_type_bias))

    self.S0_type_w_1 = tf.get_variable(
                  name='S0_type_w_1',
                  dtype=tf.float32,
                  shape=(self.args.word_dim,self.args.s2_width_0),
                  initializer=tf.contrib.layers.xavier_initializer()
                  #initializer=tf.initializers.random_normal(stddev=1.0/math.sqrt(self.args.s2_width_0))
              )
    self.S0_type_bias_1 = tf.get_variable(
                  name='S0_type_bias_1',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0),
                  initializer=tf.initializers.zeros()
            )
    self.S0_type_1 = tf.nn.relu(tf.add(tf.einsum('aij,jk->aik',self.type_entity,self.S0_type_w_1),
                                       self.S0_type_bias_1))


  def _S1_score(self):
    #(ment,cand,k,ment)
    self.S1_cand_ment_mask = tf.tile(tf.expand_dims(self.cand_mask_4d[:,:,:,0],2),[1,1,self.args.s2_width_1,1])
    print('S1_cand_ment_mask:',self.S1_cand_ment_mask)


    #S^(1)
    self.S1_cand_local_score_list = self._gen_ment_score(self.ment_cand_id_embed,
                                                         self.score_diag_w_dict['S1_cand_diag_w'])
    self.S1_type_local_score_list = self._gen_ment_score(self.cand_type_embed,
                                                        self.score_diag_w_dict['S1_type_diag_w'])

    self.ment_surface_elmo_embed_ft = tf.layers.dense(self.ment_surface_elmo_embed,20,activation=None)
    self.ment_cand_id_embed_ft = tf.layers.dense(self.ment_cand_id_embed,20,activation=None)
    self.cand_type_embed_ft = tf.layers.dense(self.cand_type_embed,20,activation=None)

    self.S1_elmo_cand_local_score_list = self._gen_elmo_ment_score(self.ment_cand_id_embed_ft,
                                                        self.score_diag_w_dict['S1_cand_elmo_w'])

    self.S1_elmo_type_local_score_list = self._gen_elmo_ment_score(self.cand_type_embed_ft,
                                                        self.score_diag_w_dict['S1_type_elmo_w'])

    self.S1_cand_verb_local_score_list = self._gen_verb_score(self.ment_cand_id_embed,
                                                              self.score_diag_w_dict['S1_verb_cand_diag_w'])
    self.S1_type_verb_local_score_list = self._gen_verb_score(self.cand_type_embed,
                                                              self.score_diag_w_dict['S1_verb_type_diag_w'])

  def _S2_score(self):
    #S_(2)
    #global
    #(ment,cand,k,ment,cand)
    self.S2_cand_mask_4_expand = tf.tile(tf.expand_dims(self.S2_cand_mask_4,2),[1,1,self.args.s2_width_2,1,1])

    self.S2_cand_local_score_list=self._loc_rel_score()
    self.S2_type_local_score_list=self._loc_type_rel_score()

  def _local_score(self):
    self.cand_type_embed = tf.nn.l2_normalize(self.ment_cand_type_embed,-1)

    ment_lent=tf.maximum(tf.cast(get_seq_lent(self.ment_surface_ids_embed),tf.float32),1.0)
    self.ment_surface_feature = tf.reduce_sum(self.ment_surface_ids_embed,
                                                 1, name='ment_surface')
    self.ment_surface_feature=tf.div(self.ment_surface_feature,tf.expand_dims(ment_lent,-1))
    if self.args.test!=True:
      print('ment_surface_feature:',self.ment_surface_feature)


    sent_lent=tf.maximum(tf.cast(get_seq_lent(self.ment_sent_ctx_embed),tf.float32),1.0)
    ment_sent_ctx_sum= tf.reduce_sum(self.ment_sent_ctx_embed,
                                               1, name='sent_ctx')

    self.relation_embed=tf.div(ment_sent_ctx_sum,tf.expand_dims(sent_lent,-1))
    if self.args.test!=True:
      print('relation_embed:',self.relation_embed)

    self.score_diag_w_dict={}


    self.score_diag_w_dict['S1_cand_elmo_w'] = tf.get_variable(
                  name='S1_cand_elmo_w',
                  dtype=tf.float32,
                  #shape=(self.args.word_dim,1024),
                  shape=(20,20),
                  initializer=tf.contrib.layers.xavier_initializer())

    self.score_diag_w_dict['S1_type_elmo_w'] = tf.get_variable(
                  name='S1_type_elmo_w',
                  dtype=tf.float32,
                  #shape=(self.args.word_dim,1024),
                  shape=(20,20),
                  initializer=tf.contrib.layers.xavier_initializer())
    self.score_diag_w_dict['S1_cand_diag_w'] = tf.get_variable(
                  name='S1_cand_diag_w',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_1,self.args.word_dim),
                  initializer=tf.contrib.layers.xavier_initializer())

    self.score_diag_w_dict['S1_type_diag_w'] = tf.get_variable(
                  name='S1_type_diag_w',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_1,self.args.word_dim),
                  initializer=tf.contrib.layers.xavier_initializer())


    self.score_diag_w_dict['S1_verb_cand_diag_w'] = tf.get_variable(
                  name='S1_verb_cand_diag_w',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0_1,self.args.word_dim),
                  initializer=tf.contrib.layers.xavier_initializer())

    self.score_diag_w_dict['S1_verb_type_diag_w'] = tf.get_variable(
                  name='S1_verb_type_diag_w',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_0_1,self.args.word_dim),
                  initializer=tf.contrib.layers.xavier_initializer())

    self.score_diag_w_dict['S2_cand_diag_w'] = tf.get_variable(
                  name='S2_cand_diag_w',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_2,self.args.word_dim),
                  initializer=tf.contrib.layers.xavier_initializer())

    self.score_diag_w_dict['S2_type_diag_w'] = tf.get_variable(
                  name='S2_type_diag_w',
                  dtype=tf.float32,
                  shape=(self.args.s2_width_2,self.args.word_dim),
                  initializer=tf.contrib.layers.xavier_initializer())

    self.semantic_entity = tf.add(self.ment_cand_id_embed,
                             tf.expand_dims(self.relation_embed,1))
    self.semantic_entity=tf.multiply(self.semantic_entity,tf.tile(tf.expand_dims(self.mask,-1),[1,1,self.args.word_dim]))

    self.type_entity = tf.add(self.cand_type_embed,
                         tf.expand_dims(self.relation_embed,1))
    self.type_entity=tf.multiply(self.type_entity,tf.tile(tf.expand_dims(self.mask,-1),[1,1,self.args.word_dim]))

    self._S0_score()
    self._S1_score()
    self._S2_score()

    self.S0_score= [tf.concat([self.S0_ment,self.S0_cand,self.S0_cand_1,self.S0_ctx,self.S0_ment_elmo],-1),
                    tf.concat([self.S0_ment,self.S0_cand,self.S0_cand_1,self.S0_ctx,self.S0_ment_elmo,
                               self.S0_type,self.S0_type_1],-1)]


    self.S1_score= [tf.concat([self.S1_cand_local_score_list,self.S1_cand_verb_local_score_list,
                               self.S1_elmo_cand_local_score_list],-1),
                    tf.concat([self.S1_cand_local_score_list,self.S1_cand_verb_local_score_list,
                               self.S1_type_local_score_list,self.S1_type_verb_local_score_list,
                               self.S1_elmo_cand_local_score_list,self.S1_elmo_type_local_score_list],-1)]

    self.S2_score = [tf.concat([self.S2_cand_local_score_list],-1), tf.concat([self.S2_cand_local_score_list,self.S2_type_local_score_list],-1)]

    if self.args.has_type =='T':
      feat_dim = 1
    else:
      feat_dim =0

    if self.args.feature=='S012':
      self.local_p_e_ctx_score = tf.concat([
                            self.S0_score[feat_dim],
                            self.S1_score[feat_dim],
                            self.S2_score[feat_dim],
                            ],-1)
    else:
      print('wrong feature',self.args.feature)
      exit(0)

    print('[****]local_p_e_ctx_score:',self.local_p_e_ctx_score)

    self.global_p_e_ctx = tf.layers.dense(self.local_p_e_ctx_score,1,
                                      activation=None
                                     )

  def _gen_elmo_ment_score(self,embeddings,M):
    #(ment,cand,dim)
    S1_local_type_1 = tf.einsum('aij,jk->aik',embeddings,M)

    #(ment,cand,ment)
    local_type_comp_score_all= tf.nn.relu(tf.einsum('aij,jk->aik',S1_local_type_1,
                                                    tf.transpose(self.ment_surface_elmo_embed_ft,[1,0]))
                                          )
    #(ment,cand,1)
    S1_type_m_local_score = tf.reduce_mean(local_type_comp_score_all,-1,keepdims=True)
    return S1_type_m_local_score

  def _gen_verb_score(self,embeddings,diags):
    #(width,300,300)
    ctx_mask = tf.cast(tf.cast(self.ment_doc_ctx,tf.bool),tf.float32)

    M=tf.matrix_diag(diags)

    #(ment_no,cand_no,300,width)
    ent_A= tf.einsum('aij,jkl->aikl',embeddings,
                           tf.transpose(M,[1,2,0]))

    #(ment_no,cand_no,width,100)
    ent_A_word = tf.nn.relu(tf.einsum('aikl,alj->aikj',tf.transpose(ent_A,[0,1,3,2]),
                                           tf.transpose(self.ment_doc_ctx_embed,[0,2,1])))
    #word_attention
    #(ment_no,cand_no,width,100)
    #word_att = tf.nn.softmax(ent_A_word,-1)
    word_att = get_top_k(ent_A_word,ctx_mask,20,self.args.s2_width_0_1,self.args.cand_nums)

    #feature
    #(ment_no,cand_no,100)
    ctx_feature = tf.einsum('aij,ajk->aik',embeddings,tf.transpose(self.ment_doc_ctx_embed,[0,2,1]))

    #(ment_no,cand_no,width,100)
    ctx_feature = tf.tile(tf.expand_dims(ctx_feature,2),[1,1,self.args.s2_width_0_1,1])
    ctx_feature_att = tf.multiply(word_att,ctx_feature)

    verb_score = tf.nn.relu(tf.reduce_sum(ctx_feature_att,-1))

    return verb_score

  def _gen_ment_score(self,embeddings,diags):
    M = tf.matrix_diag(diags)
    #(ment,cand,dim,k)
    S1_local_type_1 = tf.einsum('aij,jkl->aikl',embeddings,tf.transpose(M,[1,2,0]))

    #(ment,cand,k,ment)
    local_type_comp_score_all= tf.nn.relu(tf.einsum('aikl,lp->aikp',tf.transpose(S1_local_type_1,[0,1,3,2]),
                                           tf.transpose(self.ment_surface_feature,[1,0])))

    #(ment,cand,k)
    S1_type_m_local_score = tf.reduce_mean(local_type_comp_score_all,-1)
    return S1_type_m_local_score

  def _loc_rel_score(self):
    M = tf.matrix_diag(self.score_diag_w_dict['S2_cand_diag_w'])
    semantic_entity_M = tf.einsum('aij,jkl->aikl',self.semantic_entity,tf.transpose(M,[1,2,0]))
    print(semantic_entity_M)

    #(ment,cand,k,ment,cand)
    ent_R_ent_1 =tf.einsum('aikj,jlm->aiklm',tf.transpose(semantic_entity_M,[0,1,3,2]),
                                             tf.transpose(self.ment_cand_id_embed,[2,0,1]))
    print(ent_R_ent_1)
    ent_R_ent_1 = tf.nn.relu(tf.multiply(ent_R_ent_1,self.S2_cand_mask_4_expand))

    #(ment,cand,k,ment)
    rel_score_top1 = tf.reduce_max(ent_R_ent_1,-1)

    #(ment,cand,k)
    semantic_convd_rel_score= tf.reduce_mean(rel_score_top1,-1)
    return semantic_convd_rel_score

  def _loc_type_rel_score(self):
    M = tf.matrix_diag(self.score_diag_w_dict['S2_type_diag_w'])
    semantic_entity_M = tf.einsum('aij,jkl->aikl',self.type_entity,tf.transpose(M,[1,2,0]))
    print(semantic_entity_M)

    #(ment,cand,k,ment,cand)
    ent_R_ent_3 = tf.einsum('aikj,jlm->aiklm',tf.transpose(semantic_entity_M,[0,1,3,2]),
                                             tf.transpose(self.cand_type_embed,[2,0,1]))

    ent_R_ent_3 = tf.nn.relu(tf.multiply(ent_R_ent_3,self.S2_cand_mask_4_expand))

    #(ment,cand,k,ment)
    type_rel_score_top1 = tf.reduce_max(ent_R_ent_3,-1)

    #(ment,cand,k)
    type_convd_rel_score= tf.reduce_mean(type_rel_score_top1,-1)

    return type_convd_rel_score

  def _init_embeddings(self):
    self.mask = tf.cast(tf.cast(self.ment_cand_id,tf.bool),tf.float32,name='ment_cand_mask')
    with tf.name_scope("init_embed"):

      self.word_embed_matrix = tf.Variable(
            initial_value=self.word_embed_pl,
            trainable=False,
            name='word_embed_matrix',
            dtype=tf.float32
          )

      self.entity_type_embed_matrix = tf.Variable(
            initial_value=self.ent_type_embed_pl,
            trainable=False,
            name='entity_type_embed_matrix',
            dtype=tf.float32
          )
      self.entity_embed_matrix = tf.Variable(
            initial_value=self.ent_embed_pl,
            trainable=False,
            name='entity_embed_matrix',
            dtype=tf.float32
          )
      #we need to build unknow entity embedding and typing
      if self.args.use_unk=='unk':
        self.unk_entity_embed =tf.get_variable(
                    name='unk_entity_embed',
                    dtype=tf.float32,
                    shape=(1,self.args.word_dim),
                    initializer=tf.zeros_initializer()
                )

        self.unk_entity_type_embed =tf.get_variable(
                    name='unk_entity_type_embed',
                    dtype=tf.float32,
                    shape=(1,self.args.word_dim),
                    initializer=tf.zeros_initializer()
                )

        self.entity_type_embed_matrix = tf.concat([self.unk_entity_type_embed,self.entity_type_embed_matrix],0)

        self.entity_embed_matrix=tf.concat([self.unk_entity_embed,self.entity_embed_matrix],0)

    with tf.name_scope("fea_embed"):
      self.ment_doc_ctx_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.ment_doc_ctx)
      self.ment_surface_ids_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.ment_surface_ids)
      self.ment_sent_ctx_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.ment_sent_ctx)
      self.ment_sent_left_ctx_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.ment_sent_left_ctx)
      self.ment_sent_right_ctx_embed = tf.nn.embedding_lookup(self.word_embed_matrix,self.ment_sent_right_ctx)
      self.ment_cand_id_embed = tf.nn.embedding_lookup(self.entity_embed_matrix,self.ment_cand_id)
      self.ment_cand_type_embed = tf.nn.embedding_lookup(self.entity_type_embed_matrix,self.ment_cand_id)


    self.ment_sent_ctx_embed = tf.nn.l2_normalize(self.ment_sent_ctx_embed,-1)
    self.ment_sent_left_ctx_embed = tf.nn.l2_normalize(self.ment_sent_left_ctx_embed,-1)
    self.ment_sent_right_ctx_embed = tf.nn.l2_normalize(self.ment_sent_right_ctx_embed,-1)
    self.ment_cand_id_embed = tf.nn.l2_normalize(self.ment_cand_id_embed,-1)
    self.ment_cand_type_embed = tf.nn.l2_normalize(self.ment_cand_type_embed,-1)
    self.ment_doc_ctx_embed = tf.nn.l2_normalize(self.ment_doc_ctx_embed,-1)

    self.ment_cand_id_embed = tf.nn.dropout(self.ment_cand_id_embed,self.keep_prob)
    self.ment_cand_type_embed = tf.nn.dropout(self.ment_cand_type_embed,self.keep_prob)
    self.ment_sent_ctx_embed=tf.nn.dropout(self.ment_sent_ctx_embed,self.keep_prob)
    self.ment_surface_ids_embed=tf.nn.dropout(self.ment_surface_ids_embed,self.keep_prob)

  def _init_placeholder(self):
    with tf.name_scope("init_placeholder"):
      self.ment_cand_id = tf.placeholder(
          dtype=tf.int32,
          shape=[None,self.args.cand_nums],
          name='ment_cand_id'
          )

      self.ment_cand_prob = tf.placeholder(
          dtype=tf.float32,
          shape=[None,self.args.cand_nums],
          name='ment_cand_prob'
          )

      self.linking_tag = tf.placeholder(
          dtype=tf.float32,
          shape=[None,self.args.cand_nums],
          name='linking_tag'
          )

      self.ment_doc_ctx = tf.placeholder(
            dtype=tf.int32,
            shape=[None,self.args.ent_ctx_lent],
            name='ment_doc_ctx'
        )

      self.ment_surface_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None,5],
            name='ment_surface_ids'
        )
      self.ment_surface_elmo_embed = tf.placeholder(
            dtype=tf.float32,
            shape=[None,1024],
            name='ment_surface_elmo_embed'
        )
      self.ment_sent_ctx = tf.placeholder(
            dtype=tf.int32,
            shape=[None,20],
            name='ment_sent_ctx'
        )

      self.ment_sent_left_ctx = tf.placeholder(
            dtype=tf.int32,
            shape=[None,10],
            name='ment_sent_left_ctx'
        )

      self.ment_sent_right_ctx = tf.placeholder(
            dtype=tf.int32,
            shape=[None,10],
            name='ment_sent_right_ctx'
        )

      self.ment_cands_lent =tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='ment_cands_lent'
        )

      self.S2_cand_mask_4 = tf.placeholder(
                tf.float32,
                shape =[None,self.args.cand_nums,None,self.args.cand_nums],
                name='S2_cand_mask_4'
                )
      self.cand_mask_pad = tf.placeholder(
              tf.float32,
              shape =[None,None],
              name='cand_mask_pad'
              )

      self.is_training = tf.placeholder(
          tf.bool,
          name='is_training'
        )

      self.cand_mask_2d = tf.placeholder(
              tf.float32,
              shape =[None,None],
              name='cand_mask_2d'
              )
      self.cand_mask_4d = tf.placeholder(
              tf.float32,
              shape =[None,self.args.cand_nums,None,self.args.cand_nums],
              name='cand_mask_4d'
              )
      self.sample_size = tf.placeholder(
          tf.int32,
          name='sample_size'
        )
      self.lr=tf.placeholder(tf.float32)
      self.ent_embed_pl = tf.placeholder(tf.float32,[self.args.ent_nums,300])
      self.ent_type_embed_pl = tf.placeholder(tf.float32,[self.args.ent_nums,300])
      self.word_embed_pl = tf.placeholder(tf.float32,[self.args.word_nums,300])

      linking_tag_shape = tf.shape(self.linking_tag)
      self.ment_no = linking_tag_shape[0]
      self.cand_no= self.args.cand_nums
