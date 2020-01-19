# -*- coding: utf-8 -*-
import tensorflow as tf

def margin_loss_sum(labels,logits,neg_mask,margin_param):
  pos_score = tf.reduce_sum(tf.multiply(labels,logits),-1,keepdims=True)

  neg_score = tf.multiply(neg_mask,logits)  #neg_scores!
  h = tf.nn.relu(tf.multiply(neg_mask,(margin_param - pos_score + neg_score)))

  h = tf.reduce_sum(h,-1)  #different mention has different candidates...
  loss = tf.reduce_sum(h,-1)
  return loss


def margin_loss(labels,logits,neg_mask,margin_param):
  pos_score = tf.reduce_sum(tf.multiply(labels,logits),-1,keepdims=True)

  h = tf.nn.relu(tf.multiply(neg_mask,(margin_param - pos_score + logits)))

  h = tf.div(tf.reduce_sum(h,-1),tf.maximum(1.0,tf.cast(tf.reduce_sum(neg_mask,-1),tf.float32)))

  loss = tf.reduce_mean(h,-1)
  return loss


class LSTM(object):
  def __init__(self,cell_size,num_layers=2,name='LSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.reuse = None
    self.trainable_weights = None
    self.name = name
    self.keep_prob=1.0
    self.cell =tf.contrib.rnn.LSTMCell(self.cell_size,name='sent_fw')

  def __call__(self,x,seq_length=None,):  #__call__ is very efficient when the state of instance changes frequently
    with tf.variable_scope(self.name):
      if seq_length ==None:  #get the real sequence length (suppose that the padding are zeros)
        used = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
        seq_length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)

      outputs,states=tf.nn.dynamic_rnn(self.cell,
                        x,
                        sequence_length=seq_length,
                        dtype=tf.float32,
                        time_major=False)
    return outputs,states,seq_length

class BiLSTM(object):
  '''
  LSTM layers using dynamic rnn
  '''
  def __init__(self,cell_size,num_layers=2,name='BiLSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.reuse = None
    self.trainable_weights = None
    self.name = name
    self.keep_prob=1.0
    fw_cell =tf.contrib.rnn.GRUCell(self.cell_size,name='sent_fw')
    bw_cell =tf.contrib.rnn.GRUCell(self.cell_size,name='sent_bw')

    self.fw_cell_drop = tf.contrib.rnn.DropoutWrapper(fw_cell,input_keep_prob=self.keep_prob)
    self.bw_cell_drop = tf.contrib.rnn.DropoutWrapper(bw_cell,input_keep_prob=self.keep_prob)

  #x() equals to x.__call___()
  def __call__(self,x,seq_length=None,):  #__call__ is very efficient when the state of instance changes frequently
    with tf.variable_scope(self.name) as vs:
      if seq_length ==None:  #get the real sequence length (suppose that the padding are zeros)
        used = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
        seq_length = tf.cast(tf.reduce_sum(used,reduction_indices=1),tf.int32)

      lstm_out_bi,(output_state_fw,output_state_bw) =  tf.nn.bidirectional_dynamic_rnn(
                                                                       self.fw_cell_drop,
                                                                       self.bw_cell_drop,
                                                                       x,
                                                                       sequence_length=seq_length,
                                                                       dtype=tf.float32,
                                                                       time_major=False)
      lstm_out_bi = tf.concat(lstm_out_bi,2)
      print('lstm_out_bi:',lstm_out_bi)
      #print('output_state_fw:',output_state_fw)
      #print('output_state_fw:',output_state_fw[1])


      lstm_out = tf.concat([output_state_fw[1],output_state_bw][1],-1)
      #print 'lstm_out: ',lstm_out

      if self.reuse is None:
        self.trainable_weights = vs.global_variables()

    self.reuse =True
    return lstm_out_bi,lstm_out,seq_length
