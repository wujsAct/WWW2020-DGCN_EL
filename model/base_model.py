# -*- coding: utf-8 -*-

import os
import tensorflow as tf


class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    self.vocab = None
    self.data = None

  def save(self, sess, checkpoint_dir, dataset_name,var_list=None,global_step=None):
    if var_list !=None:
      saver = tf.train.Saver(var_list=var_list)
    else:
      saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__ or "Reader"
    if self.batch_size:
      model_dir = "%s_%s_%s" % (model_name, dataset_name, self.batch_size)
    else:
      model_dir = dataset_name

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver.save(sess,
        os.path.join(checkpoint_dir, model_name), global_step=global_step)

  def load(self, sess, checkpoint_dir, dataset_name,var_list=None,global_step=None):
    model_name = type(self).__name__ or "Reader"
    if var_list !=None:
      saver = tf.train.Saver(var_list=var_list)
    else:
      saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    if self.batch_size:
      model_dir = "%s_%s_%s" % (model_name, dataset_name, self.batch_size)
    else:
      model_dir = dataset_name
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False
