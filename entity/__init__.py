# -*- coding: utf-8 -*-
from sys import version_info
version = version_info.major
if version==2:
  '''
  #include all the functions for mention pre-process
  '''
  from get_mention_cands_utils import *

  '''
  #search candidate entity by using mention string in mongoDB
  '''
  from mongo_utils import mongoUtils,preprocess_mention

  from load_entity_vector import load_entity_vector,load_ent_w2v_embed,load_ent_transE_embed
  from load_entity_vector_unk import load_entity_vector_unk
else:
  from .get_mention_cands_utils import *
  from .mongo_utils import mongoUtils,preprocess_mention
  from .load_entity_vector import load_entity_vector,load_ent_w2v_embed,load_ent_transE_embed
  from .load_entity_vector_unk import load_entity_vector_unk
