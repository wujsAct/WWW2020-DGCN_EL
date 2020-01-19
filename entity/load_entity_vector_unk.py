# -*- coding: utf-8 -*-

import sys
python_type = sys.version_info.major
if python_type==2:
  import cPickle
else:
  import pickle as cPickle
import numpy as np
from tqdm import tqdm

def load_entity_vector_unk(word2id,word_embedding):
  ent_id = 0
  ent2id={}
  ent_embedd=[]
  ent2id['PAD']=0
  '''
  @revise time: 2019-2-12 we append the unknown entity embedding and typing during the training process...
  '''
  #ent_embedd.append([0]*300)

  ent_2_mid_type = cPickle.load(open('data/intermediate/ent2midtype_no_padding.p','rb'))


  old_ent2id = cPickle.load(open('data/intermediate/ent2id_no_padding.p','rb'))
  old_id2ent = {old_ent2id[enti]:enti for enti in old_ent2id}

  print('start to load candidate entity embedding...')
  ent_2_mid ={}
  ent_2_mid[0] = 'None'

  ent2type=[]
  #ent2type.append([0]*300)
  #543
  fname_ent_embed = 'data/intermediate/entity2vector_emnlp_pretrain.txt'

  #fname_ent_embed = 'data/entity_embed_ret/mid2vector_train.txt_Relateness_20_5_adagrad_0.3_epoch_201'
  line_no=-1
  with open(fname_ent_embed) as file_:
    for line in tqdm(file_):
      line = line.strip()
      line_no+=1
      items = line.split(' ')[1:]

      ent = old_id2ent[line_no]
      if ent in ent_2_mid_type:
        mid,types = ent_2_mid_type[ent]
      else:
        mid,types=None,None


      if ent not in ent2id:
        ent_id += 1
        ent2id[ent] = ent_id
        ent_2_mid[ent_id] = mid
        ent_embedd.append(items)

        if types == None:
          ent2type.append(np.random.normal(size=(300,))*0.01)
          continue

        type_list = types.replace('"','').replace('/','_').split('_')
        type_vector = np.zeros((300,))
        flag=True
        type_lent = 0
        for wi in type_list:
          if wi!=' ':
            if wi in word2id:
              type_vector += word_embedding[word2id[wi]]
              flag=False
              type_lent+=1
        if flag:
          ent2type.append(np.random.normal(size=(300,))*0.01)
        else:
          ent2type.append(type_vector*(1.0/type_lent))

  #cPickle.dump(ent2id,open('data/ent2id.p','w'))
  assert(len(ent2id)==len(ent_2_mid)==len(old_ent2id)+1)

  return ent2id,ent2type, np.asarray(ent_embedd,dtype=np.float32),ent_2_mid
