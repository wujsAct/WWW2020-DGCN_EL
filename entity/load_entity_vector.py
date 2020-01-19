# -*- coding: utf-8 -*-

import sys
python_type = sys.version_info.major
if python_type==2:
  import cPickle
else:
  import pickle as cPickle


import numpy as np
from tqdm import tqdm
import gensim
def load_ent_transE_embed(ent2id):
  mid2id={}
  id2mid={}
  line_num=0
  with open('data/Freebase/kg/entity2id.txt') as file_:
    for line in tqdm(file_):
      line_num+=1
      if line_num==1:
        continue
      line=line.strip()

      items=line.split('\t')

      mid,ids=items
      mid=mid.split('.')[-1]
      mid2id[mid]=ids
      id2mid[ids]=mid

  vec = np.memmap('data/Freebase/'+\
                  'embeddings/dimension_50/transe/entity2vec.bin' ,
                  dtype='float32', mode='r')

  vec = vec.reshape((len(mid2id),50))

  id2ent={ent2id[ent]:ent for ent in ent2id}

  ent_transE_embedd=[]
  non_transE_ents=0
  ent_transE_embedd.append(np.zeros((50,)))

  if python_type==2:
    ent_2_mid_type = cPickle.load(open('data/intermediate/ent2midtype_no_padding.p','rb'))
  else:
    ent_2_mid_type = cPickle.load(open("data/intermediate/ent2midtype_no_padding.p","rb"),
                                  encoding='iso-8859-1')

  for idx in range(1,len(ent2id)):
    ent= id2ent[idx]
    if ent in ent_2_mid_type:
      mid,types = ent_2_mid_type[ent]
    else:
      mid,types=None,None

    if mid!=None:
      mid = mid.split('/')[-1]
    if mid in mid2id:
      ids = mid2id[mid]
      ent_transE_embedd.append(vec[int(ids)])

    else:
      ent_transE_embedd.append(np.zeros((50,)))
      non_transE_ents+=1

  ent_transE_embedd=np.array(ent_transE_embedd,np.float32)
  assert(len(ent_transE_embedd)==len(ent2id))
  print('non_transE_ents:',non_transE_ents)
  np.save('data/ent_transE_embed',ent_transE_embedd)

  return ent_transE_embedd


def load_ent_w2v_embed(ent2id):
  id2ent={ent2id[ent]:ent for ent in ent2id}
  model=gensim.models.KeyedVectors.load_word2vec_format('data/freebase-vectors-skipgram1000.bin.gz',binary=True)

  print(len(model.vocab.keys()))
  word2vec_vocab = model.vocab

  ent_w2v_embedd=[]
  non_w2v_ents=0
  ent_w2v_embedd.append(np.zeros((1000,)))

  if python_type==2:
    ent_2_mid_type = cPickle.load(open('data/intermediate/ent2midtype_no_padding.p','rb'))
  else:
    ent_2_mid_type = cPickle.load(open("data/intermediate/ent2midtype_no_padding.p","rb"),
                                  encoding='iso-8859-1')

  for idx in range(1,len(id2ent)):
    ent= id2ent[idx]
    if ent in ent_2_mid_type:
      mid,types = ent_2_mid_type[ent]
    else:
      mid,types=None,None

    if mid in word2vec_vocab:
      ent_w2v_embedd.append(model[mid])
      print(np.shape(model[mid]))
    else:
      #ent_w2v_embedd.append(np.random.normal(size=(1000,))*0.01)
      ent_w2v_embedd.append(np.zeros((1000,)))
      non_w2v_ents +=1

  ent_w2v_embedd=np.array(ent_w2v_embedd,np.float32)
  assert(len(ent_w2v_embedd)==len(ent2id))
  print('non_w2v_ents:',non_w2v_ents)
  np.save('data/ent_w2v_embed',ent_w2v_embedd)


  return ent_w2v_embedd

def load_entity_vector(word2id,word_embedding):
  ent_id = 0
  ent2id={}
  ent_embedd=[]
  ent2id['PAD']=0
  ent_embedd.append(np.zeros((300,)))

  if python_type==2:
    ent_2_mid_type = cPickle.load(open('data/intermediate/ent2midtype_no_padding.p','rb'))
    old_ent2id = cPickle.load(open('data/intermediate/ent2id_no_padding.p','rb'))
  else:
    ent_2_mid_type = cPickle.load(open('data/intermediate/ent2midtype_no_padding.p','rb'),
                                  encoding='iso-8859-1')
    old_ent2id = cPickle.load(open('data/intermediate/ent2id_no_padding.p','rb'),
                              encoding='iso-8859-1')

  old_id2ent = {old_ent2id[enti]:enti for enti in old_ent2id}

  print('start to load candidate entity embedding...')
  ent_2_mid ={}
  ent_2_mid[0] = 'None'

  ent2type=[]
  ent2type.append(np.zeros((300,)))

  non_type_ents=0
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
          ent2type.append(np.zeros((300,)))
          non_type_ents+=1
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
          ent2type.append(np.zeros((300,)))
          non_type_ents+=1
        else:
          ent2type.append(type_vector*(1.0/type_lent))

  #cPickle.dump(ent2id,open('data/ent2id.p','w'))
  assert(len(ent2id)==len(ent_2_mid)==len(old_ent2id)+1)

  print('non_type_ents:',non_type_ents)
  # np.asarray(ent_w2v_embedd,dtype=np.float32)
  return ent2id,ent2type, np.asarray(ent_embedd,dtype=np.float32),ent_2_mid
