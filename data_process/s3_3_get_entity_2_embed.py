# -*- coding: utf-8 -*-
import sys
sys.path.append('Project Absolute Path')
import os
from entity import get_disam_wiki_id_name,get_wiki_redirect
from tqdm import tqdm
import cPickle
import codecs
import numpy as np
from decimal import Decimal

if __name__ == '__main__':
  ent2id=cPickle.load(open('data/intermediate/ent2id_no_padding.p'))
  id2ent = {ent2id[key]:key for key in ent2id}
  if os.path.isfile('data/intermediate/rltd_name_2_idx.p')==False:
    wiki_name_2_id,wiki_id_2_name= get_disam_wiki_id_name()
    wiki_2_redir_title=get_wiki_redirect()

    emnlp_ent_file = 'data/deep-ed/data/generated/embeddings/word_ent_embs/dict.entity'
    rltd_name_2_idx = {}

    idx=0
    '''
    @error 1:
    @utf-8 replace(u'%22',u'"')
    '''
    with codecs.open(emnlp_ent_file,'r','utf-8') as file_:
      for line in tqdm(file_):
        line=line.strip()
        name=line.split(u'\t')[0].replace(u'%22',u'"').replace(u'en.wikipedia.org/wiki/',u'').replace(u'_',u' ')
        if name in wiki_name_2_id:
          wiki_idx = wiki_name_2_id[name]
          rltd_name_2_idx[name]=[idx,wiki_idx]
          idx += 1
        else:
          print(name)
    cPickle.dump(rltd_name_2_idx,open('data/intermediate/rltd_name_2_idx.p','w'))
    print(len(rltd_name_2_idx))
  else:
    rltd_name_2_idx = cPickle.load(open('data/intermediate/rltd_name_2_idx.p'))

  print('rltd_name_2_idx:',len(rltd_name_2_idx))
  emnlp_ent_embed_file = 'data/deep-ed/data/generated/embeddings/word_ent_embs/entity_embeddings.npy'
  rltd_name_2_embeddings = np.load(emnlp_ent_embed_file)
  wiki_idx_2_embed_idx = {}
  for name in rltd_name_2_idx:
    idx,wiki_idx = rltd_name_2_idx[name]
    wiki_idx_2_embed_idx[wiki_idx]=idx
  non_exist_wiki_idx=0.0
  wiki_name_2_id,wiki_id_2_name= get_disam_wiki_id_name()
  ent2embedding={}
  for ent in ent2id:
    wiki_idx = ent
    if wiki_idx not in wiki_idx_2_embed_idx:
      print(wiki_idx)
      non_exist_wiki_idx+=1
      continue
    idx = wiki_idx_2_embed_idx[wiki_idx]
    if ent not in ent2embedding:
      ent2embedding[ent]=rltd_name_2_embeddings[idx]
  #assert(len(ent2embedding)==len(ent2id))
  print('non exist ents:',non_exist_wiki_idx)
  print('all_ents:',len(ent2id))
  fname = 'data/intermediate/entity2vector_emnlp_pretrain.txt'
  fout = codecs.open(fname,'w','utf-8')
  for idx in range(len(ent2id)):
    wiki_idx = id2ent[idx]
    embedding_str=None
    if wiki_idx not in ent2embedding:
      random_embed = np.random.normal(size=(300))
      embedding_str = ['{:.3f}'.format(Decimal(str(random_embed[i]))) for i in range(300)]
    else:
      embedding = ent2embedding[wiki_idx]
      embedding_str = map(str,list(embedding))
    assert(len(embedding_str)==300)

    fout.write(wiki_idx+' '+' '.join(embedding_str)+'\n')
    fout.flush()
  fout.close()
