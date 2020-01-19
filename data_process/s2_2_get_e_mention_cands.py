# -*- coding: utf-8 -*-
import sys
sys.path.append('Project Absolute Path')
from tqdm import tqdm
import codecs
import cPickle

def get_emnlp_ment_cands(emnlp_fname,dir_path,feature_path):
  docName_2_id = {}
  id_2_docName={}
  ids = 0
  with codecs.open(dir_path+'process/'+'aid2Name.txt','r','utf-8') as file_:
    for line in file_:
      line = line.strip()
      items = line.split('\t')
      if 'aida' in emnlp_fname:
        name=items[1][:-4]
      else:
        name = items[0]

      docName_2_id[name]=ids
      id_2_docName[ids]=name
      ids += 1

  emnlp_doc2ents = {}

  fname ='data/deep-ed/data/generated/test_train_data/'+emnlp_fname
  with codecs.open(fname,'r','utf-8') as file_:
    for line in tqdm(file_):
      line = line.strip()
      items = line.split('\t')

      if 'aida_train' in emnlp_fname:
        doc_name = items[0].split(' ')[0]
      else:
        doc_name = items[0]

      if doc_name not in docName_2_id:
        print('wrong ...')
        exit(0)
      doc_id = docName_2_id[doc_name]
      mention = items[2]
      cands= '\t'.join(items[6:-2])


      if doc_id not in emnlp_doc2ents:
        emnlp_doc2ents[doc_id]={}

      emnlp_doc2ents[doc_id][mention]=cands
  cPickle.dump(emnlp_doc2ents,open(feature_path+'emnlp_doc2ents.p','w'))

if __name__ == "__main__":

  dataset_list=['aida','aida','aida','ace2004','msnbc','aquaint','wikipedia','clueweb']
  tag_list=['testa','testb','train','ace2004','msnbc','aquaint','wikipedia','clueweb']
  emnlp_file_list =['aida_testA.csv','aida_testB.csv','aida_train.csv',
                    'wned-ace2004.csv','wned-msnbc.csv','wned-aquaint.csv',
                    'wned-wikipedia.csv','wned-clueweb.csv']


  for i in range(3):
    dataset = dataset_list[i]
    tag=tag_list[i]

    print('dataset:',dataset)
    print('tag:',tag)


    if dataset == 'aida' or dataset == 'KBP2014':
      dir_path = 'data/'+dataset+'/'+tag+'/'
    else:
      dir_path ='data/'+dataset+'/'


    feature_path = dir_path+"features/"
    emnlp_file=emnlp_file_list[i]
    print('emnlp_file:',emnlp_file)
    get_emnlp_ment_cands(emnlp_file,dir_path,feature_path)
