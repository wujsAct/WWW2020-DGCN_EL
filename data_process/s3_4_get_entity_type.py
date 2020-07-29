# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 20:25:18 2018

@author: wujs
"""
import sys
sys.path.append('/home/wjs/demo/entityType/EntLinking')
from entity import get_disam_wiki_id_name
from tqdm import tqdm
import cPickle
from pymongo import MongoClient
import codecs

client = MongoClient('mongodb://192.168.3.196:27017')
db = client["freebase_full"] 
collection = db["freebase"]  # collection freebase  

def get_name_2_mid():
  title2mid = {}
  
  with codecs.open('data/mid2name.tsv','r','utf-8') as file_:
    for line in tqdm(file_):
      items = line[:-1].split('\t')
      mid = items[0]
      title = ' '.join(items[1:])
      title2mid[title] = mid
  return title2mid


def find_notable_type(mid):
  mid = mid[1:].replace('/','.')
  ent = '<http://rdf.freebase.com/ns/'+mid+'>'
  type_ = None
  #we need to iterate the fb twice time, but it also faster to search the google online 
  try:
      type_mid = collection.find_one({"head": ent,'rel':'<http://rdf.freebase.com/ns/common.topic.notable_types>'})['tail']
      type_ = db.freebase.find_one({"head": type_mid,'rel':'<http://rdf.freebase.com/ns/type.object.key>'})['tail']
  except:
      return None
  
  return type_
     
def get_ent_notable_type(mid):
  if mid==None:
    return None
  return find_notable_type(mid)
  
def get_mid(ent_id):
  if ent_id in wiki_id_2_name:
    title =wiki_id_2_name[ent_id]
  else:
    return None
  
  mid = None
  if title in title2mid:
    mid = title2mid[title]
  if mid == None:
    return None
  
  return mid

if __name__ == '__main__':
  wiki_name_2_id,wiki_id_2_name= get_disam_wiki_id_name()
  title2mid = get_name_2_mid()
  
  ent_2_mid_type = {}
  mid_none = 0
  ent2id = cPickle.load(open('data/intermediate/ent2id_no_padding.p','rb'))
  for ent in tqdm(ent2id):
    mid = get_mid(ent)
    types = get_ent_notable_type(mid)
    ent_2_mid_type[ent]=[mid,types]
    if mid==None:
      mid_none+=1
  print(mid_none)
  
  cPickle.dump(ent_2_mid_type,open('data/intermediate/ent2midtype_no_padding.p','wb'))
  
  
  ent_2_mid_type = cPickle.load(open('data/intermediate/ent2midtype_no_padding.p','rb'))
  mid_dict={}
  
  for ent in ent_2_mid_type:
    mid,types = ent_2_mid_type[ent]
    if mid not in mid_dict:
      mid_dict[mid]=1
  cPickle.dump(mid_dict,open('data/intermediate/mid2id.p','wb'))