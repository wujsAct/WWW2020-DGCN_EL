# -*- coding: utf-8 -*-
'''
function: get coreference for entity mentions
'''
import sys
sys.path.append('Project Absolute Path')
from entity import get_aid2ents
import codecs
import cPickle
from collections import Counter
import argparse
from tqdm import tqdm

def get_ment_2_key():
  entMent_2_tags={}
  with codecs.open(dir_path+'process/'+'entMen2aNosNoid.txt','r','utf8') as file_:
    for line in tqdm(file_):
      line = line.strip()
      items = line.split('\t')
      mention =items[0]
      aNosNo = items[1]
      start = items[2]
      end = items[3]
      key = aNosNo + '\t'+start +'\t'+end
      tag = items[-1]

      entMent_2_tags[key] = [mention.lower(),mention,tag,line]
  return entMent_2_tags

def get_rep_ent_in_one_cluster():
  Line2WordDict = {}
  Line2entRep={}  #a little complex
  with codecs.open(dir_path+'process/'+"corefRet.txt",'r','utf-8') as file_:
    for line in tqdm(file_):
      line =line.strip()
      items = line.split(u'\t\t')
      entInDict={}
      wordList = []
      for enti in items:
        if len(enti.split(u'\t'))!=4:
          print(enti)
        aNosNo, start, end, mention = enti.split(u'\t')
        key = aNosNo +u'\t'+start+u'\t'+end

        #if key in entMent_2_tags:
        for word in mention.split(u' '):
          wordList.append(word.lower())
      wordDict = Counter(wordList)
      wordDict= sorted(wordDict.iteritems(), key=lambda d:d[1], reverse = True)

      for enti in items:
        aNosNo, start, end, mention = enti.split('\t')
        key = aNosNo +'\t'+start+'\t'+end

        #representive entity is the longest entities and exist in the extracted entity mentions!
        flag = False
        if key in entMent_2_tags:
          for iment in mention.split(' '):
            if len(wordDict)>0:
              if iment.lower() == wordDict[0][0]:
                flag = True
          if flag:
            entInDict[enti] = len(mention.split(u' '))
        else:
          for keyi in entMent_2_tags:  #relative cluase deleted!
            aNosNok, startk, endk = keyi.split('\t')
            mentionk = entMent_2_tags[keyi][1]
            if aNosNok == aNosNo and int(start) >= int(startk) and int(end) <= int(endk):
              entInDict[keyi+'\t'+mentionk]=len(mentionk.split(' '))


      entInDict = sorted(entInDict.iteritems(), key=lambda d:d[1], reverse = True)
      if len(entInDict)>0:
        Line2entRep[line] = entInDict[0][0]
        Line2WordDict[line] = wordDict
  return Line2entRep

def get_rep_ent_for_mention():
  same_entMent2repMent = {}  #entity mentions that need to refine
  entMent2repMent={}
  for line in tqdm(Line2entRep):
    line = line.strip()
    entRep = Line2entRep[line]
    entMention = entRep.split('\t')[-1]

    for item in line.split('\t\t'):
      if item != entRep:
        aNosNo,start,end,mention = item.split('\t')
        itemkey = '\t'.join(item.split('\t')[0:3])
        if itemkey in entMent_2_tags:
          entMent2repMent[item] = [entRep]
          itemment=entMent_2_tags[itemkey][0]
          if itemment.lower() == entMention.lower() or (itemment.lower() not in entMention.lower()) :
            same_entMent2repMent[item] = entRep
  return same_entMent2repMent, entMent2repMent


def get_refine_rep_ent():
  for key in entMent_2_tags:
    aNo = key.split('\t')[0].split('_')[0]
    sNo = key.split('\t')[0].split('_')[1]

    mention = entMent_2_tags[key][1]


    key_s = int(entMent_2_tags[key][3].split('\t')[2])
    key_e = int(entMent_2_tags[key][3].split('\t')[3])

    aNo_ments = aid2ents[int(aNo)]

    keyii = key+'\t'+mention

    if (keyii not in entMent2repMent) or (keyii in same_entMent2repMent):
      if mention=='Major':
        print(sNo,entMent_2_tags[key])
      retP=None
      temps = 100000

      if sNo =='0':
        #has no antecedent, we choose the first mention or the most
        #frequent entity mention as its coreferent mention?
        print(aNo_ments)
        for menti in aNo_ments:
          print(menti)
          ment_item = menti.split('\t')
          ment_str =ment_item[0]
          ment_sNo = ment_item[1].split('_')[1]

          ment_key = '\t'.join(ment_item[1:4])+'\t'+ment_str


          if mention.lower() in ment_str.lower().split(' ') and ment_str.lower() not in mention.lower():

            if ment_key in entMent2repMent:
              retP = entMent2repMent[ment_key]
            else:
              retP = [ment_key]
            break
      else:
        for menti in aNo_ments:
          ment_item = menti.split('\t')
          ment_str =ment_item[0]
          ment_sNo = ment_item[1].split('_')[1]

          ment_key = '\t'.join(ment_item[1:4])+'\t'+ment_str
          ment_s = int(ment_item[2])
          ment_e = int(ment_item[3])

          dist = int(sNo) - int(ment_sNo)


          if mention.lower() in ment_str.lower().split(' ') and \
                          ment_str.lower() not in mention.lower():
            if mention=='Major':
              print(ment_str,menti,dist,temps)
              print('----------------------')
            if (dist< temps or (dist==temps and ment_s<=key_s and ment_e<key_e)) and dist>=0:
              temps = dist
              if ment_key in entMent2repMent:
                retP = entMent2repMent[ment_key]

              else:
                retP = [ment_key]

      if retP!=None:
        if mention=='Major':
          print(retP)
        if keyii not in entMent2repMent:
          entMent2repMent[keyii] = []
        entMent2repMent[keyii] +=retP
  return entMent2repMent

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_tag', type=str, help='which data file(ace or msnbc or kbp)', required=True)
  parser.add_argument('--dataset', type=str, help='train or eval(but " " for ace and msnbc)', required=True)
  parser.add_argument('--dir_path', type=str, help='data directory path(data/ace or data/msnbc) ', required=True)

  data_args = parser.parse_args()
  data_tag = data_args.data_tag
  dir_path = data_args.dir_path
  dataset = data_args.dataset

  entMent_2_tags = get_ment_2_key()
  aid2ents = get_aid2ents(dir_path)

  Line2entRep = get_rep_ent_in_one_cluster()
  same_entMent2repMent, entMent2repMent = get_rep_ent_for_mention()
  print('reference entity nums:',len(entMent2repMent) )

#  aNoEntstr2repMent={}
#  for item in entMent2repMent:
#    aNosNo,start,end,mention = item.split('\t')
#    aNo = aNosNo.split('_')[0]
#    if aNo not in aNoEntstr2repMent:
#      aNoEntstr2repMent[aNo]={}
#
#    aNoEntstr2repMent[aNo][mention] = entMent2repMent[item]
#
#  entMent2repMent = get_refine_rep_ent()
#  print('refine reference entity nums:',len(entMent2repMent))

  #for key in entMent2repMent:
  #  print key, entMent2repMent[key]
  #  print '--------------------'
  cPickle.dump(entMent2repMent,open(dir_path+'process/entMent2repMent.p','wb'))

  newEntsFile = codecs.open(dir_path+'process/'+'new_entMen2aNosNoid.txt','w','utf-8')
  for key in tqdm(entMent_2_tags):
    val = entMent_2_tags[key]

    line = entMent_2_tags[key][3]
    aNo = key.split('\t')[0].split('_')[0]
    mention = entMent_2_tags[key][1]
    item = key+'\t'+mention
    if item in entMent2repMent:# and entMent2repMent.get(key) in entMentsTags:  #we do not need to do entity linking for this kind of entity!
      repEntsMents = u'\t\t'.join(entMent2repMent[item])

      newEntsFile.write(line+'\t\t'+repEntsMents+'\n')
    else:
      #repEntsMents is itself
      line_items= line.split('\t')
      repEntsMents = '\t'.join(line_items[1:4])+'\t'+line_items[0]
      newEntsFile.write(line+'\t\t'+repEntsMents+'\n')
    print(line)
    print(repEntsMents)
    print('----------------')
