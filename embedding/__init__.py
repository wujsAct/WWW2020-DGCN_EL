# -*- coding: utf-8 -*-
from sys import version_info
version = version_info.major
if version==2:
  from get_ent_embed_data import EntEmbedUtils
  from get_ent_relateness_data import RelatenessDataReader
  from get_word_info_utils import WordFreqVectorLoader
else:
  from .get_ent_embed_data import EntEmbedUtils
  from .get_ent_relateness_data import RelatenessDataReader
  from .get_word_info_utils import WordFreqVectorLoader
