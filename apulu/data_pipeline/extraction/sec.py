"""
methods for creating companies embeddings based on 10-Ks report
mainly used keyword extraction techniques and sentence embedding techniques
"""
from .base import MatrixConstructor
import numpy as np
import json
import os
import re
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sentence_transformers import SentenceTransformer, util

def _text_preproc(x):
  x = x.lower()
  x = x.encode('ascii', 'ignore').decode()
  x = re.sub(r'\w*\d+\w*', '', x)
  x = re.sub(r'\s{2,}', ' ', x)
  return x

def _keyword_extract(comp_fp,top_k=20):
  comp = {}
  try:
    comp = json.load(open(comp_fp)) # "./data/results/MSFT.json"
  except:
    comp = eval(open(comp_fp).read())
  comp_text = []
  for k,v in comp.items():
    v.pop("item1b",None)
    comp_text.extend([_text_preproc(v1) for k1,v1 in v.items()])
  rake_nltk_var = Rake()
  rake_nltk_var.extract_keywords_from_text(" ".join(comp_text))
  keyword_extracted = list(set(rake_nltk_var.get_ranked_phrases()[:top_k]))
  return keyword_extracted

def _sentence_encode(keyword_extracted,model_name='all-mpnet-base-v2'):
  model = SentenceTransformer(model_name) # 'all-mpnet-base-v2'
  sentence_embeddings = model.encode(keyword_extracted)
  embs = np.mean(sentence_embeddings,axis=0)
  return embs

def _out_emb(input_dir,top_k,model_name):
    result = {}
    for comp_fp in tqdm(os.listdir(input_dir)):
        keywords = _keyword_extract(os.path.join(input_dir,comp_fp),top_k)
        emb = _sentence_encode(keywords,model_name)
        result[comp_fp.split(".")[0]] = emb
    return result

class SecMatrixConstructor(MatrixConstructor):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_matrix(self):
        return _out_emb(**self.sec_config["emb_config"])
