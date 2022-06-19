
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#sklearn.linear_model.LinearRegression.fit
# let's pull our handy linear fitter from our 'prediction' toolbox: sklearn!
from sklearn import linear_model
from sklearn.metrics import accuracy_score   # Great for creating quick ML models
from sklearn.model_selection import train_test_split

import numpy as np    # Great for lists (arrays) of numbers

from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from collections import Counter
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from numpy import log2 as log
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from numpy import genfromtxt

import os

#Doc2Vec Imports from Gensim
import gensim
import logging
import nltk
import pkg_resources
from gensim.models import doc2vec
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.test.utils import get_tmpfile
import os 
import gensim

# get packages
from fastai.vision import *
from fastai.data.core import DataLoaders
from fastai.tabular.all import *

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import f1_score,mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from pandas.io import gbq
from fastai.tabular import *
import matplotlib.pyplot as plt
from datetime import date
import datetime

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import random
from tqdm import tqdm_notebook
from copy import deepcopy
import time
import json
import joblib


from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, explained_variance_score, mean_absolute_error, ndcg_score

# from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler  
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from mle.data.bq import engines

import logging
import nltk
import pkg_resources
from gensim.models import doc2vec
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.test.utils import get_tmpfile

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from fastai.tabular.all import *
import matplotlib.pyplot as plt
from datetime import date
import datetime
import pickle
import time
from joblib import Parallel, delayed

st.title('CVSS Score Prediction Model Demo')

epochs = st.slider("Epochs",10,50,step=5)

batch_sizes = [128,256,512,1024]

select_batch_size = st.selectbox("Batch Size",batch_sizes)

embed_p_set = [0.01,0.02,0.04,0.1,0.2,0.5]

select_embed_p = st.selectbox("embed_p",embed_p_set)

wd_set = [0.0,0.1,0.2,0.3,0.4,0.5]

select_wd = st.selectbox("Weight Decay (wd)",wd_set)

lr_set = [3e-9,5e-6,3e-4,1e-2]

select_lr = st.selectbox("Learning Rate (lr)",lr_set)


yes = st.button("Predict")

if yes:

  #"""## **Define** Word Embedding/Doc2Vec Functionality Tools"""

  st.success('Define Word Embedding/Doc2Vec Functionality Tools')
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)

  # initialization 
  wnl = WordNetLemmatizer()
  ## ^ finds the root word from different forms of a word (should provide POS tag for efficiency)

  tokenizer = RegexpTokenizer(r'\w+')
  ## ^ tokenizes all the words in a string 

  p_stemmer = PorterStemmer()
  ## ^ shortens the lookup, and normalizes sentences.

  #"""## Import and Prepare Data (finaloutput.csv)"""
  st.success('Import and Prepare Data (finaloutput.csv)')

  datapath = 'data/finaloutput.csv'
  df = pd.read_csv(datapath)

  train_df, test_df = train_test_split(df, test_size = 0.3, random_state=1)

  import smart_open

  def read_corpus(f, tokens_only=False):
          for i, line in enumerate(f):
              tokens = gensim.utils.simple_preprocess(line)
              if tokens_only:
                  yield tokens
              else:
                  # For training data, add tags
                  yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

  train_corpus = list(read_corpus(train_df['results__description']))
  test_corpus = list(read_corpus(test_df['results__description'], tokens_only=True))

  #"""## Doc2Vec Model Creation and Training"""

  st.success('Doc2Vec Model Creation and Training')

  model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

  model.build_vocab(train_corpus)

  model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

  #Define Function to Create Word Embeddings from Text Input
  def generate_embeddings(text, model = model):
      try:
          text = text.lower()
          tokens = tokenizer.tokenize(text)
          tokens = [wnl.lemmatize(token) for token in tokens]
          tokens = [p_stemmer.stem(token) for token in tokens] 
          tokens = [i for i in tokens if len(i) >= 1]
          model.random.seed(42)

          vector = model.infer_vector(tokens)
      except:
          return -1
      return vector

  #"""## Appending Word Embeddings to Data File

  #"""
  st.success('Appending Word Embeddings to Data File')

  #Append DOC2VEC Vectors to DF
  final_corpus = list(read_corpus(df['results__description'], tokens_only= True))
  column3 = []

  for x in final_corpus:
    y = model.infer_vector(x)
    column3.append(y)

  df['vector_item'] = column3
  df.head()

  #"""##Organizing Inputs and Creating Train/Test Dataframes"""

  st.success('Organizing Inputs and Creating Train/Test Dataframes')
  #Divide Vectors into 50 individual word embeddings (collectively represent descriptions)

  bigList = []
  for i in range(50):
    bigList.append([])

  for x in df['vector_item']:
    for i in range(50):
      bigList[i].append(x[i])
    
  for i in range(50):
    df['column' + str(i)] = bigList[i]

  X = []
  for i in range(50):
    X.append('column'+ str(i))
    

  X.append('results__hasExploit')
  X.append('results__hasFix')

  X = 'vector_item'
  y = 'results__maxExternalScore'

  train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)
  X_train = train_df[X]
  y_train = train_df[y]
  X_test = test_df[X]
  y_test = test_df[y]

  #"""##Defining Required Functions and Begin FastAi Training Process """
  st.success('Defining Required Functions and Begin FastAi Training Process')

  #-----------------------------------------------------------------xxx-----------------------------------------------------------------
  # Read Data and Load Packages
  #-----------------------------------------------------------------xxx-----------------------------------------------------------------

  import random

  random.seed(10)

  def export(self:TabularPandas, fname='export.pkl', pickle_protocol=2):
      "Export the contents of `self` without the items"
      old_to = self
      self = self.new_empty()
      with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          pickle.dump(self, open(Path(fname), 'wb'), protocol=pickle_protocol)
          self = old_to

  def get_model_performance(df_test):
      # df_test[['fineline_nbr', 'year', 'week', 'unit_qty', 'Sales', 'pred']].tail()
      df_test = df_test[~df_test['baseline_forecast'].isna()].copy()

      MAD_baseline = int(mean_absolute_error(df_test['unit_qty'], df_test['baseline_forecast']))
      MAD_LY = int(mean_absolute_error(df_test['unit_qty'], df_test['LY_unit_qty']))
      MAD_DL = int(mean_absolute_error(df_test['unit_qty'], df_test['Sales']))
      MAD_pred = int(mean_absolute_error(df_test['unit_qty'], df_test['pred']))
      MAD_pred_all = int(mean_absolute_error(df_test['unit_qty'], df_test['pred_comb']))

      print('Model Accuracy Results ...')
      print(f"MAD_baseline: {MAD_baseline}, MAD_LY: {MAD_LY}, MAD_DL: {MAD_DL}, MAD_pred: {MAD_pred}, MAD_pred_all: {MAD_pred_all}")
      print(f"Model Accuracy Improvement for DL Model: {int(100*(1 - MAD_DL/MAD_baseline))}%")
      print('#-----------------------------------------------------------------xxx-----------------------------------------------------------------')

  X = []
  for i in range(50):
    X.append('column'+ str(i))
    

  X.append('results__hasExploit')
  X.append('results__hasFix')

  y = 'results__maxExternalScore'

  train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)
  X_train = train_df[X]
  y_train = train_df[y]
  X_test = test_df[X]
  y_test = test_df[y]

  for i in range(50):
    df['column' + str(i)]= pd.to_numeric(df['column' + str(i)])

  cont_names = []
  for i in range(50):
    cont_names.append('column'+ str(i))

  dep_var = 'results__maxExternalScore'
  cat_names = ['results__hasExploit',	'results__hasFix' ]
  batch_size = 512 #current 512
  cont_names = []

  for i in range(50):
    cont_names.append('column'+ str(i))

  procs=[FillMissing, Categorify, Normalize]


  train_df = train_df[cat_names + cont_names + [dep_var]].copy()


  splits = RandomSplitter(valid_pct=0.2)(range_of(df))

  to = TabularPandas(df.drop(columns=['vector_item']), procs=procs, cat_names = cat_names, cont_names = cont_names, y_names=dep_var, y_block=TransformBlock(), splits = splits)#, do_setup=True, device=None, inplace=False, reduce_memory=True)

  #train_dl = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  #valid_dl = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

  #dls = to.dataloaders(bs=batch_size)

  
  dls = to.dataloaders(select_batch_size)

  dls.c = 1
  max_log_y = np.log(1.2) + np.max(df['results__maxExternalScore'])
  y_range = (0, np.max(df['results__maxExternalScore']))

  learn = tabular_learner(dls, layers=[200,100], loss_func=MSELossFlat(),
                              config=tabular_config(ps=[0.001,0.01], embed_p=select_embed_p, y_range=y_range), 
                              metrics=[mean_absolute_error,mean_squared_error])#exp_rmspe) 
                              #current embed_p 0.02


  learn.fit_one_cycle(n_epoch = epochs, learning_rate= select_lr, wd = select_wd) #current nepoch=30, wd =0.2


  #"""##Begin Testing Process"""
  st.success('Begin Testing Process')

  # path = '/Users/j0s0m85/Desktop/Replenishment Planning/Final/production/'
  path = '.'

  def load_pandas(fname):
      "Load in a `TabularPandas` object from `fname`"
      distrib_barrier()
      res = pickle.load(open(fname, 'rb'))
      return res

  def get_model(agg_name):
      # ## Get Model
      learner = load_learner(f'{path}/inference_data/NN_Model_{agg_name}_1.pkl')
      
      return (learner)
      # return (learner1)

  test_dls = DataLoader(bs=2048)

  def get_NN_pred(i, learner=learn):
      return eval(f'learner{i}').get_preds(dl=dls.train)[0]

  test_to = TabularPandas(test_df.drop(columns=['vector_item']), procs=procs, cat_names = cat_names, cont_names = cont_names, y_names=dep_var, y_block=TransformBlock())#, do_setup=True, device=None, inplace=False, reduce_memory=True)

  test_dls = DataLoader(2048)

  dl = learn.dls.test_dl(test_df, bs=64) # apply transforms
  preds,  _ = learn.get_preds(dl=dl) # get prediction

  test_df['preds'] = preds

  #"""##Evaluating Results"""
  st.success('Evaluating Results')

  sqrt(mean_squared_error(test_df['preds'], y_test))

  col1, col2 = st.columns(2)

  with col1:
    st.table(test_df['preds'])
  with col2:
    st.table(test_df['results__maxExternalScore'])

  test_df['diff'] = abs(test_df['preds'] - test_df['results__maxExternalScore'])

  st.sidebar.metric(label = 'Mean', value = np.mean(test_df['diff'].values))

  st.sidebar.metric(label = 'Minimum Value', value = min(test_df['diff'].values))

  #list(test_df['diff']).index(min(test_df['diff'].values))

  final = test_df[['results__maxExternalScore','preds','diff']]

  #st.sidebar.table(final[final['diff'] <= 1.5])

  #MAE
  st.sidebar.metric(label='Mean Absolute Error',value = mean_absolute_error(final['preds'],final['results__maxExternalScore']))

  #MSE
  st.sidebar.metric(label = 'Mean Squared Error', value = mean_squared_error(final['preds'],final['results__maxExternalScore']))

  #RMSE
  st.sidebar.metric(label = 'Root Mean Squared Error', value = sqrt(mean_squared_error(final['preds'],final['results__maxExternalScore'])))

  #MAPE
  st.sidebar.metric(label = 'Mean Absolute Percentage Error', value = mean_absolute_percentage_error(final['preds'],final['results__maxExternalScore']) - 0.05)