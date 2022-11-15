import os
import os.path as osp
import numpy as np
import pandas as pd
from RBOAA.RBOAA_class import _RBOAA
from RBOAA.AAM import AA

def extract_answers(df):
  answer_cols = [f'Answer.rating{i}' for i in range(20)]
  na_cols = [f'Answer.na{i}' for i in range(20)]

  answers = df[answer_cols].copy()
  na = df[na_cols].copy()

  na_mask = (na == 'na').to_numpy()
  masked = answers.mask(na_mask, -1).round().to_numpy()

  return masked

def replace_nans_with_mean(a):
  a[a == -1] = np.mean(a[a != -1])

def convert_to_likert(a, p):
  bins = np.round(100 * (np.arange(p)/p),0)
  return np.array([np.digitize(arr, bins) for arr in a])

def preprocess_data(df):
  # Get only answer columns
  data = extract_answers(df)

  # Split into the sets of 50
  data = data.reshape([21, 50, 20])

  # Impute nan values with mean of the non-nan values
  for dataset in data:
    np.apply_along_axis(replace_nans_with_mean, axis=0, arr=dataset)
  
  return data

def entropy(a):
  return -sum(a*np.log(a))

def analysis(X, plot=False):
  RBOAA = _RBOAA()
  RBAA = AA()
  if len(X.shape) > 2:
    for i in range(X.shape[0]):
      RBAA.load_data(X[i].T, columns=['ph'+str(i+1) for i in range(X.shape[1])])
      RBAA.analyse(K=3, n_iter=20000, AA_type='RBOAA')
  else:
    RBAA.load_data(X.T, columns=['ph'+str(i+1) for i in range(X.shape[1])])
    RBAA.analyse(K=3, n_iter=20000, AA_type='RBOAA')
  
  if plot:
    RBAA.plot('RBOAA', plot_type='PCA_scatter_plot')
    RBAA.plot('RBOAA', plot_type='barplot', archetype_number=0)
    RBAA.plot('RBOAA', plot_type='barplot', archetype_number=1)
    RBAA.plot('RBOAA', plot_type='barplot', archetype_number=2)
    RBAA.plot('RBOAA', plot_type='barplot_all')
  
  return RBAA

def main():
  #  = pd.read_json(path_or_buf=, lines=True)
  # NLIsent = pd.read_json(path_or_buf=, lines=True)
  # NLIcontext = 'data/NLI-variation-data/context-analysis/preprocessed-context-data.jsonl'
  # NLIsent = 'data/NLI-variation-data/sentence-pair-analysis/preprocessed-data.jsonl'
  # sent1 ='data/NLI-variation-data/sentence-pair-analysis/raw/batch1.csv'
  # sent2 ='data/NLI-variation-data/sentence-pair-analysis/raw/batch2.csv'
  # sent3 ='data/NLI-variation-data/sentence-pair-analysis/raw/batch3.csv'
  context = 'data/NLI-variation-data/context-analysis/raw/batch1.csv'

  filepath = osp.join(os.getcwd(), context)

  df = pd.read_csv(filepath)
  data = preprocess_data(df)

  # Number of ordinal values (bins)
  p = 7
  likert = convert_to_likert(data.copy(), p)
  
  num_datasets = 1
  subset = likert[:num_datasets]
  
  anal = analysis(subset)

  result1 = anal._results['RBOAA'][0]

  A = result1.A
  

if __name__ == '__main__':
  main()