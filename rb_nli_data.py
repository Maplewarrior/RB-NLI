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

def main():
  RBOAA = _RBOAA()
  RBAA = AA()

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

  # Get only answer columns
  raw_data = extract_answers(df)

  # Split into the sets of 50
  raw_data = raw_data.reshape([21, 50, 20])

  # Impute nan values with mean of the non-nan values
  for dataset in raw_data:
    np.apply_along_axis(replace_nans_with_mean, axis=0, arr=dataset)

  # Number of ordinal values (bins)
  p = 7
  likert = convert_to_likert(raw_data.copy(), p)
  print(likert)

if __name__ == '__main__':
  main()