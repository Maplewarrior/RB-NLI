import os
import numpy as np
import pandas as pd
from RBOAA.RBOAA_class import _RBOAA

chaosSNLI = pd.read_json(path_or_buf='/content/drive/MyDrive/RB NLI/data/chaosNLI_v1.0/chaosNLI_snli.jsonl', lines=True)
chaosMNLI = pd.read_json(path_or_buf='/content/drive/MyDrive/RB NLI/data/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl', lines=True)
chaosANLI = pd.read_json(path_or_buf='/content/drive/MyDrive/RB NLI/data/chaosNLI_v1.0/chaosNLI_alphanli.jsonl', lines=True)

# Initialize model
RBOAA = _RBOAA()

chaosSNLI.head()
#print(chaosSNLI['example'][0])

def extractSingleAnntotations(df, n_annotators=100):
  # n = 1, e = 2, c = 3
  N = len(df)
  label_counts = df['label_counter']

  answer_dist = [{'n':0, 'e':0, 'c':0} for _ in range(N)]
  #annotations = [np.empty((n_annotators, 1)) for _ in range(N)]
  annotations = []
  for i in range(N):
    answer_dist[i]['n'] = label_counts[i]['n'] if 'n' in list(label_counts[i].keys()) else 0
    answer_dist[i]['e'] = label_counts[i]['e'] if 'e' in list(label_counts[i].keys()) else 0
    answer_dist[i]['c'] = label_counts[i]['c'] if 'c' in list(label_counts[i].keys()) else 0
    # baseline annotations method...
    annotations.append(np.expand_dims(np.repeat([1,2,3], [answer_dist[i]['n'], answer_dist[i]['e'], answer_dist[i]['c']]), axis=0))
  
  new_df = df[['example', 'entropy']].copy()
  new_df['answer_dist'] = answer_dist
  new_df['annotations'] = annotations
  return new_df

label_counts = chaosSNLI['label_counter']
answer_dist = [{'n':0, 'e':0, 'c':0} for _ in range(len(label_counts))]
for i in range(len(label_counts)):
  answer_dist[i]['n'] = label_counts[i]['n'] if 'n' in list(label_counts[i].keys()) else 0
  answer_dist[i]['e'] = label_counts[i]['e'] if 'e' in list(label_counts[i].keys()) else 0
  answer_dist[i]['c'] = label_counts[i]['c'] if 'c' in list(label_counts[i].keys()) else 0

df_s = extractSingleAnntotations(chaosSNLI) 
df_s.head()

#df_s['annotations'][0].shape
res = RBOAA._compute_archetypes(X=df_s['annotations'][0], K=3, n_iter=1000, lr=0.01, mute=False, columns=['Q1'])

print(f'Z shape: \t {res.Z.shape}')
print(f'A shape: \t{res.A.shape}')

"""## Problems/questions

* We have no way of knowing the actual answers amongst respondents → How do we deal with this? 
* Using archetypal analysis might just loose information from the existing dataset... 
* How can we optimally use the information gathered by AA?

* Pivot? → Cluster groups of people based on their closest  archetype across examples?


## Appropriate datasets:
https://github.com/epavlick/NLI-variation-data/tree/master/context-analysis/raw

https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00293/43531/Inherent-Disagreements-in-Human-Textual-Inferences?fbclid=IwAR0KokaN1zCGAl1x9Z3BhWCFaSL04WSs2eINdwqlvuz9lZd2tuKiLset61M


* UNLI dataset (Chen et al., 2020) only has two annotators (maybe 3) per example

"""

def entropy(a):
  return -sum([e*np.log(e) for e in a])

# A = archetype weighting matrix, n_subjects = how many subjects should be considered
def get_gold_answer(df, A, n_subjects):
  entropies = [entropy(a) for a in res.A.T]
  candidates = []
  for i in range(n_subjects):
    s = np.argmax(entropies)
    # i is added to account for the entropies list getting shorter
    candidates.append(s+i)
    entropies.pop(s)


  answers = [df['annotations'][candidates[i]] for i in range(n_subjects)]

  return candidates


  
ans = get_gold_answer(df_s, res.A, 10)

  

subject = np.argmax([entropy(a) for a in res.A.T])

df_s['annotations'][0][0][0]
ans

"""# $\textbf{NLI-variation dataset}$"""

NLIcontext = pd.read_json(path_or_buf='/content/drive/MyDrive/RB NLI/data/NLI-variation-data/context-analysis/preprocessed-context-data.jsonl', lines=True)
NLIsent = pd.read_json(path_or_buf='/content/drive/MyDrive/RB NLI/data/NLI-variation-data/sentence-pair-analysis/preprocessed-data.jsonl', lines=True)

print(NLIsent['premise'][0])
print(NLIsent['hypothesis'][0])
print(NLIsent['labels'][0])

"""# Load raw data"""

context_data = pd.read_csv('/content/drive/MyDrive/RB NLI/data/NLI-variation-data/context-analysis/raw/batch1.csv')
sent1 = pd.read_csv('/content/drive/MyDrive/RB NLI/data/NLI-variation-data/sentence-pair-analysis/raw/batch1.csv')
sent2 = pd.read_csv('/content/drive/MyDrive/RB NLI/data/NLI-variation-data/sentence-pair-analysis/raw/batch2.csv')
sent3 = pd.read_csv('/content/drive/MyDrive/RB NLI/data/NLI-variation-data/sentence-pair-analysis/raw/batch3.csv')

"""# Extract relevant columns"""

answer_cols = [f'Answer.rating{i}' for i in range(20)]
na_cols = [f'Answer.na{i}' for i in range(20)]


# extract data
context_answers = context_data[answer_cols].copy()
context_na = context_data[na_cols].copy()

# Set unanswered questions to -1 and round answers to nearest int
na_mask = (context_na == 'na').to_numpy()
context_masked = context_answers.mask(na_mask, -1).round().to_numpy()

context = context_masked.reshape([21, 50, 20])

context[0][1]

np.set_printoptions(linewidth=np.inf)

for i, question_set in enumerate(context):
    print(f'############################ SET {i} ############################')
    nans_rows = 0
    nans_columns = 0
    print(f'QUESTION SET: \n\n{question_set}')
    for j in range(50):
        nans_rows += 1 if -1 in question_set[j,:] else 0
    for j in range(20):
        nans_columns += 1 if -1 in question_set[:,j] else 0
    print(f'{nans_rows} rows have NaNs')
    print(f'{nans_columns} columns have NaNs')

# AAM - instance of class for plots
#     - mute = False -> can see analysis running
#     - feed dataset, RBOAA, dataset, num archytypes, columns
# TSAA - comment out in AAM

os.chdir('/content/drive/MyDrive/RB NLI/RBOAA/')
from AAM import AA
os.chdir('/content/drive/MyDrive/RB NLI')

np.set_printoptions(linewidth=np.inf)

nans_imputed = context.copy()

def replace_nans_with_mean(a):
    a[a == -1] = np.mean(a[a != -1])

for arr in nans_imputed:
    np.apply_along_axis(replace_nans_with_mean, axis=0, arr=arr)
    
for i, question_set in enumerate(nans_imputed):
    print(f'############################ SET {i} ############################')
    nans_rows = 0
    nans_columns = 0
    print(f'QUESTION SET: \n\n{question_set}')

RBAA = AA()

converted = nans_imputed.copy()

def convert_to_likert(a, p):
    bins = np.round(100 * (np.arange(p)/p),0)
    return np.array([np.digitize(arr, bins) for arr in a])


#print(convert_to_likert(converted, 7))
#print(converted)
converted = convert_to_likert(converted, 7)

X = converted[0]
RBAA.load_data(X.T, columns=['ph'+str(i+1) for i in range(X.shape[1])])

def getAnalyses:
  
RBAA.analyse(K=3, n_iter=20000, AA_type='RBOAA')

RBAA.plot('RBOAA', plot_type='PCA_scatter_plot')

RBAA.plot('RBOAA', plot_type='barplot', archetype_number=0)

RBAA.plot('RBOAA', plot_type='barplot', archetype_number=1)

RBAA.plot('RBOAA', plot_type='barplot', archetype_number=2)

RBAA.plot('RBOAA', plot_type='barplot_all')

RBAA.plot('RBOAA', plot_type='barplot_all')


def entropy(a):
  return -sum([e*np.log(e) for e in a])

# (e, n, c) --> (0.76, 0.2, 0.04)
probs = [0.76, 0.2, 0.04]
entropy(probs)

RBAA._results['RBOAA'][0].Z

RBAA._results['RBOAA'][0].A.T

import matplotlib.pyplot as plt
#np.hist(np.argmax(RBAA._results['RBOAA'][0].A.T, axis=1))
plt.hist(np.argmax(RBAA._results['RBOAA'][0].A.T, axis=1))

"""# What do we do with this

* Cluster p → h pairs based on whether there is strong disagreement about the label. 

* Cluster respondents based on how strongly they belong to a single archetype.

* Across (50 x 20) matrices can we relate them in a meaningful way?
"""

RBAA._results['RBOAA'][0].A.T

RBAA._results['RBOAA'][1].A.T

