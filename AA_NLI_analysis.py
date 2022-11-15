import os
import numpy as np
import pandas as pd
cwd = os.getcwd()
os.chdir(os.path.join(os.getcwd(), 'RBOAA'))
from AAM import AA
os.chdir(cwd)



############ Load data ############
NLIcontext = pd.read_json(path_or_buf='data/NLI-variation-data/context-analysis/preprocessed-context-data.jsonl', lines=True)
NLIsent = pd.read_json(path_or_buf='data/NLI-variation-data/sentence-pair-analysis/preprocessed-data.jsonl', lines=True)

context_data = pd.read_csv('data/NLI-variation-data/context-analysis/raw/batch1.csv')
sent1 = pd.read_csv('data/NLI-variation-data/sentence-pair-analysis/raw/batch1.csv')
sent2 = pd.read_csv('data/NLI-variation-data/sentence-pair-analysis/raw/batch2.csv')
sent3 = pd.read_csv('data/NLI-variation-data/sentence-pair-analysis/raw/batch3.csv')

answer_cols = [f'Answer.rating{i}' for i in range(20)]
na_cols = [f'Answer.na{i}' for i in range(20)]


############ Preprocess data ############
context_answers = context_data[answer_cols].copy()
context_na = context_data[na_cols].copy()

# Set unanswered questions to -1 and round answers to nearest int
na_mask = (context_na == 'na').to_numpy()
context_masked = context_answers.mask(na_mask, -1).round().to_numpy()
context = context_masked.reshape([21, 50, 20])


# np.set_printoptions(linewidth=np.inf)

# for i, question_set in enumerate(context):
#     print(f'############################ SET {i} ############################')
#     nans_rows = 0
#     nans_columns = 0
#     print(f'QUESTION SET: \n\n{question_set}')
#     for j in range(50):
#         nans_rows += 1 if -1 in question_set[j,:] else 0
#     for j in range(20):
#         nans_columns += 1 if -1 in question_set[:,j] else 0
#     print(f'{nans_rows} rows have NaNs')
#     print(f'{nans_columns} columns have NaNs')


nans_imputed = context.copy()

def replace_nans_with_mean(a):
    a[a == -1] = np.mean(a[a != -1])

for arr in nans_imputed:
    np.apply_along_axis(replace_nans_with_mean, axis=0, arr=arr)



converted = nans_imputed.copy()

def convert_to_likert(a, p):
    bins = np.round(100 * (np.arange(p)/p),0)
    return np.array([np.digitize(arr, bins) for arr in a])

converted = convert_to_likert(converted, 7)


############ Make analyses ############
RBAA = AA()
def createAnalyses(data, K, n_iter, columns, model=RBAA, n_analyses=2):
    for i in range(n_analyses):
        model.load_data(data[i].T, columns)
        model.analyse(K=K, n_iter=n_iter, AA_type='RBOAA')
        
    return model

# get plots
def plot(model, result_number, plot_type):
    model.plot('RBOAA', result_number=result_number, plot_type=plot_type)
    
    
def getAAMatrices(model, result_number):
    return model._results['RBOAA'][result_number].Z, model._results['RBOAA'][result_number].A

    
        


