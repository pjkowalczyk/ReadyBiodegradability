import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy import stats
from scipy.spatial import distance_matrix
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

df = pd.read_csv('data/processed/alles02.csv', header = 0)
df = df.drop(['Unnamed: 0'], axis=1)

nms = [x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
for i in range(len(df)):
    try:
        descrs = calc.CalcDescriptors(Chem.MolFromSmiles(df.iloc[i, 1]))
        for x in range(len(descrs)):
            df.at[i, str(nms[x])] = descrs[x]
    except:
        for x in range(len(descrs)):
            df.at[i, str(nms[x])] = 'NaN'   
            
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index(drop=True)

dfFiltered = df[df['MolWt'] <= 700]
dfFiltered = dfFiltered[dfFiltered['HeavyAtomCount'] <= 50]
dfFiltered = dfFiltered[dfFiltered['NumRotatableBonds'] <= 12]
dfFiltered = dfFiltered[dfFiltered['NumHDonors'] <= 5]
dfFiltered = dfFiltered[dfFiltered['NumHAcceptors'] <= 10]
dfFiltered = dfFiltered[dfFiltered['MolLogP'] <= 7.5]
dfFiltered = dfFiltered[dfFiltered['MolLogP'] >= -5.0]

dfFiltered = dfFiltered.reset_index(drop=True)

X = dfFiltered.drop(columns = ['SMILES', 'SMILESbeta', 'EndPt', 'ReadyBiodeg'])
y = np.ravel(dfFiltered[['ReadyBiodeg']])
chemistry = dfFiltered[['SMILES', 'SMILESbeta', 'EndPt', 'ReadyBiodeg']]

def variance_threshold_selector(data, threshold = 0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices = True)]]

nzv = variance_threshold_selector(X, 0.0)

X = X[nzv.columns]

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k = 1).astype(np.bool))
to_drop = [column for column in upper.columns
           if any(upper[column] > 0.85)]

X = X[X.columns.drop(to_drop)]

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)

# distance matrix
dist = distance_matrix(X, X, p = 1, threshold = 1000000)

# 1-nearest neighbor
# the predicted endpoint == the endpoint of the nearest neighbor
## TODO: correlation(?) / relation(?) correct assignment & distance

col_names =  ['record', 'expt', 'nn', 'binary']
results = pd.DataFrame(columns = col_names)

tn, fp, fn, tp = 0, 0, 0, 0

for i in range(len(X)):
    results.loc[i, 'record'] = i
    results.loc[i, 'expt'] = y[i]
    results.loc[i, 'nn'] = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).\
               sort_values(0).head(1).index.values.astype(int)[0]]
    results.loc[i, 'binary'] = 2 * results.loc[i, 'expt'] + results.loc[i, 'nn']
    if results.loc[i, 'binary'] == 0:
        tn += 1
    elif results.loc[i, 'binary'] == 1:
        fp += 1
    elif results.loc[i, 'binary'] == 2:
        fn += 1
    else:
        tp += 1
    
# 3-nearest neighbors

col_names =  ['record', 'expt', 'nn', 'binary']
results = pd.DataFrame(columns = col_names)

tn, fp, fn, tp = 0, 0, 0, 0

for i in range(len(X)):
    results.loc[i, 'record'] = i
    results.loc[i, 'expt'] = y[i]
    a = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(3).index.values.astype(int)[0]]
    b = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(3).index.values.astype(int)[1]]
    c = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(3).index.values.astype(int)[2]]
    abc = a + b + c
    if abc < 2:
        results.loc[i, 'nn'] = 0
    else:
        results.loc[i, 'nn'] = 1
    
    results.loc[i, 'binary'] = 2 * results.loc[i, 'expt'] + results.loc[i, 'nn']
    
    if results.loc[i, 'binary'] == 0:
        tn += 1
    elif results.loc[i, 'binary'] == 1:
        fp += 1
    elif results.loc[i, 'binary'] == 2:
        fn += 1
    else:
        tp += 1
        
# 5-nearest neighbors

col_names =  ['record', 'expt', 'nn', 'binary']
results = pd.DataFrame(columns = col_names)

tn, fp, fn, tp = 0, 0, 0, 0

for i in range(len(X)):
    results.loc[i, 'record'] = i
    results.loc[i, 'expt'] = y[i]
    a = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(5).index.values.astype(int)[0]]
    b = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(5).index.values.astype(int)[1]]
    c = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(5).index.values.astype(int)[2]]
    d = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(5).index.values.astype(int)[3]]
    e = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(5).index.values.astype(int)[4]]
    abc = a + b + c + d + e
    if abc < 3:
        results.loc[i, 'nn'] = 0
    else:
        results.loc[i, 'nn'] = 1
    
    results.loc[i, 'binary'] = 2 * results.loc[i, 'expt'] + results.loc[i, 'nn']
    
    if results.loc[i, 'binary'] == 0:
        tn += 1
    elif results.loc[i, 'binary'] == 1:
        fp += 1
    elif results.loc[i, 'binary'] == 2:
        fn += 1
    else:
        tp += 1

# 7-nearest neighbors
        
col_names =  ['record', 'expt', 'nn', 'binary']
results = pd.DataFrame(columns = col_names)

tn, fp, fn, tp = 0, 0, 0, 0

for i in range(len(X)):
    results.loc[i, 'record'] = i
    results.loc[i, 'expt'] = y[i]
    a = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(7).index.values.astype(int)[0]]
    b = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(7).index.values.astype(int)[1]]
    c = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(7).index.values.astype(int)[2]]
    d = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(7).index.values.astype(int)[3]]
    e = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(7).index.values.astype(int)[4]]
    f = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(7).index.values.astype(int)[5]]
    g = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(7).index.values.astype(int)[6]]
    abc = a + b + c + d + e + f + g
    if abc < 4:
        results.loc[i, 'nn'] = 0
    else:
        results.loc[i, 'nn'] = 1
    
    results.loc[i, 'binary'] = 2 * results.loc[i, 'expt'] + results.loc[i, 'nn']
    
    if results.loc[i, 'binary'] == 0:
        tn += 1
    elif results.loc[i, 'binary'] == 1:
        fp += 1
    elif results.loc[i, 'binary'] == 2:
        fn += 1
    else:
        tp += 1       
        
# 9-nearest neighbors
        
col_names =  ['record', 'expt', 'nn', 'binary']
results = pd.DataFrame(columns = col_names)

tn, fp, fn, tp = 0, 0, 0, 0

for i in range(len(X)):
    results.loc[i, 'record'] = i
    results.loc[i, 'expt'] = y[i]
    a = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[0]]
    b = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[1]]
    c = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[2]]
    d = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[3]]
    e = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[4]]
    f = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[5]]
    g = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[6]]
    h = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[7]]
    j = y[pd.DataFrame(np.delete(np.take(dist, i, 0), i, 0)).sort_values(0).head(9).index.values.astype(int)[8]]
    abc = a + b + c + d + e + f + g + h + j
    if abc < 5:
        results.loc[i, 'nn'] = 0
    else:
        results.loc[i, 'nn'] = 1
    
    results.loc[i, 'binary'] = 2 * results.loc[i, 'expt'] + results.loc[i, 'nn']
    
    if results.loc[i, 'binary'] == 0:
        tn += 1
    elif results.loc[i, 'binary'] == 1:
        fp += 1
    elif results.loc[i, 'binary'] == 2:
        fn += 1
    else:
        tp += 1       
        
        