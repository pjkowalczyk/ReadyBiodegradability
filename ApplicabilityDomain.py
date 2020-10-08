##
# C:\Users\us16120\Projects\Cheminformatics\readybiodegradability
                                                        # Applicability Domain
##

import pandas as pd
# from pandas import Series
# from pandas.Series import astype
import numpy as np
from numpy import linalg
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import scipy as scipy
from scipy import spatial
from scipy.spatial import distance
from scipy import stats
import astropy as astropy
from astropy import stats
from astropy.stats import funcs
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

df = pd.read_csv('data/processed/alles02.csv', header = 0)
df = df.drop(['Unnamed: 0'], axis=1)
df.sample(5).head()

nms = [x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
for i in range(len(df)):
    try:
        descrs = calc.CalcDescriptors(Chem.MolFromSmiles(df.iloc[i, 0]))
        for x in range(len(descrs)):
            df.at[i, str(nms[x])] = descrs[x]
    except:
        for x in range(len(descrs)):
            df.at[i, str(nms[x])] = 'NaN'   
            
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index(drop=True)

train, test = train_test_split(df, test_size = 0.2, random_state = 42,
                               stratify=df[['ReadyBiodeg']])
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

X_train = train.drop(columns=['SMILES', 'EndPt', 'InChI', 'ReadyBiodeg'])
X_test = test.drop(columns=['SMILES', 'EndPt', 'InChI', 'ReadyBiodeg'])
y_train = np.ravel(train[['ReadyBiodeg']])
y_test = np.ravel(test[['ReadyBiodeg']])

def variance_threshold_selector(data, threshold = 0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices = True)]]

nzv = variance_threshold_selector(X_train, 0.0)

X_train = X_train[nzv.columns]
X_test = X_test[nzv.columns]

corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k = 1).astype(np.bool))
to_drop = [column for column in upper.columns
           if any(upper[column] > 0.85)]

X_train = X_train[X_train.columns.drop(to_drop)]
X_test = X_test[X_test.columns.drop(to_drop)]

scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#########################

col_names =  ['dist']
distances = pd.DataFrame(columns = col_names)
avgDist5 = pd.DataFrame(columns = col_names)

for i in range(len(X_train_std)):
    for j in range(len(X_train_std)):
        distances.loc[j, 'dist'] = np.linalg.norm(X_train_std[i] - X_train_std[j])
    euclid = distances.sort_values('dist').head(6).mean()
    avgDist5.loc[i, 'dist'] = euclid[0]
    if (i % 5 == 0):
        print(i)

X = train.copy()[['SMILES', 'InChI', 'EndPt', 'ReadyBiodeg']]
X['dist'] = avgDist5['dist']

X.to_csv('X.csv')
X = pd.read_csv('X.csv', header = 0)
X = X.drop(['Unnamed: 0'], axis=1)
X.sample(5).head()
X.describe()

Q1 = np.percentile(X.dist, 25)
median = np.percentile(X.dist, 50)
Q3 = np.percentile(X.dist, 75)
IQR = Q3 - Q1
MAD = funcs.median_absolute_deviation(X.dist)
upper_outlier = median + 2.5 * MAD
lower_outlier = median - 2.5 * MAD
avgDist5['dist'][0].describe()

###

col_names =  ['dist']
testDistances = pd.DataFrame(columns = col_names)
testAvgDist5 = pd.DataFrame(columns = col_names)

for i in range(len(X_test_std)):
    for j in range(len(X_train_std)):
        distances.loc[j, 'dist'] = np.linalg.norm(X_test_std[i] - X_train_std[j])
    euclid = distances.sort_values('dist').head(6).mean()
    testAvgDist5.loc[i, 'dist'] = euclid[0]
    if (i % 5 == 0):
        print(i)

Y = test.copy()[['SMILES', 'InChI', 'EndPt', 'ReadyBiodeg']]
Y['dist'] = testAvgDist5['dist']

Y.to_csv('Y.csv')
Y = pd.read_csv('Y.csv', header = 0)
Y = Y.drop(['Unnamed: 0'], axis=1)
Y.sample(5).head()
Y.describe()

Q1 = np.percentile(Y.dist, 25)
median = np.percentile(Y.dist, 50)
Q3 = np.percentile(Y.dist, 75)
IQR = Q3 - Q1
MAD = funcs.median_absolute_deviation(Y.dist)
upper_outlier = median + 2.5 * MAD
lower_outlier = median - 2.5 * MAD

###

sns.distplot(X.dist)
plt.axvline(7.497, 0, 0.75)
plt.axvline(16.830, 0, 0.75)

from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

xy = np.array(X.dist)
plt.hist(x, density=True, bins=30)
plt.ylabel('Probability');

N_points = 100000
n_bins = 20

# Generate a normal distribution, center at x=0 and y=5
x = np.random.randn(N_points)
y = .4 * x + np.random.randn(100000) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(x, bins=n_bins)
axs[1].hist(y, bins=n_bins)
