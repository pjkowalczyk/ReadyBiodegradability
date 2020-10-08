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
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy import stats
from statsmodels import robust
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

df.head()

k2, p = stats.normaltest(df['MolWt'])
print(k2)
print(p)

stats.skew(df['MolWt'])

stats.skewtest(df['MolWt'])

sns.distplot(df['MolWt'])

df['MolWt'].describe()

sns.boxplot(df['MolWt'])


# In[56]:


mean = np.mean(df['MolWt'], axis=0)
print(mean)
sd = np.std(df['MolWt'], axis=0)
print(sd)
print(mean + 3 * sd)


# # Build training set & test set

# In[4]:


train, test = train_test_split(df, test_size = 0.2, random_state = 42,
                               stratify=df[['ReadyBiodeg']])
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


# # X & y

# In[5]:


X_train = train.drop(columns=['SMILES', 'EndPt', 'InChI', 'ReadyBiodeg'])
X_test = test.drop(columns=['SMILES', 'EndPt', 'InChI', 'ReadyBiodeg'])
y_train = np.ravel(train[['ReadyBiodeg']])
y_test = np.ravel(test[['ReadyBiodeg']])


# In[6]:


print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))


# # Feature Engineering

# ## Identify / remove near-zero variance descriptors

# In[7]:


def variance_threshold_selector(data, threshold = 0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices = True)]]

nzv = variance_threshold_selector(X_train, 0.0)

X_train = X_train[nzv.columns]
X_test = X_test[nzv.columns]


# In[8]:


print(X_train.shape)
print(X_test.shape)


# ## Identify / remove highly correlated descriptors

# In[9]:


corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k = 1).astype(np.bool))
to_drop = [column for column in upper.columns
           if any(upper[column] > 0.85)]

X_train = X_train[X_train.columns.drop(to_drop)]
X_test = X_test[X_test.columns.drop(to_drop)]


# In[10]:


print(X_train.shape)
print(X_test.shape)


# In[11]:


list(X_train.columns);


# ## Standardize features by removing the mean and scaling to unit variance

# In[11]:


scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


# # Models

# ## Random Forest Classification

# In[16]:


rf = RandomForestClassifier(random_state = 42)


# ##### Look at parameters used by our current forest

# In[17]:


print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[18]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# ##### Create the random grid

# In[19]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


# ##### Use the random grid to search for the best hyperparameters

# In[20]:


# First create the base model to tune
rf = RandomForestClassifier(random_state = 42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                              n_iter = 100, scoring = 'neg_mean_absolute_error', 
                              cv = 3, verbose = 2, random_state = 42,
                              n_jobs = -1, return_train_score = True)


# ##### Fit the random search model

# In[21]:


rf_random.fit(X_train_std, y_train);


# In[22]:


rf_random.best_params_


# In[23]:


import pickle
f = open('RandomForest.pkl', 'wb')
pickle.dump(rf_random, f)
f.close()


# In[24]:


pred = rf_random.predict(X_test_std)


# In[25]:


kappa = metrics.cohen_kappa_score(y_test, pred)
print('Kappa: {:.2f}'.format(kappa))


# In[26]:


probs = rf_random.predict_proba(X_test_std)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[27]:


print(confusion_matrix(y_test, pred))


# In[29]:


rf_random.best_params_


# In[30]:


col_names =  ['TrueLabel', 'Nomine', 'Probability']
plotDF  = pd.DataFrame(columns = col_names)


# In[31]:


plotDF['TrueLabel'] = y_test
plotDF['Nomine'] = 'Prediction'
plotDF['Probability'] = preds


# In[32]:


plotDF.head()


# In[33]:


confusionPlot = sns.stripplot(x = 'Probability', y = 'Nomine', hue = 'TrueLabel', jitter = 0.4, dodge = True, data = plotDF)


# ## Support Vector Classification

# In[34]:


svm = SVC()


# In[35]:


pprint(svm.get_params())


# In[36]:


best_kappa_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # train an SVC for each combination of gamma and C
        svm = SVC(gamma = gamma, C = C)
        svm.fit(X_train_std, y_train)
        pred = svm.predict(X_test_std)
        # evaluate the SVC on the test set
        kappa = metrics.cohen_kappa_score(y_test, pred)
        # better kappa? store kappa and parameters
        if kappa > best_kappa_score:
            best_kappa_score = kappa
            best_parameters = {'C' : C, 'gamma' : gamma}
            
print("Best score: {:.2f}".format(best_kappa_score))
print("Best parameters: {}".format(best_parameters)) 


# In[37]:


svm = SVC(C = 100, gamma = 0.1, kernel = 'rbf', probability = True)
svm.fit(X_train_std, y_train)


# In[38]:


import pickle
f = open('SupportVectorMachine.pkl', 'wb')
pickle.dump(svm, f)
f.close()


# In[39]:


pred_svm = svm.predict(X_test_std)


# In[40]:


kappa = metrics.cohen_kappa_score(y_test, pred_svm)
print('Kappa: {:.2f}'.format(kappa))


# In[41]:


probs_svm = svm.predict_proba(X_test_std)
preds_svm = probs_svm[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_svm)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[42]:


print(confusion_matrix(y_test, pred_svm))


# In[43]:


col_names =  ['TrueLabel', 'Nomine', 'Probability']
plotDF  = pd.DataFrame(columns = col_names)


# In[44]:


plotDF['TrueLabel'] = y_test
plotDF['Nomine'] = 'Prediction'
plotDF['Probability'] = preds_svm


# In[45]:


confusionPlot = sns.stripplot(x = 'Probability', y = 'Nomine', hue = 'TrueLabel',                              jitter = 0.4, dodge = True, data = plotDF)


# ## Logistic Regression

# In[46]:


lr = LogisticRegression()


# In[47]:


pprint(lr.get_params())


# In[48]:


best_kappa_score = 0

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    # train logistic regression for each value of C
    lr = LogisticRegression(C = C)
    lr.fit(X_train_std, y_train)
    pred_lr = lr.predict(X_test_std)
    # evaluate the lr on the test set
    kappa = metrics.cohen_kappa_score(y_test, pred)
    # better kappa? store kappa and parameters
    if kappa > best_kappa_score:
        best_kappa_score = kappa
        best_parameters = {'C' : C}
            
print("Best score: {:.2f}".format(best_kappa_score))
print("Best parameters: {}".format(best_parameters)) 


# In[49]:


lr = LogisticRegression(C = 0.001)
lr.fit(X_train_std, y_train)


# In[50]:


import pickle
f = open('LogisticRegressionj.pkl', 'wb')
pickle.dump(lr, f)
f.close()


# In[51]:


kappa = metrics.cohen_kappa_score(y_test, pred_lr)
print('Kappa: {:.2f}'.format(kappa))


# In[52]:


probs_lr = lr.predict_proba(X_test_std)
preds_lr = probs_lr[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_lr)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[53]:


print(confusion_matrix(y_test, pred_lr))


# In[54]:


col_names =  ['TrueLabel', 'Nomine', 'Probability']
plotDF  = pd.DataFrame(columns = col_names)


# In[55]:


plotDF['TrueLabel'] = y_test
plotDF['Nomine'] = 'Prediction'
plotDF['Probability'] = preds_lr


# In[56]:


confusionPlot = sns.stripplot(x = 'Probability', y = 'Nomine', hue = 'TrueLabel',                              jitter = 0.4, dodge = True, data = plotDF)


# ## Gaussian naive Bayes

# In[13]:


# Create a Gaussian Classifier
gnb = GaussianNB()


# In[22]:


pprint(gnb.get_params())


# In[14]:


# Train the model using the training sets 
gnb.fit(X_train_std, y_train)


# In[15]:


pred_gnb = gnb.predict(X_test_std)


# In[16]:


kappa = metrics.cohen_kappa_score(y_test, pred_gnb)
print('Kappa: {:.2f}'.format(kappa))


# In[17]:


probs_gnb = gnb.predict_proba(X_test_std)
preds_gnb = probs_gnb[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_gnb)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[18]:


print(confusion_matrix(y_test, pred_gnb))


# In[19]:


col_names =  ['TrueLabel', 'Nomine', 'Probability']
plotDF  = pd.DataFrame(columns = col_names)


# In[20]:


plotDF['TrueLabel'] = y_test
plotDF['Nomine'] = 'Prediction'
plotDF['Probability'] = preds_gnb


# In[21]:


confusionPlot = sns.stripplot(x = 'Probability', y = 'Nomine', hue = 'TrueLabel',                              jitter = 0.4, dodge = True, data = plotDF)


# In[23]:


from sklearn.ensemble import AdaBoostClassifier


# In[24]:


# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
abc.fit(X_train_std, y_train)

#Predict the response for test dataset
pred_abc = abc.predict(X_test_std)


# In[25]:


kappa = metrics.cohen_kappa_score(y_test, pred_abc)
print('Kappa: {:.2f}'.format(kappa))


# In[26]:


probs_abc = abc.predict_proba(X_test_std)
preds_abc = probs_abc[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_abc)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[27]:


print(confusion_matrix(y_test, pred_abc))


# In[28]:


col_names =  ['TrueLabel', 'Nomine', 'Probability']
plotDF  = pd.DataFrame(columns = col_names)

plotDF['TrueLabel'] = y_test
plotDF['Nomine'] = 'Prediction'
plotDF['Probability'] = preds_abc

confusionPlot = sns.stripplot(x = 'Probability', y = 'Nomine', hue = 'TrueLabel',                              jitter = 0.4, dodge = True, data = plotDF)


# In[35]:


svc = SVC(probability=True, kernel = 'rbf')
abc = AdaBoostClassifier(n_estimators = 50, base_estimator = svc, learning_rate = 1)

# Train Adaboost Classifer
abc.fit(X_train_std, y_train)

#Predict the response for test dataset
pred_abc = abc.predict(X_test_std)


# In[36]:


kappa = metrics.cohen_kappa_score(y_test, pred_abc)
print('Kappa: {:.2f}'.format(kappa))


# In[37]:


probs_abc = abc.predict_proba(X_test_std)
preds_abc = probs_abc[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_abc)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[38]:


print(confusion_matrix(y_test, pred_abc))


# In[39]:


col_names =  ['TrueLabel', 'Nomine', 'Probability']
plotDF  = pd.DataFrame(columns = col_names)

plotDF['TrueLabel'] = y_test
plotDF['Nomine'] = 'Prediction'
plotDF['Probability'] = preds_abc

confusionPlot = sns.stripplot(x = 'Probability', y = 'Nomine', hue = 'TrueLabel',                              jitter = 0.4, dodge = True, data = plotDF)


# # Applicability Domain

# ## Projections

# #### PCA

# In[57]:


pca = PCA(n_components=2)
pca.fit(X_train_std)
train_projected = pd.DataFrame(pca.transform(X_train_std))
test_projected = pd.DataFrame(pca.transform(X_test_std))


# In[58]:


print(X_train_std.shape)


# In[59]:


print(train_projected.shape)


# In[60]:


col_names =  ['PC1', 'PC2', 'Set']
trainPCAplot  = pd.DataFrame(columns = col_names)
testPCAplot  = pd.DataFrame(columns = col_names)

trainPCAplot['PC1'] = train_projected[0]
trainPCAplot['PC2'] = train_projected[1]
trainPCAplot['Set'] = 'train'

testPCAplot['PC1'] = test_projected[0]
testPCAplot['PC2'] = test_projected[1]
testPCAplot['Set'] = 'test'

result = pd.concat([trainPCAplot, testPCAplot])


# In[61]:


sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Set', data = result)


# #### t-SNE
# t-distributed Stochastic Neighbor Embedding

# In[62]:


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne.fit(X_train_std)


# In[63]:


train_projected = pd.DataFrame(tsne.fit_transform(X_train_std))
test_projected = pd.DataFrame(tsne.fit_transform(X_test_std))


# In[64]:


col_names =  ['PC1', 'PC2', 'Set']
trainPCAplot  = pd.DataFrame(columns = col_names)
testPCAplot  = pd.DataFrame(columns = col_names)

trainPCAplot['PC1'] = train_projected[0]
trainPCAplot['PC2'] = train_projected[1]
trainPCAplot['Set'] = 'train'

testPCAplot['PC1'] = test_projected[0]
testPCAplot['PC2'] = test_projected[1]
testPCAplot['Set'] = 'test'

result = pd.concat([trainPCAplot, testPCAplot])


# In[65]:


sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Set', data = result)


# #### MDS

# In[66]:


mds = MDS(n_components=2)
mds.fit(X_train_std)


# In[67]:


train_projected = pd.DataFrame(mds.fit_transform(X_train_std))
test_projected = pd.DataFrame(mds.fit_transform(X_test_std))


# In[68]:


col_names =  ['PC1', 'PC2', 'Set']
trainPCAplot  = pd.DataFrame(columns = col_names)
testPCAplot  = pd.DataFrame(columns = col_names)

trainPCAplot['PC1'] = train_projected[0]
trainPCAplot['PC2'] = train_projected[1]
trainPCAplot['Set'] = 'train'

testPCAplot['PC1'] = test_projected[0]
testPCAplot['PC2'] = test_projected[1]
testPCAplot['Set'] = 'test'

result = pd.concat([trainPCAplot, testPCAplot])


# In[69]:


sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Set', data = result)


# In[84]:


def fsi_numpy(item_id):
    tmp_arr = arr - arr[item_id]
    tmp_ser = np.sum( np.square( tmp_arr ), axis=1 )
    return tmp_ser


# In[95]:


Xtrain = pd.DataFrame(X_train_std)
arr = Xtrain.values
Xtrain['dist'] = fsi_numpy(400)
Xtrain = Xtrain.sort_values('dist').head(6)
Xtrain['dist']


# In[97]:


col_names =  ['dist']
tmp  = pd.DataFrame(columns = col_names)
col_names =  ['index', 'avgDist']
neighbors  = pd.DataFrame(columns = col_names)


# In[98]:


Xtrain = pd.DataFrame(X_train_std)
arr = Xtrain.values
for i, row in Xtrain.iterrows():
    tmp['dist'] = fsi_numpy(i)
    tmp = tmp.sort_values('dist').head(6)
    neighbors.loc[i, 'index'] = i
    neighbors.loc[i, 'avgDist'] = tmp['dist'].mean()


# In[ ]:




