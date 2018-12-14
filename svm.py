
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import pandas as pd
from pathlib import Path
import os
from sklearn.svm import SVC

from log_l import *
from var_param import *
from m_step import *
from preprocessing import *


# ## Preprocess text

# In[2]:


filepath = "Articles.csv"
docs, lab, vocab = preprocess(filepath)

#permute
M = len(docs)
np.random.seed(0)
indeces = np.arange(M)
np.random.shuffle(indeces)

documents = []
labels = []

for d in range(M):
    documents.append(docs[indeces[d]])
    labels.append(lab[indeces[d]])

#documents = [docs[indeces[d]] for d in range(M)]
#labels = [lab[indeces[d]] for d in range(M)]

y = labels
#documents = docs[:M]
#labels =  labels[:M]
V = vocab.index.shape[0]
V


# In[3]:


def get_gammas(k):
    alpha_file = 'alpha_k=' + str(k) + '.npy'
    beta_file = 'beta_k=' + str(k) + '.npy'
    alpha_val = np.load(alpha_file)
    beta_val = np.load(beta_file)

    gamma = []

    for d in range(M):
        phi_val, gamma_val = var_parameters(documents[d], alpha_val, beta_val)
        gamma.append(gamma_val)

    X = np.asarray(gamma)
    return X


# In[4]:


def train_svm(X, y, n_train, C = 1, gamma = 5*1e-5):
    clf = SVC(C, gamma=gamma)

    #split in the train and test
    #n_train = 2000
    n_test = M - n_train

    X_train = X[:n_train,:]
    y_train = y[:n_train]

    X_test = X[n_train:,:]
    y_test = y[n_train:]

    clf.fit(X_train, y_train) 

    predictclass = clf.predict(X_test)
    from sklearn import metrics
    yacc = metrics.accuracy_score(y_test,predictclass)  
    
    return yacc


# In[5]:


#if we simply convert all words into dummies and run svm on that:
X_all = np.zeros((M, V))
for d in range(M):
    for w in range(V):
        if w in documents[d]:
            X_all[d, w] = 1
            
n_train = 2000
all_feat = train_svm(X_all, y, n_train)
print("Using words as features: ", all_feat)


# In[6]:


#estimate SVM for each gamma
k_s = np.arange(2, 20)
res = np.zeros(18)
n_train = 2000
for k in k_s:
    X = get_gammas(k)
    res[k-2] = train_svm(X, y, n_train)


# In[7]:


plt.figure(figsize = (8, 5))
plt.plot(k_s, res, label = "LDA Features")
plt.plot(k_s, np.ones(18)*all_feat, label = "Word Features")
plt.ylim(0.8, 1)
plt.xlabel("Number Of Topics")
plt.ylabel("Accuracy on the Test Set")
plt.legend(loc = 'best')
plt.savefig("svm_ntopics.png")


# In[8]:


#estimate SVM for different train size
train_sizes = np.arange(100, 1000, 10)
res = []
res_all = []


X = get_gammas(10)
for n_train in train_sizes:
    res.append(train_svm(X, y, n_train))
    res_all.append(train_svm(X_all, y, n_train))


# In[15]:


plt.figure(figsize = (8, 5))
plt.plot(train_sizes/M, res, label = "LDA Features")
plt.plot(train_sizes/M, res_all, label = "Word Features")
plt.ylim(0.4, 1)
plt.xlabel("Proportion of Data Used for Training")
plt.ylabel("Accuracy on the Test Set")
plt.legend(loc = 'best')
plt.savefig('svm_trainsize.png')

