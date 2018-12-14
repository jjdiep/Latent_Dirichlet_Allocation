
# coding: utf-8

# In[49]:


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

# In[50]:


filepath = "Articles.csv"
docs, lab, vocab = preprocess(filepath)

#permute
M = len(docs)
np.random.seed(0)
indeces = np.arange(M)
documents = [docs[d] for d in indeces]
labels = [lab[d] for d in indeces]


#documents = docs[:M]
#labels =  labels[:M]
V = vocab.index.shape[0]
V


# In[5]:


data = pd.read_csv("Articles.csv", encoding = "ISO-8859-1")


# ## Run LDA

# In[12]:


len(docs)


# In[13]:


#reduce number of documents
#M = 1500
N_d = np.zeros(M)
for d in range(M):
    N_d[d] = len(docs[d])
N = int(np.sum(N_d))
N


# In[14]:


#initialize everything
documents = docs[:M]
M = len(documents)


# In[15]:


def estimate(documents, k):
    
    #get initial values
    new_alpha = np.arange(1, k + 1)/k
    new_beta = np.random.normal(size = (k,V))
    new_beta = np.abs(new_beta)
    for i in range(k):
        new_beta[i,:] = new_beta[i,:]/np.sum(new_beta[i,:])

    
    print("Entering the first one ")
    new_phi, new_gamma = e_step(documents, new_alpha, new_beta)

    old_obj = 0
    new_obj = total_log_l(documents, new_phi, new_gamma, new_alpha, new_beta)

    eps = 1e-6

    n_iter = 0
    while np.abs((new_obj - old_obj)/new_obj) > eps:
        print("Diff in objectives: ", np.abs(new_obj - old_obj))
        n_iter = n_iter + 1
        alpha = new_alpha
        beta = new_beta
        gamma = new_gamma
        phi = new_phi
        old_obj = new_obj


        #E-step
        new_phi, new_gamma = e_step(documents, alpha, beta)
        #M-step
        new_alpha, new_beta = m_step(documents, new_phi, new_gamma, V)

        new_obj = total_log_l(documents, new_phi, new_gamma, new_alpha, new_beta)


    #after the estimation save the results
    outfile_alpha = "alpha_k=" + str(k)
    outfile_beta = "beta_k=" + str(k)
    np.save(outfile_alpha, new_alpha)
    np.save(outfile_beta, new_beta)


# In[16]:


for k in [40]:#range(2, 21):
    print("Estimating for k = ", k)
    estimate(documents, k)


# In[128]:


k = 6
outfile_alpha = r"C:\Users\elakkis\Box Sync\Courses\EECS545\Project\alpha_k=" + str(k) + ".npy"
outfile_beta = r"C:\Users\elakkis\Box Sync\Courses\EECS545\Project\beta_k=" + str(k) + ".npy"
new_alpha = np.load(outfile_alpha)
new_beta = np.load(outfile_beta)


vdf = vocab.reset_index()
for i in range(k):
    x = new_beta[i,:].argsort()[-12:][::-1]

    print("TOPIC ", i + 1)
    for i in x:
        print(vdf[vdf["index"] == i]["word"].values[0])

