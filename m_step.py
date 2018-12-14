# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:27:17 2018

@author: elakkis
"""
import numpy as np
from scipy.special import digamma
from scipy.special import polygamma
from log_l import *

def m_step(documents, phi, gamma, V):
    #compute betas first
    #this is for one document only!
    #if more - need to fix
    #beta = document.dot(phi).T
    
    #number of words and topics
    M = len(documents)
    k = gamma.shape[1]
    
    N_d = np.zeros(M)
    for d in range(M):
        N_d[d] = len(documents[d])
    
    #beta = np.ones((k, V))*1e-12
    #for i in range(k):
    #    for j in range(V):
    #        for n in range(N):
    #            beta[i,j] = beta[i,j] + phi[n,i]*document[n,j]
    
    #beta = phi.T.dot(document)
    beta = np.ones((k, V))*1e-9
    for d in range(M):
        
        #get phi_d
        low_ind = int(np.sum(N_d[:d]))
        if d == M - 1:
            up_ind = int(np.sum(N_d))
        else:
            up_ind = int(np.sum(N_d[:d + 1]))
            
        phi_d = phi[low_ind:up_ind,:]
        
        
        for n in range(int(N_d[d])):
            for i in range(k):
                beta[i,documents[d][n]] += phi_d[n,i]
    #print("Vectorization test: ", beta_test - beta)
    
    #now need to normalize betas
    #in each row they should sum up to 1
    for i in range(k):
        norm_const = np.sum(beta[i,:])
        beta[i,:] = beta[i,:]/norm_const

    
    alphas = np.zeros(k)
    alphas_new = np.arange(1, k + 1)/100000
    
    eps = 1e-8
    
    n_iter = 0
    
    old_obj = 0
    new_obj = log_likelihood_bound_alpha(documents, phi, gamma, alphas_new, beta)
    while np.abs((new_obj - old_obj)/new_obj) > eps:
        n_iter = n_iter + 1
        
        old_obj = new_obj
        alphas = alphas_new
        
        #compute that diagonal matrix
        h = M*polygamma(1, alphas)
        
        #element of the other matrix
        sum_alphas = np.sum(alphas)
        z = -polygamma(1, sum_alphas)

        #now gradient
        #here again, need to adjust for several documents
        grad = M*(digamma(sum_alphas) - digamma(alphas))
        #here we would sum over documents
        #grad = grad + digamma(gamma) - digamma(np.sum(gamma))
        #this is long ugly expression - to vectorize stuff...
        #for d in range(M):
        #    new_gamma = gamma[d,:]
        grad += np.sum(digamma(gamma) - digamma(np.sum(gamma, axis = 1).reshape((M,1)).repeat(k, axis = 1)), axis = 0)

        #now compute c - page 1019
        c = np.sum(grad/h)
        c = c/(1/z + np.sum(1/h))
        
        update = (grad - c)/h

        """
        print("Z is ", z)
        print("Update c ", c)
        print("Grad is ", grad)
        print("H is ", h)
        """
        #print("Old Alphas ", alphas)
        alphas_new = alphas + update
        #print("New Alphas ", alphas_new)
        new_obj = log_likelihood_bound_alpha(documents, phi, gamma, alphas_new, beta)
        #print(new_obj)
    print("M-step in ", n_iter, " iterations")
    return alphas, beta