# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:23:46 2018

@author: elakkis
"""
import numpy as np
from scipy.special import digamma
from log_l import log_likelihood_bound_gamma_phi


def var_parameters(document, alpha, beta):
    """
    Estimate variational parameters
    Pg 1005
    
    document - an array of N_d indexes
    indexes map words in the document to dictionary
    
    
    Arguments: Document D, beta
    Returns: phi, gamma
    """
    #initialize
    N = document.shape[0]
    k = alpha.shape[0]
    
    new_phi = np.ones((N, k))/k
    new_gamma = alpha + N/k
    #print("Initial gamma is ", new_gamma)
    phi = np.zeros((N, k))
    gamma = np.zeros((N, 1))
   
    eps = 1e-5
    
    old_obj = 1
    new_obj = log_likelihood_bound_gamma_phi(document, new_phi, new_gamma, alpha, beta)

    while np.abs((new_obj - old_obj)/old_obj) > eps:            
        old_obj = new_obj    
        gamma = new_gamma
        
        col_indexes = document
        beta_test = beta[:,col_indexes]
        phi_test = beta_test.T*np.exp(digamma(gamma))
        normal = np.sum(phi_test, axis = 1).reshape((N, 1))
        normal = np.repeat(normal, k, axis = 1)
        
        new_phi = phi_test/normal
        
        new_gamma = alpha + np.sum(new_phi, axis = 0)

        new_obj = log_likelihood_bound_gamma_phi(document, new_phi, new_gamma, alpha, beta)
    #print("E-step in ", n_iter, " iterations")
    return new_phi, new_gamma


def e_step(documents, alpha, beta):
    #perform E-step on all documents
    
    #n_docs and n_topics
    M = len(documents)
    k = alpha.shape[0]
    
        
    #get number of words in each document
    N_d = np.zeros(M)
    for d in range(M):
        N_d[d] = len(documents[d])
    N = int(np.sum(N_d)) 
    
    
    new_gamma = np.zeros((M, k))
    new_phi = np.zeros((N, k))
    


    #for each document estimate gamma and phi
    for d in range(M):
        p, g = var_parameters(documents[d], alpha, beta)
        new_gamma[d,:] = g

        low_ind = int(np.sum(N_d[:d]))
        if d == M - 1:
            up_ind = int(np.sum(N_d))
        else:
            up_ind = int(np.sum(N_d[:d + 1]))
        new_phi[low_ind:up_ind,:] = p
        
    return new_phi, new_gamma