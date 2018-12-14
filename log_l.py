# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:22:11 2018

@author: elakkis
"""

import numpy as np
from scipy.special import digamma
from scipy.special import gammaln



def log_likelihood_bound_gamma_phi(document, phi, gamma, alpha, beta):
    """
    Computes lower bound on log-likelihood given parameter values
    Includes only terms with gamma and phi for better performance
    Equation (15), page 1020
    """
    N = document.shape[0]
    k = alpha.shape[0]
    
    result = 0
    
    #n of words and number of topics
    sum_gamma = np.sum(gamma)
    
    dig_gamma = digamma(gamma)
    dig_sum_gamma = digamma(sum_gamma)
    
    
    #line 1
    result += np.sum((alpha - 1)*(dig_gamma - dig_sum_gamma))

    #line 2
    #checked - vectorization is ok
    second_term = phi.dot(dig_gamma - dig_sum_gamma)
    result += np.sum(second_term)
    
    #line 3
    #checked - vectorization is ok
    third_term = np.sum(phi*np.log(beta[:, document]).T)
    result = result + third_term


    
    #line 4
    result += -gammaln(sum_gamma) + np.sum(gammaln(gamma))

    result += -np.sum((gamma - 1)*(dig_gamma - dig_sum_gamma))
    #line 5
    #checked - vectorization ok
    #in case phi is negative - ignore thoe elements
    temp_phi = phi.copy()
    temp_phi[temp_phi <= 0] = 1
    result += - np.sum(temp_phi*np.log(temp_phi))
    
    
    return result




def log_likelihood_bound_alpha(document, phi, gamma, alpha, beta):
    """
    Computes lower bound on log-likelihood given parameter values
    Includes only the terms with alpha
    Equation (15), page 1020
    """
    result = 0
    
    sum_alpha = np.sum(alpha)
    M = len(document)
    
    for d in range(M):
        sum_gamma = np.sum(gamma[d,:])
        gamma_d = gamma[d,:]
        #line 1
        #this is the only part that contains alphas only
        result += gammaln(sum_alpha) - np.sum(gammaln(alpha))
        result += np.sum((alpha - 1)*(digamma(gamma_d) - digamma(sum_gamma)))

    
    return result


def log_likelihood_bound(document, phi, gamma, alpha, beta):
    """
    Computes lower bound on log-likelihood given parameter values
    Equation (15), page 1020
    """
    result = 0
    N = len(document)
    k = alpha.shape[0]
    
    sum_alpha = np.sum(alpha)
    sum_gamma = np.sum(gamma)
    
    dig_gamma = digamma(gamma)
    dig_sum_gamma = digamma(sum_gamma)
    
    #line 1
    result += gammaln(sum_alpha) - np.sum(gammaln(alpha))
    result += np.sum((alpha - 1)*(dig_gamma - dig_sum_gamma))

    #line 2
    #checked - vectorization is ok
    second_term = phi.dot(dig_gamma - dig_sum_gamma)
    result += np.sum(second_term)
    #print(result)
    
    #line 3
    #checked - vectorization is ok
    third_term = np.sum(phi*np.log(beta[:, document]).T)
    result = result + third_term
    

    
    #line 4
    result += -gammaln(sum_gamma) + np.sum(gammaln(gamma))

    result += -np.sum((gamma - 1)*(dig_gamma - dig_sum_gamma))
    #line 5
    #checked - vectorization ok
    
    
    #in case phi is negative - ignore thoe elements
    temp_phi = phi.copy()
    temp_phi[temp_phi <= 0] = 1
    result += - np.sum(temp_phi*np.log(temp_phi))
    

    
    return result


def total_log_l(documents, phi, gamma, alpha, beta):
    
    result = 0
    
    #get lengths of documents
    M = len(documents)
    N_d = np.zeros(M)
    for d in range(M):
        N_d[d] = len(documents[d])
        
    
    for d in range(M):
        #get phi_d
        low_ind = int(np.sum(N_d[:d]))
        if d == M - 1:
            up_ind = int(np.sum(N_d))
        else:
            up_ind = int(np.sum(N_d[:d + 1]))
            
        phi_d = phi[low_ind:up_ind,:]
        gamma_d = gamma[d,:]
        
        result += log_likelihood_bound(documents[d], phi_d, gamma_d, alpha, beta)

    return result

    