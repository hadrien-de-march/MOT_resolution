#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 00:56:34 2019

@author: hadriendemarch
"""

import numpy as np

#TODO: right Multi-dimensional log-normal
#TODO: Monte-Carlo Generator (not here)

def uniform(x):
    return 1.

def gaussian(x, sigma = 1.):
    return np.exp(-np.linalg.norm(x)**2/(2*sigma**2))

def lognormal(x, mu = 0, sigma = None, zero = 1e-10):
    if np.all(x >= zero):
        if sigma is None:
             sigma = np.eye(len(x))
        logx = np.log(np.maximum(x, zero))
        sigma_inv_square = np.linalg.inv(np.dot(sigma, sigma))
        argument = np.dot(logx-mu, np.dot(sigma_inv_square, logx-mu))
        Gaussian = np.exp(-argument/2)
        return Gaussian/np.prod(x)
    else:
        return 0.

def lognormal_MC_grid(mean = np.array([1.]), sigma = None, MC_iter = 40000):
    dim = len(mean)
    if sigma is None:
        sigma = np.eye(dim)
    mu = np.log(mean)-np.diag(sigma)**2/2
    Gauss = np.random.normal(size = (MC_iter, dim))
    Gauss_vol = np.dot(Gauss, sigma)+np.reshape(mu, (1, dim))
    lognorm = np.exp(Gauss_vol)
    return lognorm

#print(lognormal_MC_grid())

def g(y):
    p = 1.5
    q = 1.
    sigma = 0.2
    lamb = 1.5
    u = 0*y+1
    u/=np.linalg.norm(u)
    return max(np.linalg.norm(y,p)**q,np.exp(lamb*np.dot(y,u))-1)+2.*np.exp(-np.linalg.norm(y-0.2*u)**2./(2*sigma**2))
    
def g_univ(y, p = 1.5, q = 1.):
    return np.linalg.norm(y,p)**q