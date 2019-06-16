#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 00:53:15 2019

@author: hadriendemarch
"""

import numpy as np
import Generic_Functions as gf

#TODO: Costs from the paper

def distance_cost(x,y,p = 1., norm = 2.):
    shapeX = list(x.shape)
    shapeY = list(y.shape)
    dim = shapeX.pop()
    dimY = shapeY.pop()
    gf.check(dimY==dim, (dimY, dim))
    X = np.reshape(x, tuple(shapeX+[1]*len(shapeY)+[dim]))
    Y = np.reshape(y, tuple([1]*len(shapeX)+shapeY+[dim]))
    final_axis = len(shapeX)+len(shapeY)
    return np.linalg.norm(X-Y, ord = norm, axis = final_axis)**p

def zero_cost(x,y):
    shapeX = list(x.shape)
    shapeY = list(y.shape)
    dim = shapeX.pop()
    dimY = shapeY.pop()
    gf.check(dimY==dim, (dimY, dim))
    zero_grid = np.zeros(tuple(shapeX+shapeY))
    return zero_grid

def straddle_cost(x,y):
    return distance_cost(x, y, p=1, norm =1)

def index_forward_cost(x,y):
    shapeX = list(x.shape)
    shapeY = list(y.shape)
    dim = shapeX.pop()
    dimY = shapeY.pop()
    gf.check(dimY==dim, (dimY, dim))
    X = np.reshape(x, tuple(shapeX+[1]*len(shapeY)+[dim]))
    Y = np.reshape(y, tuple([1]*len(shapeX)+shapeY+[dim]))
    final_axis = len(shapeX)+len(shapeY)
    return np.maximum(np.sum(Y-X, axis = final_axis), 0)

def basket_option_cost(x,y, strike = None):
    shapeX = list(x.shape)
    shapeY = list(y.shape)
    dim = shapeX.pop()
    dimY = shapeY.pop()
    gf.check(dimY==dim, (dimY, dim))
    X = np.reshape(x, tuple(shapeX+[1]*len(shapeY)+[dim]))
    Y = np.reshape(y, tuple([1]*len(shapeX)+shapeY+[dim]))
    final_axis = len(shapeX)+len(shapeY)
    if strike is None:
        strike = dim
    return np.maximum(np.sum(Y/X, axis = final_axis)-strike, 0)

def random_cost(x,y,omega = 1.):
    shapeX = list(x.shape)
    shapeY = list(y.shape)
    dim = shapeX.pop()
    dimY = shapeY.pop()
    gf.check(dimY==dim, (dimY, dim))
    X = np.reshape(x, tuple(shapeX+[1]*len(shapeY)+[dim]))
    Y = np.reshape(y, tuple([1]*len(shapeX)+shapeY+[dim]))
    final_axis = len(shapeX)+len(shapeY)
    return 1./omega*np.sin(omega*np.sum(X*Y, axis = final_axis))
 
def left_curtain_cost(x,y):
    shapeX = list(x.shape)
    shapeY = list(y.shape)
    dim = shapeX.pop()
    dimY = shapeY.pop()
    gf.check(dimY==dim, (dimY, dim))
    X = np.reshape(x, tuple(shapeX+[1]*len(shapeY)+[dim]))
    Y = np.reshape(y, tuple([1]*len(shapeX)+shapeY+[dim]))
    final_axis = len(shapeX)+len(shapeY)
    Ysqu = Y*Y
    return np.sum(X, axis = final_axis)*np.sum(Ysqu, axis = final_axis)+np.sum(X*Ysqu, axis = final_axis)

def left_curtain_cost_3(x,y):
    shapeX = list(x.shape)
    shapeY = list(y.shape)
    dim = shapeX.pop()
    dimY = shapeY.pop()
    gf.check(dimY==dim, (dimY, dim))
    X = np.reshape(x, tuple(shapeX+[1]*len(shapeY)+[dim]))
    Y = np.reshape(y, tuple([1]*len(shapeX)+shapeY+[dim]))
    final_axis = len(shapeX)+len(shapeY)
    Ysqu = Y*Y*Y*Y/4.-Y*Y/8.
    return np.sum(X, axis = final_axis)*np.sum(Ysqu, axis = final_axis)+np.sum(X*Ysqu, axis = final_axis)

def Brenier_cost(x,y):
    shapeX = list(x.shape)
    shapeY = list(y.shape)
    dim = shapeX.pop()
    dimY = shapeY.pop()
    gf.check(dimY==dim, (dimY, dim))
    X = np.reshape(x, tuple(shapeX+[1]*len(shapeY)+[dim]))
    Y = np.reshape(y, tuple([1]*len(shapeX)+shapeY+[dim]))
    final_axis = len(shapeX)+len(shapeY)
    return np.sum(X*Y, axis = final_axis)
