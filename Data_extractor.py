#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:23:43 2018

@author: hadriendemarch
"""

import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import random as rd
from os import listdir
from os.path import isfile, join

def curves_extractor(search = 'sinkhorn'):
    prefix = 'new_images/'
    names = [f for f in listdir(prefix) if isfile(join(prefix, f))]
    print(names)
    for name in names:
        print(name)
        if name != '.DS_Store' and search in name:
            to_save = np.load(prefix+name, encoding = 'latin1')
            data_perf_grad = list(to_save)
            fignum = re.findall('(\d+)', name)[0]
            fignum = int(fignum)
            print(fignum)
            plt.figure(1000+fignum)
            for plot in data_perf_grad:
                if fignum == plot['fignum']:
                    coord_x = plot['times']#[i+1 for i in range(len(plot['grad_norms']))]
                    plt.semilogy(coord_x, plot['grad_norms'], label = plot['name'])
                    plt.legend()
                    plt.savefig(prefix+'get_perf/'+name+'.png')
        plt.close()
        
def dim2_extractor(points = 20, search = 'hybrid'):
    prefix = 'new_images/comp_dim2_stockage/'
    names = [f for f in listdir(prefix) if isfile(join(prefix, f))]
    subnames = rd.sample(names, points)
    fignum = 42
    for name in subnames:
        print(name)
        if name != '.DS_Store' and search in name:
            to_save = np.load(prefix+name, encoding = 'latin1')
            plt.figure(fignum)
            plt.imshow(to_save+1e-10, extent=[-1., 1., -1., 1.],
                origin = 'lower', cmap=cm.hot, norm=LogNorm())
            plt.colorbar()
            plt.savefig('new_images/get_dim2/'+name+'.png')
            plt.close()
            fignum +=1
            
def plotter_entropy_error(epsilon_vs_error, dim=1):
    plt.figure(43)
    coord_eps = []
    coord_err_prec = []
    coord_error = []
    d_div2 = []
    for elem in epsilon_vs_error:
        coord_eps.append(elem['epsilon'])
        coord_err_prec.append(elem['error_prec']/elem['epsilon'])
        coord_error.append(elem['error']/elem['epsilon'])
        d_div2.append(dim*0.5)
    plt.semilogx(coord_eps, coord_err_prec, label = "concave hull error/epsilon")
    plt.semilogx(coord_eps, coord_error, label = "supremum error/epsilon")
    plt.semilogx(coord_eps, d_div2, label = "d/2")
    plt.legend()
    plt.savefig('new_images/convergence_entropy_error.png')
    plt.close()
                
