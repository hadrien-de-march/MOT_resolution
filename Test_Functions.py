# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:09:00 2014

@author: hdemarch
"""

import numpy as np
import matplotlib.pyplot as plt

import Generic_Functions as gf
import Functions as fn




def plot_test_convex_hull(grid, float_x = False, value_x_0 = None,
                          psi = None, zero = 1e-10, debug_mode = False):
    cost = grid.cost
    d = grid.dim
    bound = grid.bound
    if len(grid.grid)==0:
        raise("Empty grid!")
    else:
        d = len(grid.grid[0])
    if value_x_0 != None:
        x_0=value_x_0
    elif float_x:
        x_0 = (np.random.rand(d)-0.5)*2*bound
    else:
        x_0 = grid.grid[int(np.random.rand()*len(grid.grid))]
    if psi == None:
        dpsi = np.random.rand(len(grid.grid))-0.5
    else:
        dpsi = psi
    func = grid.func_grid(lambda y: cost(x_0,y))
    func -= dpsi
    DATA = fn.loc_convex_hull(x_0,func, grid.grid, zero = zero, debug_mode = debug_mode)
    value = DATA['value']
    contact = DATA['contact']
    indices = DATA['argcontact']
    if d==1:
        plt.plot([x_0],[value],'ro')
        values_contact = list(map(lambda x: func[int(x)],indices))
        plt.plot(contact,values_contact,'bs')
        plt.plot(contact,values_contact)
        plt.plot(grid.grid,func)
        if grid.plot_save:
            plt.savefig('test_convex_hull'+str(plt.gcf().number)+'.png')
        else:
            plt.show()
    if d==2:
        print("code the printing for d=2!!!!! PS: you stink")          
    return {'x_0':x_0,'psi':dpsi,'convex_hull':DATA}



#                    Ords_local = list(Ords)
#                    total_contact_local = list(total_contact)
#                    minimum = 0
#                    store = False
#                    i_store = -1
#                    j_store = -1
#                    while(minimum < 5.*grid.bound/grid.steps):
#                        if store:
#                            Ords[i_store]['abs'].append(x_0)
#                            Ords[i_store]['ord'].append(total_contact_local[j_store])
#                            Ords[i_store]['last']=total_contact_local[j_store]
#                            Ords_local.pop(i_store)
#                            total_contact_local.pop(j_store)
#                            store = False
#                        minimum = 1e10
#                        for i in range(len(Ords_local)):
#                            for j in range(len(total_contact_local)):
#                                if np.linalg.norm(Ords_local[i]['last']
#                                -total_contact_local[j])<minimum:
#                                    minimum = (np.linalg.norm(Ords_local[i]['last']
#                                    -total_contact_local[j]))
#                                    i_store = i
#                                    j_store = j
#                                    store = True
#                    indices_to_pop = []
#                    for Ord in Ords_local:
#                        plt.plot(Ord['abs'],Ord['ord'])
#                        indices_to_pop.append(Ord['index'])
#                    Ords = list([ item for i,item in enumerate(Ords) if i not in indices_to_pop ])
#                    for point in total_contact_local:
#                        Ords.append({'abs' : [x_0], 'ord' : [point],
#                                     'last' : point , 'index' : len(Ords)})

#        if plot_maps:
#            if d==1:
#                    Ords = []