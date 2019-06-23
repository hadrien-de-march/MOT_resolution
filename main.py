# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:09:00 2017

@author: hdemarch
"""

#Version 11/08 11h29

#import math

import Class_grid
import numpy as np
from multiprocessing import cpu_count
import Data_extractor as dt
import Cost_Functions as cst
import Distribution_Functions as dst

print("number of CPUs = ", cpu_count())

d = 50#You need to use MC if the dimension is higher than 32

size_grids_init = {'x': 10, 'y': 10}
boundX = 1.
boundY = boundX*1.01
zero = 1e-10
epsilon = 1e-1
nb_threads = 4*cpu_count()
tasks_per_thread = 200
size_memory_max = 1e7

use_pool = 1
smart_timing_pool = 0#pool activates when the problem becomes big
print_time_pool = False
plot_perf = 1#Builds the list of time computations
plot_save = 0#Save the figures and prevent the plotting
plot_data_save = 1#Saves the data that serves to plot the figures
tag = 'Mac'

martingale = 1


#methods = ['Newton-CG']#'L-BFGS-B'#'Newton-CG'#'trust-ncg'#BFGS
methods = ['hybrid']#'Newton-CG']#]#, , 'sinkhorn']#, 'L-BFGS-B']#['trust-ncg', 'sinkhorn']#]#, 'trust-ncg', 'BFGS']
#methods = ['bregman phi implied', 'sinkhorn', 'bregman']

tolerance = 1e-7
entropic = True
compute_entropy_error = 1
newNewton = 1#Set gtol_for_newton False is this is set True
gtol_for_newton = 1-newNewton#Set is False if newNewton = True
additional_step_CG = 0
precond_CG = 1
maxiter_CG = 20
max_sinkhorn_steps_hybrid = 60
impl_psi = 0
impl_phi_h = 1
include_phi = 1###Do not consider that the martingale constraint is satisfied when computing the hessian. 1 is better from observation
compute_phi_h = 1###Compute phi and h when computing the concave hull, and computes a precise duality bound
sparse = 0
scale = 1
lift_when_scaling = 1
grid_MC = 1#You need to use MC if the dimension is higher than 32
#implieds = [(True,False)]#(impl_phi_h, grid.impl_psi)
nmax_Newton_h = 20
tol_Newton_h = 1e-7
penalization = zero
penalization_type = "tempered measure/func"#"tempered measure"#"measure"##"uniform"#
penalization_power = 2.

times_compute_phi_psi = 1

##debug_mode code: #0: nothing #1: hessian #2: gradient #3: martingale prop
                   #4: marginals #5: disp #6: print phi,psi,h #7: preconditioning
                   #8: comparing multiply size grid #9: disp_CG #10: Test CG #11: print line search
                   #12: check gap convex hull #13: experiment norm A for CG
debug_mode = 0

fignum = 1 #2, 3 busy

pow_distance = 1

omega =8.
sigma = 0.25
dimensional_rescale = np.sqrt(d)
sigma_1 = 0.1*np.eye(d)
sigma_2 = 0.2*np.eye(d)
mu = np.ones(d)/dimensional_rescale
norm = 1.
p = 1.

proba_min = 1e-7
purify_proba = True

def cost(x,y):
    return cst.index_forward_cost(x,y)#straddle_cost(x,y)#distance_cost(x,y, p=p, norm = norm)#random_cost(x,y, omega = omega)#left_curtain_cost(x,y)#distance_cost(x,y, p=p, norm = norm)##distance_cost(x,y, p=p, norm = norm)#

def grid_MC_creator(axis, MC_iter = None):
    if axis == 'x':
        sigma = sigma_1
    elif axis == 'y':
        sigma = sigma_2
    else:
        print("Axis ", axis, " is unknown.")
        raise("Unknown axis.")
    return dst.lognormal_MC_grid(mean = mu, sigma = sigma, MC_iter = MC_iter[axis])
    

print(tag)
                                    
for method in methods:
#  for implied in implieds:
#    grid.impl_phi_h = implied[0] 
#    grid.impl_psi = implied[1]

    grid = Class_grid.Grid(dim = d, size_grids_init = size_grids_init, boundX = boundX, boundY = boundY,
               epsilon = epsilon, nb_threads = nb_threads, grid_MC_creator = grid_MC_creator,
               cost = cost , use_pool = use_pool, smart_timing_pool = smart_timing_pool,
               plot_save = plot_save, tolerance = tolerance, compute_entropy_error = compute_entropy_error,
               entropic = entropic, nmax_Newton_h = nmax_Newton_h, tol_Newton_h = tol_Newton_h,
               penalization = penalization, method = method,
               max_sinkhorn_steps_hybrid = max_sinkhorn_steps_hybrid,
               times_compute_phi_psi = times_compute_phi_psi, debug_mode = debug_mode,
               purify_proba = purify_proba, grid_MC = grid_MC, MC_iter = size_grids_init,
               d_mu = lambda x: dst.lognormal(x+1.,sigma = sigma_1),#dst.uniform(x),#####,#dst.gaussian(x, sigma = sigma_1),+
               d_nu = lambda x: dst.lognormal(x+1.,sigma = sigma_2),#None,#None,##None,# dst.gaussian(x, sigma = sigma_2),#
               d_nu_d_mu = None,#lambda x: dst.g_univ(x)+zero,
               zero=zero, compute_phi_h=compute_phi_h, fignum = fignum,
               proba_min = proba_min, tasks_per_thread = tasks_per_thread,
               print_time_pool = print_time_pool, tag = tag,
               impl_psi = impl_psi, plot_perf = plot_perf, impl_phi_h = impl_phi_h,
               pow_distance = pow_distance, plot_data_save = plot_data_save,
               gtol_for_newton = gtol_for_newton, newNewton = newNewton,
               precond_CG = precond_CG, additional_step_CG = additional_step_CG, maxiter_CG = maxiter_CG,
               sparse = sparse, size_memory_max = size_memory_max, include_phi = include_phi,
               scale = scale, lift_when_scaling = lift_when_scaling,
               penalization_type = penalization_type, penalization_power = penalization_power)

#    grid2 = Class_grid.Grid(dim = d, size_grids_init = size_grids_init, boundX = boundX, boundY = boundY,
#               epsilon = epsilon,
#               cost = None,
#               d_mu = lambda x: lognormal(x+1,sigma_1),
#               d_nu = lambda x: lognormal(x+1,sigma_2),
#               d_nu_d_mu = None,#lambda x: g(x),
#               zero=zero)
#
#    grid.mu = (grid.mu+grid2.mu)/2.
#    grid.nu = (grid.nu+grid2.nu)/2.

    #grid.mu = grid2.mu
    #grid.nu = grid2.nu

    #grid.regularize_measures()

    #grid.purify_grid()

    #grid.test_convex_order(tol = zero)
    grid.set_convex_order(tol_min = 1e-5, tol_h = 1e-10, nmax_Newton = 20)


    grid.martingale = martingale

    grid.init_phi()
    grid.init_psi()
    grid.init_h()
    grid.Optimization_entropic_decay(iterations = None, epsilon_start = 1e-0,
                                     epsilon_final = 1e-4,
                                     intermediate_iter = 10000,#max number of entropic algo iterations
                                     final_size = 10000,#In case scale is on
                                     final_granularity = None,#1e-3,#In case scale is on
                                     r_0 = 0.5, r_f = 0.5,
                                     entropy_tol = None,#1e-4,
                                     pen = 1e-2, tol = 1e-3,
                                     pen_0 = 1e-1, pen_f = 1e-2,#to make it evolve during epsilon scaling
                                     tol_0 = 1e-2, tol_f = 1e-3#to make it evolve during epsilon scaling
                                     )
#    grid.plot_entropic_proba()
#    grid.tag+='more'
#    for elem in grid.epsilon_vs_error:
#        print(elem)



#dt.plotter_entropy_error(grid.epsilon_vs_error, dim=d)




