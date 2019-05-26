# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:09:00 2014

@author: hdemarch
"""
import numpy as np
import Generic_Functions as gf
import Functions as fn
import scipy.optimize
import Test_Functions as test
import Cost_Functions as cst
import matplotlib.pyplot as plt
import timeit
from scipy.sparse import csr_matrix as csr
from copy import deepcopy






class Grid:
    dim = 0
    stepsX = 0
    boundX = 0.
    stepsY = 0
    boundY = 0.
    zero = 0.
    epsilon = 0.
    nb_threads = 0
    time_init = 0.
    times_comp = []
    nmax_Newton_h = 0
    tol_Newton_h = 0.
    lenX = 0
    lenY = 0
    size_x = 0
    tasks_per_thread = 0
    fignum = 0
    tag = ''
    tolerance = 0.
    penalization = 0.
    penalization_type = ""
    penalization_power = 0.
    proba_min = 0.
    pow_distance = 0.
    size_memory_max = 0
    memory_sparse = 0
    nb_pool_fail = -1
    use_pool = False
    smart_timing_pool = False
    sparse = False
    scale = False
    values_opt =[]
    eval_nb = 0
    times_compute_phi_psi = 0
    plot_save = False
    compute_phi_h = False
    entropic = False
    compute_entropy_error = False
    martingale = True
    debug_mode = False
    purify_proba = False
    impl_psi = False
    impl_phi_h = False
    print_time_pool = False
    plot_perf = False
    plot_data_save = False
    gtol_for_newton = False
    newNewton = True
    grid_MC = False
    MC_iter = None
    precond_CG = False
    additional_step_CG = False
    sparse_is_on = False
    include_phi = True
    hess_pos_safe = False
    grid_just_dobled = False
    method = ''
    data_perf = []
    data_perf_grad = []
    data_perf_val = []
    current_pack = np.array([])
    current_pack_hess = np.array([])
    current_pack_cond = np.array([])
    current_gradient = np.array([])
    base_package = np.array([])
    current_hess_h_inv = np.array([])
    sparse_gridXY = csr(np.array([]))
    sparse_gridYX = csr(np.array([]))
    result_opt = np.array([])
    result_found = False
    data_test_convexity = []
    data_points = []
    line = []
    current_value = 0.
    current_gradient_norm = 0.
    values = []
    grad_norms = []
    epsilon_vs_error = []
    nb_calls_list = []
    nb_calls = 0
    muX = np.array([])
    nuY = np.array([])
    gridX =np.array([])
    gridY =np.array([])
    mu = np.array([])
    nu = np.array([])
    nu_original = None
    phi = np.array([])
    psi = np.array([])
    h = np.array([])
    h_sparse = np.array([])
    Proba = np.array([])
    result = np.array([])
    cost = lambda x : 0.
    d_mu = lambda x : 0.
    d_nu = lambda x : 0.
    d_nu_d_mu = lambda x : 0.
    grid_MC_creator = None
    
### GENERIC CREATOR FUNCTIONS

    def make_grid(self, axis, dim = None, basic_inclusion = [], line = []):
        if self.grid_MC:
            print("MC grid created for "+axis)
            return self.grid_MC_creator(axis, MC_iter = self.MC_iter)
        else:
            if dim is None:
                dim = self.dim
            if axis == 'x':
                steps = self.stepsX
                bound = self.boundX
            elif axis == 'y':
                steps = self.stepsY
                bound = self.boundY
            # List of ranges across all dimensions
            start = dim*[0]
            stop = dim*[steps+1]
            L = [np.arange(start[i] , stop[i]) for i in range(dim)]

            # Finally use meshgrid to form all combinations corresponding to all 
            # dimensions and stack them as M x ndims array
            grid_int = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(dim,-1).T
            origin = np.reshape(np.zeros(dim)+bound, (1, dim))
            grid = grid_int*2*bound/steps-origin
            return grid
        
        
    def copy(self):
        instance = deepcopy(self)
        return instance
    
    def init_marginals(self):
        if self.grid_MC:
            self.d_mu = lambda x: 1.
            self.d_nu = lambda x: 1.
        d_mu = self.d_mu
        d_nu = self.d_nu
        d_nu_d_mu = self.d_nu_d_mu
        if d_mu is not None:
            self.mu = self.proba_grid(d_mu, axis = 'X')
        if d_nu is not None:
            self.nu = self.proba_grid(d_nu, 'Y')
        elif d_nu_d_mu is not None:
            if self.lenX != self.lenY:
                raise("The density structure is not adapted to mismatching grids.")
            self.nu = self.proba_grid(lambda y : d_mu(y)*d_nu_d_mu(y), 'Y')
                
    def init_phi(self):
        self.phi = np.zeros(self.lenX)
        
    def init_psi(self):
        self.psi = np.zeros(self.lenY)
    
    def init_h(self):
        self.h = np.reshape(np.zeros(self.lenX*self.dim), (self.lenX, self.dim))
                
    def purify_grid(self):
        deleteX = []
        deleteY = []
        for i in range(self.lenX):
            if self.mu[i]<self.proba_min/self.lenX:
                deleteX.append(i)
        for j in range(self.lenY):
            if self.nu[j]<self.proba_min/self.lenY:
                deleteY.append(j)
        self.gridX = np.delete(self.gridX, deleteX, axis = 0)
        self.lenX = len(self.gridX)
        self.gridY = np.delete(self.gridY, deleteY, axis = 0)
        self.lenY = len(self.gridY)
        self.init_psi()
        if  self.compute_phi_h or self.entropic:
            self.init_phi()
            self.init_h()
        self.mu = np.delete(self.mu, deleteX)
        self.mu /= np.sum(self.mu)
        self.nu = np.delete(self.nu, deleteY)
        self.nu /= np.sum(self.nu)
  
    
    def proba_grid(self, density, axis = 'X'):
        if axis == 'X':
            grid = self.gridX
        else:
            grid = self.gridY
        result = np.zeros(len(grid))
        for i in range(len(grid)):
            result[i] = density(grid[i])
        total_mass = np.sum(x for x in result)
        result /= total_mass
        return result
    
    def func_grid(self, f, axis = 'X'):
        if axis == 'X':
            grid = self.gridX
        else:
            grid = self.gridY
        example = f(grid[0])
        if not np.isscalar(example):
            shape = example.shape
            size = example.size
            shape_result = tuple([len(grid)]+list(shape))
        else:
            size = 1
            shape_result = (len(grid))
        result = np.zeros(len(grid)*size)
        result = np.reshape(result, shape_result)
        for i in range(len(grid)):
            result[i] = f(grid[i])
        return result
    
    def mean(self, func, proba = None, axis = 'X'):
        if proba is None:
            if axis == 'X':
                proba = self.mu
            elif axis == 'Y':
                proba = self.nu
            else:
                raise("Wrong axis.")
        if len(proba) != len(func):
            raise("mu is not initialized or func has wrong size.")
        return np.tensordot(proba,func,axes=(0,0))

    
        
### Initialization
    
    def __init__(self, dim = 0, stepsX = 0, boundX = 0, stepsY = 0, boundY = 0, epsilon = 0,
                 cost = None, d_mu = None, d_nu = None, nmax_Newton_h = 200, tol_Newton_h = 1e-10,
                 d_nu_d_mu = None, zero = 1e-10, grid_MC_creator = None,
                 compute_entropy_error = False,
                 nb_threads = 2,  compute_phi_h = False, use_pool = False, smart_timing_pool = False,
                 plot_save = False, entropic = False, method = 'BFGS',
                 tolerance = 1e-7, penalization = 1e-4, times_compute_phi_psi = 1, MC_iter = None,
                 debug_mode = False, purify_proba = False, proba_min = 1e-5, grid_MC = False,
                 tasks_per_thread = 4, fignum = 1,
                 print_time_pool = False, tag = '',
                 impl_psi = False, plot_perf = False, impl_phi_h = False,
                 pow_distance = 2., plot_data_save = False, gtol_for_newton = False,
                 newNewton = True, precond_CG = False, additional_step_CG = False,
                 sparse = False, size_memory_max = 1e8, include_phi = True,
                 scale = False, penalization_type = "uniform", penalization_power = 2.):
        self.dim = dim
        self.stepsX = stepsX
        self.boundX = boundX
        self.stepsY = stepsY
        self.boundY = boundY
        self.method = method
        self.nmax_Newton_h = nmax_Newton_h
        self.tol_Newton_h = tol_Newton_h
        self.impl_psi = impl_psi
        self.proba_min = proba_min
        self.times_compute_phi_psi = times_compute_phi_psi
        self.tolerance = tolerance
        self.purify_proba = purify_proba
        self.penalization = penalization
        self.penalization_type = penalization_type
        self.penalization_power = penalization_power
        self.entropic = entropic
        self.compute_entropy_error = compute_entropy_error
        self.include_phi = include_phi
        self.sparse = sparse
        self.scale = scale
        self.impl_phi_h = impl_phi_h
        if self.impl_phi_h and self.impl_psi:
            raise("phi, psi, and h cannot be implied at the same time.")
        self.nb_threads = nb_threads
        self.zero = zero
        self.epsilon = epsilon
        self.smart_timing_pool = smart_timing_pool
        self.use_pool = use_pool
        self.pow_distance = pow_distance
        self.print_time_pool = print_time_pool
        self.tasks_per_thread = tasks_per_thread
        #self.fuck_memory = fuck_memory
        self.size_memory_max = size_memory_max
        self.debug_mode = debug_mode
        self.plot_save = plot_save
        self.plot_perf = plot_perf
        self.plot_data_save = plot_data_save
        self.tag = tag
        self.fignum = fignum
        self.compute_phi_h = compute_phi_h
        self.gtol_for_newton = gtol_for_newton
        self.newNewton = newNewton
        self.precond_CG = precond_CG
        self.additional_step_CG = additional_step_CG
        self.d_mu = d_mu
        self.d_nu = d_nu
        self.d_nu_d_mu = d_nu_d_mu
        self.grid_MC = grid_MC
        self.MC_iter = MC_iter
        self.grid_MC_creator = grid_MC_creator
        #Building the grids
        gridX = np.array(self.make_grid('x'))
        gridY = np.array(self.make_grid('y'))
        self.gridX = gridX
        self.gridY = gridY
        #Grids built
        self.lenX = len(self.gridX)
        self.lenY = len(self.gridY)
        if self.impl_psi:
            self.size_x = self.lenX*(self.dim+1)
        elif self.impl_phi_h:
            self.size_x = self.lenY
        else:
            self.size_x = self.lenX*(self.dim+1)+self.lenY
        if cost!=None:
            self.cost = cost
        self.init_marginals()
        if self.purify_proba:
            self.purify_grid()
        if  compute_phi_h or self.entropic:
            self.init_phi()
            self.init_h()
        self.init_psi()
        
        
    def penalization_func(self, pack, diff = 0):
        if self.penalization == 0.:
            return 0.
        else:
            if "uniform" in self.penalization_type:
                multi = 1./self.size_x
            elif "measure" in self.penalization_type:
                meas_h = self.h*0.+1.
                meas_h = meas_h*np.reshape(self.mu, (self.lenX, 1))
                multi = self.package(self.mu, self.nu, meas_h)
            elif "temper" in self.penalization_type:
                multi_1 = 1./self.size_x
                multi_2 = multi
                coeff = 0.5
                multi = coeff*multi_1+(1.-coeff)*multi_2
            elif  "func" in self.penalization_type:
                divider = np.maximum(np.absolute(self.base_package), 0.01)
                divider = np.minimum(divider, 100.)
                multi /= divider
            else:
                print("Unknown penalization type "+ self.penalization_type)
                raise("Unknown penalization type.")
            multi /= np.sum(multi)
            multi*=self.penalization
            p = self.penalization_power
            if diff == 0:
                return np.sum(pack**p*multi/p)
            elif diff == 1:
                return pack**(p-1.)*multi
            elif diff == 2:
                return (p-1.)*pack**(p-2.)*multi
                    
                
            



    def Optimization_grad(self, use_gradient=True,
                          compare_grad_diff = False,
                          psi_rand = None, lift_psi = False,
                          plot_perf = True , nb_iter = 400):
        if self.martingale:
            mart = 'mart'
        else:
            mart = ''
        psi_0 = self.psi
        self.nb_pool_fail = -1
        self.eval_nb = 0
        self.values_opt = []
        method = self.method

        def Value_func_grad_for_optimization(psi):
            print("\n")
            print("calling value/gradient function")
            t_0 = timeit.default_timer()
            self.psi = psi
            DATA = self.Value_func_grad(use_gradient = use_gradient,
                                        compare_grad_diff = compare_grad_diff)
            if use_gradient:
                value = DATA[0]
                print("Value =", value)
                print("gradient norm =",np.linalg.norm(DATA[1],1))
            else:
                value = DATA
                print("Value =",value)
            if plot_perf:
                self.values_opt.append(value)
                print("evaluation number ",self.eval_nb)
                self.eval_nb +=1
            if lift_psi:
                if not self.compute_phi_h:
                    raise("you need to compute phi and h to lift psi.")
                self.psi_reduction(good_phi_h = True)
                Aff = self.psi - psi
                nu_mu = self.nu - self.mu
                corr = self.mean(Aff, proba = nu_mu)
                print("correction = ",corr)
                if use_gradient:
                    corr_2 = self.mean(Aff, proba = DATA[1])
                    print("correction gradient = ", corr_2)
                if use_gradient:
                    DATA = ( value+corr + corr, DATA[1])
                else:
                    DATA = value+corr
            print("total time for calculation = ", timeit.default_timer()-t_0)
            return DATA
        if method == 'nonsmooth_convex_optim':

            result = self.non_smooth_convex_minimize(Value_func_grad_for_optimization, psi_0,
                                                nb_iter = nb_iter, Omega = 2*self.bound**2)
            self.psi = result['x']
        else:
            result = scipy.optimize.minimize(Value_func_grad_for_optimization, psi_0, method=method,
                                     jac=use_gradient, tol=self.tolerance, options={'maxiter': nb_iter})
            self.psi = result.x
        self.result = result
        if lift_psi:
            self.psi_reduction()
        if plot_perf:
            self.data_perf.append((range(self.eval_nb), self.values_opt))
            if self.plot_data_save:
                to_save = np.asarray(self.data_perf)
                np.save('new_images/performance'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.npy', to_save)
            else:
                plt.figure(4)
                for plot in self.data_perf:
                    plt.plot(plot[0],plot[1])
                if self.plot_save:
                    plt.savefig('new_images/performance'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.png')
                    plt.close()
                else:
                    plt.show()
        return result
    
               
    def non_smooth_convex_minimize(self, f, x_0, nb_iter = 400, Omega = 2., L= np.sqrt(2.)):
        x = np.array(x_0)
        calc = f(x)
        minimum = calc[0]
        argmin = np.array(x)
        gradient = calc[1]
        for n in range(nb_iter):
            N = np.linalg.norm(gradient)
            print("norm 2 of gradient = ",N)
            gamma = np.sqrt(2.*Omega/(n+1))/N
            x -= gamma*gradient
            calc = f(x)
            if calc[0]<= minimum:
                minimum = calc[0]
                argmin = np.array(x)
            gradient = calc[1]
        return {'x' : argmin , 'minimum' : minimum, 'gradient' : np.linalg.norm(gradient,1)}                
            
    

    
    
    def psi_reduction(self, x_0 = None, good_phi_h = False):
        if x_0 is None:
            x_0 = self.muX
        if not good_phi_h:
            self.Value_func_grad()
        elem_scalar = x_0 - self.grid
        phi_plus_h = np.array(list([self.phi[i]+np.dot(elem_scalar[i],
                                    self.h[i]) for i in range(len(elem_scalar))]))
        argmin = np.argmin(phi_plus_h)
        h_0 = self.h[argmin] 
        phi_0 = self.phi[argmin]
        elem_scalar_round = self.grid - self.grid[argmin]
        tangent = np.array(list([phi_0+np.dot(elem_scalar_round[i],
                                              h_0) for i in range(len(elem_scalar_round))]))
        for i in range(len(tangent)):
            self.psi[i] += tangent[i]
        print("min = ", np.amin(self.psi))
        print("max = ", np.amax(self.psi))
        if np.amin(self.psi)<-self.zero:
            plt.show()
            raise("woot")
        
        
    def plot(self, func, axis = ''):
        if self.martingale:
            mart = 'mart'
        else:
            mart = ''
        if self.dim !=1:
            raise("dimension fail")
        if axis == 'X':
            grid = np.array(self.gridX)
        elif axis == 'Y':
            grid = np.array(self.gridY)
        else:
            grid = np.array(range(len(func)))
        func = np.array(func)
        if self.plot_data_save:
            to_save = np.asarray((grid, func))
            np.save('new_images/plot func'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.npy', to_save)
        else:
            plt.plot(list(grid.flatten()),list(func.flatten()))
            if self.plot_save:
                plt.savefig('new_images/plot func'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.png')
                plt.close()
            else:
                plt.show()
    
                
    def plot_test_convexity(self, length_test = 100,
                            pas = 0.1, trace_value=False, float_x = False, value_x_0 = None,
                            time_print = False, zero = 1e-10, Graph = True,
                            check_convexity = False, tol_non_convex = 1e-10):
        if self.martingale:
            mart = 'mart'
        else:
            mart = ''
        d = self.dim
        bound = self.bound
        grid = self.grid
        if self.nb_pool_fail>=1:
            self.use_pool = False
        psi = self.psi
        cost = self.cost
        dpsi = np.random.rand(len(grid))-0.5
        if not gf.approx_Equal(psi,-psi):
            Size = np.linalg.norm(psi)
            Size_random = np.linalg.norm(dpsi)
            dpsi = (Size/Size_random)*dpsi
        line = list(map(lambda x: (x-length_test/2.)*pas,np.arange(length_test+1)))
        if trace_value:
             if value_x_0 is not None:
                 x_0 = value_x_0
             elif float_x:
                 x_0 = (np.random.rand(d)-0.5)*2*bound
             else:
                 x_0 = grid[int(np.random.rand()*len(grid))]
             def convex_hull_trace(psi_local,i):
                 if time_print:
                     print(i)
                 func = self.func_grid(lambda y: cost(x_0,y))
                 func -= psi_local
                 DATA = fn.loc_convex_hull(x_0,func, self.grid, zero = zero)
                 return DATA['value']
             values = list(map(lambda x:convex_hull_trace(psi+x*dpsi,x),line))
             if check_convexity:
                 for i in range(len(values)-2):
                     if values[i+2]+values[i]-2*values[i+1]< -tol_non_convex:
                         print("convexité = ",values[i+2]+values[i]-2*values[i+1])
                         plt.figure(1)
                         psi_fail = psi+line[i]*dpsi
                         DATA_1 = test.plot_test_convex_hull(cost, self, value_x_0=x_0 , psi = psi_fail,
                                                        zero = zero, debug_mode = True, plot = True)
                         plt.figure(2)
                         psi_fail = psi+line[i+1]*dpsi
                         DATA_2 = test.plot_test_convex_hull(cost, self, value_x_0=x_0 , psi = psi_fail,
                                                        zero = zero, debug_mode = True)
                         plt.figure(3)
                         psi_fail = psi+line[i+2]*dpsi
                         DATA_3 = test.plot_test_convex_hull(cost, self, value_x_0=x_0 , psi = psi_fail,
                                                        zero = zero, debug_mode = True)
                         value_1=DATA_1['convex_hull']['value']
                         value_2=DATA_2['convex_hull']['value']
                         value_3=DATA_3['convex_hull']['value']
                         print("value 1",value_1)
                         print("value 2",value_2)
                         print("value 3",value_3)
                         print("delta +",value_3-value_2)
                         print("delta -",value_2-value_1)
                         raise("NON CONVEX!")
        else:
            values = list(map(lambda x: self.Value_func_grad( psi = psi+x*dpsi,
                         to_print = x)[0],line))
        if Graph:
            self.data_test_convexity.append((list(line),list(values)))
            if self.plot_data_save:
                to_save = np.asarray(self.data_test_convexity)
                np.save('new_images/Convexity test'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.npy', to_save)
            else:
                plt.figure(1)
                for plot in self.data_test_convexity:
                    plt.plot(plot[0],plot[1])
                if self.plot_save:
                    plt.savefig('new_images/Convexity test'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.png')
                    plt.close()
                else:
                    plt.show()
        return 0





    def doble_grid(self, axis = 'xy', sparse = False, lift = False):
        if sparse:
            print("Sparse is still to code.")#I have no clue what this is for lol
        axis_do = []
        if 'x' in axis:
            axis_do.append('x')
        if 'y' in axis:
            axis_do.append('y')
        dim = self.dim
        for axis in axis_do:
            if axis == 'x':
                grid = self.gridX
                bound = self.boundX
                steps = self.stepsX
                lenX = self.lenX
            elif axis == 'y':
                grid = self.gridY
                bound = self.boundY
                steps = self.stepsY
                lenY = self.lenY
                self.nu_original = None
            else:
                raise("You should not be here")
            if self.grid_MC:
                grid_2 = self.grid_MC_creator(axis, MC_iter = self.MC_iter)
                grid = np.concatenate((grid, grid_2))
                self.MC_iter[axis] *= 2
                if axis == 'x':
                    self.phi = np.concatenate((self.phi, self.phi*0+1./self.zero))
                    if self.martingale:
                            self.h= np.concatenate((self.h, self.h*0))
                elif axis == 'y':
                    self.psi = np.concatenate((self.psi, self.psi*0+1./self.zero))
                else:
                    raise("You should not be here")
            else:
              for i in range(dim):
                grid_1 = np.array(grid)
                grid_2 = np.array(grid)
                grid_1[:, i] = (grid[:, i]+bound)*(2*steps-1.)/(2*steps)-bound
                grid_2[:, i] = (grid[:, i]-bound)*(2*steps-1.)/(2*steps)+bound
                grid = np.concatenate((grid_1, grid_2))
                if axis == 'x':
                    self.phi = np.concatenate((self.phi, self.phi))
                    if self.martingale:
                            self.h= np.concatenate((self.h, self.h))
                elif axis == 'y':
                    self.psi = np.concatenate((self.psi, self.psi))
                else:
                    raise("You should not be here")
            if axis == 'x':
                self.gridX = grid
                self.lenX = len(self.gridX)
                self.stepsX *= 2
            elif axis == 'y':
                self.gridY = grid
                self.lenY = len(self.gridY)
                self.stepsY *= 2
            else:
                raise("You should not be here")
                
            
            
                

        if self.impl_psi:
            self.size_x = self.lenX*(self.dim+1)
        elif self.impl_phi_h:
            self.size_x = self.lenY
        else:
            self.size_x = self.lenX*(self.dim+1)+self.lenY
        self.init_marginals()
        if self.purify_proba:
            self.purify_grid()
        if self.martingale:
            self.set_convex_order(tol = 1e-5)
        if lift:
            if not self.impl_phi_h:
                self.psi_from_phi_h()
            if not self.impl_psi:
                if self.martingale:
                    self.h_from_phi_psi(calc_phi = True)
                else:
                    self.phi_from_psi_h()
        for axis in axis_do:
            if axis == 'x':
                print("Grid dobled. lenX from ", lenX, " to ", self.lenX)
            if axis == 'y':
                print("Grid dobled. lenY from ", lenY, " to ", self.lenY)
        self.grid_just_dobled = True
        return 0

        




    def base_arg(self, phi = True, psi = True, h = True,
                 mu = True, nu = True, gridX = True, gridY = True,
                 sparse_gridXY = True, sparse_gridYX = True):
        arg = {'lenX' : self.lenX, 'lenY' : self.lenY,
               'cost' : self.cost, 'martingale' : self.martingale,
               'epsilon' : self.epsilon, 'dim' : self.dim,
               'zero' : self.zero, 'tasks_per_thread' : self.tasks_per_thread,
               'nb_threads' : self.nb_threads, 'use_pool' : self.use_pool,
               'print_time_pool' : self.print_time_pool, 'sparse_is_on' : self.sparse_is_on,
               'method' : self.method, 'nmax_Newton_h' : self.nmax_Newton_h,
               'tol_Newton_h' : self.tol_Newton_h,
               'tolerance' : self.tolerance, 'infty' : 1./self.zero,
               'debug_mode' : self.debug_mode, 'newNewton' : self.newNewton,
               'pow_distance' : self.pow_distance, 'proba_min' : self.proba_min,
               'penalization' : self.penalization, 'include_phi' : self.include_phi,
               'compute_phi_h' : self.compute_phi_h}
        if phi:
            arg['phi'] = self.phi
        if psi:
            arg['psi'] = self.psi
        if h:
            arg['h'] = self.h
        if mu:
            arg['mu'] = self.mu
        if nu:
            arg['nu'] = self.nu
        if gridX:
            arg['gridX'] = self.gridX
        if gridY:
            arg['gridY'] = self.gridY
        if self.sparse_is_on:
            arg['sparse_gridXY'] = self.sparse_gridXY
            arg['sparse_gridYX'] = self.sparse_gridYX
            if h and self.martingale:
                arg['h_sparse'] = self.h.transpose()
            if gridX:
                arg['gridX_sparse'] = self.gridX.transpose()
            if gridY:
                arg['gridY_sparse'] = self.gridY.transpose()
            arg['sparsify'] = fn.sparsify
        return arg
            
       
       

    def Value_func_grad(self, psi = None, h = None, to_print = None,
                        compare_grad_diff = False, psi_rand = None,
                        calc_Gamma = False, param_hidden = 1e-10, plot_maps = False,
                        use_gradient = True):
        if psi is None:
            psi = self.psi
        else:
            self.psi = psi
            
        if h is None:
            h = self.h
            
        if self.martingale:
            mart = 'mart'
        else:
            mart = ''
        
        if to_print is not None:
            print(to_print)
        
        code_name = "value_grad_conv"
        axis = 'x'
        auxiliary = fn.auxiliary_Tan      
        base_arg = self.base_arg(phi = False, psi = False, h = False)
        
        base_arg['psi'] = psi
        base_arg['h'] = h
        base_arg['calc_Gamma'] = calc_Gamma
        base_arg['param_hidden'] = param_hidden
        base_arg['loc_convex_hull'] = fn.loc_convex_hull
        
        var = {'phi' : self.phi, 'h' : self.h, 'compute_phi_h' : self.compute_phi_h,
               'value' : np.dot(psi,self.nu)+self.penalization_func(psi), 'mu' : self.mu,
               'time convex hull' : 0., 'time shit' : 0., 'calc_Gamma' : calc_Gamma,
               'plot_maps' : plot_maps, 'gridX' : self.gridX, 'd' : self.dim,
               'to_plot' : [], 'degree_distrib' : np.array([]), 'use_gradient' : use_gradient,
               'zero' : self.zero, 'check' : gf.check,
               'grad' : np.array(self.nu)+self.penalization_func(psi, diff = 1),
               'print_dots' : gf.print_dots}
            
        def apply_elem(elem, var):
            global_index = elem['index']
            if var['compute_phi_h']:
                var['phi'][global_index] = elem['value']
                var['h'][global_index] = elem['gradient']
            var['value'] += var['mu'][global_index]*elem['value']
            arguments = elem['argcontact']
            Coeff_bary = elem['coeffs']
            var['time convex hull'] += elem['time_comp']
            var['time shit'] += elem['time_shit']
            if var['calc_Gamma']:
                total_contact = elem['total_contact']
                nb_contact = len(total_contact)
                if var['plot_maps']:
                    x_0 = var['gridX'][global_index]
                    if var['d']== 2:
                        var['print_dots'](x_0, total_contact, nb_figure = global_index, plot_save = True)
                    if var['d']==1:
                        for point in total_contact:
                            var['to_plot'].append((x_0, point))
                    max_distrib = len(var['degree_distrib'])
                if max_distrib <= nb_contact:
                    z = np.zeros(nb_contact-max_distrib+1)
                    var['degree_distrib'] = np.concatenate((var['degree_distrib'],z))
                var['degree_distrib'][nb_contact]+=1
            for j in range(len(arguments)):
                arg = int(arguments[j])
                if var['use_gradient']:
                    var['grad'][arg]-= Coeff_bary[j]*var['mu'][global_index]
                var['check'](Coeff_bary[j]> - var['zero'])
            return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        self.phi = var['phi']
        self.h = var['h']
        
        if calc_Gamma and plot_maps:
            if self.dim==1:
                self.data_points.append(var['to_plot'])
                if self.plot_data_save:
                    to_save = np.asarray(self.data_points)
                    np.save('new_images/maps'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.npy', to_save)
                else:
                    index = 5
                    for to_plot in self.data_points:
                        plt.figure(index)
                        for plot in to_plot:
                            plt.plot(plot[0], plot[1], marker='.', markersize=2, linestyle='None', color='r')
                        if self.plot_save:
                            plt.savefig('new_images/maps'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.png')
                            plt.close()
                        else:
                            plt.show()
                        index += 1
        if compare_grad_diff:
            Size = np.linalg.norm(psi)
            if psi_rand is None:
                psi_rand = np.random.rand(len(self.grid))-0.5
                Size_rand = np.linalg.norm(psi_rand)
                if Size !=0:
                    psi_rand *= (Size/Size_rand)
            Value_mid = self.Value_func_grad(
                            psi_rand = psi_rand)[0]
            Value_up = self.Value_func_grad(psi = psi+0.001*psi_rand,
                psi_rand = psi_rand)[0]
            Value_down = self.Value_func_grad(psi = psi-0.001*psi_rand,
                psi_rand = psi_rand)[0]
            finite_difference_up =1000*(Value_up-Value_mid)
            finite_difference_down =1000*(Value_mid-Value_down)
            if use_gradient:
                derivative =np.dot(var['grad'], psi_rand)
            print("finite difference up  =",finite_difference_up)
            if use_gradient:
                print("gradient              =",derivative)
            print("finite difference down=",finite_difference_down)
            if use_gradient:
                bool_test=finite_difference_down<=derivative+self.zero
                bool_test = bool_test and derivative<=finite_difference_up+self.zero
            else:
                bool_test = finite_difference_down<=finite_difference_up+self.zero
            gf.check(bool_test,psi)
        if calc_Gamma:
            if use_gradient:
                result = (var['value'], var['grad'], var['degree_distrib'])
            else:
                result = (var['value'], var['degree_distrib'])
        elif use_gradient:
            result = (var['value'], var['grad'])
        else:
            result = var['value']
        return result






    def test_convex_order(self, tol = 0):
        if self.dim != 1:
            print("Not supported yet. Not completely accurate")
        flag_error = False
        for x in self.gridX:
            f = lambda y : np.linalg.norm(y-x)
            funcY = self.func_grid(f, axis = 'Y')
            proba_testY = self.mean(funcY,proba = self.nu)
            funcX = self.func_grid(f, axis = 'X')
            proba_testX = self.mean(funcX,proba = self.mu)
            proba_test = proba_testY-proba_testX
            if proba_test < -tol:
                print("x = ",x,"erreur = ",proba_test)
                flag_error = True
        if flag_error:
            raise("Not in convex order.")
        print("checking done")
        
        
        
    def set_convex_order(self, penalization_type = "uniform", tol = zero):
        print("\n")
        print("Finding the closest nu in convex order...")
        grid_for_computation = self.copy()
        gfc = grid_for_computation
        gfc.cost = cst.zero_cost
        gfc.method = 'Newton-CG'
        gfc.penalization = 1.
        gfc.epsilon = 1.
        gfc.tolerance = tol
        gfc.penalization_type = penalization_type
        gfc.Optimization_entropic_newton()
        marginal_Y = gfc.marginal(project_on = 'y')['marginal']
        if self.nu_original is None:
            self.nu_original = self.nu
        self.nu = marginal_Y
        print("Computed!")
        print("\n")
        
        
        
    def regularize_measures(self):
        if self.purify_proba:
            raise("disable purification first.")
        mu = self.mu
        nu = self.nu
        self.mu = (3.*mu+nu)/4.
        self.nu = (mu+3.*nu)/4.
        
        
        
### ENTROPIC OPTIMIZATION FUNCTIONS
        
        
    def otimes(self, h, index = None):
        if index is None:
            Y_X = np.reshape(self.gridY,(1, self.lenY,self.dim))-np.reshape(self.gridX,
                            (self.lenX, 1, self.dim))
            return np.sum(np.reshape(h,(self.lenX,1,self.dim))*Y_X,axis = 2)
        else:
            Y_X = np.reshape(self.gridY,(self.lenY,self.dim))-np.reshape(self.gridX[index],(1, self.dim))
            result = np.sum(np.reshape(h,(1,self.dim))*Y_X,axis = 1)
            return result



    def psi_from_phi_h(self, phi = None, h = None):
        if phi is None:
            phi = self.phi
        if h is None:
            h = self.h
        
        code_name = "psi"
        axis = 'y'
        auxiliary = fn.auxiliary_psi          
        base_arg = self.base_arg(phi = False, psi = False, h = False, mu = False)
        
        base_arg['phi'] = phi
        base_arg['h'] = h
        
        var = {'psi' : self.psi}
            
        def apply_elem(elem, var):
                var['psi'][elem['j']] = elem['psi_j']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        self.psi = var['psi']
        
        if self.debug_mode==4:
              gf.check(gf.approx_Equal(np.sum(self.compute_proba(), axis = 0),self.nu),
              ("marginal Y fail: ", np.sum(self.compute_proba(), axis = 0), self.nu))
        return self.psi




    def phi_from_psi_h(self, psi = None, h = None):
        if psi is None:
            psi = self.psi
        if h is None:
            h = self.h
            
        code_name = "phi"
        axis = 'x'
        auxiliary = fn.auxiliary_phi          
        base_arg = self.base_arg(phi = False, psi = False, h = False, nu = False)
        
        base_arg['psi'] = psi
        base_arg['h'] = h
        
        var = {'phi' : self.phi}
            
        def apply_elem(elem, var):
                var['phi'][elem['i']] = elem['phi_i']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        self.phi = var['phi']
        
        if self.debug_mode==4:
            gf.check(gf.approx_Equal(np.sum(self.compute_proba(), axis = 1),self.mu),
            ("marginal X fail: ",np.sum(self.compute_proba(), axis = 1),self.mu))
        return self.phi







    def h_from_phi_psi(self, calc_phi = False, phi = None, psi = None, h = None):  
        if not self.martingale:
            raise("you should not be here.")
        if self.phi is None:
            self.init_phi()
        if phi is None:
            phi = self.phi
        if psi is None:
            psi = self.psi
        if self.h is None:
            self.init_h()
        if h is None:
            h = self.h
        if self.newNewton:
            minimize = None
        else:
            minimize = scipy.optimize.minimize
            
        if calc_phi:
            mu = self.mu
        else:
            mu = None
        code_name = "h"
        axis = 'x'
        auxiliary = fn.auxiliary_h           
        base_arg = self.base_arg(phi = False, psi = False, h = False, mu = False, nu = False)
        
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        base_arg['mu'] = mu
        base_arg['minimize'] = minimize
        base_arg['calc_phi'] = calc_phi
        
        var = {'phi' : self.phi, 'h' : self.h, 'calc_phi' : calc_phi}
            
        def apply_elem(elem, var):
                var['h'][elem['i']] = elem['hi']
                if var['calc_phi']:
                    var['phi'][elem['i']] = elem['phi_i']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        self.h = var['h']
        if calc_phi:
            self.phi = var['phi']
        if self.debug_mode==4:
            Proba = self.compute_proba()
            mu = np.sum(Proba, axis = 1)
            gf.check(gf.approx_Equal(mu, self.mu), ("Marginal mu fail, distance =", np.linalg.norm(mu-self.mu)))
        if self.debug_mode==3:
            Proba = self.compute_proba()
            Esp_Y_X = np.sum(np.reshape(Proba, (self.lenX, self.lenY,
              1))*(np.reshape(self.gridY,(1,self.lenY, self.dim))-np.reshape(self.gridX,
                  (self.lenX, 1, self.dim))), axis = 1)
            print("max", np.max(Esp_Y_X))
            
            print("min", np.min(Esp_Y_X))
            
            print("sum", np.linalg.norm(Esp_Y_X, self.pow_distance))
            
        phi_h = {'phi' : self.phi , 'h' : self.h}
        return phi_h




    def marginal(self, project_on = 'y'):
        code_name = "marginal_"+project_on
        if project_on == 'x':
            auxiliary = fn.auxiliary_vg_phi
            axis = 'x'
            index = 'i'
            marginal = self.phi*0
        elif project_on == 'y':
            auxiliary = fn.auxiliary_vg_psi
            axis = 'y'
            index = 'j'
            marginal = self.psi*0
        elif project_on == 'bary':
            auxiliary = fn.auxiliary_vg_h
            axis = 'x'
            index = 'i'
            marginal = self.h*0
        else:
            print("The axis ", project_on, " does not exist")
            raise("Wrong axis")
        base_arg = self.base_arg(mu = False, nu = False)
     
        var = {'marginal' : marginal}
                
        def apply_elem(elem, var):
            var['marginal'][elem[index]] += -elem['gradient']
            return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        return {'marginal' : var['marginal']}




    def value_grad_phi(self, phi, psi, h):
        code_name = "vg_phi"
        axis = 'x'
        auxiliary = fn.auxiliary_vg_phi
        base_arg = self.base_arg(mu = False, nu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        
        value = np.sum(self.mu*phi)+np.sum(self.nu*psi)
        gradient = np.array(self.mu)     
        var = {'value' : value, 'gradient' : gradient}
            
        def apply_elem(elem, var):
                var['value'] += elem['value']
                var['gradient'][elem['i']] += elem['gradient']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        return {'value' : var['value'], 'gradient' : var['gradient']}
        


    def value_grad_psi(self, phi, psi, h):
        code_name = "vg_psi"
        axis = 'y'
        auxiliary = fn.auxiliary_vg_psi
        base_arg = self.base_arg(mu = False, nu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        
        value = np.sum(self.mu*phi)+np.sum(self.nu*psi)
        gradient = np.array(self.nu)     
        var = {'value' : value, 'gradient' : gradient}
            
        def apply_elem(elem, var):
                var['value'] += elem['value']
                var['gradient'][elem['j']] += elem['gradient']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        return {'value' : var['value'], 'gradient' : var['gradient']}



    def value_grad_h(self, phi, psi, h):
        if not self.martingale:
            raise("you lost yourself.")
        code_name = "vg_h"
        axis = 'x'
        auxiliary = fn.auxiliary_vg_h
        base_arg = self.base_arg(mu = False, nu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        
        value = np.sum(self.mu*phi)+np.sum(self.nu*psi)
#        value = np.sum(self.mu*self.phi)+np.sum(self.nu*self.psi)
        gradient = np.zeros(self.lenX*self.dim)
        gradient = np.reshape(gradient, (self.lenX, self.dim))   
        var = {'value' : value, 'gradient' : gradient}
            
        def apply_elem(elem, var):
                var['value'] += elem['value']
                var['gradient'][elem['i']] += elem['gradient']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING

        return {'value' : var['value'], 'gradient' : var['gradient']}
  
    
    def hess_p_phi(self, phi, psi, h, p_phi, p_psi, p_h):
        code_name = "hess_p_phi"
        axis = 'x'
        auxiliary = fn.auxiliary_hess_phi
        base_arg = self.base_arg(mu = False, nu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        base_arg['p_phi'] = p_phi
        base_arg['p_psi'] = p_psi
        base_arg['p_h'] = p_h
        if self.sparse_is_on and p_h is not None:
            base_arg['p_h_sparse'] = p_h.transpose()
        
        hess_p = np.zeros(self.lenX)
            
        var = {'hess_p' : hess_p}
            
        def apply_elem(elem, var):
                var['hess_p'][elem['i']] = elem['hess_p']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING

        return var['hess_p']



    def hess_p_psi(self, phi, psi, h, p_phi, p_psi, p_h):
        code_name = "hess_p_psi"
        axis = 'y'
        auxiliary = fn.auxiliary_hess_psi
        base_arg = self.base_arg(mu = False, nu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        base_arg['p_phi'] = p_phi
        base_arg['p_psi'] = p_psi
        base_arg['p_h'] = p_h
        if self.sparse_is_on and p_h is not None:
            base_arg['p_h_sparse'] = p_h.transpose()

        hess_p = np.zeros(self.lenY)
        var = {'hess_p' : hess_p}
            
        def apply_elem(elem, var):
                var['hess_p'][elem['j']] = elem['hess_p']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING

        return var['hess_p']



    def hess_p_h(self, phi, psi, h, p_phi, p_psi, p_h):
        if not self.martingale:
            raise("you should not be here.")
        code_name = "hess_p_h"
        axis = 'x'
        auxiliary = fn.auxiliary_hess_h
        base_arg = self.base_arg(mu = False, nu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        base_arg['p_phi'] = p_phi
        base_arg['p_psi'] = p_psi
        base_arg['p_h'] = p_h
        if self.sparse_is_on and p_h is not None:
            base_arg['p_h_sparse'] = p_h.transpose()
        
        hess_p = np.zeros(self.lenX*self.dim)
        hess_p = np.reshape(hess_p, (self.lenX, self.dim))
            
        var = {'hess_p' : hess_p}
            
        def apply_elem(elem, var):
                var['hess_p'][elem['i']] = elem['hess_p']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING

        return var['hess_p']


    def hess_h_inv(self, phi = None, psi= None, h = None):
        if not self.martingale:
            raise("You should not be here.")
        if phi is None:
            phi = self.phi
        if psi is None:
            psi = self.psi
        if h is None:
            h = self.h
        pack = self.package(phi, psi, h)
        if np.size(self.current_pack_hess) == np.size(pack) and gf.approx_Equal(pack,
                      self.current_pack_hess, tolerance = self.zero**2):
            return self.current_hess_h_inv
        
        code_name = "hess_h_inv"
        axis = 'x'
        auxiliary = fn.auxiliary_hess_h_inv
        base_arg = self.base_arg(nu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h

        if self.include_phi:
            subdim = self.dim +1
        else:
            subdim = self.dim
        hess_h_inv = np.zeros(self.lenX*(subdim)**2)
        hess_h_inv = np.reshape(hess_h_inv, (self.lenX, subdim, subdim))
            
        var = {'hess_h_inv' : hess_h_inv}

        def apply_elem(elem, var):
                var['hess_h_inv'][elem['i']] = elem['hess_h_inv']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        hess_h_inv = var['hess_h_inv']
        self.current_pack_hess = pack
        self.current_hess_h_inv = hess_h_inv
        return hess_h_inv 




    def test_differentiability(self, phi, psi, h, use_gradient = True, phi_rand = None,
                               psi_rand = None, h_rand = None, p_phi = None,
                               p_psi = None, p_h = None):
        ### Definition of the test variables
        if phi_rand is None:
            phi_rand = np.random.rand(self.lenX)-0.5
            Size_rand = np.linalg.norm(phi_rand)
            phi_rand /= Size_rand
        if p_phi is None:
            p_phi = np.random.rand(self.lenX)-0.5
            Size_rand = np.linalg.norm(p_phi)
            p_phi /= Size_rand
        if psi_rand is None:
            psi_rand = np.random.rand(self.lenY)-0.5
            Size_rand = np.linalg.norm(psi_rand)
            psi_rand /= Size_rand
        if p_psi is None:
            p_psi = np.random.rand(self.lenY)-0.5
            Size_rand = np.linalg.norm(p_psi)
            p_psi /= Size_rand
        if h_rand is None:
            h_rand = np.random.rand(self.lenX, self.dim)-0.5
            Size_rand = np.linalg.norm(h_rand)
            h_rand /= Size_rand
        if p_h is None:
            p_h = np.random.rand(self.lenX, self.dim)-0.5
            Size_rand = np.linalg.norm(p_h)
            p_h /= Size_rand
                
        ### Definition of the finite differences
        ### phi
        print('phi')
        Value_mid = self.value_grad_phi(phi, psi, h)['value']
        Value_up = self.value_grad_phi(phi + 1e-2*self.epsilon*phi_rand, psi, h)['value']
        Value_down = self.value_grad_phi(phi - 1e-2*self.epsilon*phi_rand, psi, h)['value']
        finite_difference_up =1e2/self.epsilon*(Value_up-Value_mid)
        finite_difference_down =1e2/self.epsilon*(Value_mid-Value_down)
        if use_gradient:
            grad = self.value_grad_phi(phi, psi, h)['gradient']
            derivative =np.dot(grad,phi_rand)
        print("finite difference up  =",finite_difference_up)
        if use_gradient:
            print("gradient              =",derivative)
        print("finite difference down=",finite_difference_down)
        if use_gradient:
            bool_test=finite_difference_down<=derivative+self.zero
            bool_test = bool_test and derivative<=finite_difference_up+self.zero
        else:
            bool_test = finite_difference_down<=finite_difference_up+self.zero
        gf.check(bool_test,"phi")
        
        ### psi
        print('psi')
        Value_mid = self.value_grad_psi(phi, psi, h)['value']
        Value_up = self.value_grad_psi(phi, psi + 1e-2*self.epsilon*psi_rand, h)['value']
        Value_down = self.value_grad_psi(phi, psi - 1e-2*self.epsilon*psi_rand, h)['value']
        finite_difference_up =1e2/self.epsilon*(Value_up-Value_mid)
        finite_difference_down =1e2/self.epsilon*(Value_mid-Value_down)
        if use_gradient:
            grad = self.value_grad_psi(phi, psi, h)['gradient']
            derivative =np.dot(grad, psi_rand)
        print("finite difference up  =",finite_difference_up)
        if use_gradient:
            print("gradient              =",derivative)
        print("finite difference down=",finite_difference_down)
        if use_gradient:
            bool_test=finite_difference_down<=derivative+self.zero
            bool_test = bool_test and derivative<=finite_difference_up+self.zero
        else:
            bool_test = finite_difference_down<=finite_difference_up+self.zero
        gf.check(bool_test,"psi")
    
        ### h
        if h is not None:
            print('h')
            Value_mid = self.value_grad_h(phi, psi, h)['value']
            Value_up = self.value_grad_h(phi, psi, h + 1e-2*self.epsilon*h_rand)['value']
            Value_down = self.value_grad_h(phi, psi, h - 1e-2*self.epsilon*h_rand)['value']
            finite_difference_up =1e2/self.epsilon*(Value_up-Value_mid)
            finite_difference_down =1e2/self.epsilon*(Value_mid-Value_down)
            if use_gradient:
                grad = self.value_grad_h(phi, psi, h)['gradient']
                derivative =np.sum(grad*h_rand)
            print("finite difference up  =",finite_difference_up)
            if use_gradient:
                print("gradient              =",derivative)
            print("finite difference down=",finite_difference_down)
            if use_gradient:
                bool_test=finite_difference_down<=derivative+self.zero
                bool_test = bool_test and derivative<=finite_difference_up+self.zero
            else:
                bool_test = finite_difference_down<=finite_difference_up+self.zero
            gf.check(bool_test,"h")



        
    def compute_proba(self, index = None):
        if index is None:
            if self.martingale:
                arg_Gibbs = (self.cost(self.gridX,self.gridY)-self.otimes(self.h)-np.reshape(self.phi,
                         (self.lenX,1))-np.reshape(self.psi,(1,self.lenY)))/self.epsilon
            else:
                arg_Gibbs = (self.cost(self.gridX,self.gridY)-np.reshape(self.phi,
                         (self.lenX,1))-np.reshape(self.psi,(1,self.lenY)))/self.epsilon
        else:
            if self.martingale:
                arg_Gibbs = (self.cost(self.gridX[index],
                     self.gridY)-self.otimes(self.h[index],
                           index = index)-self.phi[index]-self.psi)/self.epsilon
            else:
                arg_Gibbs = (self.cost(self.gridX[index],
                                       self.gridY)-self.phi[index]-self.psi)/self.epsilon
        arg_Gibbs -= np.max(arg_Gibbs)
        Proba = np.exp(arg_Gibbs)
        Proba /= np.sum(Proba)
        return Proba




   
    def Optimization_entropic(self, iterations = 20):
        self.nb_calls = 0
        self.values = []
        self.grad_norms = []
        self.times_comp = []
        self.current_pack = None
        self.time_init = 0.
        self.base_package = self.package(self.phi, self.psi, self.h)
        
        if self.martingale:
            mart = 'mart'
        else:
            mart = ''
        if self.method == 'sinkhorn':
            self.Optimization_entropic_sinkhorn(iterations = iterations)
        elif self.method == 'hybrid':
            self.Optimization_entropic_hybrid(iterations = iterations)
        else:
            self.Optimization_entropic_newton(iterations = iterations)
            
        if self.plot_perf:
            
            self.data_perf_grad.append({'nb_call' : self.nb_calls, 'grad_norms' : list(self.grad_norms),
                                   'name' : self.method+', eps = '+str(self.epsilon), 'fignum' : self.fignum,
                                   'times' : list(self.times_comp)})
            if self.plot_data_save:
                to_save = np.asarray(self.data_perf_grad)
                np.save('new_images/performance_gradients'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.npy', to_save)
            else:
                plt.figure(1000+self.fignum)
                for plot in self.data_perf_grad:
                    if self.fignum == plot['fignum']:
                        coord_x = plot['times']
                        plt.semilogy(coord_x, plot['grad_norms'], label = plot['name'])
                plt.legend()
                if self.plot_save:
                    plt.savefig('new_images/performance_gradients'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.png')
                    plt.close()
                else:
                    plt.show()
    

    def Optimization_entropic_sinkhorn(self, iterations = 20):
        self.grid_just_dobled = False
        impl_phi_h = self.impl_phi_h
        impl_psi = self.impl_psi
        self.impl_phi_h = 0
        self.impl_psi = 0
        right_mu = False
        time_compute = -self.time_init
        self.penalization = 0
        the_time = timeit.default_timer()
        if impl_psi:
            t_0 = timeit.default_timer()
            self.psi_from_phi_h()
            print("time for psi = ", timeit.default_timer()-t_0)
            if self.debug_mode == 6:
                print("psi=",self.psi)
            right_mu = False
        elif impl_phi_h:
            if self.martingale:
                
                t_0 = timeit.default_timer()
                self.h_from_phi_psi(calc_phi = True)
                print("time for h = ",timeit.default_timer()-t_0)
                if self.debug_mode == 6:
                    print("h = ",self.h)                
                    print("phi=",self.phi)
                right_mu = True
            else:
                if not right_mu:
                    t_0 = timeit.default_timer()
                    self.phi_from_psi_h()
                    print("time for phi = ", timeit.default_timer()-t_0)
                    if self.debug_mode == 6:
                        print("phi=",self.phi)
                    right_mu = True
        time_compute += timeit.default_timer()-the_time
        self.times_comp.append(time_compute)
        error = self.value_grad_ext(self.package(self.phi, self.psi, self.h))[1]
        if np.linalg.norm(error, self.pow_distance) <= self.tolerance:
            self.impl_phi_h = impl_phi_h
            self.impl_psi = impl_psi
            self.time_init -= time_compute
            return 0
        for i in range(iterations):
            the_time = timeit.default_timer()
            print('\n')
            print("iteration ",i+1)
            for j in range(self.times_compute_phi_psi):
                if not right_mu:
                    t_0 = timeit.default_timer()
                    self.phi_from_psi_h()
                    print("time for phi = ", timeit.default_timer()-t_0)
                    if self.debug_mode == 6:
                        print("phi=",self.phi)
                    right_mu = True
                t_0 = timeit.default_timer()
                self.psi_from_phi_h()
                print("time for psi = ", timeit.default_timer()-t_0)
                if self.debug_mode == 6:
                    print("psi=",self.psi)
                right_mu = False
            if self.martingale:
                
                t_0 = timeit.default_timer()
                self.h_from_phi_psi(calc_phi = True)
                print("time for h = ",timeit.default_timer()-t_0)
                if self.debug_mode == 6:
                    print("h = ",self.h)
                    print("phi=",self.phi)
                right_mu = True
            time_compute += timeit.default_timer()-the_time
            self.times_comp.append(time_compute)
            if i % 1 == 0:
                self.penalization = 0
                error = self.value_grad_ext(self.package(self.phi, self.psi, self.h))[1]
                if np.linalg.norm(error, self.pow_distance) <= self.tolerance:
                    break
        self.impl_phi_h = impl_phi_h
        self.impl_psi = impl_psi
        self.time_init -= time_compute
        return 0
                
                
    def Optimization_entropic_hybrid(self, iterations = 20):
        tol = self.tolerance
        pen = self.penalization
        self.time_init = timeit.default_timer()
        self.penalization = 0.
        error = self.value_grad_ext(self.package(self.phi, self.psi, self.h))[1]
        error_size = np.linalg.norm(error, self.pow_distance)
        if self.grid_just_dobled:
            self.tolerance = min(0.5, max(0.5*error_size, tol))
        else:
            self.tolerance = min(0.5, max(0.5*error_size, (1.-error_size)*error_size, tol))
        self.time_init = -(timeit.default_timer()-self.time_init)
        self.Optimization_entropic_sinkhorn(iterations = 500)
        self.tolerance = tol
        self.penalization = pen
        self.Optimization_entropic_newton(iterations = iterations)


    def package(self, phi, psi, h):
        if self.impl_phi_h:
            return np.array(psi)
        lenX = self.lenX
        lenY = self.lenY
        if self.impl_psi:
            lenY = 0
        if not self.martingale:
            dim = 0
        else:
            dim = self.dim
        pack = np.zeros(lenX*(dim+1)+lenY)
        for i in range(lenX):
            pack[i] = phi[i]
            for k in range(dim):
                pack[lenX+lenY+i*dim+k] = h[i][k]
        for j in range(lenY):
            pack[lenX+j] = psi[j]
            if self.impl_psi:
                raise("You should not be here.")
        return pack


    def unpack(self, pack):
        if not self.impl_phi_h:
            phi = pack[:self.lenX]
            gf.check(len(phi) == self.lenX)
        if self.impl_psi:
            lenY = 0
            psi = None
        elif self.impl_phi_h:
            return {'phi' : None, 'psi' : pack, 'h' : None}
        else:
            lenY = self.lenY
            psi = pack[self.lenX:self.lenX+lenY]
            gf.check(len(psi) == lenY)
        if self.martingale:
            h = pack[self.lenX+lenY:]
            h = np.reshape(h, (self.lenX,self.dim))
        else:
            h = None
        return {'phi' : phi, 'psi' : psi, 'h' : h}

    
    
    
    def sinkh_step(self, pack):
                unpack = self.unpack(pack)
                phi = unpack['phi']
                psi = unpack['psi']
                h = unpack['h']
                if self.impl_psi:
                    if self.current_pack is not None:
                        if gf.approx_Equal(pack, self.current_pack, tolerance = self.zero**2):
                            psi = self.psi
                    else:
                        psi = self.psi_from_phi_h(phi = phi, h = h)
                    DATA = self.h_from_phi_psi(calc_phi = True, psi = psi)
                    phi = DATA['phi']
                    h = DATA['h']
                elif self.impl_phi_h:
                    if self.current_pack is not None:
                        if gf.approx_Equal(pack, self.current_pack, tolerance = self.zero**2):
                            phi = self.phi
                            h = self.h
                    else:
                        DATA = self.h_from_phi_psi(calc_phi = True, psi = psi)
                        phi = DATA['phi']
                        h = DATA['h']
                        psi = self.psi_from_phi_h(phi = phi, h=h)
                else:
                    DATA = self.h_from_phi_psi(calc_phi = True, psi = psi)
                    phi = DATA['phi']
                    h = DATA['h']
                    self.phi = phi
                    self.h = h
                new_pack =self.package(phi, psi, h)
                return new_pack


    def diag_hess_psi(self, phi = None, psi = None, h = None, no_impl = False):
        code_name = "diag_hess_psi"
        axis = 'y'
        auxiliary = fn.auxiliary_diag_hess_psi
        if phi is None:
            phi = self.phi
        if phi is None:
            psi = self.psi
        if phi is None:
            h = self.h
        if self.martingale:
            hess_h_inv = self.hess_h_inv(phi, psi, h)
        else:
            hess_h_inv = None
        base_arg = self.base_arg(nu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        base_arg['no_impl'] = no_impl
        base_arg['hess_h_inv'] = hess_h_inv
        base_arg['include_phi'] = self.include_phi
        
        diag_hess = np.zeros(self.lenY)        
            
        var = {'diag_hess' : diag_hess}
            
        def apply_elem(elem, var):
                var['diag_hess'][elem['j']] = elem['diag'] 
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        return var['diag_hess']
        
        
        
    def diag_hess_phi_h(self, phi, psi, h, no_impl = False):
        code_name = "diag_hess_phi_h"
        axis = 'x'
        auxiliary = fn.auxiliary_diag_hess_phi_h

        base_arg = self.base_arg(mu = False)
        base_arg['phi'] = phi
        base_arg['psi'] = psi
        base_arg['h'] = h
        base_arg['no_impl'] = no_impl
        
        diag_phi = np.zeros(self.lenX)
        diag_h = np.zeros(self.lenX*self.dim)
        diag_h = np.reshape(diag_h, (self.lenX, self.dim))     
            
        var = {'diag_phi' : diag_phi, 'diag_h' : diag_h}
            
        def apply_elem(elem, var):
                index = elem['i']
                var['diag_phi'][index] = elem['diag_phi']
                var['diag_h'][index] = elem['diag_h']
                return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        return {'phi' : var['diag_phi'], 'h' : var['diag_h']}

            
            
            
    def preconditioner_CG(self, pack):
        if self.hess_pos_safe:
            impl_phi_h = self.impl_phi_h
            impl_psi = self.impl_psi
            self.impl_phi_h = False
            self.impl_psi = False
            pack = self.package(self.phi, self.psi, self.h)
        if self.impl_psi:
            data = self.unpack(pack)
            phi = data['phi']
            h = data['h']
            diag_phi_h = self.diag_hess_phi_h(phi, self.psi, h)
            diag_phi = diag_phi_h['phi']
            diag_h = diag_phi_h['h']
            self.current_pack_cond = self.package(diag_phi, None, diag_h)
            self.current_pack_cond += self.penalization_func(pack, diff = 2)
        elif self.impl_phi_h:
            data = self.unpack(pack)
            psi = data['psi']
            if gf.approx_Equal(pack, self.current_pack, tolerance = self.zero**2):
                phi = self.phi
                h = self.h
            else:
                DATA = self.h_from_phi_psi(calc_phi = True, psi = psi)
                phi = DATA['phi']
                h = DATA['h']
            diag_psi = self.diag_hess_psi(phi = phi, psi = psi, h = h, no_impl = False)
            self.current_pack_cond = self.package(None, diag_psi, None)
            self.current_pack_cond += self.penalization_func(pack, diff = 2)
        else:
            data = self.unpack(pack)
            phi = data['phi']
            psi = data['psi']
            h = data['h']
            diag_phi_h = self.diag_hess_phi_h(phi, psi, h, no_impl = True)
            diag_phi = diag_phi_h['phi']
            diag_psi = self.diag_hess_psi(self.phi, psi, self.h, no_impl = True)
            diag_h = diag_phi_h['h']
            self.current_pack_cond = self.package(diag_phi, diag_psi, diag_h)
            self.current_pack_cond += self.penalization_func(pack, diff = 2)
        if self.hess_pos_safe:
            self.impl_phi_h = impl_phi_h
            self.impl_psi = impl_psi
        return (lambda x : x/self.current_pack_cond)

        
        
        
    def save_h(self, h_sto = None):
        if h_sto is not None:
            self.h = h_sto
        return self.h
    
        
    def value_grad_ext(self, pack, print_grad = True):
        self.current_pack = pack
        unpack = self.unpack(pack)
        phi = unpack['phi']
        psi = unpack['psi']
        h = unpack['h']
        if self.impl_psi:
            psi = self.psi_from_phi_h(phi = phi, h = h)
        elif self.impl_phi_h:
            self.psi = psi
            if self.martingale:
                DATA = self.h_from_phi_psi(calc_phi = True, psi = psi)
                phi = DATA['phi']
                self.phi = phi
                h = DATA['h']
                self.h = h
            else:
                phi = self.phi_from_psi_h(psi = psi, h = None)
        if self.hess_pos_safe:
            impl_phi_h = self.impl_phi_h
            impl_psi = self.impl_psi
            self.impl_phi_h = False
            self.impl_psi = False
            pack = self.package(phi, psi, h)
        if not self.martingale:
            h = None
        else:
            if not self.impl_phi_h:
                vg_h = self.value_grad_h(phi, psi, h)
        if self.impl_phi_h:
            vg_phi = {'gradient' : None}
            vg_h = {'gradient' : None}
        else:
            vg_phi = self.value_grad_phi(phi, psi, h)
        if self.impl_psi:
            vg_psi = {'gradient' : None}
        else:
            vg_psi = self.value_grad_psi(phi, psi, h)
        if self.debug_mode==2 and not self.impl_phi_h:
            gf.check(gf.approx_Equal(vg_phi['value'],vg_psi['value']),
                 (vg_phi['value'],vg_psi['value']))
            if self.martingale:
                gf.check(gf.approx_Equal(vg_phi['value'],vg_h['value']),
                         (vg_phi['value'],vg_h['value']))
        if self.impl_phi_h:
            value = vg_psi['value']
        else:
            value = vg_phi['value']
        if not self.martingale:
            vg_h = {'gradient' : None}
        gradient = self.package(vg_phi['gradient'], vg_psi['gradient'],
                                vg_h['gradient'])
        ###penalization
        value += self.penalization_func(pack)
        gradient += self.penalization_func(pack, diff = 1)
 
        grad_norm = np.linalg.norm(gradient, self.pow_distance)###The norm       
        if print_grad:
            print("value = ", value)
            print("gradient norm = ", grad_norm)
        self.current_value = value
        self.current_gradient_norm = grad_norm
        self.nb_calls += 1
        if self.plot_perf:
            self.values.append(self.current_value)
            self.grad_norms.append(self.current_gradient_norm)
            if len(self.times_comp) < len(self.grad_norms):
                self.times_comp.append(timeit.default_timer()- self.time_init)
        if self.debug_mode == 2:
            self.test_differentiability(phi,psi,h)
        self.phi = phi
        self.psi = psi
        self.h = h
        self.current_gradient = gradient
        if self.hess_pos_safe:
            self.impl_phi_h = impl_phi_h
            self.impl_psi = impl_psi
        return (value, gradient)



  
    
    def hessian_p_ext(self, pack, p):
            unpack = self.unpack(pack)
            phi = unpack['phi']
            psi = unpack['psi']
            h = unpack['h']
            if self.impl_psi:
                if gf.approx_Equal(pack, self.current_pack, tolerance = self.zero**2):
                    psi = self.psi
                else:
                    psi = self.psi_from_phi_h(phi = phi, h = h)
            if self.impl_phi_h:
                if gf.approx_Equal(pack, self.current_pack, tolerance = self.zero**2):
                    phi = self.phi
                    h = self.h
                else:
                    DATA = self.h_from_phi_psi(calc_phi = True, psi = psi)
                    phi = DATA['phi']
                    h = DATA['h']
            self.current_pack = pack
            if self.hess_pos_safe:
                impl_phi_h = self.impl_phi_h
                impl_psi = self.impl_psi
                self.impl_phi_h = False
                self.impl_psi = False
                pack = self.package(phi, psi, h)
            unpack_p = self.unpack(p)
            p_phi = unpack_p['phi']
            p_psi = unpack_p['psi']
            p_h = unpack_p['h']
            if not self.martingale:
                h = None
            if self.impl_psi:
                hess_psi = None
                p_psi = self.hess_p_psi(phi, psi, h, p_phi, None, p_h)
                p_psi *= -self.epsilon/np.maximum(self.nu, self.zero)
            if self.impl_phi_h:
                hess_phi = None
                hess_p_phi = self.hess_p_phi(phi, psi, h, None, p_psi, None)
                if self.martingale:
                    hess_p_h = self.hess_p_h(phi, psi, h, None, p_psi, None)
                    hess_h_inv = self.hess_h_inv(phi = phi, psi = psi, h = h)
                    if self.include_phi:
                        dim_here = self.dim+1
                        hess_p_h = np.concatenate((np.reshape(hess_p_phi, (self.lenX, 1)),
                                                   hess_p_h), axis = 1)
                    else:
                        dim_here = self.dim
                    p_h = -np.sum(hess_h_inv*np.reshape(hess_p_h,
                             (self.lenX, 1, dim_here)), axis = 2)#axis = 2 because matrix product.
                    if self.include_phi:
                        p_phi = p_h[:, 0:1]
                        p_phi = np.reshape(p_phi, (self.lenX))
                        p_h = p_h[:, 1:]
                    else:
                        p_phi = -hess_p_phi*self.epsilon/np.maximum(self.mu,
                                                                    self.zero/self.lenX)
                else:
                    p_h = None
                    p_phi = -hess_p_phi*self.epsilon/np.maximum(self.mu,
                                                self.zero/self.lenX)
            else:
                hess_phi = self.hess_p_phi(phi, psi, h, p_phi, p_psi, p_h)
            if not self.impl_psi:
                hess_psi = self.hess_p_psi(phi, psi, h, p_phi, p_psi, p_h)
            if self.martingale and not self.impl_phi_h:
                hess_h   = self.hess_p_h(phi, psi, h, p_phi, p_psi, p_h)
            else:
                hess_h = None
            hess_p   = self.package(hess_phi, hess_psi, hess_h)
            ###penalization
            hess_p += self.penalization_func(pack, diff = 2)*p
            
            if self.debug_mode==1:
                hess_p_test = (self.value_grad_ext(pack+1e-4*p)[1]-self.value_grad_ext(pack-1e-4*p)[1])/(2e-4)
                print("finite diff hess = ", np.linalg.norm(hess_p-hess_p_test, self.pow_distance))
                print("reference to compare the latest figure: ",
                               np.linalg.norm(hess_p+hess_p_test, self.pow_distance))
            self.nb_calls += 1
            if self.plot_perf:
                self.values.append(self.current_value)
                self.grad_norms.append(self.current_gradient_norm)
                self.times_comp.append(timeit.default_timer()- self.time_init)
            print("hess, call", self.nb_calls,
                  ", grad norm = ", self.current_gradient_norm)
            if self.hess_pos_safe:
                self.impl_phi_h = impl_phi_h
                self.impl_psi = impl_psi
            return hess_p



    def Optimization_entropic_newton(self, iterations = 20):
        
        self.grid_just_dobled = False
        self.time_init += timeit.default_timer()
        
        def value_grad(pack):
            if self.result_found:
                return (0, 0*pack)
            value_grad = self.value_grad_ext(pack)
            if self.gtol_for_newton:
                if np.linalg.norm(value_grad[1], self.pow_distance) <= self.tolerance:
                    print("result found.")
                    self.result_opt = pack
                    self.result_found = True
            return value_grad
        
        def hessian_p(pack, p):
            if self.result_found:
                return 0*p
            else:
                return self.hessian_p_ext(pack, p)
        
        x0 = self.package(self.phi, self.psi, self.h)
        if self.debug_mode==2:
            print("matching = ", scipy.optimize.check_grad(lambda x: value_grad(x)[0],
                                  lambda x: value_grad(x)[1], x0 = x0))
            
        if self.gtol_for_newton:
            tolerance = self.zero**2
        else:
            tolerance = self.tolerance

        if self.newNewton:
            if self.precond_CG:
                cond_inv = self.preconditioner_CG
            else:
                cond_inv = None
            
            if self.additional_step_CG:
                add_step = self.sinkh_step
            else:
                add_step = None
                
            if self.debug_mode==7:
                check_cond = True
            else:
                check_cond = False
            
            if self.debug_mode == 9:
                disp_CG = True
            else:
                disp_CG = False
            debug_mode_CG = self.debug_mode
            if self.debug_mode == 11:
                print_line_search = True
            else:
                print_line_search = False
            if self.hess_pos_safe:
                def adjust_for_ls(x):
                    impl_phi_h = self.impl_phi_h
                    impl_psi = self.impl_psi
                    self.impl_phi_h = False
                    self.impl_psi = False
                    data = self.unpack(x)
                    phi = data['phi']
                    psi = data['psi']
                    h = data['h']
                    self.impl_phi_h = impl_phi_h
                    self.impl_psi = impl_psi
                    return self.package(phi, psi, h)
            else:
                adjust_for_ls = None
            
            result = gf.Newton_CG(value_grad, x0 = x0, hessp = hessian_p,
                              tol = tolerance, disp = True, pow_distance = self.pow_distance,
                              maxiter = iterations,
                              maxiter_CG = 100, add_step = add_step,
                              cond_inv = cond_inv, check_cond = check_cond,
                              save_action = self.save_h, disp_CG = disp_CG,
                              debug_mode_CG = debug_mode_CG,
                              print_line_search = print_line_search,
                              order_hess = 1./self.epsilon,
                              adjust_for_ls = adjust_for_ls)
        else:
            result = scipy.optimize.minimize(value_grad, x0 = x0,
                                    method=self.method, jac=True, hessp = hessian_p,
                                    tol=tolerance,
                                    options={'maxiter': iterations,
                                             'disp' : True,'xtol' : tolerance,
                                             'gtol' : tolerance})
        if self.result_found:
            DATA = self.unpack(self.result_opt)
            self.result_found = False
        else:
            if self.newNewton:
                DATA = self.unpack(result['x'])
            else:
                DATA = self.unpack(result.x)
        if self.impl_psi:
            self.phi = DATA['phi']
            self.h = DATA['h']
            self.psi_from_phi_h()
        elif self.impl_phi_h:
            self.psi = DATA['psi']
            if self.martingale:
                self.h_from_phi_psi(calc_phi = True, psi = self.psi)
            else:
                self.phi_from_psi_h()
        else:
            self.phi = DATA['phi']
            self.psi = DATA['psi']
            self.h = DATA['h']



    def init_sparse(self, no_sto = False):
        self.sparse_is_on = False
        code_name = "sparse"
        axis = 'x'
        auxiliary = fn.auxiliary_sparse_grid
        base_arg = self.base_arg()
        base_arg['no_sto'] = no_sto
            
        var = {'Xx' : [], 'Yx' : [], 'Xy' : [], 'Yy' : [], 'non_zero_x' : 0,
               'non_zero_y' : 0, 'no_sto' : no_sto}
            
        def apply_elem(elem, var):
            if not var['no_sto']:
                var['Xx'] += elem['Xx']
                var['Yx'] += elem['Yx']
                var['Xy'] += elem['Xy']
                var['Yy'] += elem['Yy']            
            var['non_zero_x'] += elem['non_zero_x']
            var['non_zero_y'] += elem['non_zero_y']
            return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING
        
        if not no_sto:
            data_x = list(1+np.zeros(var['non_zero_x']))
            data_y = list(1+np.zeros(var['non_zero_y']))
            self.sparse_gridXY = csr( (data_x, (var['Xx'],var['Yx'])),
                                 shape=(self.lenX, self.lenY) )
            if True:
                self.sparse_gridYX = self.sparse_gridXY.transpose() 
            else:#Creating instabilities with the hessian
                self.sparse_gridYX = csr( (data_y, (var['Yy'],var['Xy'])),
                                 shape=(self.lenY, self.lenX) )
        
        self.memory_sparse = var['non_zero_x']+var['non_zero_y']

        return (var['non_zero_x'], var['non_zero_y'])



    def expectation_cost(self, dual = False, check_mass = True, phi = None,
                         psi = None, h = None):
        code_name = "expectation"
        axis = 'x'
        auxiliary = fn.auxiliary_expectation_cost
        if phi is None:
            phi = self.phi
        if psi is None:
            psi = self.psi
        if h is None:
            h = self.h
        base_arg = self.base_arg(mu = False, nu = False)
        base_arg['dual'] = dual
        base_arg['phi_test'] = phi
        base_arg['psi_test'] = psi
        base_arg['h_test'] = h
            
        var = {'expectation' : 0., 'total_mass' : 0., 'dual' : dual}
        if dual:
            var['expect_dual'] = 0.
            var['gap_phi'] = 0.
            
        def apply_elem(elem, var):
            var['expectation'] += elem['expectation_i']
            var['total_mass'] += elem['mass']
            if var['dual']:
                var['expect_dual'] += elem['expect_dual_i']
                var['gap_phi'] += elem['gap_phi']
            return var

        #DO NOT TOUCH THIS PART
        var = fn.action_pool(auxiliary = auxiliary, apply_elem = apply_elem,
        axis = axis, base_arg = base_arg, var = var, code_name = code_name)
        #END OF NOT TOUCHING

        total_mass = var['total_mass']
        if check_mass:
            gf.check(gf.approx_Equal(total_mass, 1.), total_mass)
        expectation = var['expectation']
        if dual:
            result = {'expectation_cost' : expectation,
                      'dual' : var['expect_dual'] , 'gap_phi' : var['gap_phi']}
        else:
            result = expectation
        return result      
    
    



    def test_accuracy(self):
        data = self.expectation_cost(dual = True)
        primal = data['expectation_cost']
        dual = data['dual']-data['gap_phi']
        if self.debug_mode == 12:
            print("gap phi = ", data['gap_phi'])
        print("primal = ", primal)
        print("dual = ", dual)
        gf.check(primal < dual)
        error = dual-primal
        print("error max <= ", error)
        error_sto = {'epsilon':self.epsilon , 'error' : error}
        
        if self.compute_phi_h and self.martingale and self.compute_entropy_error:
            phi = np.array(self.phi)
            h = np.array(self.h)
        
            self.Value_func_grad()
            phi_test = self.phi
            h_test = self.h
            self.phi = phi
            self.h = h
        
            data_prec = self.expectation_cost(dual = True, phi = phi_test, h = h_test)
            dual_prec = data_prec['dual']
            error_prec = dual_prec - primal
            print("error conc = ", error_prec)
            if self.debug_mode == 12:
                print("gap phi prec = ", data_prec['gap_phi'])
            error = error_prec
            error_sto['error_prec'] = error_prec
        self.epsilon_vs_error.append(error_sto)
        
        return error


        
    def Optimization_entropic_decay(self, iterations = None, epsilon_start = 1., final_size = 0.,
                                    final_granularity = None,
                                    r_0 = 0.5, r_f = None, entrop_error = 1e-4, tol = 1e-7,
                                    epsilon_final = 1e-5, intermediate_iter = 40, pen = 1e-2,
                                    tol_0 = 1e-1, tol_f = None,
                                    pen_0 = None, pen_f = None):
        def test_size():
            if final_granularity is not None:
                if self.grid_MC:
                    stepX = 1./self.MC_iter['x']
                    stepY = 1./self.MC_iter['y']
                else:
                    stepX = 2*self.boundX/self.stepsX
                    stepY = 2*self.boundY/self.stepsY
                if self.impl_psi:
                    step = stepX
                elif self.impl_phi_h:
                    step = stepY
                else:
                    step = np.sqrt(stepX*stepY)
                return (step > final_granularity)
            else:
                return ((self.lenX+self.lenY)<final_size)
        if iterations is None:
                if r_f is None:
                    r_f = r_0
                iterations = int(2*np.log(epsilon_final/epsilon_start)/np.log(r_0*r_f))
                if iterations <= 1:
                    raise("wrong parameters choice.")
                r_f = (epsilon_final/epsilon_start)**(2./(iterations))/r_0
                print("")
                print("rf = ", r_f)
                print(iterations+1, " iterations")
        else:
                r_0 = (epsilon_final/epsilon_start)**(1./(iterations))
                r_f = r_0
        tot = iterations - 1.
        if tol_f is None:
                tol_f = np.sqrt(epsilon_final)*1e-1
        if pen_0 is None:
                pen_0 = np.sqrt(epsilon_start)*np.sqrt(epsilon_final)*1e-1
        if pen_f is None:
                pen_f = epsilon_final*1e-1
        self.fignum = 1
        self.epsilon = epsilon_start
        for i in range(iterations+1):
                print('\n')
                print("BIG iteration",i+1)
                if i>0:
                    self.epsilon *= r_0**((tot-(i-1))/tot)*r_f**((i-1)/tot)
                print("epsilon = ",self.epsilon)
                if i < iterations:
                    self.tolerance = tol_0**((tot-i)/tot)*tol_f**(i/tot)
                    self.penalization = pen_0**((tot-i)/tot)*pen_f**(i/tot)
                    self.Optimization_entropic(iterations = intermediate_iter)
                    self.fignum += 1
                else:
                    self.tolerance = tol_f
                    self.penalization = pen_f
                    self.Optimization_entropic(iterations = intermediate_iter)
                    self.fignum += 1
                self.penalization = 0.
                error = self.value_grad_ext(self.package(self.phi, self.psi, self.h), print_grad = False)[1]
                print("error_real = ", np.linalg.norm(error,self.pow_distance))
                if self.epsilon<=1e-0:
                    test_accuracy = self.test_accuracy()
                    if self.scale and test_size():
                        self.sparse_gridXY = None
                        self.sparse_gridYX = None
                        self.sparse_in_on = False
                        if self.grid_MC:
                            stepX = 1./self.MC_iter['x']
                            stepY = 1./self.MC_iter['y']
                        else:
                            stepX = 2*self.boundX/self.stepsX
                            stepY = 2*self.boundY/self.stepsY
                        if self.impl_psi:
                            step = stepX
                        elif self.impl_phi_h:
                            step = stepY
                        else:
                            step = np.sqrt(stepX*stepY)
                        while self.epsilon < step and test_size():
                            self.doble_grid(axis = 'xy', sparse = False, lift = True)
                            if self.grid_MC:
                                stepX = 1./self.MC_iter['x']
                                stepY = 1./self.MC_iter['y']
                            else:
                                stepX = 2*self.boundX/self.stepsX
                                stepY = 2*self.boundY/self.stepsY
                            if self.impl_psi:
                                step = stepX
                            elif self.impl_phi_h:
                                step = stepY
                            else:
                                step = np.sqrt(stepX*stepY)
                        print("granularity = ", step)
                
                    if self.smart_timing_pool and self.lenX+self.lenY >= 2000:
                        if self.use_pool:
                            print("Pool still on.")
                        else:
                            self.use_pool = True
                            print("Pool on.")
                                       
                
                    if self.sparse:
                        sparsity_limit = 0.33
                        data = self.init_sparse(no_sto = True)
                        super_ratio = (data[0]+data[1])/(2.*self.lenX*self.lenY)
                        if super_ratio <= sparsity_limit and data[0]+data[1] <= self.size_memory_max:
                            data = self.init_sparse(no_sto = False)
                            self.sparse_in_on = True
                        else:
                            self.sparse_gridXY = None
                            self.sparse_gridYX = None
                            self.sparse_in_on = False
                        ratio_x = data[0]*1./(self.lenX*self.lenY)
                        ratio_y = data[1]*1./(self.lenX*self.lenY)
                        print("non_zero_x = ", data[0]," ratio = ", ratio_x)
                        print("non_zero_y = ", data[1]," ratio = ", ratio_y)
                if entrop_error is not None and (test_accuracy <= entrop_error):
                    break
        print('\n')
        print("Adjustement iteration")
        while self.scale and test_size():
            self.doble_grid(axis = 'xy', sparse = False, lift = True) 
        self.penalization = pen_f
        self.tolerance = tol
        self.Optimization_entropic(iterations = intermediate_iter)
        self.fignum += 1
        print('\n')
        print("Final iteration")
        self.penalization = pen
        print("penalization = ", self.penalization)
        print("tolerance = ", self.tolerance)
        self.Optimization_entropic(iterations = intermediate_iter)
        self.fignum += 1
        self.penalization = 0.
        error = self.value_grad_ext(self.package(self.phi, self.psi, self.h), print_grad = False)[1]
        print("error_real = ", np.linalg.norm(error,self.pow_distance))
        if entrop_error is not None:
            gf.check(self.test_accuracy() <= entrop_error)
                
                



    def plot_entropic_proba(self):
        if self.martingale:
            mart = 'mart'
        else:
            mart = ''
        if self.plot_data_save:
            to_save = np.asarray((self.phi, self.psi, self.h))
            np.save('data/phi,psi,h'+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.npy', to_save)
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import LogNorm
        if self.dim == 1:
            Proba_cond = self.compute_proba()
            Proba_cond /= np.reshape(self.mu,(self.lenX,1))
            print(Proba_cond)
            xmin = self.gridX[0][0]
            xmax = self.gridX[self.lenX - 1][0]
            ymin = self.gridY[0][0]
            ymax = self.gridY[self.lenY - 1][0]
            stepX = 2*self.boundX/self.stepsX
            stepY = 2*self.boundY/self.stepsY
            NX = int((xmax-xmin)/stepX+self.zero)+1
            NY = int((ymax-ymin)/stepY+self.zero)+1
            z = np.zeros(NX*NY)
            z = np.reshape(z, (NY, NX))
            for i in range(self.lenX):
                for j in range(self.lenY):
                    X = self.gridX[i][0]
                    Y = self.gridY[j][0]
                    nX = int((X-xmin)/stepX+self.zero)
                    nY = int((Y-ymin)/stepY+self.zero)
                    z[nY][nX] = Proba_cond[i][j]
            if self.plot_data_save:
                to_save = np.asarray(z)
                np.save('new_images/Optimal probability'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+'.npy', to_save)
            else:
                plt.figure(self.fignum)
                plt.imshow(self.zero+z, extent=[xmin, xmax, ymin, ymax],
                origin = 'lower', cmap=cm.hot, norm=LogNorm())
                plt.colorbar()
                if self.plot_save:
                    plt.savefig('new_images/Optimal probability'+str(self.fignum)+self.tag+', dim'+str(self.dim)+', size'+str(self.stepsX)+'.png')
                    plt.close()
                else:
                    plt.show()
        elif self.dim == 2:
            if not self.plot_save:
                raise("Displaying plot is not adapted to dimension 2.")
            for index in range(self.lenX):
                x= self.gridX[index]
                Proba_cond_i = self.compute_proba(index = index)
                minima = np.min(self.gridY,axis = 0)
                maxima = np.max(self.gridY,axis = 0)
                xmin = minima[0]
                xmax = maxima[0]
                ymin = minima[1]
                ymax = maxima[1]
                stepX = 2*self.boundX/self.stepsX
                stepY = 2*self.boundY/self.stepsY
                NX = int((xmax-xmin)/stepX+self.zero)+1
                NY = int((ymax-ymin)/stepY+self.zero)+1
                z = np.zeros(NX*NY)
                z = np.reshape(z, (NY, NX))
                for j in range(self.lenY):
                    Y1 = self.gridY[j][0]
                    Y2 = self.gridY[j][1]
                    nY1 = int((Y1-xmin)/stepY+self.zero)
                    nY2 = int((Y2-ymin)/stepY+self.zero)
                    z[nY2][nY1] = Proba_cond_i[j]
                dataname = 'Optimal_probability'+str(self.fignum)+self.tag+self.method+', dim'+str(self.dim)+', size'+str(self.stepsX)+mart+', x= '+str(x)
                if self.plot_data_save:
                    to_save = np.asarray(z)
                    np.save('new_images/comp_dim2/'+dataname+'.npy', to_save)
                else:
                    plt.figure(self.fignum)
                    plt.imshow(self.zero+z, extent=[xmin, xmax, ymin, ymax],
                    origin = 'lower', cmap=cm.hot, norm=LogNorm())
                    plt.colorbar()
                    if self.plot_save:
                        plt.savefig('new_images/comp_dim2/'+dataname+'.png')
                        plt.close()
                    else:
                        plt.show()
        else:
            print("Dimension not working yet for plot.")
