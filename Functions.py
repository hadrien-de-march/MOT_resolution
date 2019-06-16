# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:09:00 2014

@author: hdemarch
"""

import numpy as np
import timeit
import math
from multiprocessing import Pool

import Generic_Functions as gf
        



def loc_convex_hull(x_0, func, grid, plot = False, 
                    zero = 1e-10, hidden_contact = False,
                debug_mode = False, param_hidden = 1e-10,
                gradient_guess = None):
    d = len(x_0)
    contact = np.array([])
    argcontact = np.array([])
    shift = -np.copy(func)
    bary_coeffs = [-1.]
    infty = 1/zero
    if gradient_guess is None:
        gradient = np.zeros(d)
    else:
        gradient = gradient_guess
        elem_scalar = grid*gradient
        scalar_product =np.sum(elem_scalar, axis=1)
        shift += scalar_product
    if plot:
        fig_num=0
    please_break = False
    while gf.approx_Negative(bary_coeffs):
        if (len(contact)==d+1 or please_break):
            argneg=np.argmin(bary_coeffs)
            contact = np.delete(contact,argneg,0)
            argcontact = np.delete(argcontact,argneg,0)
        please_break = False
        if len(contact)>0:
            p=gf.projection(x_0,np.copy(contact))
            if gf.approx_Equal(p,x_0):
                bary_DATA =gf.barycenter(x_0,contact,argcontact)
                bary_coeffs = bary_DATA['coeffs']
                contact = bary_DATA["vectors"]
                argcontact = bary_DATA["argvectors"]
                please_break = True
                if plot:
                    gf.print_dots(x_0,contact,fig_num)
                    fig_num = fig_num+1
            else:
                diff = x_0-p
                u=diff/np.linalg.norm(diff)
                elem_scalar = grid-p
                elem_scalar *=u
                scalar_product =np.sum(elem_scalar, axis=1)
                scalar_product_plus = np.maximum(scalar_product,zero)
                prohibition = np.maximum(np.minimum(2*zero-scalar_product_plus,zero),0)
                prohibition *= infty**2
                quotient = shift/scalar_product_plus
                quotient += prohibition
        elif len(contact)==0:
            quotient = shift
            u=0*x_0
            p=0*x_0
        if not please_break:
            argument = np.argmin(quotient)
            if debug_mode:
                print("x_0",x_0)
                print("shift before",shift)
                for quo in quotient:
                    print("quotient",quo)
                print("argument",argument)
                for con in contact:
                    print("contact = ",con)
                print("quotient[argument]before", quotient[argument])
            quot_min = quotient[argument]
            if len(contact)>0:
                shift -=  quot_min*scalar_product
                gradient -= quot_min*u
            else:
                shift -=  quot_min
                if debug_mode:
                    print("quotient[argument]", quot_min)
            if debug_mode:
                print("shift after",shift)        
            xmin=grid[argument]
            if len(contact)==0:
                contact = np.array([xmin])
            else:
                contact = np.append(contact,[xmin],axis = 0)
            argcontact = np.append(argcontact,argument)
            if len(contact)==d+1:
                bary_DATA =gf.barycenter(x_0,contact,argcontact)
                bary_coeffs = bary_DATA['coeffs']
                contact = bary_DATA["vectors"]
                if plot:
                    gf.print_dots(x_0,contact,fig_num)
                    fig_num = fig_num+1
                argcontact = bary_DATA["argvectors"]
    nb_contacts = len(contact)
    values = np.zeros(nb_contacts)
    for i in range(nb_contacts):
        values[i] = func[int(argcontact[i])]
    value = np.dot(bary_coeffs,values)
    total_contact = list(contact)
    total_argcontact = list(argcontact)
    if hidden_contact:
        for index in argcontact:
            shift[int(index)] += infty
        argmin = np.argmin(shift)
        while shift[argmin] < param_hidden:
            total_contact.append(grid[int(argmin)])
            total_argcontact.append(argmin)
            shift[argmin] += infty
            argmin = np.argmin(shift)
    return {'value':value , 'contact' : contact , 'argcontact' : argcontact,
            'coeffs' : bary_coeffs , 'total_contact' : total_contact,
            'total_argcontact' : total_argcontact , 'gradient' : gradient }





def Gradient_stochastique(psi_0, grid, cost,
                          tolerance = 1e-4, nbpas=40000, epsilon = 0):
    mu_0 = grid.mu
    nu_0 = grid.nu
    psi = np.copy(psi_0)
    for n in range(nbpas):
        grad = epsilon*psi
        rand = np.random.rand(2)
        prob_cumul_x = 0
        prob_cumul_y = 0
        for i in range(len(grid.grid)):
            prob_cumul_x = prob_cumul_x+mu_0[i]
            if prob_cumul_x>=rand[0]:
                index_x = i
                break
        for i in range(len(grid.grid)):
            prob_cumul_y = prob_cumul_y+nu_0[i]
            if prob_cumul_y>=rand[0]:
                index_y = i
                break
        gamma = 1./(n+1)**(0.6)
        x_0=grid.grid[index_x]
        func = grid.func_grid(lambda y: cost(x_0,y))
        func -= psi
        DATA_convexhull=loc_convex_hull(x_0,func,grid)
        arguments = DATA_convexhull['argcontact']
        Coeff_bary = DATA_convexhull['coeffs']
        for j in range(len(arguments)):
            arg = int(arguments[j])
            grad[arg]= grad[arg]-Coeff_bary[j]*mu_0[arg]
        grad[index_y]=grad[index_y]-1
        psi = psi - gamma*grad
        if n % 40 == 0:
            print(n)
    return {'x': psi}


def auxiliary_phi(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                mu = arg['mu']
                infty = arg['infty']
                martingale = arg['martingale']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX[i],gridY)
                if martingale:
                    Y_X = gridY-np.reshape(gridX[i],(1,dim))
                    arg_Gibbs = (cost_array-psi-np.sum(np.reshape(h[i],(1,dim))*Y_X,
                                                       axis = 1))/epsilon
                else:
                    arg_Gibbs = (cost_array-psi)/epsilon
                maximum = np.max(arg_Gibbs)
                arg_Gibbs -= maximum
                Gibbs = np.sum(np.exp(arg_Gibbs))
                if mu[i] == 0.:
                    logmu_i = -infty
                    print("mu equals zero at x="+str(gridX[i]))
                else:
                    logmu_i = np.log(mu[i])
                phi_i = -epsilon*(logmu_i- (np.log(Gibbs)+maximum) )
                
                time_comp = timeit.default_timer()-t_0
                return {'i' : i , 'time_comp' : time_comp, 'phi_i' : phi_i}


def auxiliary_psi(j, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                nu = arg['nu']
                infty = arg['infty']
                martingale = arg['martingale']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX,gridY[j])
                if martingale:
                    Y_X = np.reshape(gridY[j],(1,dim))-gridX
                    arg_Gibbs = (cost_array-phi-np.sum(h*Y_X, axis = 1))/epsilon
                else:
                    arg_Gibbs = (cost_array-phi)/epsilon
                maximum = np.max(arg_Gibbs)
                arg_Gibbs -= maximum
                Gibbs = np.sum(np.exp(arg_Gibbs))
                if nu[j] == 0.:
                    lognu_j = -infty
                    print("nu equals zero at y="+str(gridY[j]))
                else:
                    lognu_j = np.log(nu[j])
                psi_j = -epsilon*(lognu_j- (np.log(Gibbs)+maximum) )
                
                time_comp = timeit.default_timer()-t_0
                return {'j' : j , 'time_comp' : time_comp, 'psi_j' : psi_j}


def auxiliary_h(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                psi = arg['psi']
                lenY = arg['lenY']
                hi_0 = arg['h'][i]
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                minimize = arg['minimize']
                nmax_Newton_h = arg['nmax_Newton_h']
                calc_phi = arg['calc_phi']
                mu = arg['mu']
                infty = arg['infty']
                zero = arg['zero']
                lenX = arg['lenX']
                debug_mode = arg['debug_mode']
                newNewton = arg['newNewton']
                pow_distance = arg['pow_distance']
                tol_Newton_h = arg['tol_Newton_h']
                precise_h = arg['precise_h']
                safe_solving = arg['safe_solving']
                if safe_solving:
                    hardcore_compute = arg['restrict_compute'][i] 
                    previous_error = arg['previous_error'][i] 
                else:
                    hardcore_compute = False
                
                t_0 = timeit.default_timer()
                if precise_h:
                    tol_Newton_h *= min(1., epsilon)
                cost_array = cost(gridX[i],gridY)
                Y_X = gridY-np.reshape(gridX[i],(1,dim))
                
                if calc_phi:#???
                    mu_i = max(mu[i], zero/lenX)
                    if mu[i] == 0.:
                        logmu_i = -infty
                    else:
                        logmu_i = np.log(mu[i])
                    DATA = {'phi_i' : None, 'hi' : None, 'gradient' : None,
                            'result_found' : False, 'result_opt' : None}
                    def phi_from_h(hi):
                        arg_Gibbs = (cost_array-psi-np.sum(np.reshape(hi,(1,dim))*Y_X,
                                                           axis = 1))/epsilon
                        maximum = np.max(arg_Gibbs)
                        arg_Gibbs -= maximum
                        Gibbs = np.sum(np.exp(arg_Gibbs))
                        phi_i_loc = -epsilon*(logmu_i- (np.log(Gibbs)+maximum) )
                        return phi_i_loc
                else:
                    arg_Gibbs_0 = (cost_array-psi-np.sum(np.reshape(hi_0,(1,dim))*Y_X,
                                                         axis = 1))/epsilon
                    arg_Gibbs_0 -= np.max(arg_Gibbs_0)
                    Gibbs_0 = np.exp(arg_Gibbs_0)
                    Z_0 = np.sum(Gibbs_0)
                
                def value_h(hi):
                    if DATA['result_found']:
                        return (0, 0*hi)
                    if calc_phi:
                        DATA['hi'] = hi
                        DATA['phi_i'] = phi_from_h(hi)
                        phi_i = DATA['phi_i']
                        arg_Gibbs = (cost_array-psi-phi_i-np.sum(np.reshape(hi,(1,dim))*Y_X,
                                                 axis = 1))/epsilon
                        Gibbs = np.exp(arg_Gibbs)
                        if debug_mode == 4:
                            gf.check(gf.approx_Equal(np.sum(Gibbs),
                                                     mu[i]), (np.sum(Gibbs), mu[i]))
                        value = (epsilon+phi_i)*mu_i
                        if debug_mode == 4:
                            gf.check(np.sum(Gibbs), mu_i)
                        gradient = -np.sum(np.reshape(Gibbs,
                                            (lenY, 1))*Y_X, axis = 0)
                        DATA['gradient'] = gradient
                    else:
                        arg_Gibbs = arg_Gibbs_0 -np.sum(np.reshape(hi,(1,dim))*Y_X, axis = 1)/epsilon
                        Gibbs = np.exp(arg_Gibbs)/Z_0
                        value = epsilon*np.sum(Gibbs)
                        gradient = -np.sum(np.reshape(Gibbs,
                                                (lenY, 1))*Y_X, axis = 0)
                    if not newNewton:
                        if np.linalg.norm(gradient, 1) <= tol_Newton_h:
                            DATA['result_found'] = True
                            DATA['result_opt'] = hi
                            if debug_mode == 5:
                                print("result found.")
                    val = value/mu_i
                    grad = gradient/mu_i
                    return (val, grad)
                def hessian_h(hi):
                    if calc_phi:
                        phi_i = DATA['phi_i']
                        arg_Gibbs = (cost_array-psi-phi_i-np.sum(np.reshape(hi,(1,dim))*Y_X,
                                                axis = 1))/epsilon
                        Gibbs = np.exp(arg_Gibbs)
                        hessian = 1/epsilon*np.sum(np.reshape(Gibbs,
                                            (lenY,1,1))*np.reshape(Y_X,
                                     (lenY,dim,1))*np.reshape(Y_X,
                                     (lenY,1,dim)), axis = 0)
                        if gf.approx_Equal(DATA['hi'], hi, tolerance = tol_Newton_h):
                            gradient = DATA['gradient']
                        else:
                            gradient = value_h(hi)[1]
                        hessian -= 1/epsilon*1/mu_i*np.reshape(gradient,
                                            (dim, 1))*np.reshape(gradient,(1, dim))
                    else:
                        arg_Gibbs = arg_Gibbs_0 -np.sum(np.reshape(hi,(1,dim))*Y_X,
                                                        axis = 1)/epsilon
                        Gibbs = np.exp(arg_Gibbs)/Z_0
                        hessian = 1/epsilon*np.sum(np.reshape(Gibbs,
                                            (lenY,1,1))*np.reshape(Y_X,
                                     (lenY,dim,1))*np.reshape(Y_X,
                                     (lenY,1,dim)),axis = 0)
                    return hessian/mu_i
                
                x0 = hi_0
                if debug_mode == 5:
                    disp = True
                else:
                    disp = False
                if not hardcore_compute:
                    if newNewton:
                        result = gf.Newton(value_h, hessian_h, x0 = x0, tol = tol_Newton_h,
                                       maxiter= nmax_Newton_h, disp= disp,
                                       pow_distance = pow_distance,
                                       order_hess = 1./epsilon)
                    else:
                        result = minimize(value_h, x0 = x0,
                                        method='Newton-CG',
                                        jac=True, hess = hessian_h,
                                        tol=tol_Newton_h,
                                        options={'maxiter': nmax_Newton_h,
                                                 'xtol' : tol_Newton_h,
                                                 'disp' : disp})
                elif hardcore_compute:
                    epsilon_sto = epsilon
                    epsilon_end = epsilon
                    epsilon_start = previous_error
                    if epsilon_end < epsilon_start:
                        nb_iter = int(2+np.rint(-np.log2(epsilon_end/epsilon_start)))
                        d_eps = (epsilon_end/epsilon_start)**(1./(nb_iter-1.))
                    else:
                        nb_iter = int(2+np.rint(np.log2(epsilon_end/epsilon_start)))
                        d_eps = (epsilon_end/epsilon_start)**(1./(nb_iter-1.))
                    for step in range(nb_iter):
                        epsilon = epsilon_start*d_eps**step
                        if newNewton:
                            result = gf.Newton(value_h, hessian_h, x0 = x0, tol = tol_Newton_h,
                                       maxiter= nmax_Newton_h, disp= disp,
                                       pow_distance = pow_distance,
                                       order_hess = 1./epsilon)
                            x0 = result['x']
                        else:
                            result = minimize(value_h, x0 = x0,
                                        method='Newton-CG',
                                        jac=True, hess = hessian_h,
                                        tol=tol_Newton_h,
                                        options={'maxiter': nmax_Newton_h,
                                                 'xtol' : tol_Newton_h,
                                                 'disp' : disp})
                        if calc_phi:
                            if newNewton:
                                x0 = result['x']
                            elif DATA['result_found']:
                                x0 = DATA['result_opt']
                            else:
                                x0 = result.x

                    epsilon = epsilon_sto
                    
                time_comp = timeit.default_timer()-t_0
                if calc_phi:
                    if newNewton:
                        hi = result['x']
                    elif DATA['result_found']:
                        hi = DATA['result_opt']
                    else:
                        hi = result.x
                    if gf.approx_Equal(hi, DATA['hi'], tolerance = zero**2):
                        phi_i = DATA['phi_i']
                    else:
                        phi_i = phi_from_h(hi)

                else:
                    phi_i = None
                    if newNewton:
                        hi += result['x']
                    elif DATA['result_found']:
                        hi += DATA['result_opt']
                    else:
                        hi += result.x
                if debug_mode == 5:
                    print("grad norm wololo", np.linalg.norm(DATA['gradient']))
                return {'hi' : hi , 'i' : i , 'time_comp' : time_comp, 'phi_i' : phi_i,
                        'gradient' : DATA['gradient']/mu_i}



def auxiliary_h_index(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                psi = arg['psi']
                lenY = arg['lenY']
                hi_0 = arg['h'][i]
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                minimize = arg['minimize']
                nmax_Newton_h = arg['nmax_Newton_h']
                calc_phi = arg['calc_phi']
                mu = arg['mu']
                infty = arg['infty']
                zero = arg['zero']
                lenX = arg['lenX']
                debug_mode = arg['debug_mode']
                newNewton = arg['newNewton']
                pow_distance = arg['pow_distance']
                tol_Newton_h = arg['tol_Newton_h']
                index = arg['index']
                pen_h = 0.#1e-10#???
                
                hi_index_0 = hi_0[index]
                x_i = gridX[i]
                
                t_0 = timeit.default_timer()
                cost_array = cost(x_i, gridY)
                DATA = {'phi_i' : None, 'gradient' : None, 'hi': hi_0, 'hi_index' : None,
                            'result_found' : False, 'result_opt' : None}
                mu_i = max(mu[i], zero/lenX)
                if mu[i] == 0.:
                        logmu_i = -infty
                else:
                        logmu_i = np.log(mu[i])
                if calc_phi:
                    Y_X = gridY-np.reshape(x_i,(1,dim))
                    Y_X_index = np.reshape(Y_X[:, index], (lenY, 1))
                    def phi_from_h(hi):
                        arg_Gibbs = (cost_array-psi-np.sum(np.reshape(hi,(1,dim))*Y_X,
                                                           axis = 1))/epsilon
                        maximum = np.max(arg_Gibbs)
                        arg_Gibbs -= maximum
                        Gibbs = np.sum(np.exp(arg_Gibbs))
                        phi_i_loc = -epsilon*(logmu_i- (np.log(Gibbs)+maximum) )
                        return phi_i_loc

                else:
                    arg_Gibbs = (cost_array-psi-np.sum(np.reshape(hi_0,(1,dim))*gridY,
                                                           axis = 1))/epsilon
                    maximum = np.max(arg_Gibbs)
                    arg_Gibbs -= maximum
                    Gibbs = np.sum(np.exp(arg_Gibbs))
                    phi_h_x = -epsilon*(logmu_i- (np.log(Gibbs)+maximum) )
                    gridY_index = np.reshape(gridY[:, index], (lenY, 1))
                    x_i_index = x_i[index]
                
                def value_h(hi_index):
                    if DATA['result_found']:
                        return (0, 0*hi_index)
                    DATA['hi'][index] = hi_index[0]
                    hi = DATA['hi']
                    DATA['hi_index'] = hi_index
                    if calc_phi:
                        DATA['phi_i'] = phi_from_h(hi)
                        phi_i = DATA['phi_i']
                        arg_Gibbs = (cost_array-psi-phi_i-np.sum(np.reshape(hi, (1, dim))*Y_X,
                                                 axis = 1))/epsilon
                    else:
                        arg_Gibbs = (cost_array-psi-phi_h_x-np.sum(np.reshape(hi, (1, dim))*gridY,
                                                 axis = 1))/epsilon
                    Gibbs = np.exp(arg_Gibbs)
                    if debug_mode == 4 and calc_phi:
                            gf.check(gf.approx_Equal(np.sum(Gibbs),
                                                     mu[i]), (np.sum(Gibbs), mu[i]))
                    if calc_phi:
                        value = (epsilon+phi_i)*mu_i
                        gradient = -np.sum(np.reshape(Gibbs,
                                            (lenY, 1))*Y_X_index, axis = 0)
                    else:
                        value = mu_i*(phi_h_x+np.dot(hi, x_i))+epsilon*np.sum(Gibbs)
                        gradient = mu_i*x_i_index-np.sum(np.reshape(Gibbs,
                                            (lenY, 1))*gridY_index, axis = 0)
                    DATA['gradient'] = gradient
                    if not newNewton:
                        if np.linalg.norm(gradient, 1) <= tol_Newton_h:
                            DATA['result_found'] = True
                            DATA['result_opt'] = hi_index
                            if debug_mode == 5:
                                print("result found.")
                    val = value/mu_i
                    grad = gradient/mu_i
                    val += pen_h*0.5*np.dot(hi_index, hi_index)
                    grad += pen_h*hi_index
                    return (val, grad)
                
                def hessian_h(hi_index):
                    DATA['hi'][index] = hi_index[0]
                    hi = DATA['hi']
                    
                    if calc_phi:
                        phi_i = DATA['phi_i']
                        arg_Gibbs = (cost_array-psi-phi_i-np.sum(np.reshape(hi,(1,dim))*Y_X,
                                                axis = 1))/epsilon
                        multiplied_by_h_index = Y_X_index
                    else:
                        arg_Gibbs = (cost_array-psi-phi_h_x-np.sum(np.reshape(hi,(1,dim))*gridY,
                                                axis = 1))/epsilon
                        multiplied_by_h_index = gridY_index
                    Gibbs = np.exp(arg_Gibbs)
                    hess = 1/epsilon*np.sum(np.reshape(Gibbs,
                                            (lenY,1,1))*np.reshape(multiplied_by_h_index,
                                     (lenY,1,1))*np.reshape(multiplied_by_h_index,
                                     (lenY,1,1)), axis = 0)
                    if calc_phi:
                        if gf.approx_Equal(DATA['hi_index'], hi_index, tolerance = tol_Newton_h):
                            gradient = DATA['gradient']
                        else:
                            gradient = value_h(hi_index)[1]
                        hess -= 1/epsilon*1/mu_i*np.reshape(gradient,
                                            (1, 1))*np.reshape(gradient,(1, 1))
                    hessian = hess/mu_i
                    hessian += pen_h*np.eye(1)
                    return hessian
                
                x0 = np.array([hi_index_0])
                if debug_mode == 5:
                    disp = True
                else:
                    disp = False
                if newNewton:
#                    result = gf.Newton(value_h, hessian_h, x0 = x0, tol = tol_Newton_h,
#                                       maxiter= nmax_Newton_h, disp= disp,
#                                       pow_distance = pow_distance,
#                                       order_hess = 1./epsilon, invertor = zero*epsilon)???
                    result = gf.Newton_CG(value_grad, x0 = x0, hess = hessian_h,
                                          tol = tol_Newton_h, maxiter = nmax_Newton_h,
                                          disp = disp, pow_distance = pow_distance,
                                          maxiter_CG = self.dim,
                                          check_cond = False, disp_CG = disp,
                                          debug_mode_CG = False, print_line_search = False,
                                          order_hess = 1.)
                else:
                    result = minimize(value_h, x0 = x0,
                                        method='Newton-CG',
                                        jac=True, hess = hessian_h,
                                        tol=tol_Newton_h,
                                        options={'maxiter': nmax_Newton_h,
                                                 'xtol' : tol_Newton_h,
                                                 'disp' : disp})
                
                time_comp = timeit.default_timer()-t_0
                hi = hi_0
                if calc_phi:
                    if newNewton:
                        hi[index] = result['x']
                    elif DATA['result_found']:
                        hi[index] = DATA['result_opt']
                    else:
                        hi[index] = result.x
                    if gf.approx_Equal(hi, DATA['hi'], tolerance = zero**2):
                        phi_i = DATA['phi_i']
                    else:
                        phi_i = phi_from_h(hi)

                else:
                    if newNewton:
                        hi[index] = result['x']
                    elif DATA['result_found']:
                        hi[index] = DATA['result_opt']
                    else:
                        hi[index] = result.x
                    phi_i = phi_h_x+np.dot(hi, x_i)
                if debug_mode == 5:
                    print("grad norm wololo", np.linalg.norm(DATA['gradient']))
                return {'hi' : hi , 'i' : i , 'time_comp' : time_comp, 'phi_i' : phi_i}


def auxiliary_cost_minimax(i, arg):
                gridY = arg['gridY']
                cost = arg['cost']
                gridX = arg['gridX']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX[i],gridY)
                min_cost = np.amin(cost_array)
                max_cost = np.amax(cost_array)
                time_comp = timeit.default_timer()-t_0
                return {'i' : i , 'time_comp' : time_comp, 'min_cost' : min_cost,
                        'max_cost': max_cost}

            
def auxiliary_vg_phi(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                martingale = arg['martingale']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX[i],gridY)
                if martingale:
                    Y_X = gridY-np.reshape(gridX[i],(1,dim))
                    arg_Gibbs = (cost_array-phi[i]-psi-np.sum(np.reshape(h[i],
                             (1,dim))*Y_X, axis = 1))/epsilon
                else:
                    arg_Gibbs = (cost_array-phi[i]-psi)/epsilon
                Gibbs = np.sum(np.exp(arg_Gibbs))
                gradient = -Gibbs
                value = epsilon*Gibbs
                time_comp = timeit.default_timer()-t_0
                return {'i' : i , 'time_comp' : time_comp, 'value' : value,
                        'gradient': gradient}





def auxiliary_vg_psi(j, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                martingale = arg['martingale']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX,gridY[j])
                if martingale:
                    Y_X = np.reshape(gridY[j],(1,dim))-gridX
                    arg_Gibbs = (cost_array-phi-psi[j]-np.sum(h*Y_X, axis = 1))/epsilon
                else:
                    arg_Gibbs = (cost_array-phi-psi[j])/epsilon
                Gibbs = np.sum(np.exp(arg_Gibbs))
                gradient = -Gibbs
                value = epsilon*Gibbs
                time_comp = timeit.default_timer()-t_0
                return {'j' : j , 'time_comp' : time_comp, 'value' : value,
                        'gradient': gradient}


def auxiliary_vg_h(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                lenY = arg['lenY']
                
                t_0 = timeit.default_timer()
                Y_X = gridY-np.reshape(gridX[i],(1,dim))
                cost_array = cost(gridX[i],gridY)
                if h is None:
                    arg_Gibbs = (cost_array-phi[i]-psi)/epsilon
                else:
                    arg_Gibbs = (cost_array-phi[i]-psi-np.sum(np.reshape(h[i],(1,dim))*Y_X,
                             axis = 1))/epsilon
                Gibbs = np.exp(arg_Gibbs)
                gradient = -np.sum(np.reshape(Gibbs, (lenY,1))*Y_X, axis = 0)
                value = epsilon*np.sum(Gibbs)
                time_comp = timeit.default_timer()-t_0
                return {'i' : i , 'time_comp' : time_comp, 'value' : value,
                        'gradient': gradient}








               
def auxiliary_hess_phi(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                p_phi = arg['p_phi']
                p_psi = arg['p_psi']
                p_h = arg['p_h']
                lenY = arg['lenY']
                martingale = arg['martingale']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX[i],gridY)
                if martingale:
                    Y_X = gridY-np.reshape(gridX[i],(1,dim))
                    arg_Gibbs = (cost_array-phi[i]-psi-np.sum(np.reshape(h[i],
                                 (1,dim))*Y_X, axis = 1))/epsilon
                else:
                    arg_Gibbs = (cost_array-phi[i]-psi)/epsilon
                Gibbs = np.exp(arg_Gibbs)
                hess_p = 0.
                if p_phi is not None:
                    hess_p += np.sum(p_phi[i]*Gibbs)
                if p_psi is not None:
                    hess_p += np.sum(p_psi*Gibbs)
                if martingale and p_h is not None:
                    hess_p += np.sum(np.reshape(p_h[i], (1, dim))*Y_X*np.reshape(Gibbs, (lenY, 1)))
                hess_p /= epsilon
                time_comp = timeit.default_timer()-t_0
                return {'i' : i , 'time_comp' : time_comp, 'hess_p' : hess_p}



def auxiliary_hess_psi(j, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                p_phi = arg['p_phi']
                p_psi = arg['p_psi']
                p_h = arg['p_h']
                lenX = arg['lenX']
                martingale = arg['martingale']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX,gridY[j])
                if martingale and p_h is not None:
                    Y_X = np.reshape(gridY[j],(1,dim))-gridX
                    arg_Gibbs = arg_Gibbs = (cost_array-phi-psi[j]-np.sum(h*Y_X,
                                             axis = 1))/epsilon
                else:
                    arg_Gibbs = arg_Gibbs = (cost_array-phi-psi[j])/epsilon
                Gibbs = np.exp(arg_Gibbs)
                hess_p = 0.
                if p_phi is not None:
                    hess_p += np.sum(p_phi*Gibbs)
                if p_psi is not None:
                    hess_p += np.sum(p_psi[j]*Gibbs)
                if martingale and p_h is not None:
                    hess_p += np.sum(p_h*Y_X*np.reshape(Gibbs, (lenX, 1)))
                hess_p /= epsilon
                time_comp = timeit.default_timer()-t_0
                return {'j' : j , 'time_comp' : time_comp, 'hess_p' : hess_p}




def auxiliary_hess_h(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                p_phi = arg['p_phi']
                p_psi = arg['p_psi']
                p_h = arg['p_h']
                lenY = arg['lenY']
                
                t_0 = timeit.default_timer()
                Y_X = gridY-np.reshape(gridX[i],(1,dim))
                cost_array = cost(gridX[i],gridY)
                arg_Gibbs = (cost_array-phi[i]-psi-np.sum(np.reshape(h[i],(1,dim))*Y_X, axis = 1))/epsilon
                Gibbs = np.exp(arg_Gibbs)
                hess_p = np.zeros(lenY)
                if p_phi is not None:
                    hess_p += p_phi[i]*Gibbs
                if p_psi is not None:
                    hess_p += p_psi*Gibbs
                if p_h is not None:
                    hess_p += np.sum(np.reshape(p_h[i], (1, dim))*Y_X*np.reshape(Gibbs,
                                 (lenY, 1)), axis = 1)
                hess_p = np.sum(np.reshape(hess_p, (lenY, 1))*Y_X, axis = 0)
                hess_p /= epsilon
                time_comp = timeit.default_timer()-t_0
                return {'i' : i , 'time_comp' : time_comp, 'hess_p' : hess_p}




def auxiliary_hess_h_inv(i, arg):
                gridX = arg['gridX']
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                dim = arg['dim']
                lenX = arg['lenX']
                lenY = arg['lenY']
                zero = arg['zero']
                include_phi = arg['include_phi']
#                mu = arg['mu']
                
                t_0 = timeit.default_timer()
                Y_X = gridY-np.reshape(gridX[i],(1,dim))
                cost_array = cost(gridX[i],gridY)
                arg_Gibbs = (cost_array-phi[i]-psi-np.sum(np.reshape(h[i],(1,dim))*Y_X,
                             axis = 1))/epsilon
                Gibbs = np.exp(arg_Gibbs)
#                print("phi in = ", phi)
#                print("psi in = ", psi)
#                print("h in = ", h)
#                print("Gibbs = ", Gibbs)
                hess_h_h = np.reshape(Y_X, (lenY, dim, 1))*np.reshape(Gibbs,
                        (lenY, 1, 1))*np.reshape(Y_X, (lenY, 1, dim))
                hess_h_h = np.sum(hess_h_h, axis = 0)
                hess_h_h /= epsilon
                dim_final = dim
                if include_phi:
                    hess_phi_phi = np.array([np.sum(Gibbs)/epsilon])
#                    gf.check(gf.approx_Equal(np.sum(Gibbs), mu[i]), (np.sum(Gibbs), mu[i]))
                    cross_phi_h = np.reshape(Gibbs, (lenY, 1))*Y_X
                    cross_phi_h = np.sum(cross_phi_h, axis = 0)
                    cross_phi_h /= epsilon
#                    print("test is zero", cross_phi_h)
                    up = np.concatenate((hess_phi_phi, cross_phi_h))
                    up = np.reshape(up, (1,dim+1))
                    down = np.concatenate((cross_phi_h.reshape(dim, 1), hess_h_h), axis = 1)
                    hess_h_h = np.concatenate((up,down))
                    dim_final +=1
                hess_h_inv = np.linalg.inv(hess_h_h+zero/lenX*np.eye(dim_final))
                time_comp = timeit.default_timer()-t_0
                return {'i' : i , 'time_comp' : time_comp, 'hess_h_inv' : hess_h_inv}



def auxiliary_diag_hess_psi(j, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                lenX = arg['lenX']
                mu = arg['mu']
                martingale = arg['martingale']
                hess_h_inv = arg['hess_h_inv']
                include_phi = arg['include_phi']
                zero = arg['zero']
                no_impl = arg['no_impl']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX,gridY[j])
                if martingale:
                    Y_X = np.reshape(gridY[j],(1,dim))-gridX
                    arg_Gibbs = arg_Gibbs = (cost_array-phi-psi[j]-np.sum(h*Y_X,
                                             axis = 1))/epsilon
                else:
                    arg_Gibbs = arg_Gibbs = (cost_array-phi-psi[j])/epsilon
                Gibbs = np.exp(arg_Gibbs)
                if not no_impl:
                    if not martingale or not include_phi:
                        diag = -np.sum(Gibbs*Gibbs/np.maximum(mu, zero/lenX))/epsilon
                    else:
                        diag = 0.
                    if martingale:
                        if include_phi:
                            ones = np.zeros(lenX)+1.
                            ones = np.reshape(ones, (lenX, 1))
                            Y_X_use = np.concatenate((ones, Y_X), axis = 1)
                            dim_use = dim+1
                        else:
                            Y_X_use = Y_X
                            dim_use = dim
                        Gibbs_Y_X = np.reshape(Gibbs,(lenX, 1))*Y_X_use/epsilon
                        Gibbs_Y_X_left = np.reshape(Gibbs_Y_X,(lenX, dim_use, 1))
                        Gibbs_Y_X_right = np.reshape(Gibbs_Y_X,(lenX, 1, dim_use))
                        diag -= np.sum(Gibbs_Y_X_left*hess_h_inv*Gibbs_Y_X_right)
                else:
                    diag = 0.
                diag += np.sum(Gibbs)/epsilon
                time_comp = timeit.default_timer()-t_0
                return {'j' : j , 'time_comp' : time_comp, 'diag' : diag}
                
             
                
def auxiliary_diag_hess_phi_h(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                nu = arg['nu']
                lenY = arg['lenY']
                martingale = arg['martingale']
                zero = arg['zero']
                no_impl = arg['no_impl']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX[i],gridY)
                if martingale:
                    Y_X = gridY-np.reshape(gridX[i],(1,dim))
                    arg_Gibbs = (cost_array-phi[i]-psi-np.sum(np.reshape(h[i],
                                 (1,dim))*Y_X, axis = 1))/epsilon
                else:
                    arg_Gibbs = (cost_array-phi[i]-psi)/epsilon
                Gibbs = np.exp(arg_Gibbs)
                if not no_impl:
                    diag_phi = -epsilon*np.sum(Gibbs*Gibbs/np.maximum(nu, zero/lenY))
                    if martingale:
                        diag_h = np.sum(np.reshape(Gibbs**2/np.maximum(nu, zero/lenY), (lenY, 1))*Y_X**2, axis = 0)
                    else:
                        diag_h = None
                else:
                    diag_phi = 0
                    if martingale:
                        diag_h = np.zeros(dim)
                    else:
                        diag_h = None
                diag_phi += np.sum(Gibbs)/epsilon
                if martingale:
                    diag_h += np.sum(np.reshape(Gibbs, (lenY, 1))*Y_X**2, axis = 0)
                time_comp = timeit.default_timer()-t_0
                return {'i' : i , 'time_comp' : time_comp, 'diag_phi' : diag_phi, 'diag_h' : diag_h}
                
                

def auxiliary_expectation_cost(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                h = arg['h']
                phi_test = arg['phi_test']
                psi_test = arg['psi_test']
                h_test = arg['h_test']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                martingale = arg['martingale']
                dual = arg['dual']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX[i],gridY)
                if martingale:
                    Y_X = gridY-np.reshape(gridX[i],(1,dim))
                    h_times = np.sum(np.reshape(h[i],(1,dim))*Y_X, axis = 1)
                    Delta = cost_array-phi[i]-psi-h_times
                    arg_Gibbs = Delta/epsilon
                else:
                    Delta = cost_array-phi[i]-psi
                    arg_Gibbs = Delta/epsilon
                Gibbs = np.exp(arg_Gibbs)
                expectation_i = np.sum(Gibbs*cost_array)
                mass = np.sum(Gibbs)
                if dual:
                    expect_dual_i = np.sum(Gibbs*psi_test)
                    expect_dual_i += mass*phi_test[i]
                    if martingale:
                        h_test_times = np.sum(np.reshape(h_test[i],(1,dim))*Y_X, axis = 1)
                        expect_dual_i += np.sum(Gibbs*h_test_times)
                    gap_phi = -mass*np.max(cost_array-phi_test[i]-psi_test-h_test_times)
                else:
                    expect_dual_i = None
                    gap_phi = None
                time_comp = timeit.default_timer()-t_0
                return {'time_comp' : time_comp, 'expectation_i' : expectation_i,
                        'expect_dual_i' : expect_dual_i, 'mass' : mass, 'gap_phi' : gap_phi}
                
                


def sparsify(vector, structure_matrix, index, dim = None, length = None):
    if dim is None:
        sparse_vector_matrix = structure_matrix[index].multiply(vector)
        sparse_vector = np.array(sparse_vector_matrix.data)
    else:
        sparse_vector = np.zeros(dim*length)
        sparse_vector = np.reshape(sparse_vector, (dim, length))
        for d in range(dim):
            sparse_vector[d] = sparsify(vector[d], structure_matrix, index)
        sparse_vector = sparse_vector.transpose()
    return sparse_vector


                


def auxiliary_sparse_grid(i, arg):
                gridY = arg['gridY']
                cost_raw = arg['cost']
                min_cost = arg['min_cost']
                max_cost = arg['max_cost']
                def cost(x, y):
                    cost_computed = cost_raw(x, y)
                    if min_cost == max_cost:
                        return 0.*cost_computed
                    else:
                        return (cost_computed+min_cost)/(max_cost-min_cost)
                phi = arg['phi']
                psi = arg['psi']
                mu = arg['mu']
                nu = arg['nu']
                h = arg['h']
                epsilon = arg['epsilon']
                gridX = arg['gridX']
                dim = arg['dim']
                martingale = arg['martingale']
                proba_min = arg['proba_min']
                no_sto = arg['no_sto']
                
                t_0 = timeit.default_timer()
                cost_array = cost(gridX[i],gridY)
                if martingale:
                    Y_X = gridY-np.reshape(gridX[i],(1,dim))
                    h_times = np.sum(np.reshape(h[i],(1,dim))*Y_X, axis = 1)
                    Delta = cost_array-phi[i]-psi-h_times
                    arg_Gibbs = Delta/epsilon
                else:
                    Delta = cost_array-phi[i]-psi
                    arg_Gibbs = Delta/epsilon
                Gibbs = np.exp(arg_Gibbs)
                Yx = np.where(Gibbs/mu[i]>= proba_min)[0]
                Yy = np.where(Gibbs/nu>= proba_min)[0]
                non_zero_x = len(Yx)
                non_zero_y = len(Yy)
                Xx = np.zeros(non_zero_x)+i
                Xy = np.zeros(non_zero_y)+i
                if no_sto:
                    Xx = None
                    Xy = None
                    Yx = None
                    Yy = None
                else:
                    Xx = list(Xx)
                    Xy = list(Xy)
                    Yx = list(Yx)
                    Yy = list(Yy)
                time_comp = timeit.default_timer()-t_0
                return {'time_comp' : time_comp, 'Xx' : Xx,
                        'Yx' : Yx, 'Xy' : Xy, 'Yy' : Yy,
                        'non_zero_x' : non_zero_x, 'non_zero_y' : non_zero_y}





def auxiliary_Tan(i, arg):
            gridX = arg[ 'gridX']
            gridY = arg[ 'gridY']
            cost_raw = arg['cost']
            min_cost = arg['min_cost']
            max_cost = arg['max_cost']
            def cost(x, y):
                cost_computed = cost_raw(x, y)
                if min_cost == max_cost:
                    return 0.*cost_computed
                else:
                    return (cost_computed+min_cost)/(max_cost-min_cost)
            psi= arg['psi']
            calc_Gamma= arg['calc_Gamma']
            zero= arg['zero']
            param_hidden= arg['param_hidden']
            loc_convex_hull= arg['loc_convex_hull']
            x = gridX[i]
            h_i = arg['h'][i]
            func = cost(x, gridY)
            func -= psi
            
            t_0 = timeit.default_timer()
            DATA = loc_convex_hull(x,func, gridY, zero = zero ,
                               hidden_contact = calc_Gamma,
                               param_hidden = param_hidden,
                               gradient_guess = h_i)
            time_CH = timeit.default_timer()-t_0
            
            time_shit = 0.
            
            
            argvectors = np.array(DATA['argcontact'])
            value_loc=DATA['value']
            gradient_loc = DATA['gradient']
            coeffs = np.array(DATA['coeffs'])
            total_argcontact = []
            total_contact = []
            if calc_Gamma:
                total_argcontact = DATA['total_argcontact']
                total_contact = DATA['total_contact']
            return {'value' : value_loc, 'argcontact' : argvectors, 'coeffs' : coeffs,
                    'total_contact' : total_contact, 'total_argcontact' : total_argcontact,
                    'gradient' : gradient_loc, 'time_comp' :  time_CH ,
                    'index' : i, 'time_shit' : time_shit }
            

        
def auxiliary_package(arg):
            i_0= arg['i_0']
            size_job= arg['size_job']
            auxiliary = arg['auxiliary']
            total_len= arg['total_len']
            if arg['sparse_is_on']:
                sparsify = arg['sparsify']
                if arg['axis']=='x':
                    structure_matrix = arg["sparse_gridXY"]
                    if 'psi' in arg.keys():
                        psi = arg['psi']
                    if 'gridY_sparse' in arg.keys():
                        gridY_sparse = arg['gridY_sparse']                   
                    if 'nu' in arg.keys():
                        nu = arg['nu']                   
                    if 'p_psi' in arg.keys():
                        p_psi = arg['p_psi']
                elif arg['axis']=='y':
                    structure_matrix = arg["sparse_gridYX"]
                    if 'phi' in arg.keys():
                        phi = arg['phi']
                    if 'h_sparse' in arg.keys() and arg['martingale']:
                        h_sparse = arg['h_sparse']
                    if 'gridX_sparse' in arg.keys():
                        gridX_sparse = arg['gridX_sparse']
                    if 'mu' in arg.keys():
                        mu = arg['mu']                   
                    if 'p_phi' in arg.keys():
                        p_phi = arg['p_phi']                   
                    if 'p_h_sparse' in arg.keys():
                        p_h_sparse = arg['p_h_sparse']
                else:
                    raise("no "+arg['axis']+" axis")

            package = []
            for k in range(size_job):
                index = k+i_0
                if index >= total_len:
                    break
                if arg['sparse_is_on']:
                    if arg['axis']=='x':
                        if 'psi' in arg.keys():
                            arg['psi'] = sparsify(psi, structure_matrix, k)
                            lenY = len(arg['psi'])
                            arg['lenY'] = lenY
                        if 'gridY_sparse' in arg.keys():
                            arg['gridY'] = sparsify(gridY_sparse, structure_matrix,
                                                    k, dim = arg['dim'], length = lenY)            
                        if 'nu' in arg.keys():
                            arg['nu'] = sparsify(nu, structure_matrix, k)           
                        if 'p_psi' in arg.keys():
                            arg['p_psi'] = sparsify(p_psi, structure_matrix, k)
                    elif arg['axis']=='y':
                        if 'phi' in arg.keys():
                            arg['phi'] = sparsify(phi, structure_matrix, k)
                            lenX = len(arg['phi'])
                            arg['lenX'] = lenX
                        if 'h_sparse' in arg.keys() and arg['martingale']:
                            arg['h'] = sparsify(h_sparse, structure_matrix,
                                                    k, dim = arg['dim'], length = lenX) 
                        if 'gridX_sparse' in arg.keys():
                            arg['gridX'] = sparsify(gridX_sparse, structure_matrix,
                                                    k, dim = arg['dim'], length = lenX) 
                        if 'mu' in arg.keys():
                            arg['mu'] = sparsify(mu, structure_matrix, k)          
                        if 'p_phi' in arg.keys():
                            arg['p_phi'] = sparsify(p_phi, structure_matrix, k)          
                        if 'p_h_sparse' in arg.keys():
                            arg['p_h'] = sparsify(p_h_sparse, structure_matrix,
                                                    k, dim = arg['dim'], length = lenX)
                package.append(auxiliary(index, arg))
            return package

        
     
def action_pool(auxiliary = None, apply_elem = None, axis = None,
                base_arg = None, var = None, code_name = ""):
    tasks_per_thread = base_arg['tasks_per_thread']
    nb_threads = base_arg['nb_threads']
    use_pool = base_arg['use_pool']
    print_time_pool = base_arg['print_time_pool']
    sparse_is_on = base_arg['sparse_is_on']
    if axis == 'x':
        total_len = base_arg['lenX']
        if sparse_is_on:
            base_arg['sparse_gridYX'] = None
    elif axis == 'y':
        total_len = base_arg['lenY']
        if sparse_is_on:
            base_arg['sparse_gridXY'] = None
    else:
        raise("Axis "+axis+" does not exist.")
    numb_paral = tasks_per_thread*nb_threads
    size_job = int(math.ceil(total_len*1./numb_paral))
    gf.check(numb_paral*size_job>=total_len,(numb_paral,size_job,total_len))
    args = []
    base_arg['auxiliary'] = auxiliary
    base_arg['size_job'] = size_job
    base_arg['total_len'] = total_len
    base_arg['axis'] = axis
    for j in range(numb_paral):
        i_0 = j*size_job
        if i_0 >= total_len:
            break
        base_arg['i_0'] = i_0
        arg = dict(base_arg)
        if sparse_is_on:
            iplus = min(total_len, (j+1)*size_job)
            if axis == 'x':
                arg['sparse_gridXY'] = arg['sparse_gridXY'][i_0:iplus]
            elif axis == 'y':
                arg['sparse_gridYX'] = arg['sparse_gridYX'][i_0:iplus]
        args.append(arg)   
        
    if use_pool:
        pool = Pool(nb_threads)
        t_0 = timeit.default_timer()
        results = list(pool.map(auxiliary_package, args))
        pool.close()
        pool.join()
        time_pool = timeit.default_timer()-t_0
        if print_time_pool:
            print("time for poolmap "+code_name, time_pool)
    else:
        t_0 = timeit.default_timer()
        results = list(map(auxiliary_package, args))
        if print_time_pool:
            print("time for map "+code_name,timeit.default_timer()-t_0)
            
    total_time = 0.
    for elems in results:
        for elem in elems:
            var = apply_elem(elem, var)
            total_time += elem['time_comp']   
            
    if use_pool:
        var['stop_pool'] = False
        if total_time < time_pool:
            var['stop_pool'] = True
    if print_time_pool:
        print("sum of computation times for "+code_name+" = ", total_time)
    return var

