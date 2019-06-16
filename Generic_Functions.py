# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:09:00 2014

@author: hdemarch
"""

import numpy as np
import matplotlib.pyplot as plt

def apply(f,x):
    def function(y):
        return f(x,y)
    return function

def gram_schmidt(vectors, tol = 1e-10):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        norm_w=np.linalg.norm(w)
        if norm_w > tol:
            basis.append(w/norm_w)
    return np.array(basis)

def approx_Equal(x, y, tolerance=1e-10):
    return np.linalg.norm(x-y) <= tolerance 

def approx_Negative(x, tolerance=1e-10):
    Isnegative = False
    for i in range(len(x)):
        Isnegative = Isnegative or x[i]< tolerance
    return Isnegative


def print_dots(x_0,vectors,nb_figure = 1,plot_save = False):
    d = len(x_0)
    if d==2:
        plt.figure(nb_figure)
        AbsRoots = list(map(lambda x:x[0],vectors))
        OrdRoots = list(map(lambda x:x[1],vectors))
        plt.plot(AbsRoots,OrdRoots, marker='o', linestyle='None', color='r')
        Absx_0 = x_0[0]
        Ordx_0 = x_0[1]
        plt.plot(Absx_0,Ordx_0, marker='o', linestyle='None', color='y')
        if plot_save:
            plt.savefig('dots'+str(plt.gcf().number)+'.png')
            plt.close()
        else:
            plt.show()
    else:
        print("Warning! Couldn't print because dimension = ",d)
    return 0


def projection(x,vectors):
    if len(vectors)==0:
        raise("projection on nothing")
    elif len(vectors)==1:
        return vectors[0]
    else:
        basis = np.array(vectors[1:])-vectors[0]
        point = np.array(x)-vectors[0]
        orthobasis = gram_schmidt(basis)
        return (np.array(vectors[0])+np.sum( np.dot(point,b)*b  for b in orthobasis ))



def check(b,x=None):
    if b:
        return 1
    else:
        if x!= None:
            print(x," has been a total LOSER")
        raise("Your checking completely FAILED")
        return 0


def barycenter(x, vectors, argvectors = []):
    dim = len(vectors)-1
    if dim==-1:
        raise("barycenter with no vectors")
    if dim==0:
        return {'coeffs' : np.array([1.]), 'vectors' : vectors, 'argvectors' : argvectors}
    basis = vectors[:dim]-np.array(vectors[dim])
    orthobasis = gram_schmidt(basis)
    orthodim = len(orthobasis)
    while(orthodim!=dim):
        for i in range(dim+1):
            vector_test = vectors[i]
            vectors_test = np.delete(vectors,i,0)
            if approx_Equal(projection(vector_test,vectors_test),vector_test):
                vectors = np.delete(vectors,i,0)
                argvectors = np.delete(argvectors,i,0)
                dim=dim-1
                break
            elif i == dim:
                raise("never found the right vector to remove")
    if dim==0:
        return {'coeffs' : [1.], 'vectors' : vectors, 'argvectors' : argvectors}        
    basis = vectors[:dim]-np.array(vectors[dim])
    point = np.array(x)-vectors[dim]
    Mat = np.transpose(np.matrix(basis))
    point_matrix = np.transpose(np.matrix(point))
    MattMat_Inv = np.linalg.inv(np.dot(np.transpose(Mat),Mat))
    Atpoint = np.dot(np.transpose(Mat),point_matrix)
    Lambdas = np.dot(MattMat_Inv,Atpoint)
    Lambdas = np.append(Lambdas, np.array(1.-np.sum(a for a in Lambdas)), axis=0)
    Lambdas = np.array(list(map(lambda y:y[0,0],Lambdas)))
    return {'coeffs' : Lambdas, 'vectors' : vectors, 'argvectors' : argvectors}



def interpolate(a_lo, a_hi, val_lo, val_hi, diff_lo, diff_hi, bound = 0.1,
                order = 'third', print_ratio = False, a_min = None, a_max = None):
    check(bound <= 0.5 , bound)
    if a_min is None:
        a_min = min(a_lo+bound*(a_hi-a_lo), a_hi+bound*(a_lo-a_hi))
    if a_max is None:
        a_max = max(a_lo+bound*(a_hi-a_lo), a_hi+bound*(a_lo-a_hi))
    fail = False
    if order == 'part affine':
        if diff_lo == diff_hi:
            fail = True
        else:
            ratio = (val_lo - val_hi + a_hi*diff_hi - a_lo*diff_lo)/(diff_hi - diff_lo)
            
    elif order == 'third':
        if a_lo == a_hi:
            fail = True
        else:
            d1 = diff_lo+diff_hi-3*(val_lo - val_hi)/(a_lo-a_hi)
            if d1**2-diff_lo*diff_hi <0.:
                fail = True
            else:
                d2 = np.sign(a_hi - a_lo)*np.sqrt(d1**2-diff_lo*diff_hi)
                if diff_hi - diff_lo + 2*d2 == 0.:
                   fail = True
                else:
                    ratio = a_hi + ( a_lo - a_hi )*(diff_hi + d2 - d1)/(diff_hi - diff_lo + 2*d2)
    
    if fail:
        if val_lo <= val_hi:
            ratio = a_lo + bound*(a_hi - a_lo)
        else:
            ratio = a_hi + bound*(a_lo - a_hi)
    if print_ratio:
        print("            match_point", ratio)
    return max(a_min, min(a_max, ratio))






def zoom(a_lo, a_hi, phi, val_0, diff_0, val_lo, diff_lo, val_hi, diff_hi,
         a_min = 1e-8, c1 = 1e-4, c2 = 0.9, maxiter = 100, 
                save_action = None, sto_save_action = None, print_evol = False):
    for j in range(maxiter):
        a_j = max(a_min, interpolate(a_lo, a_hi, val_lo, val_hi,
                                     diff_lo, diff_hi, print_ratio = print_evol))
        if print_evol:
            print("a_j, a_lo, a_hi = ", (a_j, a_lo, a_hi))
        data = phi(a_j)
        val_j = data['val']
        diff_j = data['diff']
        if a_j == a_min:
            if print_evol:
                print("got to minimum alpha")
            return {'value' : val_j, 'gradient' : data['grad'], 'alpha' : a_j}
        if val_j > val_0 + c1*a_j*diff_0 or val_j>=val_lo:
            if save_action is not None:
                    save_action(sto_save_action)
            a_hi = a_j
            val_hi = val_j
            diff_hi = diff_j
        else:
            if np.abs(diff_j) <= - c2*diff_0:
                return {'value' : val_j, 'gradient' : data['grad'], 'alpha' : a_j}
            if diff_j*(a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_j
            if save_action is not None:
                    save_action(sto_save_action)
    return {'value' : val_j, 'gradient' : data['grad'], 'alpha' : a_j}
    raise("YOU LOOOOOOOSE in zoom")
                
            
    



def line_search(func_grad, x0, val_0, grad_0, p, old_val = None, c1 = 1e-4, c2 = 0.9,
                a_max = 50., a_min = 1e-8, maxiter = 100, 
                save_action = None, sto_save_action = None, print_evol = False):
    a_i_1 = 0.
    val_i_1 = val_0
    diff_0 = -np.dot(p, grad_0)
    if old_val is None or diff_0 == 0.:
        a_i = 1.
    else:
        a_i_guess = 2*(val_0-old_val)/diff_0
        if a_i_guess < 1:
            if print_evol:
                print("2*(val_0-old_val)/diff_0      = ", a_i_guess)
        a_i = max(a_min, min(1., 1.01*a_i_guess))
    
    diff_i_1 = diff_0
    val_i = val_0
    diff_i = diff_0
    def phi(a):
        data = func_grad(x0 - a*p)
        return {'val' : data[0], 'diff' : - np.dot(data[1], p), 'grad' : data[1]}
    for i in range(maxiter):
        i += 1
        data = phi(a_i)
        if i>1:
            val_i_1 = val_i
            diff_i_1 = diff_i
        val_i = data['val']
        diff_i = data['diff']
        if (val_i > val_0 + c1*a_i*diff_0) or (i>1 and val_i >= val_i_1):
            if save_action is not None:
                    save_action(sto_save_action)
            return zoom(a_i_1, a_i, phi, val_0, diff_0, val_i_1, diff_i_1,
                        val_i, diff_i, a_min = a_min, c1 = c1, c2 = c2,
                        maxiter = maxiter, print_evol = print_evol,
                        save_action = save_action, sto_save_action = sto_save_action)
        if np.abs(diff_i) <= - c2*diff_0:
            return {'value' : val_i , 'gradient' : data['grad'] , 'alpha' : a_i}
        if diff_i >= 0:
            if save_action is not None:
                    save_action(sto_save_action)
            return zoom(a_i, a_i_1, phi, val_0, diff_0, val_i, diff_i,
                         val_i_1, diff_i_1, a_min = a_min, c1 = c1, c2 = c2,
                         maxiter = maxiter, print_evol = print_evol,
                         save_action = save_action, sto_save_action = sto_save_action)
        if a_i == a_max:
            return {'value' : val_i , 'gradient' : data['grad'] , 'alpha' : a_i}
        else:
            if save_action is not None:
                    save_action(sto_save_action)
            if diff_0 == 0.:
                raise("you should not be here...")
            a_i_plus = interpolate(a_i, a_i_1, val_i, val_i_1, diff_i, diff_i_1,
                print_ratio = print_evol, a_min = a_i*1.1, a_max = a_max)
            a_i_1 = a_i
            a_i = a_i_plus
            if print_evol:
                print("RISING: a_i, a_i_1", (a_i, a_i_1))
    print("YOU LOOOOOOOSE in line search")
    return {'value' : val_i , 'gradient' : data['grad'] , 'alpha' : a_i_1}


    
    
def CG(b, prodA = None, tol = 1e-7, x_0 = None, maxiter = 100,
          cond_inv = None, disp = False, debug_mode = 0,
          pow_distance = 2):#solves Ax = b
       
    following_evolution = {'best_x' : b, 'best_norm' : np.infty,
                           'times_rising' : 0, 'last_norm_A' : np.infty}
    if x_0 is None:
        x_0 = 0.*b
        r = b
    else:
        r = b - prodA(x_0)
    x = x_0
    if cond_inv is None:
        p = r
        rr = np.dot(r,r)
    else:
        z = cond_inv(r)
        p = z
        rz = np.dot(r,z)
    Ap = 0.*b
    beta = 0.
    for k in range(maxiter):     
        Ap = prodA(p)
        pAp = np.dot(p,Ap)
        if pAp<= 0:
            print("Hessian non positive definite: pAp/pp = ",
                  pAp/np.dot(p, p), ", and pp = ", np.dot(p, p))
            break
        if cond_inv is None:
            a = rr/pAp
        else:
            a = rz/pAp
        x = x+a*p
        r_sto = r
        r = r-a*Ap
        if cond_inv is None:
            rr_sto = rr
            rr = np.dot(r,r)
        else:
            z = cond_inv(r)
            rz_sto = rz
            rz = np.dot(r,z)
        norm = np.linalg.norm(r, pow_distance)
        if cond_inv is None:
            Ar = prodA(r)
            rAr = np.dot(r, Ar)
            norm_A = np.sqrt(rAr)
        else:
            Ar = prodA(r)
            rAr = np.dot(r, Ar)
            norm_A = np.sqrt(rAr)
            if debug_mode == 13:
                print("Ap-A*p", Ap-prodA(p))
                print("Ar-A*r_sto", Ar-prodA(r_sto))
                print("Ar-A*r", Ar-prodA(r))
                pAr = np.dot(p, Ar)
                rAp = np.dot(r, Ap)
                Np = np.sqrt(np.dot(p,Ap))
                print("rAr2", np.dot(r,Ar))
                Nr = np.sqrt(np.dot(r,Ar))
                diff = (pAr-rAp)/(Np*Nr)
                print("pAr-rAp = ", diff)
                print("norm A p = ", Np, "norm A r = ", Nr)
        if norm < following_evolution['best_norm']:
            following_evolution['best_x'] = x
            following_evolution['best_norm'] = norm
        if norm_A >= following_evolution['last_norm_A']:
            following_evolution['times_rising'] +=1
        else:
            following_evolution['times_rising'] = 0
        following_evolution['last_norm_A'] = norm_A
        if disp:
            print("CG error = ", norm, "norm A = ", norm_A)
        if norm <= tol:
            if debug_mode == 10:
                norm_res = np.linalg.norm(prodA(x)-b, pow_distance)
                
                check(norm_res <= tol, ('norm_res', norm_res,
                        'tol', tol, 'norm', norm, 'scalar', np.sum(x*b)))
            return x
        if following_evolution['times_rising'] == 10:
            print("Precision lost")
            break
        if cond_inv is None:
            beta = rr/rr_sto
            p = r + beta*p
        else:
            beta = rz/rz_sto
            p = z + beta*p
    print("CG FAIL")
    return following_evolution['best_x']
    

    
def Newton_CG(value_grad, x0 = 0, hessp = None, hess = None, tol=1e-7, maxiter = 100,
              disp = False,
              bump = 1e-9, pow_distance = 2, maxiter_CG = 100, add_step = None,
              cond_inv = None, check_cond = False, save_action = None, disp_CG = False,
              debug_mode_CG = False, print_line_search = False, order_hess = 1.,
              adjust_for_ls = None):
    if hessp is None:
        if hess is not None:
            storage_hess = {'hess': None, 'x': None}
            def hessp(x, p):
                if approx_Equal(x, storage_hess['x'], tolerance=1e-20):
                    hessian = storage_hess['hess']
                else:
                    hessian = hess(x)
                    storage_hess['x'] = np.array(x)
                    storage_hess['hess'] = hessian
                return np.dot(hessian, p)
        else:
            def hessp(x, p):
                return (value_grad(x+bump*p)[1]-value_grad(x-bump*p)[1])/(2*bump)
    x = x0
    if add_step is not None:
        x_new = add_step(x)
        if approx_Equal(x, x_new):
            raise("it should have been modified!")
        x = x_new
    data = value_grad(x)
    value = data[0]
    gradient = data[1]
    p = gradient
    val_sto = None
    for k in range(maxiter):
        if np.linalg.norm(gradient, pow_distance)<= tol:
            result = {'iterations' : k, 'success' : True,
                'value' : value, 'grad_norm' : np.linalg.norm(gradient, pow_distance)}
            if disp:
                print(result)
                result['x'] = x
            return result
        if add_step is not None:
            x = add_step(x)
            data = value_grad(x)
            value = data[0]
            gradient = data[1]
        norm_grad = np.linalg.norm(gradient, pow_distance)
        tol_CG = norm_grad*min(0.5, np.sqrt(norm_grad))
        x_hess = np.array(x)
        def prodHess(p_loc):
            return hessp( x_hess, p_loc)
        if cond_inv is not None:
            cond_inv_x = cond_inv(x_hess)
        elif hess is not None:
            if approx_Equal(x_hess, storage_hess['x'], tolerance=1e-20):
                hessian = storage_hess['hess']
            else:
                hessian = hess(x_hess)
                storage_hess['x'] = np.array(x_hess)
                storage_hess['hess'] = hessian
            cond_inv_x = 1./np.diag(hessian)
        else:
            cond_inv_x = None
        p_sto = p
        p = CG(gradient, prodA = prodHess, tol = tol_CG, x_0 = None,
                         maxiter = maxiter_CG,
                         cond_inv = cond_inv_x, disp = disp_CG,
                         debug_mode = debug_mode_CG, pow_distance = pow_distance)
        if print_line_search:
            print("angle with last direction = ",
                np.dot(p, p_sto)/(np.linalg.norm(p)*np.linalg.norm(p_sto)))
        if np.dot(gradient, p)<0:
            p = -p
            print("BIG INVERSION")
        if save_action is not None:
            sto = save_action()
        else:
            sto = None
        a_min = 1e-8/order_hess/max(1., np.linalg.norm(p))
        if adjust_for_ls is not None:
            def value_grad_adjust(x):
                return adjust_for_ls(x)
        else:
            value_grad_adjust = value_grad
        data_ls = line_search(value_grad_adjust, x, val_0 = value, grad_0 = gradient, p = p,
                              old_val = val_sto, c1 = 1e-4, c2 = 0.9,
                              a_max = 50., a_min = a_min, maxiter = 100,
                              save_action = save_action, sto_save_action = sto,
                              print_evol = print_line_search)
        if print_line_search:
            print("alpha = ", data_ls['alpha'])
        x -= data_ls['alpha']*p
        val_sto = value
        value = data_ls['value']
        gradient = data_ls['gradient']
        if check_cond:
            if cond_inv is None:
                raise("Conditioning is not running.")
            dim = len(gradient)
            u = np.zeros(dim)*1.
            index = int(np.random.rand()*dim)
            u[index]=1.
            app = cond_inv_x(prodHess(u))
            scal_test = np.dot(u,app)
            print("u cond-1 Hess u = ", scal_test, "index =" , index)
    if np.linalg.norm(gradient, pow_distance)<= tol:
        result = {'iterations' : k, 'success' : True, 'value' : value,
        'grad_norm' : np.linalg.norm(gradient, pow_distance)}
    else:
        result = {'iterations' : k, 'success' : False, 'value' : value,
        'grad_norm' : np.linalg.norm(gradient, pow_distance)}
    if disp:
        print(result)
    result['x'] = x
    return result
        


def Newton(value_grad, hess, x0 = 0, tol=1e-7, maxiter = 100, disp = False,
              pow_distance = 2, add_step = None, invertor = 1e-10, save_action = None,
              print_evol = False, order_hess = 1.):
    x = x0
    dim = len(x)
    if add_step is not None:
        x = add_step(x)
    data = value_grad(x)
    value = data[0]
    gradient = data[1]
    val_sto = None
    for k in range(maxiter):
        if np.linalg.norm(gradient, pow_distance)<= tol:
            result = {'x' : x, 'iterations' : k, 'success' : True, 'value' : value,
            'grad_norm' : np.linalg.norm(gradient, pow_distance)}
            if disp:
                print(result)
            return result
        if add_step is not None:
            x = add_step(x)
            data = value_grad(x)
            value = data[0]
            gradient = data[1]
        p = np.linalg.solve(hess(x)+invertor*np.eye(dim),gradient)
        if save_action is not None:
            sto = save_action()
        else:
            sto = None
        a_min = 1e-8*order_hess/max(1., np.linalg.norm(p))
        data_ls = line_search(value_grad, x, val_0 = value, grad_0 = gradient, p = p,
                              old_val = val_sto, c1 = 1e-4, c2 = 0.9,
                              a_max = 50., a_min = a_min, maxiter = 100,
                              save_action = save_action, sto_save_action = sto,
                              print_evol = print_evol)
        if print_evol:
            print("alpha = ", data_ls['alpha'])
        x -= data_ls['alpha']*p
        val_sto = value
        value = data_ls['value']
        gradient = data_ls['gradient']
    if np.linalg.norm(gradient, pow_distance)<= tol:
        result = {'x' : x, 'iterations' : k, 'success' : True, 'value' : value,
        'grad_norm' : np.linalg.norm(gradient, pow_distance)}
    else:
        result = {'x' : x, 'iterations' : k, 'success' : False, 'value' : value,
        'grad_norm' : np.linalg.norm(gradient, pow_distance)}
    if disp:
        print(result)
    return result

