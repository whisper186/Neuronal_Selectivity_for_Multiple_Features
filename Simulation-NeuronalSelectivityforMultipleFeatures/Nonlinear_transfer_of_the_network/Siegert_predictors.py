####### Wenqing Wei ###############################################
####### Calculate the steady state result of Siegert formula. #####
###################################################################

import time
import numpy as np

import scipy.linalg
import scipy.sparse, scipy.sparse.linalg, scipy.integrate
from importlib import reload
import output; reload(output); import output as out

import Siegert_model; reload(Siegert_model); from Siegert_model import Siegert_model_class

#import matplotlib.pyplot as plt

class Siegert_predictor_class(Siegert_model_class):
    
    def siegPredict(self, Nu, V_FFI, plot=False, prnt=True, **kwargs):
        
        t0 = time.time()
        
        if 'der_tol' not in kwargs:
            kwargs['der_tol'] = 1e-6
        if 's_step' not in kwargs:
            kwargs['s_step'] = 10.
        if 's_max' not in kwargs:
            kwargs['s_max'] = 100
        if 'init' not in kwargs:
            kwargs['init'] = 'zero'
        
        if prnt:
            print ('der_tol:', '%.1e'%kwargs['der_tol'])
            print ('s_step', kwargs['s_step'])
            print ('s_max', kwargs['s_max'])
        
        s_step = kwargs['s_step']
        
        
        dvds = lambda t, v, nu, v_FFI: (self.F(v, nu, v_FFI) - v)  # t not used; v: rates; i: input current
        
        if type(kwargs['init']) is np.ndarray and kwargs['init'].size==self.N:
            init = kwargs['init']
        elif kwargs['init'] == 'zero':
            init = np.zeros(self.N)
        elif kwargs['init'] == 'rand':
            init = np.random.rand(self.N)
        else:
            init = np.ones(self.N) * kwargs['init']
        
        if prnt: print ('integrating ... \n')
        
        t0 = time.time()
        
        ode_obj = scipy.integrate.ode(dvds)
        ode_obj.set_f_params(Nu, V_FFI)      # set extra parameters for user-supplied function f
        
        ode_obj.set_integrator('dopri5', max_step = 1e-1)
        
        ode_obj.set_initial_value(init, 0.)
        
        if prnt: print ('[')
        
        if plot:
            times = [0.]
            res = [init.reshape((1, -1))]
        
        s = 0.
        n = 0.
        while True:
            t1 = time.time()
            n += 1
            
            s += s_step
            result = ode_obj.integrate(s)
            
            if plot:
                times.append(s)
                res.append(result.reshape((1, -1)))
                
            last_der = dvds(None, result, Nu, V_FFI)
            
            if prnt: print (n, s, '%.2e'%np.max(np.abs(last_der)), out.timeConversion(time.time() - t1))
            if not ode_obj.successful():
                raise UserWarning('Integration not successful')
            elif np.all(np.abs(last_der) < kwargs['der_tol']):
                break
            elif s == kwargs['s_max']:
                print ('\nder:' + str(np.max(np.abs(last_der))))
                print('=============== WARNING: s_max reached without converging ===============')
                break
        
        if prnt: print (']...done (%s)'%out.timeConversion(time.time() - t0))
        
        if plot:
            
            print ('plotting')
            
            n_tc = 10
            
            res = np.concatenate(res, axis = 0)
            res = np.split(res, np.cumsum(self.n_neurons)[:-1], axis = 1)  # [((n, 10000)), ((n, 2500))]

            figsize = (16, 8)
            f, axs_rec = plt.subplots(2,1,figsize=figsize,sharex=True)
            axs = axs_rec.flatten()
            for p, pop in enumerate(self.populations):
                ax = axs[p]
                
                for i in range(n_tc):
                    ax.plot(times, res[p][:, i], lw = 2.)
                
                ylims = ax.get_ylim()
                xmax = float(ax.get_xlim()[1])
                k = np.int(np.ceil(xmax/kwargs['s_step']))
                for i in xrange(k):
                    ax.plot(np.ones(2) * i * kwargs['s_step'], [-1000, 1000], 'k--', lw = 1.)
                
                ax.set_ylim(ylims)
                ax.set_title(pop)
            
            plt.suptitle('rates')
            axs_rec[-1].set_xlabel('s [a.u.]')
            axs_rec[-1].set_ylabel(r'$\nu$ [Hz]')
            
            plt.show()
            
            print ('done')
        
        return result
