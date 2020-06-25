from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt
from time import time as time

"""
##-------------------------------------------------------------------------
##         Fully Adaptive Bayesian Algorithm for Data Analysis
##-------------------------------------------------------------------------
"""

def running_mean(dat):
    
    mean = np.array(dat)
    dim = len(mean.shape)
    
    if dim==1:
        mean[:-1] += dat[1:]
        mean[1:] += dat[:-1]
        mean[1:-1] /= 3
        mean[0] /= 2
        mean[-1] /= 2
    elif dim==2:
        mean[:-1,:] += dat[1:,:]
        mean[1:,:] += dat[:-1,:]
        mean[:,:-1] += dat[:,1:]
        mean[:,1:] += dat[:,:-1]
        mean[1:-1,1:-1] /= 5
        mean[0,1:-1] /= 4
        mean[-1,1:-1] /= 4
        mean[1:-1,0] /= 4
        mean[1:-1,-1] /= 4
        mean[0,0] /= 3
        mean[-1,-1] /= 3
        mean[0,-1] /= 3
        mean[-1,0] /= 3
    else:
        print("Warning: Size of array not supported")
    return mean 


def Evidence(mu1,mu2,var1,var2):
    return np.exp(-(mu1-mu2)**2/(2*(var1+var2)))/np.sqrt(
            2*np.pi*(var1+ var2))


def FABADA_SM(data, variance, **kwargs):    
    data = np.array(data/1.0)
    variance = np.array(variance/1.0)
    if not kwargs: kwargs = {} ; kwargs["debug"] = False
    if len(data.shape) == 1: print("FABADA 1-D - Single Mode")
    elif len(data.shape) ==2: print("FABADA 2-D - Single Mode")
    else: print("Warning: Size of array not supported")
    if variance.size != data.size : variance = variance * np.ones_like(data)
    
    
    t = time()
    posterior_mean = data
    posterior_variance = variance
    extra_margin = 3*np.sqrt(variance)
    data_range = np.nanmax(data+extra_margin) - np.nanmin(data-extra_margin)
    evidence = [np.ones_like(data)/data_range]
    param = np.mean(evidence[-1])
    i=0
    try:
        while param >0:
            prior_mean = running_mean(posterior_mean)  
            prior_variance = posterior_variance                 
            posterior_variance = 1/(1/prior_variance + 1/variance)
            posterior_mean = (prior_mean/prior_variance + data/variance
                               )*posterior_variance
            evidence.append(Evidence(prior_mean, data,
                                     prior_variance, variance))
            param = np.mean(evidence[-1]) - np.mean(evidence[-2])
            
            i=i+1 # for debugging
            
    except:
        print("Error: Data size not supported in running mean")

    
    model_result = posterior_mean
    t = time() - t
    print("Finish at ",i," iterations and with an execute time of "
                   ,round(t,6), "seconds.")
    try: 
        if kwargs["debug"] == True:
            debug(len(data.shape),i,posterior_mean,evidence,**kwargs)
    except:
        print("Debug could not be done!")
    
    return model_result



def FABADA_BM(data, variance, **kwargs):
    
    data = np.array(data/1.0)
    variance = np.array(variance/1.0)
    if not kwargs: kwargs = {} ; kwargs["debug"] = False
    if len(data.shape) == 1: print("FABADA 1-D Bayesian mode")
    elif len(data.shape) ==2: print("FABADA 2-D - Bayesian mode")
    else: print("Warning: Size of array not supported")
    if variance.size != data.size : variance = variance * np.ones_like(data)
    
    
    t = time()
    posterior_mean = data
    posterior_variance = variance
    extra_margin = 3*np.sqrt(variance)
    data_range = np.nanmax(data+extra_margin) - np.nanmin(data-extra_margin)
    evidence = np.ones_like(data)/data_range
    devidence = np.mean(evidence)
    evidence_pre,devidence_pre,ddevidence = 0,0,0
    ev_sum , model_result,i = 0,0,0


    try:
        while (devidence > 0) or (ddevidence < 0):
            evidence_pre =  evidence
            devidence_pre = devidence
            prior_mean = running_mean(posterior_mean)  
            prior_variance = posterior_variance                 
            posterior_variance = 1/(1/prior_variance + 1/variance)
            posterior_mean = (prior_mean/prior_variance + data/variance
                               )*posterior_variance
            evidence = Evidence(prior_mean, data,
                                     prior_variance, variance)
            devidence = np.mean(evidence) - np.mean(evidence_pre)
            ddevidence = devidence - devidence_pre
            ev_sum = ev_sum + evidence
            model_result +=  np.array(posterior_mean)*np.array(evidence)
            i=i+1 # for debugging
            
    except:
        print("Something happened")
    
    model_result = model_result/ev_sum
    
    t = time()-t
    print("Finish at ",i," iterations and with an execute time of "
                  ,round(t,5), "seconds.")
    return model_result



def FABADA_SMopt(data, variance, real):
    
    data = np.array(data)/1.0
    variance = np.array(variance)/1.0
    real = np.array(real)/1.0
    if len(data.shape) == 1: print("FABADA 1-D - Single OPTIMAL Mode")
    elif len(data.shape) ==2: print("FABADA 2-D - Single OPTIMAL Mode")
    else: print("Warning: Size of array not supported")
    if variance.size != data.size : variance = variance * np.ones_like(data)
    
    if len(data.shape) == 1: N = 1000
    if len(data.shape) == 2: N = 1000
    
    t = time()
    posterior_mean = data
    posterior_variance = variance
    mse = np.mean((data-real)**2/variance)
    dmse,mse_pre = -1,0
    i=0
    try:
        while (dmse < 0) & (i<N): 
            mse_pre = mse
            prior_mean = running_mean(posterior_mean)  
            prior_variance = posterior_variance                 
            posterior_variance = 1/(1/prior_variance + 1/variance)
            posterior_mean = (prior_mean/prior_variance + data/variance
                               )*posterior_variance
            mse = np.mean((posterior_mean-real)**2/variance)
            dmse = mse - mse_pre
            i=i+1 # for debugging
            
    except:
        print("Error: Data size not supported in running mean")
    
    model_result = posterior_mean
    
    t = time()-t
    print("Finish at ",i," iterations and with an execute time of "
                  ,round(t,5), "seconds.")
    return model_result

def FABADA_BMopt(data, variance, real):
    
    data = np.array(data)/1.0
    variance = np.array(variance)/1.0
    real = np.array(real)/1.0
    if len(data.shape) == 1: print("FABADA 1-D - Bayesian OPTIMAL Mode")
    elif len(data.shape) == 2: print("FABADA 2-D - Bayesian OPTIMAL Mode")
    else: print("Warning: Size of array not supported")
    if variance.size != data.size : variance = variance * np.ones_like(data)
    
    if len(data.shape) == 1: N = 1000
    if len(data.shape) == 2: N = 1000
        
    t = time()
    posterior_mean = data
    posterior_variance = variance
    extra_margin = 3*np.sqrt(variance)
    data_range = np.nanmax(data+extra_margin) - np.nanmin(data-extra_margin)
    evidence = np.ones_like(data)/data_range
    mse = np.mean((data-real)**2/variance)
    mse_pre,dmse,i= 0,-1,0
    model_result,ev_sum = 0,0
    try:
        while (dmse < 0) & (i<N):
            mse_pre = mse
            prior_mean = running_mean(posterior_mean)  
            prior_variance = posterior_variance                 
            posterior_variance = 1/(1/prior_variance + 1/variance)
            posterior_mean = (prior_mean/prior_variance + data/variance
                               )*posterior_variance
            evidence = Evidence(prior_mean, data,
                                     prior_variance, variance)

            ev_sum = ev_sum + evidence
            model_result +=  np.array(posterior_mean)*np.array(evidence)
            mse = np.mean((model_result/ev_sum-real)**2/variance)
            dmse = mse - mse_pre
            i=i+1 # for debugging
            
    except:
        print("Something happened :S")
    
    model_result = model_result / ev_sum
    
    t = time() - t
    print("Finish at ",i," iterations and with an execute time of "
                   ,round(t,6), "seconds.")
    
    return model_result


def debug(dim,i,mean,ev,**kwargs):
    print("Debugging started")
    posterior_mean = mean
    evidence = ev
    if dim == 1:
        plt.figure()
        debug_result = []
        N_iter = range(i+1)
        for j in range(1,i+2):
            debug_result.append(np.sum(
                np.multiply(np.array(posterior_mean)[:j,:],
                            np.array(evidence)[:j,:]),axis=0
                )/np.sum(np.array(evidence)[:j,:],axis=0))
        if "data_real" in kwargs:    
            data_real = kwargs["data_real"]/1.0
            if "sigma" in kwargs:
                sig = kwargs["sigma"]
                chi = (np.array(debug_result) - data_real)**2/sig**2
            else:
                chi = (debug_result - data_real)**2/(
                    (debug_result-data_real)**2).max()**2
            chi_mean = np.array([np.mean(chi[i,:]) for i in range(len(chi))])
            plt.plot(N_iter,chi_mean,"g-",label="$\mathcal{X}^2_O$")
            plt.axvline(x=N_iter[np.argmin(chi_mean)]
                        ,ymin=0,ymax=1, color='g', linestyle=':')
        ev_mean = np.array([np.mean(evidence[i]) for i in range(len(evidence))])
        ev_pro = np.array([np.prod(evidence[i]) for i in range(len(evidence))])
        ev_pro = (ev_pro / ev_pro.max())*ev_mean.max()
        plt.plot(N_iter,ev_mean,"r-",label="Evidence")
        plt.plot(N_iter,ev_pro,"c-",label="Evidence Product")
        plt.legend()
        plt.xlabel("$N_{iter}$",fontsize=15)
        plt.ylabel("$\mathcal{X}^2$",fontsize=15) 
        
    elif dim == 2:
        debug_result = []
        N_iter = range(i+1)
        plt.figure()
        posterior_mean = np.array(posterior_mean)
        evidence = np.array(evidence)
        for j in range(1,i+2):
            debug_result.append(np.sum(posterior_mean[:j,:,:]*
                                       evidence[:j,:,:],axis=0)
                                /np.sum(evidence[:j,:,:],axis=0))
        if "data_real" in kwargs:    
            data_real = kwargs["data_real"]
            if "sigma" in kwargs:
                sig = kwargs["sigma"]
                chi = (debug_result - data_real)**2/sig**2
            else:
                chi = (debug_result -data_real)**2/((
                    debug_result-data_real)**2).max()**2
            chi_mean = np.array([np.mean(chi[i,:,:]) for i in range(len(chi))])
            plt.plot(N_iter,chi_mean,"g.",label="$\mathcal{X}^2_O$")
            plt.axvline(x=N_iter[np.argmin(chi_mean)]
                        ,ymin=0,ymax=1, color='g', linestyle=':')
        ev_mean = np.array([np.mean(evidence[i]) for i in range(len(evidence))])
        ev_pro = np.array([np.prod(evidence[i]) for i in range(len(evidence))])
        plt.plot(N_iter,ev_mean,"r.",label="Evidence")
        plt.plot(N_iter,ev_pro,"c-",label="Evidence Product")
        plt.legend()
        plt.xlabel("$N_{iter}$",fontsize=15)
        plt.ylabel("$\mathcal{X}^2$",fontsize=15) 



                          