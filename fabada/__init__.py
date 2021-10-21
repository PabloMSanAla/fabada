"""
FABADA is a non-parametric noise reduction technique based on Bayesian
inference that iteratively evaluates possibles moothed  models  of  
the  data introduced,  obtaining  an  estimation  of the  underlying  
signal that is statistically  compatible  with the  noisy  measurements.

based on P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.
"""

from __future__ import print_function, division
import numpy as np
from typing import Union
from time import time as time
from scipy import ndimage
import scipy.stats as stats
import sys



def fabada(data : Union[np.array,list], data_variance: Union[np.array,list,float],
	       max_iter: int = 3000, verbose: bool = False, **kwargs) -> np.array:

	"""
	FABADA for any kind of data (1D or 2D). Performs noise reduction in input.
	:param data: Noisy measurements, either 1 dimension (M) or 2 dimensions (MxN)
	:param data_variance: Estimated variance of the input, either MxN array, list
                          or float assuming all point have same variance. 
    :param max_iter: 3000 (default). Maximum of iterations to converge in solution. 
    :param verbose: False (default) or True. Spits some informations about process.
	:param **kwargs: Future Work.
    :return bayes: denoised estimation of the data with same size as input.
    """
  	 
	data = np.array(data/1.0)
	data_variance = np.array(data_variance/1.0)

	if not kwargs: kwargs = {} ; kwargs["debug"] = False
    
	if verbose:
		if   len(data.shape) == 1: print("FABADA 1-D initialize")
		elif len(data.shape) == 2: print("FABADA 2-D initialize")
		else: print("Warning: Size of array not supported")
        
	if data_variance.size != data.size: 
		data_variance = data_variance * np.ones_like(data)

	# INITIALIZING ALGORITMH ITERATION ZERO
	t = time()
	posterior_mean = data
	posterior_variance = data_variance
	evidence  = Evidence(0, np.sqrt(data_variance), 0, data_variance)
	initial_evidence = evidence
	chi2_pdf,chi2_data,iteration = 0,data.size,0
	chi2_pdf_derivative,chi2_data_min = 0,data.size
	bayesian_weight = 0
	bayesian_model = 0

	converged = False
    
	try:
		while not converged:
            
			chi2_pdf_previous = chi2_pdf
			chi2_pdf_derivative_previous = chi2_pdf_derivative
			evidence_previous = np.mean(evidence)

			iteration += 1 # Check number of iterartions done


			# GENERATES PRIORS
			prior_mean = running_mean(posterior_mean)
			prior_variance = posterior_variance


			# APPLIY BAYES' THEOREM
			posterior_variance = 1/(1/prior_variance + 1/data_variance)
			posterior_mean = (prior_mean/prior_variance + data/data_variance
			                  ) * posterior_variance


			# EVALUATE EVIDENCE
			evidence = Evidence(prior_mean,data,prior_variance,data_variance)
			evidence_derivative = np.mean(evidence) - evidence_previous

			# EVALUATE CHI2
			chi2_data = np.sum((data-posterior_mean)**2/data_variance)
			chi2_pdf = stats.chi2.pdf(chi2_data, df=data.size)
			chi2_pdf_derivative = chi2_pdf - chi2_pdf_previous
			chi2_pdf_snd_derivative = (chi2_pdf_derivative - 
			                           chi2_pdf_derivative_previous)

			# COMBINE MODELS FOR THE ESTIMATION
			model_weight = evidence * chi2_data
			bayesian_weight += model_weight
			bayesian_model  += model_weight*posterior_mean
            
			if iteration ==1:
				chi2_data_min = chi2_data

			# CHECK CONVERGENCE
			if (chi2_data > data.size and chi2_pdf_snd_derivative >= 0
				) and  ( evidence_derivative < 0 
				) or (iteration > max_iter ):
                
                
				converged = True

				# COMBINE ITERATION ZERO
				model_weight = initial_evidence  *  chi2_data_min
				bayesian_weight += model_weight
				bayesian_model  += model_weight * data
                
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise
    
	bayes = bayesian_model/bayesian_weight
    
	if verbose:
		print("Finish at {} iterations".format(iteration),
			" and with an execute time of {:3.2f} seconds.".format(time()-t))

	return bayes
    



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

def PSNR(recover,signal,L=255):
    MSE = np.sum((recover-signal)**2)/(recover.size)
    return 10*np.log10((L)**2 / MSE)