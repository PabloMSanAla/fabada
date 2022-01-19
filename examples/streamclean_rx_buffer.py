"""
FABADA is a non-parametric noise reduction technique based on Bayesian
inference that iteratively evaluates possible smoothed  models  of
the  data introduced,  obtaining  an  estimation  of the  underlying
signal that is statistically  compatible  with the  noisy  measurements.

based on P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"

Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.


Instructions:
Save the code as a .pyw file to disable the console. Save as a .py to enable it.
If you already have python installed and knowledge of python, use python 3.10,
packages: numba => 0.55.0, np_rw_buffer, dearpygui,pyaudio, numpy.

If you do not have basic understanding of programming terms or python:
This file is an executable script that effectively turns into a program when run with an interpreter.
That interpreter is called "cpython" and the programming language the script is written in is called "python".

In order to run this program, use the following instructions:
If you are not on windows, use your best judgement and change audio device detection as appropriate.

If you are on windows:
Save this file to your C:/Users/yourusername/ folder as streamclean.py

Download the following file and install it:
https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe
During the installation, it will ask :install for just me or for all users
It is mandatory that you select "just me"

Open the command line prompt from the "miniforge3" folder in the start menu.
In that command line interface, run the following commands, agreeing to the prompts:

conda upgrade --all -c conda-forge
conda install -c conda-forge python=3.10
conda install -c conda-forge numba numpy pip
pip install pipwin np_rw_buffer dearpygui
pipwin install pyaudio

pythonw.exe streamclean.py

Usage:
You'll need a line-in device or virtual audio cable you can configure so you can loop the output to input.
The streaming example included here looks for the windows muxer that configures audio devices- whatever you set
in your windows settings for default mic and speaker, respectively, this program will treat as input and output.
So, you can configure another program to output noisy sound to the speaker side of a virtual audio device, and configure
the microphone end of that device as your system microphone, then this program will automatically pick it up and run it.
https://vb-audio.com/Cable/ is an example of a free audio cable.
The program expects 44100hz audio, 16 bit, two channel, but can be configured to work with anything
Additional thanks to Justin Engel.
"""

from __future__ import division


import numpy
import pyaudio
import numba
from np_rw_buffer import AudioFramingBuffer
from threading import Thread
import math
import time
from time import sleep
import dearpygui.dearpygui as dpg
from decimal import *

@numba.jit(numba.float64[:](numba.float64[:], numba.int32, numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift1d(arr : list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = numpy.empty_like(arr)
    if num > 0:
        result[:num] = fill_value[:num]
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value[:num]
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

@numba.jit(numba.float64[:,:](numba.float64[:,:], numba.int32, numba.float64[:,:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift2dy(arr: list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = numpy.empty_like(arr)
    if num > 0:
        result[:,:num] = fill_value[:,:num]
        result[:,num:] = arr[:,:-num]
    elif num < 0:
        result[:,num:] = fill_value[:,:num]
        result[:,:num] = arr[:,-num:]
    else:
        result[::] = arr
    return result

@numba.jit(numba.float64[:,:,:](numba.float64[:,:,:], numba.int32, numba.float64[:,:,:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift3dx(arr: list[numpy.float64], num: int, fill_value: list[numpy.float64]) -> list[numpy.float64] :
    result = numpy.empty_like(arr)
    if num > 0:
        result[:num,:,:] = fill_value[:num,:,:]
        result[num:,:,:] = arr[:-num,:,:]
    elif num < 0:
        result[num:,:,:] = fill_value[:num,:,:]
        result[:num,:,:] = arr[-num:,:,:]
    else:
        result[:] = arr
    return result

@numba.jit(numba.float32[:,:,:](numba.float32[:,:,:], numba.int32, numba.float32[:,:,:]), nopython=True, parallel=True, nogil=True,cache=True)
def shift3dximg(arr: list[numpy.float32], num: int, fill_value: list[numpy.float32]) -> list[numpy.float32] :
    result = numpy.empty_like(arr)
    if num > 0:
        result[:,:num,:] = fill_value
        result[:,num:,:] = arr[:,:-num,:]
    elif num < 0:
        result[:,num:,:] = fill_value
        result[:,:num,:] = arr[:,-num:,:]
    else:
        result[:] = arr
    return result

@numba.jit(numba.float64[:](numba.float64[:], numba.int32), nopython=True, parallel=True, nogil=True,cache=True)
def round(data: list[float], num: int) -> list[float] :
    X = num
    for i in numba.prange(X):
        data[i] = data[i] * (i / X)  # perform a linear fadein
    return data

@numba.jit(numba.float64[:](numba.float64[:], numba.int32), nopython=True, parallel=True, nogil=True,cache=True)
def unround(data: list[float], num: int) -> list[float] :
    X = num
    for i in numba.prange(X):
        data[i] = data[i] * ((1 + X - i) / X)#perform a linear fadeout
    return data
#because numpy's zeroth array is the Y axis, we have to do this in the 1st dimension to shift the X axis
#if the arrays are not the same size, don't attempt to use coordinates for fill value- it will fail.

#https://stackoverflow.com/questions/53550764/python-numpy-is-there-a-faster-way-to-convolve

#https://link.springer.com/chapter/10.1007/978-3-030-51935-3_39
#undecimated discrete wavelet transforms
@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def wavelet(data: list[numpy.float64]):
    data_pad = numpy.zeros_like(data)
    xod = data_pad.size//2

    for i in numba.prange(xod,2):
        data_pad[i] = ((data[i] + data[i+1])/2)
    for i in numba.prange(xod,2):
        data_pad[xod + i] = ((data[i*2] - data[i*2-1])/2)
        #obtain the first order wavelet transforms
    return data_pad

@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def waveletnth(data: list[numpy.float64]):
    xod = data.size//2 #6,4,7,2
    dog = xod//2       #5,4       s2: 1,5
                       #4.5       # 0.5
    #each wavelet is half the size of the previous one.
    #the first time this is run, data's size is 24,000.
    data_pad = numpy.zeros((data.size))
    #data_pad[0:xod] = data[0:xod] #first 12000 are our input values
    data_pad[0:xod] = wavelet(data[0:xod])
    for i in numba.prange(xod):#from 12,000 through 24,000, write in the differences.
        data_pad[xod + i] = data[xod+i]/2
        #array now contains 24,000 elements.
        #the central half is the core of the wavelet envelope.
        #obtain the wavelet transform for subsequent runs
    return data_pad

#what's the maximum wavelet degree we can compose that will work for this problem?
#1st: 24000 -> 12000
#2nd: 12000 -> 6000
#3rd: 6000 -> 3000
#4th: 3000 -> 1500
#5th: 1500 -> 750
#6th: 750 -> 375
#The maximum HAAR discrete envelopes we can calculate are 6.



@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def waveletinverse(data: list[numpy.float64]):
    data_pad = numpy.zeros((data.size))
    xod = data.size//2

    for i in numba.prange(xod):
        data_pad[i*2] = ((data[i] + data[xod + i]))
        data_pad[i*2+1] = ((data[i] - data[xod + i]))

    return data_pad
    #obtain the inverse wavelet transform for the first generation wavelet
#how do we decompose higher order wavelets?



@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def adjacentaverage(data):
    returndata = numpy.zeros_like(data)
    data_pad = numpy.zeros((data.size + 2))
    data_pad[1:data.size+1] = data[:]
    data_pad[0] =  (data[0] + data[1])/2
    data_pad[-1] = (data[-1] + data[-2])/2

    for i in numba.prange(1,data.size):
        returndata[i-1] = (data_pad[i-1] + data_pad[i] + data_pad[i+1])/3


    return returndata

#median filter
@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def medianfilter(data):
    data_pad = numpy.zeros((data.size + 2))
    data_pad[1:data.size+1] = data[:]
    data_pad[0] = 2 * data[0] - data[1]
    data_pad[-1] = 2 * data[-1] - data[-2]
    returndata = numpy.zeros((data.size,3))
    returndata[0,0] = data[0]
    returndata[-1,-1] = data[-1]

    for i in numba.prange(1,data.size-1):
        for x in numba.prange(3):
            returndata[i,x] = data[i + x - 1]
    for i in numba.prange(1,data.size-1):
        returndata[i, :]= numpy.sort(returndata[i, :])



    returndata2 = numpy.zeros((data.size))

    for i in numba.prange(data.size):
        returndata2[i] = returndata[1,i] #return the median value

    return returndata2




#5th order autoregressive rolling mean
@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def arma(data):

    y = numpy.zeros_like(data)
    x = numpy.zeros((data.size, 6))  # create array for outputs
    for i in numba.prange(data.size):
        x[i , 0] = data[max(0,i -5)] * 0.2
        x[i , 1] = data[max(0,i -4)] * 0.2
        x[i , 2] = data[max(0,i -3)] * 0.2
        x[i , 3] = data[max(0,i -2)] * 0.2
        x[i , 4] = data[max(0,i -1)] * 0.2
        x[i , 5] = data[i]

    y[0] = x[0,5]#pad first value


    for i in numba.prange(data.size -1):
        y[i+1] = numpy.sum(x[i,:])/2

    return y



@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def adjacentmeansquared(data):
    data_pad = numpy.zeros((data.size + 2))
    data_pad[1:data.size] = data[:]
    data_pad[0] = 2 * data[0] - data[1]
    data_pad[-1] = 2 * data[-1] - data[-2]
    returndata = numpy.zeros((data.size))

    for i in numba.prange(1,data.size):
        returndata[i-1] = math.sqrt((data_pad[i-1]**2 + data_pad[i]**2 + data_pad[i+1]**2)/3)
    return returndata



#5 width
@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, parallel=True, nogil=True,cache=True)
def savgol(data: list[numpy.float64]):

    coeff = numpy.asarray([-0.08571429,  0.34285714,  0.48571430,  0.34285714, -0.08571429])
    #pad_length = h * (width - 1) // 2# for width of 5, this defaults to 2...
    data_pad = numpy.zeros(data.size + 4)
    data_pad[2:data.size + 2] = data[0:data.size]#leaving two on each side
    firstval =  2* data[0] - data[2:0:-1]
    lastvals = 2* data[-1] - data[-1:-3:-1]
    data_pad[0] = firstval[0]
    data_pad[1] = firstval[1]
    data_pad[-1] = lastvals[1] 
    data_pad[-2] = lastvals[0]
    new_data = numpy.zeros(data.size)
    x = numpy.zeros(((data.size-2),6)) #create array for outputs
    for i in numba.prange(2, data.size):
        x[ i - 2,0] =  data_pad[i - 2]
        x[ i - 2,1] =  data_pad[i - 1]
        x[ i - 2,2] =  data_pad[i]
        x[i - 2,3] =  data_pad[i + 1]
        x[i - 2,4] =  data_pad[i + 2]
        x[i - 2,5]=  data_pad[i + 3]

   
    #
    #multiply vec2 by vec1[0] = 2    4   6
    #multiply vec2 by vec1[1] = -    3   6   9
    #multiply vec2 by vec1[2] = -    -   4   8   12
    #-----------------------------------------------
    #add the above three      = 2    7   16  17  12 

    z = numpy.zeros((data.size, 30))

    for i in numba.prange(data.size):
        for n in numba.prange(30):
            for k in numba.prange(5):
                for gk in numba.prange(6):
                    z[i,n] = coeff[gk] * x[i, k]
    

        #create the array of each set of averaging values
    for i in numba.prange(data.size):
        new_data[i] = numpy.sum(z[i,:])
    return new_data

def chi2pdffullprecision(chi2_data):
            # chi2.pdf(x, df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
            #decimal mathematics are required here, because for data sets over 1490 in size,
            #floating point numbers will just divide down to zero
            #so, here, we use decimal precision.
            #this allows us to also prevent overflows if the number is very large.
            df = 5 #if df is large, it is impossible to get a useful value out of this.
            gamman = Decimal(chi2_data / 2.)
            gammas = gamman ** Decimal(df/2-1) # TODO
            gammaq = Decimal(-chi2_data / 2.).exp() #math.exp doesnt work over -1000
            gammaz = Decimal(2. * math.lgamma(df / 2.))
            gamma1 = Decimal(1)
            gammax = gamma1 / gammaz
            gammmalarge = gammax * gammas * gammaq
            return float(gammmalarge)
    #this code can be used with an objmode lift to calculate the chi2pdf for large or small values.
    #however, it will still return 0.0 if the value is smaller than 53 points of precision,
    #and will still return InF if the value is over 19 0s in length


@numba.jit( numba.float64[:](numba.float64[:], numba.float64, numba.int32), nopython=True, parallel=True, nogil=True,cache=True)
def numba_fabada(data: list[numpy.float64], timex: float,iterationcontrol: int) -> ( list[numpy.float64]):
    # notes:
    # The pythonic way to COPY an array is to do x[:] = y[:]
    # do x=y and it wont copy it, so any changes made to X will also be made to Y.
    # also, += does an append instead of a +
    # math.sqrt is 7x faster than numpy.sqrt but not designed for complex numbers.
    # specifying the type and size in advance of all variables accelerates their use, particularily when JIT is used.
    # However, JIT does not permit arbitrary types like UNION? maybe it does but i havnt figured out how.
    # this implementation uses a lot of for loops because they can be easily vectorized by simply replacing
    # range with numba.prange and also because they will translate well to other languages
    # this implementation of FABADA is not optimized for 2d arrays, however, it is easily swapped by changing the means
    # estimation and by simply changing all other code to iterate over 2d instead of 1d
    # care must be taken with numba parallelization/vectorization

    with numba.objmode(start=numba.float64):
        start = time.time()

    iterations: int = 1
    TAU: float = 2 * math.pi
    N = data.size

    # must establish zeros for the model or otherwise when data is empty, algorithm will return noise
    bayesian_weight = numpy.zeros_like(data)
    bayesian_model = numpy.zeros_like(data)
    model_weight = numpy.zeros_like(data)

    # pre-declaring all arrays allows their memory to be allocated in advance
    posterior_mean = numpy.zeros_like(data)
    posterior_variance = numpy.zeros_like(data)
    initial_evidence = numpy.zeros_like(data)
    evidence = numpy.zeros_like(data)
    prior_mean = numpy.zeros_like(data)
    prior_variance = numpy.zeros_like(data)
    boolv = numpy.zeros_like(data)

    waveletx1 = numpy.zeros_like(data)
    waveletx2 = numpy.zeros_like(data)
    waveletx3 = numpy.zeros_like(data)
    waveletx4 = numpy.zeros_like(data)
    waveletx5 = numpy.zeros_like(data)
    waveletx6 = numpy.zeros_like(data)

    inverted1 = numpy.zeros((waveletx1.size))
    inverted2 = numpy.zeros((waveletx1.size))
    inverted3 = numpy.zeros((waveletx2.size))
    inverted4 = numpy.zeros((waveletx3.size))
    inverted5 = numpy.zeros((waveletx4.size))
    inverted6 = numpy.zeros((waveletx5.size))
    inverted7 = numpy.zeros((waveletx6.size))


    savigol1x = numpy.zeros_like(waveletx1)
    savigol2x = numpy.zeros_like(waveletx2)
    savigol3x = numpy.zeros_like(waveletx3)
    savigol4x = numpy.zeros_like(waveletx4)
    savigol5x = numpy.zeros_like(waveletx5)
    savigol6x = numpy.zeros_like(waveletx6)



    arma1x = numpy.zeros_like(waveletx1)
    arma2x = numpy.zeros_like(waveletx2)
    arma3x = numpy.zeros_like(waveletx3)
    arma4x = numpy.zeros_like(waveletx4)
    arma5x = numpy.zeros_like(waveletx5)
    arma6x = numpy.zeros_like(waveletx6)



    avg1 = numpy.zeros_like(waveletx1)
    avg2 = numpy.zeros_like(waveletx2)
    avg3 = numpy.zeros_like(waveletx3)
    avg4 = numpy.zeros_like(waveletx4)
    avg5 = numpy.zeros_like(waveletx5)
    avg6 = numpy.zeros_like(waveletx6)



    # working set arrays, no real meaning, just to have work space
    ja1 = numpy.zeros_like(data)
    ja2 = numpy.zeros_like(data)
    ja3 = numpy.zeros_like(data)
    ja4 = numpy.zeros_like(data)

    # eliminate divide by zero
    min_d: float = numpy.nanmin(data)
    max_d: float = numpy.ptp(data)

 
    for i in numba.prange(N):
        if data[i] == 0.0:
            boolv[i] = False
        else:
            boolv[i] = True

    data_mean = numpy.mean(data[data != 0.0])  # get the mean, but only for the data that's not zero, to avoid disto
    waveletx1 = wavelet(data)#48000 hz wavelet envelope
    waveletx2[:] = waveletnth(waveletx1)#24000
    waveletx3[:] = waveletnth(waveletx2)#12000
    waveletx4[:] = waveletnth(waveletx3)#6000
    waveletx5[:] = waveletnth(waveletx4)#3000
    waveletx6[:] = waveletnth(waveletx5)#1500


    #achieve 6th order wavelet decomposition from input datum.




    data_variance = numpy.zeros_like(data)
    for i in numba.prange(N):
            data_variance[i] = 1+ (data_mean - data[i])*(data_mean - data[i])
            #standard deviation plus normalization
            #https://prvnk10.medium.com/batch-normalization-d6e402add220

    for i in numba.prange(N):
            data[i] = (data[i] - min_d) / (max_d - min_d)

    posterior_mean[:]= data[:]
    prior_mean[:] = data[:]
    posterior_variance[:] = data_variance[:]
    #fabada figure 14
    for i in numba.prange(N):
            ja1[i] = (0.0 - math.sqrt(data[i]))** 2
            ja2[i] = (data_variance[i]* 2)
            ja3[i] = math.sqrt(TAU * data_variance[i])
    for i in numba.prange(N):
            ja4[i] = math.exp(-ja1[i] / ja2[i])
    for i in numba.prange(N):
            evidence[i] = ja4[i] / ja3[i]
    initial_evidence[:] = evidence[:]

    for i in numba.prange(N):
            ja1[i] = data[i] - posterior_mean[i]
    for i in numba.prange(N):
            ja1[i] =  ja1[i] ** 2.0 / data_variance[i]

    chi2_data_min = numpy.sum(ja1)


    while 1:

        # GENERATES PRIORS
        #combines linear regression with weighted smoothing
        

        prior_mean[:]  = posterior_mean[:]

        waveletx1 = wavelet(prior_mean)  # 48000 hz wavelet envelope
        waveletx2[:] = waveletnth(waveletx1)  # 24000
        waveletx3[:] = waveletnth(waveletx2)  # 12000
        waveletx4[:] = waveletnth(waveletx3)  # 6000
        waveletx5[:] = waveletnth(waveletx4)  # 3000
        waveletx6[:] = waveletnth(waveletx5)  # 1500

        savigol6x = adjacentaverage(waveletx6)
        arma6x = arma(waveletx6)
        for i in numba.prange(waveletx6.size):
            waveletx6[i] = (savigol6x[i] + arma6x[i])/2 #average our results for best fit

        inverted5 = waveletinverse(waveletx6) #deconvolute our 6th wavelet

        savigol5x = adjacentaverage(waveletx5)
        arma5x = arma(waveletx5)

        for i in numba.prange(waveletx5.size):
            waveletx5[i] = (savigol5x[i] + arma5x[i])/2 #average our results for best fit

        for i in numba.prange(waveletx5.size):
            waveletx5[i] = (waveletx5[i]  + inverted5[i])/2
            #reconstitute our coefficients from the higher array

        inverted4 = waveletinverse(waveletx5)

        savigol4x = adjacentaverage(waveletx4)
        arma4x = arma(savigol4x)

        for i in numba.prange(waveletx4.size):
            waveletx4[i] = (savigol4x[i] + arma4x[i]) / 2  # average our results for best fit

        for i in numba.prange(waveletx5.size):
            waveletx4[i] = (waveletx4[i] + inverted4[i]) / 2
                # reconstitute our coefficients from the higher array

        inverted3 = waveletinverse(waveletx4)

        savigol3x = adjacentaverage(waveletx3)
        arma3x = arma(savigol3x)

        for i in numba.prange(waveletx3.size):
            waveletx3[i] = (savigol3x[i] + arma3x[i]) / 2  # average our results for best fit

        for i in numba.prange(waveletx5.size):
            waveletx4[i] = (waveletx3[i] + inverted3[i]) / 2
                # reconstitute our coefficients from the higher array

        inverted2 = waveletinverse(waveletx3)

        savigol2x = adjacentaverage(waveletx2)
        arma2x = arma(waveletx2)

        for i in numba.prange(waveletx2.size):
            waveletx2[i] = (savigol2x[i] + arma2x[i]) / 2  # average our results for best fit

        for i in numba.prange(waveletx5.size):
            waveletx2[i] = (waveletx2[i] + inverted2[i]) / 2
                # reconstitute our coefficients from the higher array

        inverted1 = waveletinverse(waveletx2)

        savigol1x = adjacentaverage(waveletx1)
        arma1x = arma(waveletx1)

        for i in numba.prange(waveletx1.size):
            waveletx1[i] = (savigol1x[i] + arma1x[i]) / 2  # average our results for best fit

        for i in numba.prange(waveletx5.size):
            waveletx1[i] = (waveletx1[i] + inverted1[i]) / 2
            # reconstitute our coefficients from the higher array

        #we now have noise smoothed wavelet transforms for 1-6 HAAR envelopes.




        prior_mean[:] = waveletinverse(waveletx1)


        #perform repeated, iterative smoothing and wavelet reconvolution.
        #its fuckin magic!
        #furthermore, because we use doubling, we can double each time.
        #we can go down to quite a bit of frequency accuracy.



        iterations = iterations +  1

        prior_variance[:] = posterior_variance[:]

        # APPLY BAYES' THEOREM 
        #fabada figure 8?
        for i in numba.prange(N):
                posterior_variance[i] = 1. / (1. / data_variance[i] + 1. / prior_variance[i])


    #fabada figure 7
        for i in numba.prange(N):
                posterior_mean[i] = (
                            ((prior_mean[i] / prior_variance[i]) + (data[i] / data_variance[i])) * posterior_variance[i])

        # EVALUATE EVIDENCE
        #fabada figure 6: probability distribution calculation	
        for i in numba.prange(N):
                ja1[i] = (prior_mean[i] - math.sqrt(data[i])) ** 2
                ja2[i] = (2. * (prior_variance[i] + data_variance[i]))
                ja3[i] = math.sqrt(TAU * (prior_variance[i] + data_variance[i]))
        for i in numba.prange(N):
                ja4[i] = math.exp((-ja1[i] / ja2[i]))
        for i in numba.prange(N):
                evidence[i] = ja4[i] / ja3[i]

        # EVALUATE CHI2
        #fabada figure 11
        for i in numba.prange(N):
            ja4[i] = math.sqrt((data[i] - posterior_mean[i]) ** 2) / (data_variance[i])
        chi2_data = numpy.sum(ja4)
        if numpy.isnan(chi2_data):
            chi2_data = data.size #typically this never overflows


        # COMBINE MODELS FOR THE ESTIMATION
        for i in numba.prange(N):
                model_weight[i] = evidence[i] * chi2_data

        for i in numba.prange(N):
                bayesian_weight[i] = bayesian_weight[i] + model_weight[i]
                bayesian_model[i] = bayesian_model[i] + (model_weight[i] * posterior_mean[i])

        if iterations >= iterationcontrol :
            break #break wherever the user wants it broken.

        with numba.objmode(current=numba.float64):
           current = time.time()
        timerun = (current - start) * 1000

        if (int(timerun) > int(timex)):
            break#  avoid exceeding our time budget, regardless

        # COMBINE ITERATION ZERO
    for i in numba.prange(N):
            model_weight[i] = initial_evidence[i] * chi2_data_min
    for i in numba.prange(N):
            bayesian_weight[i] = (bayesian_weight[i] + model_weight[i])
            bayesian_model[i] = bayesian_model[i] + (model_weight[i] * data[i])

    for i in numba.prange(N):
            data[i] = bayesian_model[i] / bayesian_weight[i]

    for i in numba.prange(N):
        #if boolv[i]:
        data[i] = data[i] * ((iterations // (iterations //2)) + (max_d - min_d) + min_d) #denormalize the data
        #((iterations // (iterations //2))
        # apply gain to compensate for the loss caused by processing. 

    for i in numba.prange(N):
        if (numpy.isnan(data[i])):
            data[i] = 0 #do not return NaN values

    return data

class FilterRun(Thread):
    def __init__(self, rb, pb, channels, processing_size, dtype,work,time,floor,run):
        super(FilterRun, self).__init__()
        self.running = True
        self.rb = rb
        self.processedrb = pb
        self.channels = channels
        self.processing_size = processing_size
        self.dtype = dtype
        self.buffer = numpy.ndarray(dtype=numpy.float64, shape=[int(self.processing_size * self.channels)])
        self.buffer2 = numpy.ndarray(dtype=numpy.float64, shape=[int(self.processing_size * self.channels)])
        self.buffer2 = self.buffer.reshape(-1, self.channels)
        self.buffer = self.buffer.reshape(-1, self.channels)
        self.work = work
        self.time = time
        self.floor = floor
        self.enabled = run
        self.NFFT = 512
        self.noverlap=446
        self.last = [0.,0.]
        self.iterationcontrol = 64





    def write_filtered_data(self):

        numpy.copyto(self.buffer, self.rb.read(self.processing_size).astype(dtype=numpy.float64))
        #self.buffer contains (44100,2) of numpy.float32 audio values sampled with pyaudio
        if self.enabled == False:
            self.processedrb.write(self.buffer.astype(dtype=self.dtype), error=True)
            return

        for i in range(self.channels):

            x = numpy.sign(self.last[i])
            fft = numpy.fft.rfft(self.buffer[:, i])
            zeros = numpy.zeros_like(fft)
            band = numpy.zeros_like(fft)
            band[18:11025] = fft[18:11025]#use a wide passband but filter stuff that will confuse us
            bandz = numba_fabada(numpy.fft.irfft(band),480,self.iterationcontrol)
            band = numpy.fft.rfft(bandz)
            zeros[18:11025] = band[18:11025]
            self.buffer2[:, i] = numpy.fft.irfft(zeros)

            v = []
            for b in range(14):
                    v.append(numpy.sign(self.buffer2[47984 +b, i]) )#get the signs of the last 16 elements
            v.append(numpy.sign(x)) #insert the last sign
            for b in range(14):
                    if (abs(v[b]) + abs(v[b+1])) == 2 and abs((v[b]) + (v[b+1])) != 2:#if -1 + -1 or 1 + 1, not a zero crossing
                     #zero crossing detected!
                        self.buffer2[47984 + b, i] = 0.0 #insert zero crossing fix

            self.last[i] = self.buffer2[-1, i] #copy out last sign of previous array
            self.buffer2[:, i] = round(self.buffer2[:, i],200)#
            self.buffer2[47800:48000, i] = unround(self.buffer2[47800:48000, i],200)
            #the rounding is necessary to prevent clicking

        self.processedrb.write(self.buffer2.astype(dtype=self.dtype), error=True)

    def run(self):
        while self.running:
            if len(self.rb) < self.processing_size * 2:
                sleep(0.05)  # idk how long we should sleep
            else:
                self.write_filtered_data()

    def stop(self):
        self.running = False


class StreamSampler(object):
    dtype_to_paformat = {
        # Numpy dtype : pyaudio enum
        'uint8': pyaudio.paUInt8,
        'int8': pyaudio.paInt8,
        'uint16': pyaudio.paInt16,
        'int16': pyaudio.paInt16,
        'uint24': pyaudio.paInt24,
        'int24': pyaudio.paInt24,
        "uint32": pyaudio.paInt32,
        'int32': pyaudio.paInt32,
        'float32': pyaudio.paFloat32,

        # Float64 is not a valid pyaudio type.
        # The encode method changes this to a float32 before sending to audio
        'float64': pyaudio.paFloat32,
        "complex128": pyaudio.paFloat32,
    }

    @classmethod
    def get_pa_format(cls, dtype):
        try:
            dtype = dtype.dtype
        except (AttributeError, Exception):
            pass
        return cls.dtype_to_paformat[dtype.name]

    def __init__(self, sample_rate=48000, channels=2, buffer_delay=1.5,  # or 1.5, measured in seconds
                 micindex=1, speakerindex=1, dtype=numpy.float32):
        self.pa = pyaudio.PyAudio()
        self._processing_size = sample_rate
        # np_rw_buffer (AudioFramingBuffer offers a delay time)
        self._sample_rate = sample_rate
        self._channels = channels
        self.ticker = 0
        self.rb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                     seconds=6,  # Buffer size (need larger than processing size)[seconds * sample_rate]
                                     buffer_delay=0,  # #this buffer doesnt need to have a size
                                     dtype=numpy.dtype(dtype))

        self.processedrb = AudioFramingBuffer(sample_rate=sample_rate, channels=channels,
                                              seconds=6,
                                              # Buffer size (need larger than processing size)[seconds * sample_rate]
                                              buffer_delay=1,
                                              # as long as fabada completes in O(n) of less than the sample size in time
                                              dtype=numpy.dtype(dtype))
        self.work = 1. #included only for completeness
        self.time = 495 #generally, set this to whatever timeframe you want it done in. 44100 samples = 500ms window.
        self.floor = 8192#unknown, seems to do better with higher values
        self.enabled = True
        self.filterthread = FilterRun(self.rb, self.processedrb, self._channels, self._processing_size, self.dtype,self.work,self.time, self.floor,self.enabled)
        self.micindex = micindex
        self.speakerindex = speakerindex
        self.micstream = None
        self.speakerstream = None
        self.speakerdevice = ""
        self.micdevice = ""


        # Set inputs for inheritance
        self.set_sample_rate(sample_rate)
        self.set_channels(channels)
        self.set_dtype(dtype)

    @property
    def processing_size(self):
        return self._processing_size

    @processing_size.setter
    def processing_size(self, value):
        self._processing_size = value
        self._update_streams()

    def get_sample_rate(self):
        return self._sample_rate

    def set_sample_rate(self, value):
        self._sample_rate = value
        try:  # RingBuffer
            self.rb.maxsize = int(value * 5)
            self.processedrb.maxsize = int(value * 5)
        except AttributeError:
            pass
        try:  # AudioFramingBuffer
            self.rb.sample_rate = value
            self.processedrb.sample_rate = value
        except AttributeError:
            pass
        self._update_streams()

    sample_rate = property(get_sample_rate, set_sample_rate)

    def get_channels(self):
        return self._channels

    def set_channels(self, value):
        self._channels = value
        try:  # RingBuffer
            self.rb.columns = value
            self.processedrb.columns = value
        except AttributeError:
            pass
        try:  # AudioFrammingBuffer
            self.rb.channels = value
            self.processedrb.channels = value
        except AttributeError:
            pass
        self._update_streams()

    channels = property(get_channels, set_channels)

    def get_dtype(self):
        return self.rb.dtype

    def set_dtype(self, value):
        try:
            self.rb.dtype = value
        except AttributeError:
            pass
        self._update_streams()

    dtype = property(get_dtype, set_dtype)

    @property
    def pa_format(self):
        return self.get_pa_format(self.dtype)

    @pa_format.setter
    def pa_format(self, value):
        for np_dtype, pa_fmt in self.dtype_to_paformat.items():
            if value == pa_fmt:
                self.dtype = numpy.dtype(np_dtype)
                return

        raise ValueError('Invalid pyaudio format given!')

    @property
    def buffer_delay(self):
        try:
            return self.rb.buffer_delay
        except (AttributeError, Exception):
            return 0

    @buffer_delay.setter
    def buffer_delay(self, value):
        try:
            self.rb.buffer_delay = value
            self.processedrb.buffer_delay = value
        except AttributeError:
            pass

    def _update_streams(self):
        """Call if sample rate, channels, dtype, or something about the stream changes."""
        was_running = self.is_running()

        self.stop()
        self.micstream = None
        self.speakerstream = None
        if was_running:
            self.listen()

    def is_running(self):
        try:
            return self.micstream.is_active() or self.speakerstream.is_active()
        except (AttributeError, Exception):
            return False

    def stop(self):
        try:
            self.micstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.speakerstream.close()
        except (AttributeError, Exception):
            pass
        try:
            self.filterthread.join()
        except (AttributeError, Exception):
            pass

    def open_mic_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxInputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        self.micdevice = devinfo["name"]
                        device_index = i
                        self.micindex = device_index

        if device_index is None:
            print("No preferred input found; using default input device.")

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              input=True,
                              input_device_index=self.micindex,  # device_index,
                              # each frame carries twice the data of the frames
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_read,
                              start=False  # Need start to be False if you don't want this to start right away
                              )

        return stream

    def open_speaker_stream(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            # print("Device %d: %s" % (i, devinfo["name"]))
            if devinfo['maxOutputChannels'] == 2:
                for keyword in ["microsoft"]:
                    if keyword in devinfo["name"].lower():
                        self.speakerdevice = devinfo["name"]
                        device_index = i
                        self.speakerindex = device_index

        if device_index is None:
            print("No preferred output found; using default output device.")

        stream = self.pa.open(format=self.pa_format,
                              channels=self.channels,
                              rate=int(self.sample_rate),
                              output=True,
                              output_device_index=self.speakerindex,
                              frames_per_buffer=int(self._processing_size),
                              stream_callback=self.non_blocking_stream_write,
                              start=False  # Need start to be False if you don't want this to start right away
                              )
        return stream

    # it is critical that this function do as little as possible, as fast as possible. numpy.ndarray is the fastest we can move.
    # attention: numpy.ndarray is actually faster than frombuffer for known buffer sizes
    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        audio_in = memoryview(numpy.ndarray(buffer=memoryview(in_data), dtype=self.dtype,
                                            shape=[int(self._processing_size * self._channels)]).reshape(-1,
                                                                                                         self.channels))
        self.rb.write(audio_in, error=False)
        return None, pyaudio.paContinue

    def non_blocking_stream_write(self, in_data, frame_count, time_info, status):
        # Read raw data
        # filtered = self.rb.read(frame_count)
        # if len(filtered) < frame_count:
        #     filtered = numpy.zeros((frame_count, self.channels), dtype=self.dtype)
        if len(self.processedrb) < self.processing_size:
            # print('Not enough data to play! Increase the buffer_delay')
            # uncomment this for debug
            audio = numpy.zeros((self.processing_size, self.channels), dtype=self.dtype)
            return audio, pyaudio.paContinue

        audio = self.processedrb.read(self.processing_size)
        chans = []
        for i in range(self.channels):
            filtered = audio[:, i]
            chans.append(filtered)

        return numpy.column_stack(chans).astype(self.dtype).tobytes(), pyaudio.paContinue



    def stream_start(self):
        if self.micstream is None:
            self.micstream = self.open_mic_stream()
        self.micstream.start_stream()

        if self.speakerstream is None:
            self.speakerstream = self.open_speaker_stream()
        self.speakerstream.start_stream()
        # Don't do this here. Do it in main. Other things may want to run while this stream is running
        # while self.micstream.is_active():
        #     eval(input("main thread is now paused"))

    listen = stream_start  # Just set a new variable to the same method




if __name__ == "__main__":
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    SS = StreamSampler(buffer_delay=0)
    SS.listen()
    SS.filterthread.start()
    def close():
        dpg.destroy_context()
        SS.filterthread.stop()
        SS.stop()
        quit()


    def fabadatoggle(sender, app_data,user_data):
        if SS.filterthread.enabled == True:
            dpg.set_item_label("toggleswitch", "Enable")
            SS.filterthread.enabled = False
        else:
            dpg.set_item_label("toggleswitch", "Disable")
            SS.filterthread.enabled = True

    def iterationset(sender, app_data,user_data):
        SS.filterthread.iterationcontrol = int(app_data)

    dpg.create_context()
    dpg.create_viewport(title='FABADA Streamclean', height=100, width=400)
    dpg.setup_dearpygui()
    dpg.configure_app(auto_device=True)

    with dpg.window(height = 100, width = 400) as main_window:
        dpg.add_text("Welcome to FABADA! 1S delay typical.")
        dpg.add_text("Adjust the slider to your preference.")
        dpg.add_slider_int(tag="iterations",max_value = 255, min_value = 1, default_value =64, callback=iterationset)
        dpg.add_button(label="Disable", tag="toggleswitch", callback=fabadatoggle)

    dpg.set_primary_window(main_window,True)  # TODO: Added Primary window, so the dpg window fills the whole Viewport

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    close() #clean up the program runtime when the user closes the window
