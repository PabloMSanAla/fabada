from __future__ import print_function, division
import numpy as np
import utilities as ut
from matplotlib import pyplot as plt
import FABADA as FAB
import scipy.signal
import statsmodels.api as smodel
import sys
import os
from time import time as time
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.lines as mlines

  
#%%

"""
##-------------------------------------------------------------------------
##          Fully Adaptive Bayesian Algorithm for Data Analysis
    
    1D Example Comparison
    
    With these code you will be able to reproduce any of the results of the
    M.Sc. Thesis of Pablo M. Sánchez for the spectra sample. 
    You only have to change the name of the spectrum you want to reproduce and 
    the noise level in the following variables (name and sig)  
    
    Possible images:
        kurucz
        SN132D
        arp256

          
##-------------------------------------------------------------------------
"""

"Inputs" 

name = "SN132D"
sig = 5


""" 
##-------------------------------------------------------------------------
                                    MAIN
##-------------------------------------------------------------------------                        
"""

main_path = os.getcwd()
spectrumname = os.path.join(main_path,'test_spectra',name+'.csv')

spectrum = pd.read_csv(spectrumname)
data_real = (spectrum.flux/spectrum.flux.max())*255

np.random.seed(1)

# Adding Noise


data = data_real + np.random.normal(0, sig, data_real.shape)
errors = sig*np.ones_like(data)
variance = errors**2
N = len(data_real)
                     
                  
# Executing the Smoothing algoriths

smooth_names = ["FABADA $SM(\mathcal{E}_{MAX})$","FABADA $BM(\mathcal{E}_{IP})$",
          "FABADA $SM(MSE_{min})$","FABADA $BM(MSE_{min})$",
          'SGF','LOWESS',"LPFF","Median"]


results = {}
kwargs = {"debug" : False,"data_real":data_real,"sigma":sig}

# FADADA

results[smooth_names[0]] = FAB.FABADA_SM(data, variance,**kwargs)
results[smooth_names[1]] = FAB.FABADA_BM(data, variance)
results[smooth_names[2]] = FAB.FABADA_SMopt(data, variance,data_real)
results[smooth_names[3]] = FAB.FABADA_BMopt(data, variance,data_real)



tf = time()
bw=10
PSNR_sg = np.zeros((bw,bw))
for w in range(3,bw,2):
    sys.stdout.write('\r'+""+str(round((w/bw)*100,3))+" % done")
    i=1
    while (i<w) & (i<30):
        a = scipy.signal.savgol_filter(data,w, i)
        PSNR_sg[w,i] = ut.PSNR(a,data_real)
        i+=2
argmax = np.unravel_index(np.argmax(PSNR_sg, axis=None), PSNR_sg.shape)
results[smooth_names[4]] = scipy.signal.savgol_filter(data,
                                                      argmax[0],
                                                      argmax[1])
tf = time() - tf
sys.stdout.write('\r'+""+str(100.00)+" % done")
print(" ---> Sg time =",tf)


# LOCALLY WEIGHTED SCATTERPLOT SMOOTHING


tf = time()
PSNR_loess = [0]
param = 1
for f in range(1,150,5):
    a = smodel.nonparametric.lowess(data,range(1,len(data)+1),frac=1./f,it=0,
                                return_sorted = False)
    PSNR_loess.append(ut.PSNR(a,data_real))
    param = PSNR_loess[-1] - PSNR_loess[-2]


results[smooth_names[5]] = smodel.nonparametric.lowess(data,range(1,
    len(data)+1),frac=1./(1+np.argmax(PSNR_loess)*5),it=0, 
                                                       return_sorted = False)
sys.stdout.write('\r'+""+str(100.00)+" % done")
print(" ---> LOWESS time =",time()-tf,np.argmax(PSNR_loess))


# LOW PASS FRECUENCY FILTER MSE OPTIMIZED

tf = time()
PSNR_fft = []
data_spec = np.fft.fftshift(np.fft.fft(data))
i_max = int(np.floor(len(data)/2))
for rad in range(1,i_max,5):
    sys.stdout.write('\r'+""+str(round(rad/i_max*100,2))+" % done")
    
    fftsmooth = np.abs(np.fft.ifft(np.fft.ifftshift(data_spec 
                                        * ut.gaussianLP_1D(rad,data.shape))))
    PSNR_fft.append(ut.PSNR(fftsmooth,data_real))

data_spec = np.fft.fftshift(np.fft.fft(data))* ut.gaussianLP_1D(
                                    (1 + np.argmax(PSNR_fft)*5),data.shape)
results[smooth_names[6]] = np.abs(np.fft.ifft(np.fft.ifftshift(data_spec)))
sys.stdout.write('\r'+""+str(100.00)+" % done")
print(" ---> FFT time =",time()-tf)

# MEDIAN

tg=time()
PSNR_med = {}
for s in range(1,20,2):
   sys.stdout.write('\r'+""+str(round(s/20*100,2))+" % done")
   results[smooth_names[7]] = scipy.signal.medfilt(data,s)
   PSNR_med[str(s)] = ut.PSNR(results[smooth_names[7]],data_real)

results[smooth_names[7]] = scipy.signal.medfilt(data,
    int(max(PSNR_med, key=PSNR_med.get)))

sys.stdout.write('\r'+""+str(100.00)+" % done")
print(" ---> Median time = ", time()-tg)


# Computing PSNR 

PSNR_all = {}
for key in results:
    PSNR_all[key] = ut.PSNR(results[key],data_real)

PSNR_all["Data"] = ut.PSNR(data,data_real)


    
# os.system('clear')

print(64*"=")
print(30*"-"+"PSNR"+30*"-")
print(27*"-"+"sigma = "+'{:02.0f}'.format(sig)+27*"-")
for sm,psnr in sorted(PSNR_all.items(),key=lambda item: item[1],reverse=True):
    print(sm,(50-len(sm))*" ",'{:02.3f}'.format(psnr)," (dB)")
    
    

# Plotting results 
#%%

if name == "kurucz":
    rect = [0.5,0.3,0.5,0.4]
    subpos = [[231,317],[33,126]]
    loc1,loc2 = 2,4
elif name == "SN132D":
    rect = [0.0,0.6,0.6,0.4]
    subpos = [[4351,4585],[-7,171]]
    loc1,loc2 = 2,4
elif name == "arp256":
    rect = [0.0,0.6,0.6,0.4]
    subpos = [[785,923],[40,100]]
    loc1,loc2 = 2,4



labels = ["$DATA_{REAL}$","NOISY ($\sigma=$"+
          '{:2.2f}'.format(sig)+" / "+
          "PSNR="+'{:2.2f}'.format(PSNR_all["Data"])+")",
          "$SM(\mathcal{E}_{MAX})$","$BM(\mathcal{E}_{IP})$",
          "$SM(MSE_{min})$","$BM(MSE_{min})$",
          "MEDIAN",'SGF','LPFF',"Photoshop", "BM3D"]

handles = [mlines.Line2D([], [], ls="-",color="green",lw=2),
           mlines.Line2D([], [], ls="-",color="red",lw=2),
           mlines.Line2D([], [], ls="-",color="blue",lw=2)]

labels2 = ["$DATA_{REAL}$",
           "NOISY ($\sigma=$"+
          '{:2.2f}'.format(sig)+" / "+
          "PSNR="+'{:2.2f}'.format(PSNR_all["Data"])+")",
           "RECOVER"]
    

cols = int(np.ceil(len(smooth_names)/2))
rows = 2
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
fig, ax = plt.subplots(rows,cols,figsize=(20,6),sharex='all',sharey='all')
plt.subplots_adjust(top=0.935,bottom=0.005,left=0.005,
                    right=0.995,hspace=0.0,wspace=0.0)

k=0
for i in range(rows):
    for j in range(cols):
        if k<len(smooth_names):
            ax[i][j].plot(data,'r-',alpha = 0.2,lw=0.5)
            ax[i][j].plot(data_real,'g-',lw=1.4)
            ax[i][j].plot(results[smooth_names[k]],"b-",lw=0.8)
            ax[i][j].text(0.1,0.03,smooth_names[k]+" / PSNR ="+
                    '{:02.2f}'.format(PSNR_all[smooth_names[k]]),
                    transform=ax[i][j].transAxes, verticalalignment='bottom',
                    horizontalalignment='left',bbox=props)
            ax[i][j].set_xticks([]) ; ax[i][j].set_yticks([])
            ax[i][j].set_ylim([-10,260])
            ax[i][j].set_xlim([-0.3,len(data)+5])
            subax = ut.add_subplot_axes(ax[i][j],rect,color="black")
            subax.plot(data_real,'g-',lw=3)
            subax.plot(data,'r-',alpha = 0.2,lw=0.8)
            subax.plot(results[smooth_names[k]],"b-",lw=1.3)
            subax.set_xlim(subpos[0])
            subax.set_ylim(subpos[1])
            subax.set_xticks([])
            subax.set_yticks([])
            mark_inset(ax[i][j], subax, loc1=loc1, loc2=loc2, fc="none", ec="0.5",
                        color="white",lw=1)
        
            if (len(data_real)>1300) & (len(data_real)<3000):
                ax[i][j].set_xlim([90,1645])    
        else:
            ax[i][j].axis("off")
        k+=1
        
fig.legend(handles, labels2,loc='upper center',
           fancybox=True, shadow=True, ncol=len(labels2))

# plt.savefig(name+"_sig"+'{:2.0f}'.format(sig)+".png",dpi=300)

