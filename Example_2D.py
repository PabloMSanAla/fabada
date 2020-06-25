from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt 
import FABADA as FAB
import utilities as ut
from bm3d import bm3d
import cv2
from time import time
import sys
import math
import scipy
import pandas as pd
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


#%%
"""
##-------------------------------------------------------------------------
##          Fully Adaptive Bayesian Algorithm for Data Analysis
    
    2D Example Comparison
    
    With these code you will be able to reproduce any of the results of the
    M.Sc. Thesis of Pablo M. Sánchez for the images sample. 
    You only have to change the name of the image you want to reproduce and 
    the noise level in the following variables (name and sig)  
    
    Possible images:
        cluster             eagle 
        ghost               stars
        bubble              crab 
        galaxies            saturn

          
##-------------------------------------------------------------------------
"""

"Inputs" 

name = "eagle"
sig = 40


""" 
##-------------------------------------------------------------------------
                                    MAIN
##-------------------------------------------------------------------------                        
"""

main_path = os.getcwd()
imagename = os.path.join(main_path,'test_images',name+'.png')



data_real = cv2.imread(imagename,0)/255


np.random.seed(1)
N=data_real.shape
mu, sig = 0, sig/255
noise = np.random.normal(mu, sig,N)
data = data_real + noise
errors = sig*np.ones_like(data)
variance = errors**2
kwargs = {"debug" : False,"data_real": data_real,"sigma":sig}

smooth_names = ["FABADA SM","FABADA BM",'FABADA SMopt',"FABADA BMopt",
                'BM3D', "Fourier Transform",'Savitzky–Golay','MEDIAN']

"Executing the algorithms"


results = {}

# FABADA

results[smooth_names[0]] = FAB.FABADA_SM(data, variance,**kwargs)
results[smooth_names[1]] = FAB.FABADA_BM(data, variance)
results[smooth_names[2]] = FAB.FABADA_SMopt(data, variance, data_real)
results[smooth_names[3]] = FAB.FABADA_BMopt(data, variance, data_real)

# BLOCK MATCHING AND 3D FILTERING
# http://www.cs.tut.fi/~foi/GCF-BM3D/

b0 = time()
results[smooth_names[4]] = bm3d(data, sig)
b0 = time()-b0

    
# LOW PASS FRECUENCY FILTER MSE OPTIMIZED

PSNR_fft = []
for rad in range(10,85,10):
    sys.stdout.write('\r'+""+str(round((rad/85)*100,3))+" % done")
    img_c3 = np.fft.fftshift(np.fft.fft2(data)) *ut.gaussianLP(rad,data.shape)
    img_c3 = np.abs(np.fft.ifft2(np.fft.ifftshift(img_c3)))
    PSNR_fft.append(ut.PSNR(img_c3,data_real))
tf = time()
img_c3 = np.fft.fftshift(np.fft.fft2(data)) * ut.gaussianLP(
    10+np.argmax(PSNR_fft)*10,data.shape)
results[smooth_names[5]] = np.abs(np.fft.ifft2(np.fft.ifftshift(img_c3)))
tf = time() - tf
sys.stdout.write('\r'+""+str(100.00)+" % done")
print(" ---> LPFF time =",tf)


# Savitzky–Golay MSE OPTIMIZED

tf = time()
bw=100
PSNR_sg = np.zeros((bw,bw))
for w in range(3,bw,2):
    sys.stdout.write('\r'+""+str(round((w/bw)*100,3))+" % done")
    i=0
    while (i<w) & (i<20):
        a = ut.sgolay2d(data,window_size=w, order=i)
        PSNR_sg[w,i]=ut.PSNR(a,data_real)
        i+=5
argmax = np.unravel_index(np.argmax(PSNR_sg, axis=None), PSNR_sg.shape)

results[smooth_names[6]] = ut.sgolay2d(
    data,window_size=argmax[0],order=argmax[1])
tf = time() - tf
sys.stdout.write('\r'+""+str(100.00)+" % done")
print(" ---> SGF time =",tf)

# MEDIAN MSE OPTIMIZED

tg=time()
PSNR_med = {}
for s in range(1,13,2):
   sys.stdout.write('\r'+""+str(round((s/13)*100,3))+" % done")
   results[smooth_names[7]] = scipy.signal.medfilt(data,s)
   PSNR_med[str(s)]= ut.PSNR(results[smooth_names[7]],data_real)

results[smooth_names[7]] = scipy.signal.medfilt(data,
    int(max(PSNR_med, key=PSNR_med.get)))

sys.stdout.write('\r'+""+str(100.00)+" % done")
print(" ---> Median time = ", time()-tg)


"Computing PSNR"

if not "Data + Noise" in results:
    smooth_names.append("Data + Noise")
    results["Data + Noise"] = data

PSNR_all = {}
for sm in smooth_names:
    PSNR_all[sm] = ut.PSNR(results[sm],data_real)
    
    
# EXTRACTING PSNR RESULTS FROM photoshop.csv

phot_file = os.path.join(main_path,"Photoshop.csv")
    
try:
    phot = pd.read_csv(phot_file, index_col="Image Name")
    phot = phot.loc[name].take(np.where(phot.loc[name,"Sigma"]==sig*255)[0])
    phot = phot.iloc[phot.loc[name,"PSNR (dB)"].argmax()]
    PSNR_all["Photoshop"] = phot['PSNR (dB)']
except:
    print("Not possible to extract PSNR values of photoshop")

print(34*"=")
print(15*"-"+"PSNR"+15*"-")
print(int((34-len(name))/2)*"-"+name+int((34-len(name))/2)*"-")
print(12*"-"+"sigma = "+'{:02.0f}'.format(sig*255)+12*"-")
for sm,psnr in sorted(PSNR_all.items(),key=lambda item: item[1],reverse=True):
    print(sm,(20-len(sm))*" ",'{:02.3f}'.format(psnr)," (dB)")

if not "Real Data" in results:
    results["Real Data"] = data_real

    

 #%%

"Plotting"


if name=="lense":
    subpos = [[178,225],[394,363]]
    rect = [0.0,0.0,0.36,0.22]
    loc1,loc2 = 1,2
elif name=="bubble":
    subpos = [[186,246],[72,38]]
    rect = [0.6,0.78,0.4,0.22]
    loc1,loc2 = 2,3
elif name=="ghost":
    subpos = [[425,489],[34,100]]
    rect = [0.0,0.6,0.4,0.4]
    loc1,loc2 = 3,4
elif name=="eagle":
    subpos = [[444,464],[482,444]]
    rect = [0.75,0.5,0.25,0.5]
    loc1,loc2 = 3,4
    
    
smooth_names = ["Real Data","Data + Noise","FABADA SM","FABADA BM",
                'FABADA SMopt',"FABADA BMopt",
                'MEDIAN','Savitzky–Golay',"Fourier Transform",'BM3D']
    

labels = ["$DATA_{REAL}$","NOISY" ,
          "$SM(\mathcal{E}_{MAX})$","$BM(\mathcal{E}_{IP})$",
          "$SM(MSE_{min})$","$BM(MSE_{min})$",
          "MEDIAN",'SGF','LPFF',"Photoshop", "BM3D"]

cols = math.ceil(len(smooth_names)/2)
rows = 2
fig, ax = plt.subplots(rows,cols,figsize=(20,8.5),sharex='all',sharey='all')
plt.subplots_adjust(top=0.950,bottom=0.0,left=0.0,
                    right=1.0,hspace=0.08,wspace=0.0)

vmin,vmax = np.nanpercentile(results["Real Data"],[5,97])

props = dict(boxstyle='round',facecolor='wheat', alpha=0.2)


k=0
for i in range(rows):
    for j in range(cols):
        if k<len(smooth_names):
            ax[i][j].imshow(results[smooth_names[k]],vmin=vmin, vmax=vmax,
                            cmap='gray')
            if smooth_names[k] == "Real Data" :
                ax[i][j].set_title(labels[k],size = 10,bbox = props)
                
            elif smooth_names[k] == "Data + Noise" :
                ax[i][j].set_title(labels[k]+"\n(PSNR = "+
                            '{:02.2f}'.format(PSNR_all[smooth_names[k]]
                            )+" / $\sigma=$"+'{:02.2f}'.format(sig*255)+")",
                            size = 10,bbox = props)
            else:
                ax[i][j].set_title(labels[k]+"(PSNR = "+
                            '{:02.2f}'.format(PSNR_all[smooth_names[k]])+")",
                            size = 10,bbox = props)
            try:
                subax = ut.add_subplot_axes(ax[i][j],rect,axisbg='w')
                subax.imshow(results[smooth_names[k]],
                             vmin=vmin,vmax=vmax,cmap='gray')
                subax.set_xlim(subpos[0])
                subax.set_ylim(subpos[1])
                subax.set_xticks([])
                subax.set_yticks([])
                mark_inset(ax[i][j], subax, loc1=loc1, loc2=loc2, 
                           fc="none",ec="1.0",lw=0.5)
            except:
                print("",end="")
            
            ax[i][j].axis("off")
        else:
            ax[i][j].axis("off")

        k+=1

# plt.savefig("Redaccion/MSE-"+name+"_"+'{:02.2f}'.format(sig*255
#                                                         )+".png",dpi=300)


