"""
FABADA example of denosing a astronomical spectrum based on

P.M. Sanchez-Alarcon, Y. Ascasibar, 2022
"Fully Adaptive Bayesian Algorithm for Data Analysis. FABADA"
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
	from skimage.metrics import structural_similarity as ssim
except:
	print("skimage is not install. SSIM would not be displayed.")
from fabada import fabada,PSNR


def main():
	# IMPORTING SPECTRUM
	y = np.array(pd.read_csv('arp256.csv').flux)[100:1530]
	y = (y/y.max())*255  # Normalize to 255

	# ADDING RANDOM GAUSSIAN NOISE
	np.random.seed(12431)
	sig      = 10 			# Standard deviation of noise
	noise    = np.random.normal(0, sig ,y.shape)
	z        = y + noise 
	variance = sig**2

	# APPLY FABABA FOR RECOVER
	y_recover = fabada(z,variance)

	# SHOW RESULTS
	show_results(y,z,y_recover)


def show_results(y,z,y_recover):
	# SHOW RESULTS
	fig,ax = plt.subplots(1,1,figsize=(11,3))

	ax.plot(y,"r-",alpha=.3,lw=1,label="Signal")

	try:
		text_noisy = "({:2.2f}/{:1.2})".format(
                PSNR(z, y, L=255),
                ssim(z, y, data_range=255))

		text_recover = "({:2.2f}/{:1.2})".format(
                PSNR(y_recover, y, L=255),
                ssim(y_recover, y, data_range=255))
	except:
		text_noisy = "({:2.2f})".format(
                PSNR(z, y, L=255))
		
		text_recover = "({:2.2f})".format(
                PSNR(y_recover, y, L=255))

	ax.plot(z,"k-",alpha=.1,lw=1,label='Noisy ' + text_noisy)
	ax.plot(y_recover,"k-",alpha=1,lw=1,
							label='Recover '+ text_recover)
	ax.set_xlim([-1,y.size])
	ax.set_ylim([-0.01*255,0.65*255])
	ax.set_xticks([]); ax.set_yticks([])
	fig.legend(loc='upper center',fancybox=True, shadow=True, ncol=3)
                    

	plt.subplots_adjust(top=0.97,bottom=0.0,left=0.01,
                    right=0.99,wspace=0.0,hspace=0.00) 
	
	plt.show()


if __name__ == '__main__':
    main()
