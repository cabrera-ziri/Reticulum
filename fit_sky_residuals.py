#!/usr/bin/env python

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from glob import glob
from sklearn.decomposition import PCA
from joblib import load
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sys import stdin, argv

# you can set here the dimensions
# of the figure
xsize = 14
ysize = 8
ncol = 3 #no. of columns in legend

# centre the figure on the CaT
xlim_lo,xlim_hi = 849,870

# turn this off if you don't want to produce figures
plot_fig = True

def Huber_continuum(x,y):
 
    # standardize    
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x[..., None])
    y_train = y_scaler.fit_transform(y[..., None])

    # fit model
    model = HuberRegressor(epsilon=1)
    model.fit(x_train, y_train.ravel())

    # do some predictions
    test_x = x
    predictions = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(test_x[..., None]))
        )
    return(predictions)

def readspec(file):
	"""
	this function reads the spectrum from the fits file
	"""
	
	hdu = fits.open(file)
	
	# wavelenght0
	ext=0 
	crval = hdu[ext].header['crval1']
	cdelt = hdu[ext].header['cdelt1']
	naxis = hdu[ext].header['naxis1']
	        
	wl = crval+cdelt*np.arange(naxis)
	flux = hdu[ext].data
	label='{:}'.format(file[:-5].split('/')[-1])
	return wl,flux,label  

if __name__ == '__main__':
                
    infiles = argv[1:]       
    if argv == ['./fit_sky_residuals.py']:
        print('reading from stdin...')
        infiles = [i[:-1] for i in stdin.readlines()]

    #here we load pca and sky pixel mask    
    sky_pixels, pca = load('./sky_residual_pca_16-01-24.joblib')
        
    print('plotting...')

    for file in infiles:

        wl,flx,label = readspec(file)
        cnt = Huber_continuum(wl,flx)

        #normalize spectra for pca fit
        norm_flx = flx/cnt

        #pca fit
        weights = pca.transform(norm_flx[sky_pixels][None,...])
        model = pca.inverse_transform(weights)

        #write model in similar array to data
        _model = np.ones_like(wl)*np.nan
        _model[sky_pixels] = model[0,:]    
        flx2 = norm_flx.copy()
        flx2[sky_pixels] = flx2[sky_pixels]-model[0,:]+1

        corr_flx = cnt*flx2 #this is the sky substracted spcetrum with continuum 

        if plot_fig:
            f, ax = plt.subplots(figsize=(xsize,ysize))
            plt.title(label)

            #this bit shows the original data
            # ax.plot(wl,flx,'-',lw=1,label='original data')
            # ax.plot(wl,_model*cnt,'-',lw=1,label='sky residual model')
            # ax.plot(wl,corr_flx,ls='-',lw=1,label='sky residual substracted - 1')

            #this plot the normalised data for clarity
            ax.plot(wl,norm_flx,'-',lw=1,label='original data')
            ax.plot(wl,_model,'-',lw=1,label='sky residual model')
            ax.plot(wl,flx2-1,ls='-',lw=1,label='sky residual substracted - 1')

            for _ in [850.04,854.44, 866.45]:
                ax.axvline(_,lw=1,ls='--',color='k')

            #set automatic yscale
            _flxs = np.r_[norm_flx[(wl>=xlim_lo)&(wl<=xlim_hi)],
                        flx2[(wl>=xlim_lo)&(wl<=xlim_hi)]-1,
                        model[0][(wl[sky_pixels]>=xlim_lo)&(wl[sky_pixels]<=xlim_hi)]]

            _flxs_min,_flxs_max = _flxs.min(),_flxs.max()

            if _flxs_min<0:
                ax.set_ylim(1.1*_flxs_min,1.1*_flxs_max)
            else:
                ax.set_ylim(0.9*_flxs_min,1.1*_flxs_max)

            plt.legend(markerscale=12,ncol=ncol)
            ax.set_xlim(xlim_lo,xlim_hi)
            ax.set_xlabel('wl [nm]')
            ax.set_ylabel('normalized flux')
            plt.tight_layout()
            plt.show(block=True)