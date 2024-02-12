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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sys import stdin, argv
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

# set output deirectory here
odir = '/Users/cabrera-ziri/Work/data/eso/reticulum/sky_residual_corrected6/'

# you can set here the dimensions
# of the figure
xsize = 14
ysize = 8
ncol = 3 #no. of columns in legend

# centre the figure on the CaT
xlim_lo,xlim_hi = 849.1,870

# turn this off if you don't want to produce figures
plot_fig = True  

def poly_continuum(y,x):
    #note the change in argument order in this version
    # this fits a poly to the continuum

    # assumes wavelength array is in nm
    # and that fluctuations are in 10nm scale
    order = int((x[-1]-x[0])/(10))

    try:
        # standardize    
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        x_train = x_scaler.fit_transform(x[..., None])
        y_train = y_scaler.fit_transform(y[..., None])

        # fit model
        model = Pipeline([('poly', PolynomialFeatures(degree=order)),
                  ('linear', HuberRegressor())])

        model.fit(x_train, y_train.ravel())

        # do some predictions
        test_x = x

        test_x_transformed = x_scaler.transform(test_x[..., None])
        y_transformed = model.predict(test_x_transformed)
        predictions = y_scaler.inverse_transform(y_transformed[..., None])

        return(predictions.ravel())
    except ValueError as e:
        print(e)
        print('couldn\'t fit continuum this time! probably too flat already...')
        return(np.ones_like(y))

def Huber_continuum(y,x):
    #note the change in arguments order in this version
    try:
        # standardize    
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        x_train = x_scaler.fit_transform(x[..., None])
        y_train = y_scaler.fit_transform(y[..., None])

        # fit model
        model = HuberRegressor(epsilon=1)
        model.fit(x_train, y_train.ravel())

        # do some predictions
        test_x = x
        
        test_x_transformed = x_scaler.transform(test_x[..., None])
        y_transformed = model.predict(test_x_transformed)
        predictions = y_scaler.inverse_transform(y_transformed[..., None])

        return(predictions.ravel())
    except ValueError as e:
        print(e)
        print('couldn\'t fit continuum this time! probably too flat already...')
        return(np.ones_like(y))

def sky_correct(model,wl,x,wl_lo=860,wl_hi=870):

    #define window to check residuals
    residual_window =  (wl >=wl_lo) & (wl<=wl_hi) 

    y = model.predict(x)

    #check rmse of uncorrected data
    x_mse = mean_squared_error(x[:,model.sky_pixels & residual_window],np.ones_like(x[:,model.sky_pixels & residual_window]))
    x_rmse = np.sqrt(x_mse)

    # rmse of corrected data
    x_predict = x.copy()
    x_predict[:,model.sky_pixels] = x_predict[:,model.sky_pixels]-y+1
    y_mse = mean_squared_error(x_predict[:,model.sky_pixels & residual_window],np.ones_like(x_predict[:,model.sky_pixels & residual_window]))
    y_rmse = np.sqrt(y_mse)
    # print('x rmse: {}'.format(x_rmse))
    # print('y rmse: {}'.format(y_rmse))

    if y_rmse<x_rmse:
        # print('correction improved the spectrum...\nReturning corrected spectrum')
        return x_predict,y
    else:
        # print('correction did not improve the spectrum...\nReturning original spectrum')
        return x,y

class MyPCA3(BaseEstimator, TransformerMixin):

    def __init__(self,sky_stddev,sky_pixels_flux_threshold,sky_pixel_scale_factor=1.03,explained_variance = 0.95):
        
        #now we carry out the sky_pixel definition just before running the PCA
        self.sky_stddev = sky_stddev
        self.sky_pixels_flux_threshold = sky_pixels_flux_threshold
        self.sky_pixel_scale_factor = sky_pixel_scale_factor
        self.sky_pixels = self.sky_stddev>=self.sky_pixels_flux_threshold*self.sky_pixel_scale_factor
        self.explained_variance = explained_variance

    def fit(self,X,y=None):
        X = check_array(X)  # checks that X is an array with finite float values

        print('training PCA...')
        # PCA run 
        self.model_ = PCA(random_state=42)
        self.model_.fit(X[:,self.sky_pixels])
        self.cumsum_ = np.cumsum(self.model_.explained_variance_ratio_)
        self.d_ = np.argmax(self.cumsum_>=self.explained_variance)+1
        
        print('{:} components explain {:}% of variance'.format(self.d_,self.explained_variance*100))

        # re-train PCA
        self.model_ = PCA(n_components=self.d_,random_state=42)
        self.model_.fit(X[:,self.sky_pixels])

        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!
    
    def predict(self,X):

        # this works for a single spec.
        self.weights = self.model_.transform(X[:,self.sky_pixels])
        y_predict = self.model_.inverse_transform(self.weights)
        return y_predict
    
    def score(self, X, y, **kwargs):
        return self.model_.score(X[:,self.sky_pixels], y, **kwargs)

if __name__ == '__main__':
                
    infiles = argv[1:]       
    if argv == ['./fit_sky_residuals.py']:
        print('reading from stdin...')
        infiles = [i[:-1] for i in stdin.readlines()]

    #here we load pca and sky pixel mask
    # pca_file = 'sky_residual_pca_18-01-24v2.joblib'
    # sky_pixels, pca = load(pca_file)
    pca_file = 'sky_residual_pca_09-02-24.joblib'
    pca = load(pca_file)
        
    print('start reading files...')

    for file in infiles:

        hdu = fits.open(file)

        #output file
        ofile = odir+file.split('/')[-1][:-5]+'_srs.fits'
	
        # wavelenght0
        ext=0 
        crval = hdu[ext].header['crval1']
        cdelt = hdu[ext].header['cdelt1']
        naxis = hdu[ext].header['naxis1']
                
        wl = crval+cdelt*np.arange(naxis)
        flx = hdu[ext].data
        label='{:}'.format(file[:-5].split('/')[-1])
    
        # cnt = Huber_continuum(wl,flx)
        cnt = poly_continuum(flx,wl)

        #normalize spectra for pca fit
        norm_flx = flx/cnt

        #pca fit
        # weights = pca.transform(norm_flx[sky_pixels][None,...])
        # model = pca.inverse_transform(weights)
        flx2,model = sky_correct(pca,wl,norm_flx[None,...],wl_lo=855.5,wl_hi=866.5)
    
        flx2 = flx2.ravel()

        # print(flx2.shape,norm_flx.shape)
        # print(flx2[:10],norm_flx[:10])

        if (flx2 == norm_flx).all():
            nothing_done = True
            print(file)
        else:
            nothing_done = False

        corr_flx = cnt*flx2 #this is the sky substracted spcetrum with continuum 

        if plot_fig:
            f, ax = plt.subplots(figsize=(xsize,ysize))
            if nothing_done:
                plt.title(label+' (not improved by sky residual subtration)')
            else:
                plt.title(label)

            #this bit shows the original data
            # ax.plot(wl,flx,'-',lw=1,label='original data')
            # ax.plot(wl,_model*cnt,'-',lw=1,label='sky residual model')
            # ax.plot(wl,corr_flx,ls='-',lw=1,label='sky residual substracted - 1')

            #this plot the normalised data for clarity
            ax.plot(wl,norm_flx,'-',lw=1,label='original data')
            _model = np.ones_like(wl)*np.nan
            _model[pca.sky_pixels] = model[0,:]
            ax.plot(wl,_model,'-',lw=1,label='sky residual model')
            
            if nothing_done:
                ax.plot(wl,flx2.ravel()-1,ls='-',lw=1,label='this spec. could not be improved')
            else:
                ax.plot(wl,flx2.ravel()-1,ls='-',lw=1,label='sky residual substracted - 1')
            
            # for reference show restframe the positions of CaT lines
            for _ in [850.04,854.44, 866.45]:
                ax.axvline(_,lw=1,ls='--',color='k')

            #set automatic yscale
            _flxs = np.r_[norm_flx[(wl>=xlim_lo)&(wl<=xlim_hi)],
                        flx2[(wl>=xlim_lo)&(wl<=xlim_hi)]-1,
                        model[0][(wl[pca.sky_pixels]>=xlim_lo)&(wl[pca.sky_pixels]<=xlim_hi)]]

            # _flxs_min,_flxs_max = _flxs.min(),_flxs.max()
            _flxs_min,_flxs_max = np.percentile(_flxs,2),np.percentile(_flxs,98)

            if _flxs_min<0:
                # ax.set_ylim(1.1*_flxs_min,1.1*_flxs_max)
                ax.set_ylim(1.5*_flxs_min-1,1.5*_flxs_max)
            else:
                ax.set_ylim(0.5*_flxs_min-1,1.5*_flxs_max)

            plt.legend(markerscale=12,ncol=ncol)
            ax.set_xlim(xlim_lo,xlim_hi)
            ax.set_xlabel('wl [nm]')
            ax.set_ylabel('normalized flux')
            plt.tight_layout()
            plt.grid()
            # plt.show(block=True)
            plt.savefig(ofile[:-5]+'.png')
            plt.close('all')

        # write sky residual corrected file
        _hdu = fits.PrimaryHDU(corr_flx,header=hdu[0].header)

        # update headear
        _hdu.header['skypca'] = pca_file 
    
        header = _hdu.header       
        header.comments['skypca'] = 'sky residual pca file used'

        # hdul = fits.HDUList([_hdu])
        _hdu.writeto(ofile,overwrite=True)