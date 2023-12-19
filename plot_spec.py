#!/usr/bin/env python

##	For a single file you could just:
##	
##		% ./plot_spec.py spec_file
##	
##	For multiple files you could input
##	the list of files using stdin, e.g.:
##	
##		% ./plot_spec.py *.fits
##	
##		% ls *.fits | ./plot_spec.py
##	
##		% ./plot_spec.py < list.txt
##	
##		% ./plot_spec.py [ENTER]
##	
##			file1.fits
##			file2.fits
##			file3.fits
##			Ctrl+D (i.e. EOF to finish the reading)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# you can set here the dimensions
# of the figure
xsize = 14
ysize = 8
ncol = 3 #no. of columns in legend

def plotspec(ax,file):
	"""
	this function plots the spectra into the figure
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
	return ax.plot(wl,flux,label='{:}'.format(label))  

if __name__ == '__main__':	

	from sys import stdin, argv, exit

	#print(argv,len(argv))
        
	infiles = argv[1:]       
	if argv == ['./plot_spec.py']:
		print('reading from stdin...')
		infiles = [i[:-1] for i in stdin.readlines()]
	
	fig, ax = plt.subplots(figsize=(14,8))

	print('plotting...')	
	for file in infiles:
		print(file)	
		plotspec(ax,file)
	ax.set_xlabel('Wavelength [nm]',weight='bold')
	ax.set_ylabel('Flux [adu]',weight='bold')
	fig.tight_layout()
	plt.legend(ncol=ncols)
	plt.show(block=True)
