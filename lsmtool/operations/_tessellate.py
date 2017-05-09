# -*- coding: utf-8 -*-
#
# Defines the tessellate functions used by the group operation
#
# Based on the IDL version of the Voronoi binning method of:
#   Cappellari M., Copin Y., 2003, MNRAS, 342, 345
# Modified by David Rafferty as required for integration into LSMTool
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, copy and distribute unmodified copies for
# non-commercial purposes is granted. Permission to modify for
# personal or internal use is granted, provided this copyright
# and disclaimer are included unchanged at the beginning of
# the file. All other rights are reserved.


from __future__ import print_function
import numpy as np
from numpy import sum, sqrt, min, max, any
from numpy import argmax, argmin, mean, abs
from numpy import int32 as Nint
from numpy import float32 as Nfloat
import copy


def get_skymodel(skymodel_fn):
    """Returns x, y, flux arrays for input skymodel
    """
    import math as m
    from astropy import WCS

    y = []
    x = []
    fluxes = []

    # Make wcs object to handle transformation from ra and dec to pixel coords.
    w = WCS(naxis=2)
    w.wcs.crpix = [-234.75, 8.3393]
    w.wcs.cdelt = np.array([-0.066667, 0.066667])
    w.wcs.crval = [0, -90]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])
    arcsec_per_pix = abs(w.wcs.cdelt[0]) * 3600.0 # arcsec/pixel

    for line in open(skymodel_fn):
        sline=line.split(',')
        if line.startswith("FORMAT"): continue
        if line[0] == '#': continue
        if line[0] == '\n': continue
        name = str(sline[0])
        srctype = str(sline[1])
        ra_src = str(sline[2]).split(':')
        ra_deg = float(ra_src[0])*15.0 + (float(ra_src[1])/60.0)*15.0 + (float(ra_src[2])
            /3600.0)*15.0
        dec_src = str(sline[3]).split('.')
        if len(dec_src) == 3:
            dec_deg = float(dec_src[0]) + (float(dec_src[1])/60.0) + (float(dec_src[2])/3600.0)
        else:
            dec_deg = float(dec_src[0]) + (float(dec_src[1])/60.0) + (float(dec_src[2]
                + '.' + dec_src[3])/3600.0)
        flux = str(sline[4])
        fluxes.append(np.float(flux))
        ra_dec = np.array([[ra_deg, dec_deg]])
        try:
            x.append(w.wcs_world2pix(ra_dec, 1)[0][0])
            y.append(w.wcs_world2pix(ra_dec, 1)[0][1])
        except AttributeError:
            x.append(w.wcs_sky2pix(ra_dec, 1)[0][0])
            y.append(w.wcs_sky2pix(ra_dec, 1)[0][1])

    minx = np.min(x)
    miny = np.min(y)
    x += abs(minx)
    y += abs(miny)
    return x, y, np.array(fluxes), arcsec_per_pix


def dist2(x1,y1,x2,y2, scale=1.0) :
    return ((x1-x2)**2 + (y1-y2)**2)/scale**2


def guess_regular_grid(xnodes, ynodes, pixelsize=None) :
    """
    Return a regular grid guessed on an irregular one (Voronoi)
    xnodes, ynodes: arrays of Voronoi bins

    Return: xunb, yunb = regular grid for x and y (unbinned)
    """
    ## First deriving a pixel size
    xn_rav, yn_rav = xnodes.ravel(), ynodes.ravel()
    if pixelsize is None :
        pixelsize = derive_pixelsize(xnodes, ynodes)
    minxn = np.int(np.min(xn_rav) / pixelsize) * pixelsize
    minyn = np.int(np.min(yn_rav) / pixelsize) * pixelsize
    xunb, yunb = np.meshgrid(np.arange(minxn, np.max(xn_rav)+pixelsize, pixelsize),
                           np.arange(minyn, np.max(yn_rav)+pixelsize, pixelsize))

    return xunb, yunb


def derive_unbinned_field(xnodes, ynodes, data, xunb=None, yunb=None) :
    """
       Provide an array of the same shape as the input xunb, and yunb
       with the values derived from the Voronoi binned data

       xnodes, ynodes: 2 arrays providing the nodes from the binning
       data : values for each node
       xunb, yunb: x and y coordinates of the unbinned data
                 if not provided (default) they will be guessed from the nodes

       Return: xunb, yunb, and unbinned_data arrays with the same shape as xunb,
    """
    if xunb is None :
        xunb, yunb = guess_regular_grid(xnodes, ynodes)

    x_rav, y_rav = xunb.ravel(), yunb.ravel()
    xnodes_rav, ynodes_rav = xnodes.ravel(), ynodes.ravel()
    data_rav = data.ravel()
    unbinned_data = np.zeros_like(x_rav)
    for i in xrange(len(x_rav)) :
        indclosestBin = argmin(dist2(x_rav[i], y_rav[i], xnodes_rav, ynodes_rav))
        unbinned_data[i] = data_rav[indclosestBin]

    return xunb, yunb, unbinned_data.reshape(xunb.shape)


listmethods = ["voronoi", "quadtree"]
class bin2D:
    """
    Class for Voronoi binning of a set of x and y coordinates
    using given data and potential associated noise
    """
    def __init__(self, xin, yin, data, target_flux=1.0, pixelsize=None,
        method="Voronoi", cvt=1, wvt=0):
        self.xin = xin.ravel()
        self.yin = yin.ravel()
        self.data = data.ravel()

        ## Sort all pixels by their distance to the maximum flux
        if pixelsize is None :
            self.pixelsize = derive_pixelsize(self.xin, self.yin, verbose=1)
        else :
            self.pixelsize = pixelsize

        self.target_flux = target_flux

        ## Binning method and options
        self.method = str.lower(method)
        self.cvt = cvt
        self.wvt = wvt
        self.scale = 1.0
        self._check_input()

    def _check_input(self):
        """
        Check consistency of input data
        """
        # Basic checks of the data
        # First about the dimensions of the datasets
        self.npix = len(self.xin)

    ### Accrete the bins ============================================
    def bin2d_accretion(self, verbose=0):
        """ Accrete the bins according to their flux
        """
        ## Status holds the bin number when assigned
        self.status = np.zeros(self.npix, dtype=Nint)
        ## Good is 1 if the bin was accepted
        self.good = np.zeros(self.xin.size, dtype=Nint)

        ## Start with the max flux in the data
        currentBin = [argmax(self.data)]

        for ind in range(1,self.npix+1):  ## Running over the index of the Voronoi BIN
            ## Only one pixel at this stage
            current_flux = sum(self.data[currentBin])
            if verbose :
                print("Bin %d"%(ind)) # TODO: Change to logging

            self.status[currentBin] = ind   # only one pixel at this stage
            ## Barycentric centroid for 1 pixel...
            xbar, ybar = self.xin[currentBin], self.yin[currentBin]

            ## Indices of remaining unbinned data
            unBinned = np.where(self.status == 0)[0]
            ##+++++++++++++++++++++++++++++++++++++++++++++++++
            ## STOP THE WHILE Loop if all pixels are now binned
            while len(unBinned) > 0 :
            ##+++++++++++++++++++++++++++++++++++++++++++++++++

                ## Coordinates of the Remaining unbinned pixels
                xunBin = self.xin[unBinned]
                yunBin = self.yin[unBinned]

                ## Location of current bin
                xcurrent = self.xin[currentBin]
                ycurrent = self.yin[currentBin]

                ## Closest unbinned pixel to the centroid making sure the highest flux is used
                indclosestBar = argmin(dist2(xbar,ybar, xunBin, yunBin))
                xclosest = xunBin[indclosestBar]
                yclosest = yunBin[indclosestBar]

                ## Distance between this closest pixel and the current pixel
                currentSqrtDist= sqrt(min(dist2(xclosest,yclosest, xcurrent,ycurrent)))

                ## Add new pixel
                possibleBin = currentBin + [unBinned[indclosestBar]]

                ## Transfer new flux to current value
                old_flux = current_flux
                current_flux = sum(self.data[possibleBin])

                ## Test if getting better for flux
                if (current_flux > 0.8 * self.target_flux):
                    if (current_flux < 1.2 * self.target_flux):
                        self.good[possibleBin] = 1
                        break
                    elif (abs(current_flux-self.target_flux) > abs(old_flux-self.target_flux)):
                        if (old_flux > 0.8 * self.target_flux):
                            self.good[currentBin] = 1
                        break
                ##++++++++++++++++++++++++++++++++++++++++++

                ## If the new Bin is ok we associate the number of the bin to that one
                self.status[unBinned[indclosestBar]] = ind
                ##   ... and we use that now as the current Bin
                currentBin = possibleBin

                ## And update the values
                ## First the centroid
                xbar, ybar = mean(self.xin[currentBin]), mean(self.yin[currentBin])

                ## New set of unbinned pixels
                unBinned = np.where(self.status == 0)[0]
                ## ----- End of While Loop --------------------------------------------

            ## Unbinned pixels
            unBinned = np.where(self.status == 0)[0]
            ## Break if all pixels are binned
            if len(unBinned) == 0 :
                break
            ## When the while loop is finished for this new BIN
            ## Find the centroid of all Binned pixels
            Binned = np.where(self.status != 0)[0]
            xbar, ybar = mean(self.xin[Binned]), mean(self.yin[Binned])

            ## Closest unbinned pixel to the centroid of all Binned pixels
            xunBin = self.xin[unBinned]
            yunBin = self.yin[unBinned]
            indclosestBar = argmin(dist2(xbar,ybar, xunBin, yunBin))
            ## Take now this closest pixel as the next one to look at
            currentBin = [unBinned[indclosestBar]]

        ## Set to zero all bins that did not reach the target flux
        self.status *= self.good

    ### Compute centroid of bins ======================================
    def bin2d_centroid(self, verbose=0):
        ## Compute the area for each node
        self.Areanode, self.bins = np.histogram(self.status, bins=np.arange(np.max(self.status+0.5))+0.5)
        ## Select the ones which have at least one bin
        self.indgoodbins = np.where(self.Areanode > 0)[0]
        self.Areanode = self.Areanode[self.indgoodbins]
        ngoodbins = self.indgoodbins.size
        ## Reset the xnode, ynode, SNnode, and statusnode (which provides the number for the node)
        self.xnode = np.zeros(ngoodbins, Nfloat)
        self.ynode = np.zeros_like(self.xnode)
        self.flux_node = np.zeros_like(self.xnode)
        self.statusnode = np.zeros_like(self.xnode)
        self.listbins = []
        for i in range(ngoodbins) :
            ## indgoodbins[i] provides the bin of the Areanode, so indgoodbins[i] + 1 is the status number
            self.statusnode[i] = self.indgoodbins[i]+1
            ## List of bins which have statusnode as a status
            listbins = np.where(self.status==self.statusnode[i])[0]
            ## Centroid of the node
            self.xnode[i], self.ynode[i] = mean(self.xin[listbins]), mean(self.yin[listbins])
            self.flux_node[i] = sum(self.data[listbins])
            self.listbins.append(listbins)

    ### Compute WEIGHTED centroid of bins ======================================
    def bin2d_weighted_centroid(self, weight=None, verbose=0):
        if weight is not None : self.weight = weight

        self.Areanode, self.bins = np.histogram(self.status, bins=np.arange(np.max(self.status+0.5))+0.5)
        ## Select the ones which have at least one bin
        self.indgoodbins = np.where(self.Areanode > 0)[0]
        self.Areanode = self.Areanode[self.indgoodbins]
        ngoodbins = self.indgoodbins.size
        ## Reset the xnode, ynode, SNnode, and statusnode (which provides the number for the node)
        self.xnode = np.zeros(ngoodbins, Nfloat)
        self.ynode = np.zeros_like(self.xnode)
        self.flux_node = np.zeros_like(self.xnode)
        self.statusnode = np.zeros_like(self.xnode)
        self.listbins = []
        for i in range(ngoodbins) :
            ## indgoodbins[i] provides the bin of the Areanode, so indgoodbins[i] + 1 is the status number
            self.statusnode[i] = self.indgoodbins[i]+1
            ## List of bins which have statusnode as a status
            listbins = np.where(self.status==self.statusnode[i])[0]
            ## Weighted centroid of the node
            self.xnode[i], self.ynode[i] = np.average(self.xin[listbins], weights=self.weight[listbins]), np.average(self.yin[listbins], weights=self.weight[listbins])
            self.flux_node[i] = sum(self.data[listbins])
            self.listbins.append(listbins)

    ### Assign bins ============================================
    def bin2d_assign_bins(self, sel_pixels=None, scale=None, verbose=0) :
        """
        Assign the bins when the nodes are derived With Scaling factor
        """
        if scale is not None: self.scale = scale
        if sel_pixels is None : sel_pixels = range(self.xin.size)
        for i in sel_pixels :
            minind = argmin(dist2(self.xin[i], self.yin[i], self.xnode, self.ynode, scale=self.scale))
            self.status[i] = self.statusnode[minind]
            if verbose :
                print("Pixel ",  self.status[i], self.xin[i], self.yin[i], self.xnode[minind], self.ynode[minind]) # TODO: Change to logging

        ## reDerive the centroid
        self.bin2d_centroid()

    ### Do  CV tesselation ============================================
    def bin2d_cvt_equal_mass(self, wvt=None, verbose=1) :
        """
        Produce a CV Tesselation

        wvt: default is None (will use preset value, see self.wvt)
        """

        ## Reset the status and statusnode for all nodes
        self.status = np.zeros(self.npix, dtype=Nint)
        self.statusnode = np.arange(self.xnode.size) + 1

        if wvt is not None : self.wvt = wvt
        if self.wvt : self.weight = np.ones_like(self.data)
        else : self.weight = self.data**4

        self.scale = 1.0

        self.niter = 0
        ## WHILE LOOP: stop when the nodes do not move anymore ============
        Oldxnode, Oldynode = copy.copy(self.xnode[-1]), copy.copy(self.ynode[-1])
        while (not np.array_equiv(self.xnode, Oldxnode)) | (not np.array_equiv(self.ynode, Oldynode)):
            Oldxnode, Oldynode = copy.copy(self.xnode), copy.copy(self.ynode)
            ## Assign the closest centroid to each bin
            self.bin2d_assign_bins()

            ## New nodes weighted centroids
            self.bin2d_weighted_centroid()

            ## Eq. (4) of Diehl & Statler (2006)
            if self.wvt : self.scale = sqrt(self.Areanode/self.flux_node)
            self.niter += 1

    ### Do Voronoi binning ============================================
    def bin_voronoi(self, wvt=None, cvt=None, verbose=0) :
        """ Actually do the Voronoi binning

        wvt: default is None (will use preset value, see self.wvt)
        cvt: default is None (will use preset value, see self.cvt)
        """
        if cvt is not None : self.cvt = cvt
        if wvt is not None : self.wvt = wvt

        self.bin2d_accretion(verbose=verbose)
        self.bin2d_centroid()
        ## Get the bad pixels, not assigned and assign them
        badpixels = np.where(self.status == 0)[0]
        self.bin2d_assign_bins(badpixels)
        if self.cvt :
            self.bin2d_cvt_equal_mass()
        else : self.scale = 1.0
        ## Final nodes weighted centroids after assigning to the final nodes
        self.bin2d_assign_bins()
        if self.wvt : self.weight = np.ones_like(self.data)
        else : self.weight = self.data
        self.bin2d_weighted_centroid()


    def show_voronoibin(self, datain=None, shownode=1, mycmap=None) :
        """
        Display the voronoi bins on a map

        datain: if None (Default), will use random colors to display the bins
                if provided, will display that with a jet (or specified mycmap) cmap
                   (should be either the length of the voronoi nodes array or the size of the initial pixels)
        shownode: default is 1 -> show the voronoi nodes, otherwise ignore (0)
        mycmap: in case datain is provide, will use that cmpa to display the bins
        """
        from distutils import version
        try:
            import matplotlib
        except ImportError:
            raise Exception("matplotlib 0.99.0 or later is required for this routine")

        if version.LooseVersion(matplotlib.__version__) < version.LooseVersion('0.99.0'):
            raise Exception("matplotlib 0.99.0 or later is required for this routine")

        from matplotlib.collections import PatchCollection
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        fig = plt.figure(1,figsize=(7,7))
        plt.clf()
        ax = plt.gca()
        patches = []
        binsize = self.pixelsize
        for i in range(len(self.xin)) :
            patches.append(mpatches.Rectangle((self.xin[i],self.yin[i]), binsize*10, binsize*10))

        if datain is None :
            dataout = self.status
            mycmap = 'prism'
        else :
            if len(datain) == self.xnode.size :
                dataout = np.zeros(self.xin.size, Nfloat)
                for i in range(self.xnode.size) :
                    listbins = self.listbins[i]
                    dataout[listbins] = [datain[i]]*len(listbins)
            elif len(datain) == self.xin.size :
                dataout = datain
            if mycmap is None : mycmap = 'jet'

        colors = dataout * 100.0 / max(dataout)
        collection = PatchCollection(patches, cmap=mycmap)
        collection.set_array(np.array(colors))
        ax.add_collection(collection)
        if shownode :
            plt.scatter(self.xnode,self.ynode, marker='o', edgecolors='k', facecolors='none', s=100)
            for i in range(self.xnode.size):
                ax.annotate(i, (self.xnode[i], self.ynode[i]))
                listbins = self.listbins[i]

        plt.axis('image')
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.title("Voronoi Map")

    def bin_data(self, datain=None, noisein=None) :
        """
        Return a Voronoi adaptive binning of your data.

        datain: if provided, will be used as data input
                if not provided (None = default), will use self.data
        noisein: if provided, will be used as noise input
                if not provided (None = default), will use self.noise

        Output = xnode, ynode, bindata, S/N
        """

        if datain is None: datain = copy.copy(self.data)

        dataout = np.zeros(self.xnode.size, Nfloat)
        xout = np.zeros_like(dataout)
        yout = np.zeros_like(dataout)
        flux_out = np.zeros_like(dataout)
        for i in range(self.xnode.size) :
            listbins = self.listbins[i]
            xout[i] = np.average(self.xin.ravel()[listbins], weights=datain[listbins])
            yout[i] = np.average(self.yin.ravel()[listbins], weights=datain[listbins])
            dataout[i] = mean(datain[listbins])
            flux_out[i] = sum(datain[listbins])

        return xout, yout, dataout, flux_out


def derive_pixelsize(x, y, verbose=0) :
    """ Find the pixelsize by looking at the minimum distance between
        pairs of x,y
        x: xaxis coordinates
        y: yaxis coordinates
        Return: pixelsize
    """
    pixelsize = 1.e30
    for i in range(len(x)-1) :
        mindist = np.min(dist2(x[i], y[i], x[i+1:], y[i+1:]))
        pixelsize = np.minimum(mindist, pixelsize)
    pixelsize = np.sqrt(pixelsize)
    return pixelsize


def bins2Patches(bin_obj, root='Patch', pad_index=False):
    """Returns a patch column based on binning"""
    patchNames = []
    for src_idx in range(len(bin_obj.data)):
        for i in range(bin_obj.xnode.size):
            listbins = bin_obj.listbins[i]
            if src_idx in listbins:
                if pad_index:
                    patchNames.append('{0}_{1}'.format(root,
                        str(i).zfill(int(np.ceil(np.log10(bin_obj.xnode.size))))))
                else:
                    patchNames.append('{0}_{1}'.format(root, i))
    return np.array(patchNames)

