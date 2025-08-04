#!/usr/bin/env python
# coding: utf-8
import warnings

import numpy as np
import math
import random

import skfda
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix, KNeighborsHatMatrix
from skfda.preprocessing.smoothing import KernelSmoother, BasisSmoother
from skfda.representation import FDataGrid
from skfda.representation.basis import FourierBasis, BSplineBasis
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch, LinearSmootherGeneralizedCVScorer, shibata
import pywt


"""
smoother.py
===========

This module provides the `Smoother` class for transforming numpy ndarrays or skfda.FDataGrid objects into smoothed functional data representations using various smoothing techniques. The class supports B-spline, Fourier, wavelet, Nadaraya-Watson kernel, and k-nearest neighbors (kNN) smoothing bases, with automatic parameter selection via generalized cross-validation (GCV) where available.

Classes
-------
Smoother
    Transforms input data into a smoothed functional data object using the specified smoothing basis.

Examples
--------
>>> import numpy as np
>>> from GPmix.smoother import Smoother
>>> data = np.random.randn(10, 100)
>>> smoother = Smoother(basis='bspline', basis_params={'n_basis': 10})
>>> fd_smooth = smoother.fit(data)
"""

class Smoother:
    """
    Transform numpy ndarray or skfda.FDataGrid to a smoothed functional data object via smoothing.

    Parameters
    ----------
    basis : str, default='bspline'
        Smoothing basis to use. Supported options are:
            - 'bspline': B-spline basis smoothing.
            - 'fourier': Fourier basis smoothing.
            - 'wavelet': Wavelet basis smoothing.
            - 'nadaraya_watson': Nadaraya-Watson kernel smoothing.
            - 'knn': k-nearest neighbors kernel smoothing.
    basis_params : dict, default={}
        Additional parameters for the smoothing basis. If not provided, required parameters are selected via generalized cross-validation (GCV) where implemented.
        Example parameters:
            - B-spline: {'order': 3, 'n_basis': 20}
            - Wavelet: {'wavelet': 'db4', 'mode': 'soft'}
            - Kernel: {'bandwidth': 1.0}
            - Fourier: {'n_basis': 20, 'period': 1}
        For wavelet basis, GCV is not implemented.
    domain_range : tuple or None, default=None
        The domain range of the functional data. If None, the domain is set to [0, 1] for array-like data, or inherited from the FDataGrid object.

    Attributes
    ----------
    gcv_score : float or None
        The best GCV score found during parameter selection, if applicable.
    fd_smooth : skfda.FDataGrid
        The smoothed functional data object.
    grid_points : ndarray
        The grid points used for the functional data.

    Methods
    -------
    fit(fd, return_data=True)
        Fit the transformation to the input data and return the smoothed functional data.
    """

    def __init__(self, basis='bspline', basis_params={}, domain_range=None):
        """
        Initialize the Smoother object.

        Parameters
        ----------
        basis : str, default='bspline'
            Smoothing basis to use. See class docstring for options.
        basis_params : dict, default={}
            Additional parameters for the smoothing basis.
        domain_range : tuple or None, default=None
            The domain range of the functional data.
        """
        self.basis = basis
        self.basis_params = basis_params
        self.domain_range = domain_range

        #check the basis_params does not contain unwanted keys
        if not all(key in ['period', 'order', 'n_basis', 'wavelet', 'mode', 'bandwidth', 'n_neighbors'] for key in self.basis_params.keys()):
            raise ValueError('basis_params contains some unknown keys. '
                            'Ensure that the dict keys are limited to the following: '
                            "'period', 'order', 'n_basis', bandwidth', 'wavelet', 'mode', 'n_basis_grid', 'bandwidth_grid'."

            )
        

    def _gcv_fourier_smoothing(self, fdata: skfda.FDataGrid):
        '''implement gcv for selecting number of basis for smoothing with fourier basis.'''
        n_basis_grid = self.basis_params.get('n_basis_grid', range(3, 51, 2))
        domain_width = fdata.domain_range[0][1] - fdata.domain_range[0][0]
        grid = SmoothingParameterSearch(
            estimator = BasisSmoother(FourierBasis(domain_range = fdata.domain_range[0])),
            param_values = [FourierBasis(domain_range = fdata.domain_range[0], n_basis=i, period = domain_width)
                    for i in n_basis_grid],
            param_name ='basis',
            scoring = LinearSmootherGeneralizedCVScorer(shibata)
        )
        grid.fit(fdata)
        gcv_score = grid.best_score_.round(4)
 
        return gcv_score, grid.transform(fdata)

    def _gcv_bspline_smoothing(self, fdata : skfda.FDataGrid):
        '''implement gcv for selecting number of basis and order of B-spline basis for smoothing.'''
        n_basis_grid = self.basis_params.get('n_basis_grid', range(5, 31, 1))
        grid = SmoothingParameterSearch(
                estimator = BasisSmoother(BSplineBasis(domain_range = fdata.domain_range[0], n_basis=5)),
                param_values = [BSplineBasis(domain_range = fdata.domain_range[0], n_basis=i, order= random.randint(3, min(i, 15)))
                                for i in n_basis_grid],
                param_name='basis',
                scoring = LinearSmootherGeneralizedCVScorer(shibata),
                )
        grid.fit(fdata)
        gcv_score = grid.best_score_.round(4)

        return gcv_score, grid.transform(fdata)
    
    def _gcv_nw_kernel_smoothing(self, fdata : skfda.FDataGrid):
        '''implement gcv for selecting bandwidth for smoothing with Nadaraya-Watson kernel.'''
        domain_width = fdata.domain_range[0][1] - fdata.domain_range[0][0]
        bandwidth_grid = self.basis_params.get('bandwidth_grid', np.arange(domain_width/20, domain_width/5, domain_width/40))
        grid = SmoothingParameterSearch(
                KernelSmoother(kernel_estimator=NadarayaWatsonHatMatrix()),
                bandwidth_grid,
                param_name='kernel_estimator__bandwidth',
                scoring = LinearSmootherGeneralizedCVScorer(shibata)
                )
        grid.fit(fdata)
        gcv_score = grid.best_score_.round(4)
        return gcv_score, grid.transform(fdata)
    
    def _gcv_knn_kernel_smoothing(self, fdata : FDataGrid):
        '''implement gcv for selecting number of neighbours for smoothing with KNeighbour kernel.'''
        n_neighbors = self.basis_params.get('n_neighbors', np.arange(2, min(30, len(fdata.grid_points[0]))))
        grid = SmoothingParameterSearch(
                KernelSmoother(kernel_estimator=KNeighborsHatMatrix()),
                n_neighbors,
                param_name='kernel_estimator__n_neighbors',
                scoring = LinearSmootherGeneralizedCVScorer(shibata)
                )
        grid.fit(fdata)
        gcv_score = grid.best_score_.round(4)
        return gcv_score, grid.transform(fdata)
    
    def _wavelet_smoothing(self, fdata : skfda.FDataGrid, wavelet_name : str, mode : str = 'soft'):
        dt_matrix = fdata.data_matrix.squeeze()
        #wavelet decomposition
        coeffs = pywt.wavedec(dt_matrix, wavelet_name)

        #thresholding coefficients
        ## estimation of noise in data
        eps = np.array([np.median(np.abs(coef))  for coef in coeffs[-1]])[:,np.newaxis] / 0.6745
        new_coeffs = [pywt.threshold(arr, eps * (2 * np.log(dt_matrix.shape[1])) ** 0.5, mode = mode)
                    for arr in coeffs]
            
        # wavelet reconstruction
        dt_matrix_sm = pywt.waverec(new_coeffs, wavelet_name)

        #To skfda.FDataGrid object: enforce (grid points) concordance btw data_matrix and fdata 
        series_ext = dt_matrix_sm.shape[1] - fdata.data_matrix.shape[1]
        #if excessive, trim off excesses
        if series_ext != 0:
            dt_matrix_sm = dt_matrix_sm[:,:-series_ext]
        return FDataGrid(data_matrix= dt_matrix_sm, grid_points= fdata.grid_points[0])

    def fit(self, fd, return_data = True):
        """
        Fit the transformation to the input array.
        
        Args
        ----
        fd : array-like or skfda.FDataGrid object
            The input data to transform.
        
        Returns:
            FDataGrid: The transformed functional data.
        """
        self.gcv_score = None
        
        # set up data as functional object
        if type(fd) is np.ndarray:
            if self.domain_range is None:
                self.grid_points = np.linspace(0, 1, fd.shape[1])
                fd = skfda.FDataGrid(data_matrix = fd, grid_points=self.grid_points)
            else:
                self.grid_points = np.linspace(self.domain_range[0], self.domain_range[1], fd.shape[1])
                fd = skfda.FDataGrid(data_matrix = fd, grid_points=self.grid_points)
            
        elif type(fd) is skfda.FDataGrid:
            self.grid_points = fd.grid_points[0]

        else:
            raise ValueError("'fd' should be either numpy array or skfda.FDataGrid object.")

        #bspline smoothing
        if self.basis == 'bspline':
            #gcv if parameters not given
            if 'n_basis' not in self.basis_params and 'order' not in self.basis_params:
                self.gcv_score, self.fd_smooth = self._gcv_bspline_smoothing(fd)
            
            #use only n_basis and default order to 4 if not given
            elif 'order' not in self.basis_params:
                n_basis = self.basis_params.get('n_basis')
                basis = skfda.representation.basis.BSplineBasis(fd.domain_range[0], n_basis=n_basis)
                smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)  
                self.fd_smooth = smoother.fit_transform(fd)
            
            #use given parameters
            else:
                order = self.basis_params.get('order')
                n_basis = self.basis_params.get('n_basis')
                basis = skfda.representation.basis.BSplineBasis(fd.domain_range[0], n_basis = n_basis, order = order)
                smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)  
                self.fd_smooth = smoother.fit_transform(fd)

        elif self.basis == 'fourier':
            #gcv if parameters not given
            if 'n_basis' not in self.basis_params:
                self.gcv_score, self.fd_smooth = self._gcv_fourier_smoothing(fd)
            
            #use given parameters
            else:
                n_basis = self.basis_params.get('n_basis')
                period = self.basis_params.get('period', self.grid_points[-1] - self.grid_points[0])
                basis = skfda.representation.basis.FourierBasis(fd.domain_range[0], n_basis=n_basis, period = period)
                smoother = skfda.preprocessing.smoothing.BasisSmoother(basis)
                self.fd_smooth = smoother.fit_transform(fd)

        elif self.basis == 'nadaraya_watson':
            if 'bandwidth' not in self.basis_params:
                self.gcv_score, self.fd_smooth = self._gcv_nw_kernel_smoothing(fd)

            else:
                bandwidth = self.basis_params.get('bandwidth')
                smoother = KernelSmoother(kernel_estimator=NadarayaWatsonHatMatrix(bandwidth = bandwidth))
                self.fd_smooth = smoother.fit_transform(fd)

        elif self.basis == 'knn':
            if 'bandwidth' not in self.basis_params:
                self.gcv_score, self.fd_smooth = self._gcv_knn_kernel_smoothing(fd)

            else:
                neighbors = self.basis_params.get('n_neighbors')
                smoother = KernelSmoother(kernel_estimator=KNeighborsHatMatrix(neighbors = neighbors))
                self.fd_smooth = smoother.fit_transform(fd)
            
        elif self.basis == 'wavelet':
            # Set the wavelet type and decomposition level
            mode = self.basis_params.get('mode', 'soft')
            wavelet_name = self.basis_params.get('wavelet', 'db5')

            self.fd_smooth = self._wavelet_smoothing(fd, wavelet_name, mode)

        else:
            raise ValueError(f"Invalid basis type: {self.basis}. Supports 'bspline', 'fourier', 'nadaraya_watson', 'knn', and 'wavelet'.")
        
        if return_data:
            return self.fd_smooth

    
    # def plot(self, **kwarg):
    #     self.fd_smooth.plot(**kwarg)
 
