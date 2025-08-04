import skfda
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import FDataBasis, FourierBasis, BSplineBasis
from skfda.misc import inner_product, inner_product_matrix
from skfda.preprocessing.dim_reduction import FPCA
from skfda.exploratory.visualization import FPCAPlot
from skfda.misc.covariances import Exponential
from skfda.datasets import make_gaussian_process

import numpy as np
import pywt

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

"""
projector.py
============

This module provides the `Projector` class for projecting functional data onto a set of univariate data
using various types of projection functions (basis). It supports Fourier, B-spline, eigenfunction (FPC),
wavelet, Ornstein-Uhlenbeck process, and random linear combinations of eigenfunctions as projection bases.

Classes
-------
Projector
    Transforms functional data to univariate data by projecting onto specified basis functions.

Dependencies
------------
- scikit-fda (skfda)
- numpy
- pywavelets (pywt)
- seaborn
- matplotlib

Example
-------
>>> from skfda.datasets import make_gaussian_process
>>> from projector import Projector
>>> fdata = make_gaussian_process(n_samples=10, n_features=100)
>>> proj = Projector(basis_type='fourier', n_proj=3)
>>> coeffs = proj.fit(fdata)
>>> proj.plot_basis()
>>> proj.plot_projection_coeffs()

"""


class Projector():
    """
    Transform functional data to a set of univariate data by projection onto specified projection functions.

    Parameters
    ----------
    basis_type : str
        Specifies the type of projection function. Supported types include:
        - 'fourier': Fourier basis functions.
        - 'fpc': Eigenfunctions from functional principal component analysis.
        - 'wavelet': Discrete wavelet basis.
        - 'bspline': B-spline basis functions.
        - 'ou': Ornstein-Uhlenbeck process realizations.
        - 'rl-fpc': Random linear combinations of eigenfunctions.

    n_proj : int, default=3
        Number of projection functions to use (i.e., number of univariate projections).

    basis_params : dict, optional
        Dictionary of hyperparameters for basis generation. Supported keys:
        - 'period': Period for Fourier basis.
        - 'order': Order for B-spline basis.
        - 'wv_name': Name of the wavelet (for wavelet basis).
        - 'resolution': Base resolution for wavelet basis.

    Attributes
    ----------
    n_features : int
        Number of grid points for the projection functions and sample curves.

    basis : skfda.FDataGrid
        The projection functions used.

    coefficients : ndarray of shape (n_proj, N)
        Projection coefficients for N samples.

    Methods
    -------
    fit(fdata)
        Compute projection coefficients for the given functional data.

    plot_basis(**kwargs)
        Plot the projection functions.

    plot_projection_coeffs(**kwargs)
        Visualize the distribution of projection coefficients.

    Notes
    -----
    The class supports orthogonalization of basis functions and can handle several types of bases.
    """

    def __init__(self, basis_type: str, n_proj: int = 3, basis_params: dict = {} ) -> None: 
        """
        Initialize the Projector.

        Parameters
        ----------
        basis_type : str
            Type of projection basis ('fourier', 'fpc', 'wavelet', 'bspline', 'ou', 'rl-fpc').
        n_proj : int, default=3
            Number of projection functions.
        basis_params : dict, optional
            Parameters for basis generation.
        """
        self.basis_type = basis_type
        self.n_proj = n_proj
        self.basis_params = basis_params
        
        # Check for unwanted keys in basis_params
        if not all(key in ['period', 'order', 'wv_name', 'resolution'] for key in self.basis_params.keys()):
            raise ValueError('basis_params contains some unknown keys. '
                            'Ensure that the dict keys are limited to the following: '
                            "'period', 'order', 'wv_name', 'resolution'."
            )

    def get_wavelet_signal(self, wavelet_name):
        """
        Retrieve the scaling and wavelet functions for a given discrete wavelet.

        Parameters
        ----------
        wavelet_name : str
            Name of the discrete wavelet.

        Returns
        -------
        scaling_function : ndarray
            The scaling (father) wavelet function.
        wavelet_function : ndarray
            The wavelet (mother) function.

        Raises
        ------
        ValueError
            If the wavelet is not discrete or is unknown.
        """
        try:
            wavelet = pywt.Wavelet(wavelet_name)
        except ValueError as e:
            if 'Use pywt.ContinuousWavelet instead' in e.args[0]:
                raise ValueError(f"The `Projector` class only works with discrete wavelets, {wavelet_name} is a continuous wavelet.")
            elif 'Unknown wavelet name' in e.args[0]:
                raise ValueError(f"Unknown wavelet name {wavelet_name}, check pywt.wavelist(kind = 'discrete') for the list of available builtin wavelets.")

        wavefuns = wavelet.wavefun()
        scaling_function, wavelet_function, x = wavefuns[0], wavefuns[1], wavefuns[-1]

        # Truncate tails of scaling and wavelet functions
        tails = 1e-1
        nonzero_idx = np.argwhere(np.abs(wavelet_function) > tails)
        wavelet_function = wavelet_function[nonzero_idx[0,0]: nonzero_idx[-1,0] + 1]
        nonzero_idx = np.argwhere(np.abs(scaling_function) > tails)
        scaling_function = scaling_function[nonzero_idx[0,0]: nonzero_idx[-1,0] + 1]

        return scaling_function, wavelet_function
    

    def dilate_translate_signal(self, signal, n_trans):
        """
        Generate dilated and translated versions of a signal over the domain.

        Parameters
        ----------
        signal : ndarray
            The signal to dilate and translate.
        n_trans : int
            Number of translations (intervals).

        Returns
        -------
        list of skfda.FDataGrid
            List of normalized, dilated, and translated signals as FDataGrid objects.
        """
        knots = np.linspace(self.domain_range[0], self.domain_range[1], n_trans + 1)
        signals_ = [skfda.FDataGrid(signal, grid_points= np.linspace(knots[i], knots[i+1], len(signal)), extrapolation= 'zeros') 
                            for i in range(n_trans)]

        # Return normalized signals
        return [signal / np.sqrt(skfda.misc.inner_product(signal, signal)) for signal in signals_]


    def get_wavelet_basis(self, wavelet_name, n):
        """
        Construct a wavelet basis using scaling and wavelet functions.

        Parameters
        ----------
        wavelet_name : str
            Name of the discrete wavelet.
        n : int
            Number of intervals at the lowest resolution.

        Returns
        -------
        skfda.FDataGrid
            The constructed wavelet basis as FDataGrid.
        """
        scaling_signal, wavelet_signal = self.get_wavelet_signal(wavelet_name)

        # Get lowest resolution father wavelet
        basis = self.dilate_translate_signal(scaling_signal, n)

        # Get lowest resolution mother wavelet
        basis = basis + self.dilate_translate_signal(wavelet_signal, n)

        # Get higher resolution wavelets
        r_basis = self.n_proj - 2 * n
        while r_basis > 0:
            n *= 2
            basis = basis + self.dilate_translate_signal(wavelet_signal, n)
            r_basis -= n

        # Evaluate the basis at grid points
        basis_grid = [skfda.FDataGrid(basis_(self.grid_points).squeeze(), grid_points= self.grid_points)
            for basis_ in basis[ :self.n_proj]
            ]

        # Return basis as a FDataGrid object
        return skfda.concatenate(basis_grid)


    def _generate_basis(self) -> FDataGrid:
        """
        Generate projection functions from the specified basis type.

        Returns
        -------
        skfda.FDataGrid
            The generated basis functions.

        Raises
        ------
        ValueError
            If the basis type is unknown.
        """
        if self.basis_type == 'fourier':
            self.period = self.basis_params.get('period', self.domain_range[1] - self.domain_range[0])
            nb = self.n_proj
            
            if (nb % 2) == 0:
                nb += 1

            coeffs = np.eye(nb)
            basis = FourierBasis(domain_range= self.domain_range, n_basis= nb, period = self.period)
            return FDataBasis(basis, coeffs).to_grid(self.grid_points)[ : self.n_proj]

        elif self.basis_type == 'bspline':
            self.order = self.basis_params.get('order', 3)
            coeffs = np.eye(self.n_proj)
            basis = BSplineBasis(domain_range= self.domain_range, n_basis=self.n_proj, order = self.order)

            return FDataBasis(basis, coeffs).to_grid(self.grid_points)
        
        elif self.basis_type == 'ou':
            # Ornstein-Uhlenbeck process: mean = 0, k(x,y) = exp(-|x - y|)
            basis = make_gaussian_process(start = self.domain_range[0], stop = self.domain_range[1], n_samples = self.n_proj, 
                                              n_features = 2 * len(self.grid_points), mean = 0, cov = Exponential(variance = 1, length_scale=1)
                                                    ).to_grid(self.grid_points)
                
            return basis

        elif self.basis_type == 'wavelet':
            wavelet_name = self.basis_params.get('wv_name', 'db5')
            n = self.basis_params.get('resolution', 1)

            return self.get_wavelet_basis(wavelet_name, n)
            
    def _compute_fpc_combination(self, fdata):
        """
        Construct projection functions as random linear combinations of eigenfunctions
        explaining at least 95% of the variation in the sample data.

        Parameters
        ----------
        fdata : skfda.FDataGrid
            The functional data.

        Returns
        -------
        skfda.FDataGrid
            The constructed basis as random linear combinations of eigenfunctions.
        """
        fpca_ = FPCA(n_components= min(fdata.data_matrix.squeeze().shape))
        fpca_.fit(fdata)
        lambdas_sq = np.square(fpca_.singular_values_) 
        jn = np.argmax(np.cumsum(lambdas_sq / lambdas_sq.sum()) >= 0.95) + 1

        s2 = [skfda.misc.inner_product(fpca_.components_[i], fdata).var() for i in range(jn)]
        ej = fpca_.components_[:jn]

        gammas = np.array([np.random.normal(0, np.sqrt(s2_), self.n_proj) for s2_ in s2])
        
        basis_ = (gammas[:,0] * ej).sum()
        for i in range(1,self.n_proj):
            basis_ = basis_.concatenate((gammas[:,i] * ej).sum())

        return basis_        
        
    def _is_orthogonal(self, basis: FDataGrid, tol: float | None = None) -> bool:
        """
        Check the orthogonality of a given set of projection functions.

        Parameters
        ----------
        basis : skfda.FDataGrid
            The basis functions to check.
        tol : float, optional
            Tolerance for orthogonality. If None, checks at 1e-15 and 1e-10.

        Returns
        -------
        bool
            True if orthogonal within tolerance, False otherwise.
        """
        basis_gram = inner_product_matrix(basis)
        basis_gram_off_diagonal = basis_gram - np.diag(np.diagonal(basis_gram))
        if not tol is None:
            nonzeros = np.count_nonzero(np.abs(basis_gram_off_diagonal) > tol)
            if nonzeros == 0:
                return True
            
        else:
            for tol in [1e-15, 1e-10]:
                nonzeros = np.count_nonzero(np.absolute(basis_gram_off_diagonal) > tol)
                if nonzeros == 0:
                    return True
        
        return False
        
    def _gram_schmidt(self, funs: FDataGrid) -> FDataGrid:
        """
        Perform Gram-Schmidt orthogonalization on a set of functions.

        Parameters
        ----------
        funs : skfda.FDataGrid
            Functions to orthogonalize.

        Returns
        -------
        skfda.FDataGrid
            Orthogonalized functions.
        """
        funs_ = funs.copy()
        num_funs = len(funs_)

        for i in range(num_funs):
            fun_ = funs_[i]
            for j in range(i):
                projection = inner_product(funs_[i], funs_[j]) / np.sqrt(inner_product(funs_[j], funs_[j]))
                fun_ -= projection * funs_[j]
                
            if i == 0:
                orthogonalized_funs = fun_.copy()
            else:
                orthogonalized_funs = orthogonalized_funs.concatenate(fun_.copy())

        return orthogonalized_funs


    def _compute_coefficients(self, fdata: FDataGrid):
        """
        Orthogonalize the basis functions if necessary and compute projection coefficients.

        Parameters
        ----------
        fdata : skfda.FDataGrid
            Functional data to project.

        Returns
        -------
        tuple
            (coefficients, basis) where coefficients is an array of projection coefficients,
            and basis is the (possibly orthogonalized) basis functions.
        """
        basis = self._generate_basis()

        assert all((basis.grid_points[0].shape == fdata.grid_points[i].shape for i in range(len(fdata.grid_points)))), 'Set the appropriate sample_points for basis functions; number of sample points for both objects, the basis and the functional sample data, must be equal.'
        assert all(((basis.grid_points[0] == fdata.grid_points[i]).all() for i in range(len(fdata.grid_points)))), 'Set the appropriate sample_points for basis functions; sample points for both objects, the basis and the functional sample data, must be equal.'
        
        # Enforce orthogonality where necessary
        if self.basis_type not in ['ou', 'wavelet']:
            while not self._is_orthogonal(basis):
                basis = self._gram_schmidt(basis)

        return inner_product_matrix(basis, fdata), basis


    def _compute_fpc(self, fdata):
        """
        Construct the eigenfunctions (principal components) from the data.

        Parameters
        ----------
        fdata : skfda.FDataGrid
            Functional data.

        Returns
        -------
        skfda.FDataGrid
            The principal component functions.
        """
        fpca_ = FPCA(n_components = self.n_proj)
        basis = fpca_.fit(fdata).components_
        return basis


    def fit(self, fdata: FDataGrid):
        """
        Compute the projection coefficients of sample functions.

        Parameters
        ----------
        fdata : skfda.FDataGrid
            Functional data to project.

        Returns
        -------
        ndarray
            Projection coefficients.
        """
        self.domain_range = fdata.domain_range[0]
        self.grid_points = fdata.grid_points[0]
        # Center data
        fdata = fdata - fdata.mean()

        if self.basis_type in ['fourier', 'ou', 'wavelet', 'bspline']:
            self.coefficients, self.basis = self._compute_coefficients(fdata)

        elif self.basis_type == 'fpc':
            self.basis = self._compute_fpc(fdata)
            self.coefficients =  inner_product_matrix(self.basis, fdata)

        elif self.basis_type == 'rl-fpc':
            self.basis = self._compute_fpc_combination(fdata)
            self.coefficients =  inner_product_matrix(self.basis, fdata)

        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}. Choose from the supported options: 'fourier', 'bspline', 'ou', 'rl-fpc', 'wavelet', 'fpc'.")
        
        return self.coefficients

    def plot_basis(self, **kwargs):
        """
        Plot the projection basis functions.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the plot function.
        """
        self.basis.plot(group = range(1, len(self.basis)+1), **kwargs)
        plt.xlabel('t')
        plt.ylabel('$\\beta_v(t)$')

    def plot_projection_coeffs(self, **kwargs):
        """
        Plot the distribution of univariate projection coefficients.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to seaborn.histplot.
        """
        if self.n_proj >= 4:
            fig, axes = plt.subplots(int(np.ceil(self.n_proj / 4)), 4, figsize=(15, 15))
        else:
            fig, axes = plt.subplots(1, self.n_proj, figsize=(10, 5))
        axes = axes.ravel()

        for coeffs, ax in zip(self.coefficients, axes):
            sns.histplot(data=coeffs, stat='density', ax=ax, **kwargs)
            ax.set(xlabel = 'projection coefficients')
            
        fig.tight_layout()