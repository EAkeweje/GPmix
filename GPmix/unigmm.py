import math
import numpy as np
import skfda
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from scipy.stats import norm
from math import comb, factorial, log, floor
from cmath import log as clog
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score as ami,  adjusted_rand_score as ari, accuracy_score
from itertools import permutations
from joblib import Parallel, delayed
from .misc import gmms_fit_plot_, silhouette_score, davies_bouldin_score
import warnings


"""
unigmm.py
=========

This module provides classes for estimating parameters of base univariate GMMs and also for formulating
a consensus clustering from these GMMs. The main classes are:

- GaussianMixtureParameterEstimator: Estimates parameters of a univariate GMM using the method of moments.
- UniGaussianMixtureEnsemble: Constructs a consensus clustering using an ensemble of univariate
"""

class GaussianMixtureParameterEstimator():
    """
    Estimate parameters of a univariate Gaussian mixture model (GMM) using the
    method of moments.

    This estimator implements a numerically stable method-of-moments approach
    for univariate GMMs, using the log-sum-exp trick for moment calculations. It
    supports flexible initialization, simple parameter constraints, and optional
    normalization of the input data.

    THIS IS STILL EXPERIMENTAL. USE WITH CAUTION.

    Parameters
    ----------
    n_comp : int
        Number of mixture components.
    n_moments : int or None, optional
        Highest order of moment to consider. If ``None``, defaults to
        ``4 * n_comp - 2``.
    epsilon : float, optional
        Lower bound on mixture weights and bounds for means/variances used to
        improve numerical stability. If set to 0.0, no additional constraints
        are applied.

    Attributes
    ----------
    n_comp : int
        Number of mixture components.
    n_moments : int
        Highest order of moment used during fitting.
    epsilon : float
        Lower bound/constraint strength used during fitting.
    init_guess_ : ndarray of shape (3 * n_comp,)
        Initial parameter guess in the order: weights, means, variances.

    Notes
    -----
    The method of moments can be sensitive to both the choice of initialization
    and the moment order. The solver minimizes a system of moment equations via
    non-linear least squares. For stability, computed log-moments use the
    log-sum-exp trick.
    """
   
    def __init__(self, n_comp: int, n_moments: int or None = None, epsilon= 0.0) -> None:
        self.n_comp = n_comp
        self.epsilon = epsilon
        if n_moments is None:
            self.n_moments= 4 * self.n_comp - 2
        else:
            self.n_moments =  n_moments

    def log_sample_moments(self, data: np.ndarray, order: int) -> float:
        '''Compute the log of sample moments using log-sum-exp trick'''
        n = len(data)
        log_data =  np.array([clog(x) for x in data])
        a = np.max(log_data).real
        sumexp = np.sum(np.exp(order * (log_data - a)))
        return order * a - log(n) + clog(sumexp)

    def log_theoretical_moments(self, parameters: np.ndarray or list, order: int) -> float:
        ''' Compute the log of theoretical moments of a univariate Gaussian using log-sum-exp trick

        Attributes
        ------
        parameters : array-like of shape (2, )
            mean and variance of univatiate Gaussian, strictly this order: mean, variance
        order : int 
            order of moment
        '''
        summand_ = lambda mu, sigma2, j : clog(comb(order, 2 * j) * factorial(2 * j) / (factorial(j) * 2 ** j)) + (order - (2 * j)) * clog(mu) + j * clog(sigma2)
        summands = np.array([summand_(*parameters, j) for j in range(floor(order / 2) + 1)])
        a = np.max(summands).real
        sumexp = np.sum(np.exp(summands - a))
        return a + clog(sumexp)

    def log_theoretical_mixture_moments(self, parameters: np.ndarray or list, order: int) -> float:
        ''' Compute the log of theoretical moments of a univariate Gaussian mixture using log-sum-exp trick

        Attributes
        ------
        parameters : array-like of shape (3 * n_comp, )
            weights, means and variances of univatiate Gaussian, strictly this order: weights, means, variances
        order : int 
            order of moment
        '''
        assert len(parameters) == 3 * self.n_comp , "Ensure that the parameters are ordered this way: p_1, p_2, ..., p_k, mu_1, mu_2, ..., mu_k, sigma2_1, sigma2_2, ... sigma2_k."
        
        weight_ = parameters[: self.n_comp]
        mu_var_ = np.array(parameters[self.n_comp : ]).reshape(-1, 2)
        log_moments = [self.log_theoretical_moments(param_, order) for param_ in mu_var_]
        exponents = np.log(weight_) + np.array(log_moments)

        max_exponent = np.max(exponents).real
        sumexp = np.sum(np.exp(exponents - max_exponent))

        return max_exponent + clog(sumexp)
    
    def fit(self, data: np.ndarray, normalize: bool = True, full_output: bool = False):
        ''' Fit data to univariate GMM using method of moment estimation
        
        Attributes
        ------
        data : array-like of shape (sample size,)
            Sample data.
        normalize : boolean, default = True.
            If True, the already centered data will be transformed to having unit variance.
        full_output : boolean, default = False
            If True, a comprehensive report of the (optimization) solver is returned, else only the estimates are returned.

        returns
        -------
        If 'full_output = False', returns an array of shape (3*n_comp,); ordered in the form weights, means, variance.
        else if 'full_output = True', a comprehensive report on the (optimization) solver is returned.
        '''

        data_std = np.std(data)
        
        if normalize:
            #normalize data to mean 0 and unit variance
            # data in the algorithm is already centered, only scale data with the std
            data = data / data_std

        if self.epsilon:
            l_bounds = [(self.epsilon) for _ in range(self.n_comp)] + [(-1/ self.epsilon) for _ in range(self.n_comp)] + [(1e-15) for _ in range(self.n_comp)]
            h_bounds = [(1 - self.epsilon) for _ in range(self.n_comp)] + [(1/self.epsilon) for _ in range(self.n_comp)] + [(1 / self.epsilon) for _ in range(self.n_comp)]
            
            #set feasible initialization
            weight_init = np.random.uniform(self.epsilon, 1/self.n_comp, self.n_comp)
            mean_init = np.random.uniform(-1/self.epsilon, 1/self.epsilon, self.n_comp)
            var_init = np.random.uniform(1e-15, 1 / self.epsilon, self.n_comp)
            
        else:
            l_bounds = [(0) for _ in range(self.n_comp)] + [(-np.inf) for _ in range(self.n_comp)] + [(1e-15) for _ in range(self.n_comp)]
            h_bounds = [(1) for _ in range(self.n_comp)] + [(np.inf) for _ in range(self.n_comp)] + [(np.inf) for _ in range(self.n_comp)]
            
            #set feasible initialization
            weight_init = np.random.uniform(0, 1/self.n_comp, self.n_comp)
            mean_init = np.random.rand(self.n_comp)
            var_init = np.abs(np.random.rand(self.n_comp))

        self.init_guess_ = np.concatenate((weight_init/np.sum(weight_init), mean_init, var_init))
        
        #the system of moments
        moment_system = lambda input_vars: [input_vars[: self.n_comp].sum() - 1, input_vars[: self.n_comp] @ np.array(input_vars[self.n_comp : 2 * self.n_comp])] + [2 * (self.log_theoretical_mixture_moments(input_vars, order).real - self.log_sample_moments(data, order)).real for order in range(2, self.n_moments + 1)]
        
        #set up the solver
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = least_squares(moment_system, self.init_guess_, bounds= (tuple(l_bounds), tuple(h_bounds)), max_nfev = 200 * len(self.init_guess_), ftol = 1e-20, gtol = 1e-6)

        ## check if iteration was successful
        if not res.success:
            warnings.warn("Numerical iteration failed to converge:{message}. EM is initialized with values from the values from last iteration. As this may not be optimal, it is recommended that other initialization techniques be employed".format(message = res.message))

        if normalize:
            #adjust the mean (scale by data std) and std (scale by data std^2)
            res.x = np.concatenate((res.x[ :self.n_comp], data_std * res.x[self.n_comp : 2 * self.n_comp], data_std ** 2 * res.x[2 * self.n_comp : ]))
            
        # if res.x[ : self.n_comp].sum() != 1:
        #     res.x[ :self.n_comp] = np.abs(res.x[:self.n_comp]) / res.x[ : self.n_comp].sum()

        if full_output:
            return res
        
        else:
            return res.x


class UniGaussianMixtureEnsemble:
    """
    Consensus clustering using an ensemble of univariate Gaussian Mixture Models (GMMs).

    This class fits univariate GMMs to multiple one-dimensional projections of the
    data, computes base clusterings, and combines them into a consensus clustering
    using spectral clustering on an affinity matrix built from binary membership
    matrices. Base clusterings are weighted by an estimate of their total
    misclassification probability.

    Parameters
    ----------
    n_clusters : int
        Number of mixture components (clusters) to fit in each GMM and in the
        consensus clustering.
    init_method : {"kmeans", "k-means++", "random", "random_from_data", "mom"}, optional
        Initialization method for GMM parameters. Default is ``"kmeans"``.
        The ``"mom"`` option uses method-of-moments initialization.
    n_init : int, optional
        Number of initializations to perform for each GMM fit. The best result is
        kept. Default is ``10``.
    mom_epsilon : float, optional
        Lower bound for GMM weights (and related constraints) when using
        ``init_method="mom"``. Ignored otherwise. Default is ``5e-2``.

    Attributes
    ----------
    n_projs : int
        Number of projections (base clusterings).
    data_size : int
        Number of samples in the data.
    gmms : tuple of sklearn.mixture.GaussianMixture
        Fitted univariate GMMs for each projection.
    MoM_res : tuple
        If ``init_method == 'mom'``, the method-of-moments solver results for each
        projection.
    clustering_weights_ : ndarray of shape (n_projs,)
        Weights assigned to each base clustering.
    labels_ : ndarray of shape (n_samples,)
        Cluster labels assigned by the consensus clustering.
    max_cca_labels_ : tuple
        Permutation of predicted labels that yields the highest classification
        accuracy when compared to ground truth.

    Notes
    -----
    The affinity matrix is constructed as a weighted sum of outer products of
    binary membership matrices (one per projection), where the weights are
    proportional to the inverse of each GMM's estimated total misclassification
    probability.
    """

    def __init__(self, n_clusters: int, init_method: str = 'kmeans', n_init: int = 10, mom_epsilon: float = 5e-2) -> None:
        """
        Initialize the UniGaussianMixtureEnsemble.

        Parameters
        ----------
        n_clusters : int
            Number of mixture components (clusters).
        init_method : str, default='kmeans'
            Initialization method for GMMs. One of 'kmeans', 'k-means++', 'random', 'random_from_data', 'mom'.
        n_init : int, default=10
            Number of initializations for each GMM.
        mom_epsilon : float, default=5e-2
            Lower bound for GMM weights when using 'mom' initialization.
        """
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.n_init = n_init
        self.mom_epsilon = mom_epsilon

        assert self.init_method in ['kmeans', 'k-means++', 'random', 'random_from_data', 'mom'], \
            "Unknown value for 'init_method'. Set to one of: 'kmeans', 'k-means++', 'random', 'random_from_data', 'mom'."


    def gmm_with_MoM_inits(self, data: np.ndarray):
        '''Fit gmms with initialization from method of moment estimation'''

        gmm_, res_ = [], []

        for _ in range(self.n_init):
            #get inits
            try:
                mom_estimator = GaussianMixtureParameterEstimator(n_comp= self.n_clusters, epsilon= self.mom_epsilon)
                res = mom_estimator.fit(data, full_output = True)
                res_.append(res)
            except ValueError as error:
                print("'mom' initialization is not appropriate for this dataset. Attempt to reduce the number of projection basis or use other initialization methods'.")
                raise(error)
                
            #process inits
            mom_inits = res.x.reshape(-1, self.n_clusters)
            #adjust to ensure weights sum to 1
            mom_inits[0] = np.abs(mom_inits[0]) / np.abs(mom_inits[0]).sum() 

            #pass inits to EM solver and fit
            gmm = GaussianMixture(n_components = self.n_clusters, covariance_type= 'spherical', weights_init=mom_inits[0], means_init= mom_inits[1].reshape(-1,1),
                                precisions_init= 1 / mom_inits[2])
            gmm.fit(data)
            gmm_.append(gmm)

        #select the highest lower bound
        ind = np.argmax([[gmm.lower_bound_ for gmm in gmm_]])

        return gmm_[ind], res_[ind]
    

    #helper function to parallelize gmm fiting with MOM inits
    def _fit_gmms_with_MoM_inits(self, proj_coeffs: np.ndarray):
        gmm, res = self.gmm_with_MoM_inits(proj_coeffs.reshape(-1,1))
        return (gmm, res)
    

    #helper function to parallelize gmm fiting with other inits
    def _fit_gmms_with_other_inits(self, proj_coeffs: np.ndarray):
        gmm = GaussianMixture(n_components = self.n_clusters, covariance_type='spherical', n_init= self.n_init, init_params= self.init_method)
        gmm.fit(proj_coeffs.reshape(-1,1))
        return gmm
    

    def fit_gmms(self, projs_coeffs: np.ndarray,  n_jobs = -1, **kwargs):
        """
        Fit projection coefficients to univariate Gaussian mixture models

        Arguments
        ---------
        projs_coeffs : array-like of shape (number of projections, number of samples)
            array of projection coefficients to fit to univariate GMMs. 
        kwargs :
            keyword arguments for joblib Parallel
        """

        self.projs_coeffs = projs_coeffs
        self.n_projs = self.projs_coeffs.shape[0]
        self.data_size = self.projs_coeffs.shape[1]

        if self.init_method == 'mom':
            #parallelize proccess with joblib
            results = Parallel(n_jobs = n_jobs, **kwargs)(delayed(self._fit_gmms_with_MoM_inits)(proj_coeffs) for proj_coeffs in self.projs_coeffs)
            self.gmms, self.MoM_res = zip(*results)
        
        elif self.init_method in ['kmeans', 'k-means++', 'random', 'random_from_data']:
            #parallelize proccess with joblib
            self.gmms = Parallel(n_jobs = n_jobs, **kwargs)(delayed(self._fit_gmms_with_other_inits)(proj_coeffs) for proj_coeffs in self.projs_coeffs)
    
    # def gmms_lower_bound_(self):
    #     return np.sum([g.lower_bound_ for g in self.gmms])


    def plot_gmms(self, ncols: int = 4, fontsize: int = 12, fig_kws = {}, **kwargs):
        '''Visualize GMM fits
        
        Parameters
        ----------
        ncols : int, optional
            Number of columns in the plot grid. Default is 4.
        fontsize : int, optional
            Font size for axis labels. Default is 12.
        fig_kws : dict, optional
            Additional keyword arguments for figure creation. Default is an empty dict.
        kwargs :  dict, optional
            Additional keyword arguments for seaborn's histplot function. Default is an empty dict.
        '''
        # create the figure and axes
        if self.n_projs == 1:
            fig, axes = plt.subplots(1, self.n_projs, **fig_kws)
            axes = [axes]
        elif self.n_projs < 4:
            fig, axes = plt.subplots(1, self.n_projs, **fig_kws)
            axes = axes.ravel()  # flattening the array makes indexing easier
        else:
            fig, axes = plt.subplots(int(np.ceil(self.n_projs / ncols)), ncols, **fig_kws)
            axes = axes.ravel()  # flattening the array makes indexing easier

        # label_ = ['alpha_{i' + str(i) + '}' for i in range(1, self.n_projs + 1)]
        for i, coeffs, ax in zip(range(self.n_projs), self.projs_coeffs, axes):
            sns.histplot(data=coeffs, stat='density', ax=ax, **kwargs)
            gmms_fit_plot_(self.gmms[i].weights_, self.gmms[i].means_.ravel(), np.sqrt(self.gmms[i].covariances_), ax= ax, color = 'r')
            label_ = 'alpha_{i' + str(i+1) + '}'
            ax.set_xlabel(fr'$\{label_}$', fontsize = fontsize)
            ax.yaxis.get_label().set_fontsize(fontsize)
        
        # fig.suptitle('Plot of fitted GMMs')
        fig.tight_layout()


    def fuzzy_membership_matrix(self) -> np.ndarray:
        """
        Construct the cluster membership matrices from GMM fits.
        """
        membership_matrix = np.array([gmm.predict_proba(proj_coeffs.reshape(-1, 1)) for proj_coeffs, gmm in zip(self.projs_coeffs, self.gmms)])
        
        return membership_matrix


    def binary_membership_matrix(self) -> np.ndarray:
        """
        Construct a binary membership indicator matrices from the cluster membership matrice.
        """
        membership_matrix = self.fuzzy_membership_matrix()
        
        binary_matrix = np.zeros_like(membership_matrix)
        max_indices = np.argmax(membership_matrix, axis=2)
        x_indices, y_indices = np.meshgrid(np.arange(membership_matrix.shape[0]), np.arange(membership_matrix.shape[1]), indexing='ij')
        binary_matrix[x_indices, y_indices, max_indices] = 1

        return binary_matrix
    

    def get_omega_prob(self, dist_a, dist_b) -> float:
        '''Construct misclassification probability omega_{b|a} for univariate GMMs
        
        Parameters
        ----------
        dist_* : array-like of shape (3,)
            THe parameters of the mixture component *: [weight, mean, variance]
        '''

        coeff_a = 1/dist_b[2] - 1/dist_a[2]
        coeff_b = 2 * (dist_a[1] / dist_a[2] - dist_b[1] / dist_b[2])
        coeff_c = dist_b[1] ** 2 / dist_b[2] - dist_a[1] ** 2 / dist_a[2] - math.log((dist_b[0]  / dist_a[0])  ** 2 * dist_a[2] / dist_b[2])
        
        #for quadratic inequality
        if coeff_a != 0:
            #for real roots
            if coeff_b ** 2 - (4 * coeff_a * coeff_c) >= 0:
                #compute zeros
                zeros = [((-1 * coeff_b) + math.sqrt(coeff_b ** 2 - (4 * coeff_a * coeff_c))) / (2 * coeff_a),
                        ((-1 * coeff_b) - math.sqrt(coeff_b ** 2 - (4 * coeff_a * coeff_c))) / (2 * coeff_a)]

                #normalize zeros with parameters of distribution a
                norm_zeros = (np.array(zeros) - dist_a[1]) / math.sqrt(dist_a[2])
                left_ = np.min(norm_zeros)
                right_ = np.max(norm_zeros)

                #compute misclassification probability
                if 2 * coeff_a > 0:
                    prob = norm.cdf(right_) - norm.cdf(left_)
                elif 2 * coeff_a < 0:
                    prob = 1 + norm.cdf(left_) - norm.cdf(right_)

            #for complex roots
            else:
                if 2 * coeff_a > 0:
                    prob = 0.0

                else:
                    prob = 1.0

        #for linear inequality
        elif coeff_a == 0 and coeff_b != 0:
            zero = -1 * coeff_c / coeff_b
            norm_zero = (zero - dist_a[1]) / math.sqrt(dist_a[2])
            if coeff_b > 0:
                prob = norm.cdf(norm_zero)
            elif coeff_b < 0:
                prob = 1 - norm.cdf(norm_zero)

        #for a constant
        elif coeff_a == 0 and coeff_b == 0:
            #if equal means and variances
            if coeff_c <= 0:
                prob = 1.0
            elif coeff_c > 0:
                prob = 0.0

        return prob


    def get_omega_map(self, weights, means, vars) -> np.ndarray:
        '''Construct matrix of misclassification probabilities'''
        omega_array = np.zeros((self.n_clusters, self.n_clusters))

        for i in range(self.n_clusters):
            omega_array[i] = np.array([self.get_omega_prob([weights[i], means[i], vars[i]], [weights[j], means[j], vars[j]]) for j in range(self.n_clusters)])
        
        return omega_array


    def get_total_omega(self, weights, means, vars, weighted_sum) -> float:
        '''Compute total misclassification probability for univariate GMM'''

        #sum of misclassification probabilities within GMM
        omega_map = self.get_omega_map(weights, means, vars)
        omega_sum = omega_map.sum(axis = 1) - np.diag(omega_map)
        
        #return the total probability of misclassification
        if weighted_sum:
            return np.array(weights) @ omega_sum
        else:
            return omega_sum.sum()
        
    
    def get_clustering_weights(self, weighted_sum, precompute_gmms = None) -> np.ndarray:
        '''Compute weights for base clusterings'''
        if precompute_gmms:
            self.gmms = precompute_gmms
        
        if len(self.gmms) == 1:
            self.clustering_weights_ = np.array([1])

        else:
            #get params from fitted gmms
            weights_ = [gmm.weights_ for gmm in self.gmms]
            means_ = [gmm.means_.ravel() for gmm in self.gmms] 
            vars_ = [gmm.covariances_ for gmm in self.gmms]
            
            #compute total probability of misclassification for the fitted gmms
            total_omega_ = np.array([self.get_total_omega(weights_[i], means_[i], vars_[i], weighted_sum) for i in range(len(self.gmms))])
            
            #catch zero omegas to deal with potential math errors as a result of undefine computations
            omega_zero = np.argwhere(total_omega_ < 1e-20)

            # #adjust total misclassification probability
            # if len(omega_zero) == 0:
            #     total_omega_inv = 1 / np.array(total_omega_)
            # elif len(omega_zero) == 1:
            #     warnings.warn('The choice of projection basis might result in unstable clusterings for the dataset. Try other kind of projection basis.')
            #     total_omega_inv = np.insert(1 / np.delete(total_omega_, int(omega_zero)), int(omega_zero), 0)
            # else:
            #     warnings.warn('The choice of projection basis might result in unstable clusterings for the dataset. Try other kind of projection basis.')
            #     total_omega_inv = np.ones(len(self.gmms))
            total_omega_[omega_zero] = 1e-30
            total_omega_inv = 1 / np.array(total_omega_)
            #compute clustering weights
            self.clustering_weights_ = total_omega_inv / total_omega_inv.sum()

        return self.clustering_weights_

    def get_affinity_matrix(self, weighted_sum: bool, precompute_gmms = None) -> np.ndarray:
        ''' Construct affinity matrix using binary membership matrices and clustering weights'''
        if precompute_gmms:
            self.gmms = precompute_gmms

        bm_matrix =  self.binary_membership_matrix()
        clustering_weights = self.get_clustering_weights(weighted_sum, self.gmms)
        affinity_matrix = np.zeros((self.data_size, self.data_size))
        
        for i in range(len(self.gmms)):
            affinity_matrix += clustering_weights[i] * np.matmul(bm_matrix[i], bm_matrix[i].T)

        return affinity_matrix
    
    def get_clustering(self, weighted_sum: bool = True, precompute_gmms = None, **kwargs) -> np.ndarray:
        '''Obtain the consensus clustering via Spectral clustering of Affinity matrix'''
        if precompute_gmms:
            self.gmms = precompute_gmms

        clustering = SpectralClustering(n_clusters = self.n_clusters, affinity='precomputed', assign_labels='discretize', **kwargs)
        clustering.fit(self.get_affinity_matrix(weighted_sum, self.gmms))
        self.labels_ = clustering.labels_
        return self.labels_

    def plot_clustering(self, fdata: skfda.FDataGrid):
        fdata.plot(group = self.labels_)

    def adjusted_mutual_info_score(self, true_labels) -> float:
        return ami(self.labels_, true_labels)

    def adjusted_rand_score(self, true_labels) -> float:
        return ari(self.labels_, true_labels)

    def correct_classification_accuracy(self, true_labels) -> float:
        true_classes = np.unique(true_labels)
        assert self.n_clusters == len(true_classes), f"number of clusters {self.n_clusters} do not match number of true clusters {len(true_classes)}."
        pred_classes = np.arange(self.n_clusters)

        pred_perm = np.zeros_like(self.labels_)
        cca_list = []

        #permute the pred labels and compute classification accuracy for each
        for perm in permutations(pred_classes):
            for i,j in enumerate(perm):
                pred_perm[self.labels_ == j] = true_classes[i]

            #compute CCA
            cca_list.append(accuracy_score(true_labels, pred_perm))

        max_ind = np.argmax(cca_list)
        self.max_cca_labels_ = list(permutations(pred_classes))[max_ind]

        #return max classification score
        return cca_list[max_ind]
    
    def silhouette_score(self, fdata: skfda.FDataGrid):
        return silhouette_score(fdata, self.labels_)
    
    def davies_bouldin_score(self, fdata: skfda.FDataGrid):
        return davies_bouldin_score(fdata, self.labels_)

