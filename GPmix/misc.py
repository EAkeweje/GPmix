"""
misc.py

This module provides utility functions for clustering and analyzing functional data, 
including cluster number estimation, cluster evaluation metrics, Gaussian mixture plotting, 
label matching, and representative sample selection.

Functions:
    - silhouette_score: Computes the silhouette score for clustering results.
    - davies_bouldin_score: Computes the Davies-Bouldin score for clustering results.
    - gmms_fit_plot_: Plots Gaussian mixture model density curves.
    - match_labels: Permutes cluster labels to match specified true labels.
    - estimate_nclusters: Estimates the optimal number of clusters in a functional dataset.
    - hybrid_representative_selection: Selects representative samples using random sampling and KMeans clustering.

Dependencies:
    - numpy
    - matplotlib
    - sklearn
    - skfda

Author: Emmanuel Akeweje
Date: August 2025
"""

import skfda
from sklearn.mixture import GaussianMixture
from skfda.preprocessing.dim_reduction import FPCA
# from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import davies_bouldin_score as dbs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def silhouette_score(fd, y, **kwargs):
    """
    Compute the silhouette score for clustering results on functional data.

    Parameters
    ----------
    fd : skfda.FDataGrid
        Functional data object containing the data matrix.
    y : array-like
        Cluster labels for each sample.
    **kwargs : dict
        Additional keyword arguments passed to sklearn.metrics.silhouette_score.

    Returns
    -------
    float
        Silhouette score.
    """
    return ss(fd.data_matrix.squeeze(), y, **kwargs)

def davies_bouldin_score(fd, y):
    """
    Compute the Davies-Bouldin score for clustering results on functional data.

    Parameters
    ----------
    fd : skfda.FDataGrid
        Functional data object containing the data matrix.
    y : array-like
        Cluster labels for each sample.

    Returns
    -------
    float
        Davies-Bouldin score.
    """
    return dbs(fd.data_matrix.squeeze(), y)

def gmms_fit_plot_(weights, means, stdev, ax = None, **kwargs):
    """
    Plot Gaussian mixture model (GMM) density curves.

    Parameters
    ----------
    weights : array-like
        Weights of each Gaussian component.
    means : array-like
        Means of each Gaussian component.
    stdev : array-like
        Standard deviations of each Gaussian component.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses the current axis.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib plot.

    Returns
    -------
    None
    """
    d_pdf = lambda x, mu, sigma: np.exp(-0.5 * (((x - mu) / sigma)) ** 2) / (sigma * np.sqrt(2 * np.pi))
    for i in range(len(weights)):
        x = np.linspace(means[i] - 3 * stdev[i], means[i] + 3 * stdev[i], 50)
        if ax:
            ax.plot(x, weights[i] * d_pdf(x, means[i], stdev[i]), linewidth = 3, **kwargs)
        else:
            plt.plot(x, weights[i] * d_pdf(x, means[i], stdev[i]), linewidth = 3, **kwargs)

def match_labels(cluster_labels, true_class_labels, cluster_class_labels_perm):
    """
    Permute cluster labels to match specified true class labels.

    Parameters
    ----------
    cluster_labels : array-like
        Cluster labels assigned by a clustering algorithm.
    true_class_labels : array-like
        True class labels to match.
    cluster_class_labels_perm : array-like
        Permutation of cluster labels to match true labels.

    Returns
    -------
    np.ndarray
        Array of matched labels.
    """
    label_match_dict = {}
    for key, value in zip(cluster_class_labels_perm, true_class_labels):
        label_match_dict[key] = value

    matched_labels = np.zeros_like(cluster_labels)
    for i in cluster_class_labels_perm:
        matched_labels[np.argwhere(cluster_labels == i)] = label_match_dict[i]
    return matched_labels

def estimate_nclusters(fdata, ncluster_grid = None):
    """
    Estimate the optimal number of clusters in a functional dataset using GMM and FPCA.

    Parameters
    ----------
    fdata : skfda.FDataGrid
        The functional dataset for which the number of clusters is to be estimated.
    ncluster_grid : array-like, optional
        List or array specifying the grid within which the number of clusters is searched. 
        Defaults to range(2, 15).

    Returns
    -------
    int
        The estimated number of clusters in the functional dataset.
    """
    if ncluster_grid is None:
        ncluster_grid = range(2,15)

    fpca_ = FPCA(n_components = 1)
    scores = fpca_.fit_transform(fdata)

    bic_ = []
    aic_ = []
    for n_comp in ncluster_grid:
        model = GaussianMixture(n_components=n_comp, n_init= 20)
        model.fit(scores)
        bic_.append(model.bic(scores))
        aic_.append(model.aic(scores))
    
    return min([ncluster_grid[np.argmin(aic_)], ncluster_grid[np.argmin(bic_)]])

def hybrid_representative_selection(data, p : float, p1 : float):
    """
    Select representative samples from a large dataset using a hybrid approach 
    combining random sampling and KMeans clustering.

    This method first randomly samples a proportion `p` of the data, then applies 
    KMeans clustering to this subset to select a smaller set of representative samples.

    Parameters
    ----------
    data : skfda.FDataGrid or np.ndarray
        Functional dataset or array to sample from.
    p : float
        Proportion of data to randomly sample (0 < p < 1).
    p1 : float
        Proportion of data to use as representative samples (0 < p1 < p).

    Returns
    -------
    skfda.FDataGrid or np.ndarray
        Representative samples selected by KMeans clustering.

    Raises
    ------
    AssertionError
        If p1 is not less than p.
    ValueError
        If data is not of type skfda.FDataGrid or np.ndarray.
    """
    # Random Selection
    N = len(data)
    n = np.ceil(p * N).astype('int') # approx. (100 * p)% of dataset
    n_idx = np.random.choice(N, size = n, replace = False)

    # Kmeans selection
    assert p1 < p, "p1 must be less that p."
    k_reps = int(p1 * N) # approx (100 *p1)% of dataset
    Kmean_reps = KMeans(n_clusters=k_reps, init= 'k-means++',)
    if type(data) == skfda.FDataGrid:
        Kmean_reps.fit(data[n_idx].data_matrix.squeeze())
        return skfda.FDataGrid(data_matrix= Kmean_reps.cluster_centers_,
                               grid_points= data.grid_points[0])

    elif type(data) == np.ndarray:
        Kmean_reps.fit(data[n_idx])
        return Kmean_reps.cluster_centers_

    else:
        raise ValueError("'data' type should be either skfda.FDataGrid or numpy.ndarray.")
