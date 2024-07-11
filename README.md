

![](projection_illustration.png)

# About GPmix

[GPmix](GPmix) is a clustering algorithm for functional data that are generated from Gaussian process mixtures. Although designed for Gaussian process mixtures, our experimental study demonstrated that GPmix works well even for functional data that are not from Gaussian process mixtures.

The main steps of the algorithm are:

<ul>
 <li><strong>Smoothing</strong>: Apply smoothing methods on the raw data to get continuous functions.</li>
 <li><strong>Projection</strong>: Project the functional data onto a few randomly generated functions.</li>
 <li><strong>Learning GMMs</strong>: For each projection function, learn a univariate Gaussian mixture model from the projection coefficients.</li>
 <li><strong>Ensemble</strong>: Extract a consensus clustering from the multiple GMMs.</li>
</ul>

If you used this package in your research, please cite it:
```latex
@InProceedings{pmlr-AK2024,
  title =        {Learning mixtures of {Gaussian} processes through random projection},
  author =       {Akeweje, Emmanuel and Zhang, Mimi},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {},
  year = 	 {2024},
  editor = 	 {},
  volume = 	 {},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
}
```


# Getting Started

This guide will demonstrate how to use the package with the [CBF](CBF) dataset, one of the real-world datasets tested in our paper. Follow these steps to prepare the dataset for analysis:

```python
import numpy as np
data = np.concatenate([np.loadtxt('CBF\CBF_TEST.txt'), np.loadtxt('CBF\CBF_TRAIN.txt')])
X, y = data[:,1:], data[:,0]
```

To use the GPmix algorithm in your project, start by importing the necessary modules. The following import statements will include all the main functionalities from the GPmix package, as well as the specific utility for estimating the number of clusters:

```python
from GPmix import *
from GPmix.misc import estimate_nclusters
```

## Smoothing

To smooth the sample curves, use the `Smoother` object: Smoother(basis = 'bspline', basis_param = {}, domain_range = None).

**Parameters**
- basis : str, default = 'bspline'.<br> Smoothing basis. Supports 'bspline', 'fourier', 'wavelet', 'nadaraya_watson', and 'knn'.
- basis_params : dict, default = { }.<br>
    Additional parameters for smoothing basis. By default, except for wavelet basis, the required parameters are selected via generalized cross-validation (GCV) technique.
    Example: <br>
    ```
    B-spline basis: {'order': 3, 'n_basis': 20}
    Wavelet basis: {'wavelet': 'db4', 'n_basis': 20, 'mode' : 'soft'}
    Kernel basis: {'bandwidth': 1.0}
    Fourier basis: {'n_basis': 20, 'period': 1}
    ```
- domain_range : tuple | None. default = None. <br>
    The domain range of the functional data. For domain_range = None, the domain range is either set to [0,1]  if data is array-like, 
    or set to the domain_range of the data if data is FDataGrid object.

**Methods**
- fit(X) :
    Construct smoothed functions from raw data X.
- plot :


Begin by initializing the `Smoother` object, specifying the type of basis for the smoothing process. The supported basis options include Fourier, B-spline, and wavelet basis. You can customize the smoothing by passing additional configurations through the `basis_params` argument. If not specified, the system will automatically determine the best configurations using methods like Random Grid Search and Generalized Cross Validation. After initialization, apply the `fit` object to your data to smooth it.

For this demonstration, we will use the Fourier basis.

```python
sm = Smoother(basis= 'fourier')
fd = sm.fit(X)
fd.plot(group = y)
```
![](cbf_smooth.png)

## Projection

To project the sample functions onto specified projection functions, use the `Projector` object. Initialize the `Projector` object with the type of projection functions and the desired number of projections. The `basis_type` argument specifies the type of projection functions. Supported `basis_type` options are: eigen-functions from the fPC decomposition (fPC), random linear combinations of eigen-functions (rl-fPC), B-splines, Fourier basis, discrete wavelets, and Ornstein-Uhlenbeck (OU) random functions. The `n_proj` argument defines the number of projections. The `basis_params` argument allows for further configuration of the projection functions.

For this demonstration, we will use wavelets as projection functions. We will specify the family of wavelets using `basis_params`. After initializing, apply the `fit` function to the smoothed functions to compute the projection coefficients. Here, we will use 14 projection functions generated from the Haar wavelet family.

```python
proj = Projector(basis_type= 'wavelet', n_proj = 14, basis_params= {'wv_name': 'haar'})
coeffs = proj.fit(fd)
```

## Ensemble Clustering

The `UniGaussianMixtureEnsemble` object facilitates ensemble clustering by fitting a univariate Gaussian Mixture Model (GMM) to each set of projection coefficients. Follow these steps:

- Initialize the `UniGaussianMixtureEnsemble` object by specifying the number of clusters (`n_clusters`) you want to identify in your dataset.
- Use the `fit_gmms` method to obtain a collection of GMMs, one for each set of projection coefficients.
- Use the `get_clustering` object, which aggregates the results from the individual GMMs to form a consensus clustering.

  
For this demonstration, we will identify 3 clusters in the sample dataset.

```python
model = UniGaussianMixtureEnsemble(n_clusters= 3)
model.fit_gmms(coeffs)
pred_labels = model.get_clustering()
```
To visualize the clustering result, apply the `plot_clustering` method to the functional data object:

```python
model.plot_clustering(fd)
```
![](cbf_clustering.png)

Furthermore, the `UniGaussianMixtureEnsemble` object supports the calculation of several clustering validation indices. For external validation (comparing generated clusters against true labels), you can calculate Adjusted Mutual Information, Adjusted Rand Index, and Correct Classification Accuracy by passing the true labels as parameters. For internal validation (assessing the internal structure of the clusters), you can calculate the Silhouette Score and Davies-Bouldin Score by passing the functional data object as parameters. These metrics help evaluate the effectiveness of the clustering process.

For this demonstration, we calculate all the clustering validation metrics.

```python
model.adjusted_mutual_info_score(y)   # Adjusted Mutual Information

model.adjusted_rand_score(y)    # Adjusted Rand Index

model.correct_classification_accuracy(y)    # Correct Classification Accuracy

model.silhouette_score(fd)    # Silhouette Score

model.davies_bouldin_score(fd)    # Davies-Bouldin Score
```


## Estimating the Number of Clusters
To effectively estimate the optimal number of clusters in a dataset, our package includes the `estimate_nclusters` function. This function employs a systematic search to identify the number of clusters that minimize the Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC), as detailed in our paper. Here’s how you can apply this function to your data:

```python
estimate_nclusters(fd)
```


# Replicating the Experiment Results
The simulation scenarios investigated in our paper are available in [simulations.py](simulations.py). To reproduce the results from the paper for each specific scenario, you will need to execute the following command after cloning the repo:

 ```bash
 python GPmix_Clustering.py data_configs/scenario_<tag>_config.yml
```

Replace `<tag>` with the appropriate scenario identifier, which ranges from A to L. Each tag corresponds to a different configuration file located in the [data_configs](data_configs). By executing the command with the relevant tag, the results for that particular scenario will be replicated. 



# Contributing

**This project is under active development. If you find a bug, or anything that needs correction, please let us know.** 

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.
