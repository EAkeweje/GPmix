

![](projection_illustration.png)

# 1. About GPmix

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
@InProceedings{pmlr-v235-akeweje24a,
  title = 	 {Learning Mixtures of {G}aussian Processes through Random Projection},
  author =       {Akeweje, Emmanuel and Zhang, Mimi},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {720--739},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
}
```


# 2. Code Example

This quick start guide will demonstrate how to use the package with the [CBF](CBF) dataset, one of the real-world datasets tested in our paper. Follow these steps to prepare the dataset for analysis:

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
Begin by initializing the `Smoother` object, specifying the type of basis for the smoothing process. You can customize the smoothing by passing additional configurations through the `basis_params` argument. If not specified, the system will automatically determine the best configurations using methods like Random Grid Search and Generalized Cross Validation. After initialization, apply the `fit` method to the raw data to obtain the fitted functional data.

For this demonstration, we will use the Fourier basis.

```python
sm = Smoother(basis= 'fourier')
fd = sm.fit(X)
fd.plot(group = y)
```
![](cbf_smooth.png)

## Projection
To project the fitted functional data onto specified projection functions, use the `Projector` object. Initialize the `Projector` object with the type of projection functions and the desired number of projections. The `basis_type` argument specifies the type of projection functions. The `n_proj` argument defines the number of projections. The `basis_params` argument allows for further configuration of the projection functions.

For this demonstration, we will use wavelets as projection functions. We will specify the family of wavelets using `basis_params`. After initializing, apply the `fit` method to the functional data object to compute the projection coefficients. Here, we will use 14 projection functions generated from the Haar wavelet family.

```python
proj = Projector(basis_type= 'wavelet', n_proj = 14, basis_params= {'wv_name': 'haar'})
coeffs = proj.fit(fd)
```

## Ensemble Clustering

The `UniGaussianMixtureEnsemble` object facilitates ensemble clustering by fitting a univariate Gaussian Mixture Model (GMM) to each set of projection coefficients. Follow these steps:

- Initialize the `UniGaussianMixtureEnsemble` object by specifying the number of clusters (n_clusters) you want to identify in your dataset.
- Use the `fit_gmms` method to obtain a collection of GMMs, one for each set of projection coefficients.
- Use the `get_clustering` method, which aggregates the results from the individual GMMs, to form a consensus clustering.

For this demonstration, we will identify 3 clusters in the functional data.

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
To estimate the optimal number of clusters in the functional data, our package includes the `estimate_nclusters` function. This function employs a systematic search to identify the number of clusters that minimize the Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC), as detailed in our paper. Here’s how you can apply this function to your data:

```python
estimate_nclusters(fd)
```

## Replicating the Experiment Results
The simulation scenarios investigated in our paper are available in [simulations.py](simulations.py). To reproduce the results from the paper for each specific scenario, you will need to execute the following command after cloning the repo:

 ```bash
 python GPmix_Clustering.py data_configs/scenario_<tag>_config.yml
```

Replace `<tag>` with the appropriate scenario identifier, which ranges from A to L. Each tag corresponds to a different configuration file located in the [data_configs](data_configs). By executing the command with the relevant tag, the results for that particular scenario will be replicated. 



# 3. Package Functions

##  `GPmix.Smoother`

Apply smoothing methods on the raw data to get continuous functions.
```python
Smoother(basis = 'bspline', basis_params = {}, domain_range = None)
```
**Parameters**<br>
- <strong> basis {'bspline', 'fourier', 'wavelet', 'nadaraya_watson', 'knn'} </strong>: a string specifying the smoothing method to use. The default value is `'bspline'`. 
- <strong> basis_params (dict) </strong>: additional parameters for the selected smoothing method.  The default value is `{}`.  Below are examples of how to specify these parameters for different smoothing methods:
    ```python
    Smoother(basis = 'bspline', basis_params = {'order': 3, 'n_basis': 20})
    Smoother(basis = 'wavelet', basis_params = {'wavelet': 'db5', 'mode': 'soft'}) 
    Smoother(basis = 'knn', basis_params = {'bandwidth': 1.0})
    Smoother(basis = 'fourier', basis_params = {'n_basis': 20, 'period': 1})
    ```
    For all smoothing methods except wavelet smoothing, if `basis_params` is not specified, the parameter values are determined by the Generalized Cross-Validation (GCV) technique.<br>
    For wavelet smoothing, the wavelet shrinkage denoising technique is implemented, requiring two parameters:
    - `'wavelet'`: The wavelet family to use for smoothing. A list of all supported discrete wavelets can be obtained by running: `print(pywt.wavelist(kind='discrete'))`.
    - `'mode'`: The method and extent of denoising. The available modes are: {'soft', 'hard', 'garrote', 'greater', 'less'}.<br>
   
   For wavelet smoothing, if `basis_params` is not specified, the default configuration `basis_params = {'wavelet': 'db5', 'mode': 'soft'}` will be used.
- <strong> domain_range (tuple) </strong>: the domain of the functions. The default value is `None`. <br>
  If `domain_range` is set to `None`, the domain range will default to `[0,1]` if the data is array-like. If the data is an `FDataGrid` object, it will use the `domain_range` of that object.

**Attributes**<br>
- <strong> fd_smooth (FDataGrid)</strong>: functional data obtained via the smoothing technique.

**Methods**<br>
- `fit(X, return_data = True)`: Apply a smoothing method on the raw data `X` to get continuous functions. <br>  
  - <strong> X </strong>: raw data, array-like of shape (n_samples, n_features) or FDataGrid object.
  - <strong> return_data (bool) </strong>: Return the functional data if True. The default value is `True`.

##  `GPmix.Projector`

Project the functional data onto a few randomly generated functions.
```python
Projector(basis_type, n_proj = 3, basis_params = {})
```
**Parameters**<br>
- <strong> basis_type {'fourier', 'fpc', 'wavelet', 'bspline', 'ou', 'rl-fpc'} </strong>: a string specifying the type of projection functions. Supported `basis_type` options are: eigen-functions from the fPC decomposition (`'fpc'`), random linear combinations of eigen-functions (`'rl-fpc'`), B-splines, Fourier basis, wavelets, and Ornstein-Uhlenbeck (`'ou'`) random functions. 
- <strong> n_proj (int) </strong>: number of projection functions to generate. The default value is `3`.
- <strong> basis_params (dict) </strong>: additional hyperparameters required by `'fourier'`, `'bspline'` and `'wavelet'`. The default value is `{}`. Below are examples of how to specify these hyperparameters:
    ```python
    Projector(basis_type = 'fourier', basis_params = {'period': 2})
    Projector(basis_type = 'bspline', basis_params = {'order': 1}) 
    Projector(basis_type = 'wavelet', basis_params = {'wv_name': 'haar', 'resolution': 1})
    ```
    For `fourier`, the default configuration is setting the `'period'` equal to the length of the domain of the functional data. This approach works well for all datasets investigated in our paper. However, we have included the period as a hyperparameter in case users want to project onto Fourier functions with lower (or higher) oscillations. For `bspline`, if `basis_params` is not specified, the default configuration `basis_params = {'order': 3}` will be applied. Similarly, for `wavelet`, if `basis_params` is not specified, the default configuration `basis_params = {'wv_name': 'db5', 'resolution': 1}` will be applied.

**Attributes** <br>
- <strong> n_features (int) </strong>: number of evaluation points for each sample curve and for the projection functions.
- <strong> basis (FDataGrid) </strong>: generated projection functions.
- <strong> coefficients (array-like of shape (n_proj, sample size)) </strong>: projection coefficients.

**Methods** <br>
- `fit(FDataGrid)` : computing the projection coefficients. Return array-like object of shape (n_proj, sample size).
- `plot_basis()` : plots the projection functions.
- `plot_projection_coeffs(**kwargs)` : plots the distribution of projection coefficients. Takes `kwargs` from `seaborn.histplot`.

##  `GPmix.UniGaussianMixtureEnsemble`

For each projection function, learn a univariate Gaussian mixture model from the projection coefficients. Then extract a consensus clustering from the multiple GMMs.
```python
UniGaussianMixtureEnsemble(n_clusters, init_method = 'kmeans', n_init = 10, mom_epsilon = 5e-2)
```
**Parameters**<br>
- <strong> n_clusters (int) </strong>: number of mixture components in the GMMs.
- <strong> init_method {'kmeans', 'k-means++', 'random', 'random_from_data', 'mom'} </strong>: method for initializing the EM algorithm (for estimating GMM parameters). The default value is `'kmeans'`.
- <strong> n_init (int) </strong>: number of repeats of the EM algorithm, each with a different initilization. The algorithm returns the best GMM fit. The default value is `10`.
- <strong> mom_epsilon (float) </strong>: lower bound for GMM weights, only applicable if `init_method = 'mom'`. The default value is `5e-2`.
    
**Attributes**<br>
- <strong> n_projs (int) </strong>: number of base clusterings (or projections).
- <strong> data_size (int) </strong>: sample size.
- <strong> gmms (list) </strong> : a list of univariate GMMs, one for each projection function.
- <strong> clustering_weights_ (array-like of shape (n_projs,)) </strong>: weights for the base clusterings.

**Methods**<br>
- `fit_gmms(projs_coeffs,  n_jobs = -1, **kwargs)`: fit GMM to projection coefficients.
  - <strong> projs_coeffs (array-like of shape (n_proj, sample size)) </strong> : projection coefficients.
  - <strong> n_jobs </strong>: number of concurrently running jobs to parrallelize fitting the gmms. The default value is `-1`, to use all available CPUs.
  - <strong> kwargs </strong>: any keyword argument of `joblib.Parallel`.
-  `plot_gmms(ncol = 4, fontsize = 12, fig_kws = { }, **kwargs)`: visualization of GMM fits.
   - <strong> ncol (int) </strong>: number of subplot columns. The default value is `4`.
   - <strong> fontsize (int) </strong>: fontsize for the plot labels. The default value is `12`.
   - <strong> fig_kws </strong>: keyword arguments for the figures (subplots). The default value is `{}`.
   - <strong> kwargs </strong>: other keyword arguments for customizing seaborn `histplot`.
- `get_clustering(weighted_sum = True, precompute_gmms = None)`: obtain the consensus clustering. Return array-like object of shape (sample size,), the cluster labels for the sample curves.
   - <strong> weighted_sum (bool) </strong>: specifying whether the total misclassification probability, which measures the overlap among the GMM components, should be weighted by the mixing proportion. The default value is `True`.
   - <strong> precompute_gmms (list) </strong>: a subset of the fitted GMMs. By default, the consensus clustering is extracted from all the fitted GMMs. This parameter allows selecting a subset of the fitted GMMs for constructing the consensus clustering.       
- `plot_clustering(FDataGrid)` : visualize the clustered functional data.
- `adjusted_mutual_info_score(true_labels)`: computing the Adjusted Mutual Information.
    - <strong> true_labels (array-like of shape (sample size,)) </strong> : true cluster labels.
- `adjusted_rand_score(true_labels)`: computing the Adjusted Rand Index.
    - <strong> true_labels (array-like of shape (sample size,)) </strong> : true cluster labels.
- `correct_classification_accuracy(true_labels)`: computing the Correct Classification Accuracy.
    - <strong> true_labels (array-like of shape (sample size,)) </strong> : true cluster labels.
- `silhouette_score(FDataGrid)`: computing the Silhouette Score.
- `davies_bouldin_score(FDataGrid)`: computing the Davies-Bouldin Score.
 
 ## `GPmix.misc.estimate_nclusters`
 
The function `estimate_nclusters` employs a systematic search to identify the number of clusters that minimize the Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC).
 ```python
estimate_nclusters(fdata, ncluster_grid = None)
```
**Parameters**<br>
- <strong> fdata (FDataGrid) </strong>: functional data object.
- <strong> ncluster_grid (array-like) </strong>: specifies the grid within which the number of clusters is searched. The default value is `[2, 3, ..., 14]`. <br>

  

# Contributing

**This project is under active development. If you find a bug, or anything that needs correction, please let us know.** 

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.
