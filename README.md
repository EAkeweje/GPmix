# About GPmix

[GPmix](GPmix) is a clustering algorithm for functional data that are generated from Gaussian process mixtures. The main steps of the algorithm are:

<ul>
 <li>Smoothing: Apply smoothing methods on the raw data to get continuous functions.</li>
 <li>Projection: Project the functional data onto a few randomly generated functions.</li>
 <li>Learning GMMs: For each projection function, learn a univariate Gaussian mixture model from the projection coefficients.</li>
 <li>Ensemble: Extract a consensus clustering from the multiple GMMs.</li>
</ul>

Although designed for Gaussian process mixtures, our experimental study demonstrated that GPmix works well even for functional data that are not from Gaussian process mixtures. If you used this package in your research, please cite it:
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

This guide will help you understand how to use the package by demonstrating it on the [CBF](CBF) dataset, which is one of the real datasets referenced in our paper. Follow these steps to prepare the dataset for analysis:

```python
import numpy as np
data = np.concatenate([np.loadtxt('CBF\CBF_TEST.txt'), np.loadtxt('CBF\CBF_TRAIN.txt')])
X, y = data[:,1:], data[:,0]
```

## Importing Necessary Modules

To use the GPmix algorithm within your project, you need to import the necessary modules. The following import statements will include all the main functionalities from the GPmix package, as well as the specific utility to estimate the number of clusters:

```python
from GPmix import *
from GPmix.misc import estimate_nclusters
```

## Smoothing Sample Curve

To smoothing the sample curve, use the `Smoother` object. Begin by initializing the `Smoother` object. You need to specify the type of basis for the smoothing process. The supported basis options include fourier, bspline and wavelet basis. You can customize the smoothing by passing additional configurations through the `basis_params` argument. If not specified, the system will automatically determine the best configurations using methods like Random Grid Search and Generalized Cross Validation. After initialization, apply the `fit` method to your data to smooth it.

For this demonstration, we use the fourier basis.

```python
sm = Smoother(basis= 'fourier')
fd = sm.fit(X)
fd.plot(group = y)
```
![](cbf_smooth.png)

## Projection of Sample Functions

Use `Projector` object to project the sample functions unto specified (projection) function(s). The `Projector` object needs to be initialized with the type of projection function and the desired number of projections. The `basis_type` is for specifying the type of projection function (basis). Supported `basis_type` are: fourier, bspline, fpc, rl-fpc, Ornstein-Uhlenbeck random functions, and wavelet. Further, Parameter `n_proj` is for number of projections and `basis_params` for further configuration of the projection function. For instance, we use wavelet projection function in this demonstration, and so we use `basis_params` to specify the family of wavelet we want to use. Afterwards, apply the `fit` method to the smoothed functions to compute the projection coefficients.

Here, we use 14 projection functions generated from the Haar wavelet family.

```python
proj = Projector(basis_type= 'wavelet', n_proj = 14, basis_params= {'wv_name': 'haar'})
coeffs = proj.fit(fd)
```

## Ensemble Clustering

The `UniGaussianMixtureEnsemble` object facilitates ensemble clustering by fitting multiple collections of scalars to Gaussian Mixture Models (GMMs). Each collection is modeled using a univariate GMM, enabling the generation of an ensemble clustering from these models as described in the paper. 

Start by initializing the `UniGaussianMixtureEnsemble` with the number of clusters `n_clusters` you wish to identify in your dataset. Use the `fit_gmms` method to fit GMMs to the projection coefficients. After fitting the GMMs, use the `get_clustering` method to generate the consensus clustering. This method aggregates the results from the individual GMMs to form a unified cluster label for each sample.

For this demonstration, there are 3 clusters in the sample dataset:
```python
model = UniGaussianMixtureEnsemble(n_clusters= 3)
model.fit_gmms(coeffs)
pred_labels = model.get_clustering()
```
To visualize the clustering, apply the `plot_clustering` method to the sample dataset:
```python
model.plot_clustering(fd)
```
![](cbf_clustering.png)

Furthermore, `UniGaussianMixtureEnsemble` object supports several methods for both internal and external validation of cluster quality. These metrics help evaluate the effectiveness of the clustering process by comparing the generated clusters against true labels or by assessing the internal structure of the clusters. Below is how to apply the metrics:

1. Adjusted Mutual Information: Pass true labels as parameters.
```python
model.adjusted_mutual_info_score(y)
```

2. Adjusted Rand Index: Pass true labels as parameters.
```python
model.adjusted_rand_score(y)
```

3. Correct Classification Accuracy: Pass true labels as parameters.
```python
model.correct_classification_accuracy(y)
```

4. Silhouette score: Pass (smoothed) sample dataset as parameters.
```python
model.silhouette_score(fd)
```

5. Davies-Bouldin Score: Pass (smoothed) sample dataset as parameters.
```python
model.davies_bouldin_score(fd)
```

## Estimating the Number of Clusters
To effectively estimate the optimal number of clusters in a dataset, our package includes the `estimate_nclusters` function. This function employs a systematic search to identify the number of clusters that minimize the Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC), as discussed in our paper. Here is how to apply this function to your data:
```python
estimate_nclusters(fd)
```
The function returns the estimated number of clusters.

# Replicating Experiment Results
The simulation scenarios described in our study are available in [simulations.py](simulations.py). To reproduce the results from the paper for each specific scenario, you will need to execute the following command after cloning the repo:

 ```bash
 python GPmix_Clustering.py data_configs/scenario_<tag>_config.yml
```

Replace `<tag>` with the appropriate scenario identifier, which ranges from A to L. Each tag corresponds to a different configuration file located in the [data_configs](data_configs). By executing the command with the relevant tag, the results for that particular scenario will be replicated. 

