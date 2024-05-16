# GPmix
---
GPmix package is the implementation of an efficient algorithm for clustering functional data using Random Projection, specifically designed to uncover latent cluster labels in data generated from Gaussian process mixtures. Our method exploits the fact that the projection coefficients of the functional data onto any given projection function follow a univariate Gaussian mixture model (GMM). By conducting multiple one-dimensional projections and learning a univariate GMM for each, we create an ensemble of GMMs. Each GMM serves as a base clustering, and applying ensemble clustering yields a consensus clustering. Our approach significantly reduces computational complexity compared to state-of-the-art methods, and we provide theoretical guarantees on the identifiability and learnability of Gaussian process mixtures. Extensive experiments on synthetic and real datasets confirm the superiority of our method over existing techniques.

## Getting started

This guide will help you understand how to use the package by demonstrating it on the CBF dataset, which is one of the real datasets referenced in our paper. Follow these steps to prepare the dataset for analysis:

```python
import numpy as np
data = np.concatenate([np.loadtxt('CBF\CBF_TEST.txt'), np.loadtxt('CBF\CBF_TRAIN.txt')])
X, y = data[:,1:], data[:,0]
```

### Importing Necessary Modules

To use the GPmix algorithm within your project, you need to import the necessary modules. The following import statements will include all the main functionalities from the GPmix package, as well as the specific utility to estimate the number of clusters:

```python
from GPmix import *
from GPmix.misc import estimate_nclusters
```

### Smoothing sample curve

To smoothing the sample curve, use the `Smoother` object. Begin by initializing the `Smoother` object. You need to specify the type of basis for the smoothing process. The supported basis options include fourier, bspline and wavelet basis. You can customize the smoothing by passing additional configurations through the `basis_params` argument. If not specified, the system will automatically determine the best configurations using methods like Random Grid Search and Generalized Cross Validation. After initialization, apply the `fit` method to your data to smooth it.

For this demonstration, we use the fourier basis.

```python
sm = Smoother(basis= 'fourier')
fd = sm.fit(X)
fd.plot(group = y)
```
![](cbf_smooth.png)

### Projection of sample functions

Use `Projector` object to project the sample functions unto specified (projection) function(s). The `Projector` object needs to be initialized with the type of projection function and the desired number of projections. The `basis_type` is for specifying the type of projection function (basis). Supported `basis_type` are: fourier, bspline, fpc, rl-fpc, Ornstein-Uhlenbeck random functions, and wavelet. Further, Parameter `n_proj` is for number of projections and `basis_params` for further configuration of the projection function. For instance, we use wavelet projection function in this demonstration, and used `basis_params` to specify the family of wavelet we want to use. Apply the `fit`method to the smoothed functions to compute the projection coefficients.

Here, we use 14 projection functions generated from the Haar wavelet family.

```python
proj = Projector(basis_type= 'wavelet', n_proj = 14, basis_params= {'wv_name': 'haar'})
coeffs = proj.fit(fd)
```

Here's a refined and structured version of the "Ensemble Clustering" section for your README file, designed to guide users through using the `UniGaussianMixtureEnsemble` object more effectively:

### Ensemble Clustering

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

### Estimating the Number of clusters
To effectively estimate the optimal number of clusters in a dataset, our package includes the estimate_nclusters function. This function employs a systematic search to identify the number of clusters that minimize the Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC), as discussed in our paper. Here is how to apply this function to your data:
```python
estimate_nclusters(fd)
```
The function returns the estimated number of clusters. This number reflects the optimal balance between model complexity and goodness of fit, according to the AIC or BIC.

## Replicating Experiment Results
The simulation scenarios described in our study are implemented in the `simulations.py` file. To reproduce the results from the paper for each specific scenario, you will need to execute the following command:

 ```bash
 python GPmix_Clustering.py data_config/scenario_<tag>_config.yml
```

Replace `<tag>` with the appropriate scenario identifier, which ranges from A to L. Each tag corresponds to a different configuration file located in the data_config directory. By executing the command with the relevant tag, the results for that particular scenario will be replicated. 

