# GPmix

## Getting started

This guide will help you understand how to use the package by demonstrating it on the CBF dataset, which is one of the real datasets referenced in our paper. Follow these steps to prepare the dataset for analysis:

```python
import numpy as np
data = np.concatenate([np.loadtxt('CBF\CBF_TEST.txt'), np.loadtxt('CBF\CBF_TRAIN.txt')])
X, y = data[:,1:], data[:,0]
```

**Importing Necessary Modules**

To use the GPmix algorithm within your project, you need to import the necessary modules. The following import statements will include all the main functionalities from the GPmix package, as well as the specific utility to estimate the number of clusters:

```python
from GPmix import *
from GPmix.misc import estimate_nclusters
```

**Smoothing sample curve**

To smoothing the sample curve, use the `Smoother` object. Begin by initializing the `Smoother` object. You need to specify the type of basis for the smoothing process. The supported basis options include fourier, bspline and wavelet basis. You can customize the smoothing by passing additional configurations through the `basis_params` argument. If not specified, the system will automatically determine the best configurations using methods like Random Grid Search and Generalized Cross Validation. For this demonstration, we use the fourier basis. After initialization, apply the `fit` method to your data to smooth it.

```python
sm = Smoother(basis= 'fourier')
fd = sm.fit(X)
fd.plot(group = y)
```
![](cbf_smooth.png)

**Projection of sample functions**

Use `Projector` object to project the sample functions unto specified (projection) function(s). The `Projector` object needs to be initialized with the type of projection function and the desired number of projections. The `basis_type` is for specifying the type of projection function (basis). Supported `basis_type` are: fourier, bspline, fpc, rl-fpc, Ornstein-Uhlenbeck random functions, and wavelet. Further, Parameter `n_proj` is for number of projections and `basis_params` for further configuration of the projection function. For instance, we use wavelet projection function in this demonstration, and used `basis_params` to specify the family of wavelet we want to use. Apply the `fit`method to the smoothed functions to compute the projection coefficients.


## Replicating Experiment Results
The simulation scenarios described in our study are implemented in the `simulations.py` file. To reproduce the results from the paper for each specific scenario, you will need to execute the following command:

 `python GPmix_Clustering.py data_config/scenario_<tag>_config.yml`

Replace `<tag>` with the appropriate scenario identifier, which ranges from A to L. Each tag corresponds to a different configuration file located in the data_config directory. By executing the command with the relevant tag, the results for that particular scenario will be replicated. 

