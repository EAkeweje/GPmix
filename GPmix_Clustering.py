##Enhanced version.
### Fits all gmms first then fit on selected gmms to evaluate performance.


import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse

from GPmix import *
from GPmix.misc import *
from simulations import *

parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()


#read config file
with open(args.config, 'r') as config_file:
    config = yaml.safe_load(config_file)

print('generating dataset ...')
X, y = generate_data(config['dataset'])

print('True number of clusters: ', len(np.unique(y)))
print('Estimated number of clusters: ', estimate_nclusters(X))

print('Clustering with GPmix ...')
#smoothing
sm = Smoother(basis = config['smoother basis'], basis_params = config['smoother basis param'])
fdata_smoothed = sm.fit(X)
#projection
proj = Projector(basis_type = config['projector basis'], n_proj = config['n proj'], basis_params = config['projector basis param'])
coefficients = proj.fit(fdata_smoothed)
#fitting univariate gmms
model = UniGaussianMixtureEnsemble(n_clusters = len(np.unique(y)), init_method= config['gmm init method'])
model.fit_gmms(coefficients)
#ensemble clustering
pred_labels = model.get_clustering()

print('##### GPmix performance ######',
    '\nARI: ', model.adjusted_rand_score(y),
      '\nAMI: ', model.adjusted_mutual_info_score(y),
      '\nCCA:', model.correct_classification_accuracy(y),
      '\nSIL: ', silhouette_score(fdata_smoothed, pred_labels),
      '\nDB: ', davies_bouldin_score(fdata_smoothed, pred_labels))

#visualize clusters
fdata_smoothed.plot(group = pred_labels)
plt.show()