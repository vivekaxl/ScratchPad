import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from sklearn import datasets

X, y = datasets.make_classification(n_samples=100, n_features=4, n_classes=2, n_informative=4, n_redundant=0, n_clusters_per_class=3)

import pdb
pdb.set_trace()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
