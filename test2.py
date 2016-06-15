import numpy as np
from scipy import sparse as sp
from sklearn.cluster import KMeans
def test_k_means_new_centers():
    # Explore the part of the code where a new center is reassigned
    X = np.array([[0, 0, 1, 1],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 1, 0, 0]])
    labels = [0, 1, 2, 1, 1, 2]
    bad_centers = np.array([[+0, 1, 0, 0],
                            [.2, 0, .2, .2],
                            [+0, 0, 0, 0]])

    km = KMeans(n_clusters=3, init=bad_centers, n_init=1, max_iter=10,
                random_state=1)
    for this_X in (X, sp.coo_matrix(X)):
        print this_X
        km.fit(this_X)
        this_labels = km.labels_
        # Reorder the labels so that the first instance is in cluster 0,
        # the second in cluster 1, ...
        import pdb
        pdb.set_trace()
        this_labels = np.unique(this_labels, return_index=True)[1][this_labels]
        np.testing.assert_array_equal(this_labels, labels)

test_k_means_new_centers()