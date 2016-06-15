import numpy as np
import pandas as pd
from random import random, uniform
from sklearn.datasets.samples_generator import make_blobs
import sys
from sklearn import datasets


def generate_spherical_clusters(number_of_samples, number_of_clusters, n_features=2,  variances=None, filename=""):
    """
    :param number_of_samples:  The total number of points equally divided among clusters.
    :param number_of_clusters: The number of clusters to generate
    :param n_features:         The number of features for each sample.
    :param variances:          The standard deviation of the clusters.
    :param filename:           The file to store the results
    :return:
    """
    if variances is None: variances = [0.5 for _ in xrange(number_of_clusters)]
    if filename == "":
        filename = "./Data/spherical_" + str(number_of_samples) + "_features_" + str(n_features) \
                                  + "_cluster_" + str(number_of_clusters) + ".csv"
    random_state = 170
    X, y = make_blobs(n_samples=number_of_samples, centers=number_of_clusters, n_features=n_features,
                      random_state=random_state, cluster_std=variances)
    features = ["features_" + str(i+1) for i in xrange(n_features)]
    df = pd.DataFrame()
    for i, feature in enumerate(features): df[feature] = X[:, i]
    df["class"] = y
    df.to_csv(filename, index=False)

    return X, y


def _generate_spherical_clusters():
    X, y = generate_spherical_clusters(10000, 4)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def generate_anisotropically_clusters(number_of_samples, number_of_clusters, n_features=2,  variances=None, filename=""):
    """
    :param number_of_samples:  The total number of points equally divided among clusters.
    :param number_of_clusters: The number of clusters to generate
    :param n_features:         The number of features for each sample.
    :param variances:          The standard deviation of the clusters.
    :param filename:           The file to store the results
    :return:
    """

    if variances is None: variances = [0.5 for _ in xrange(number_of_clusters)]
    if filename == "":
        filename = "./Data/anisotropically_" + str(number_of_samples) + "_features_" + str(n_features) \
                   + "_cluster_" + str(number_of_clusters) + ".csv"
    random_state = 170
    X, y = make_blobs(n_samples=number_of_samples, centers=number_of_clusters, n_features=n_features,
                      random_state=random_state, cluster_std=variances)
    transformation = np.array([[random() if i == j else uniform(-1, 1) for j in xrange(n_features)] for i in xrange(n_features)])
    X = np.dot(X, transformation)

    features = ["features_" + str(i + 1) for i in xrange(n_features)]
    df = pd.DataFrame()
    for i, feature in enumerate(features): df[feature] = X[:, i]
    df["class"] = y
    df.to_csv(filename, index=False)

    return X, y


def _generate_anisotropically_clusters():

    X, y = generate_anisotropically_clusters(10000, 4)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def generate_varied_clusters(number_of_samples, number_of_clusters, n_features=2,  variances=None, filename=""):
    """
    :param number_of_samples:  The total number of points equally divided among clusters.
    :param number_of_clusters: The number of clusters to generate
    :param n_features:         The number of features for each sample.
    :param variances:          The standard deviation of the clusters.
    :param filename:           The file to store the results
    :return:
    """
    if variances is None: variances = [uniform(1, 3) for _ in xrange(number_of_clusters)]
    if filename == "":
        filename = "./Data/varied_" + str(number_of_samples) + "_features_" + str(n_features) \
                                  + "_cluster_" + str(number_of_clusters) + ".csv"
    random_state = 170
    X, y = make_blobs(n_samples=number_of_samples, centers=number_of_clusters, n_features=n_features,
                      random_state=random_state, cluster_std=variances)
    features = ["features_" + str(i+1) for i in xrange(n_features)]
    df = pd.DataFrame()
    for i, feature in enumerate(features): df[feature] = X[:, i]
    df["class"] = y
    df.to_csv(filename, index=False)

    return X, y


def _generate_varied_clusters():
    X, y = generate_varied_clusters(10000, 4)
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def points_for_clusters():
    no_instances = [100, 1000, 10000, 100000, 1000000]
    no_features = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    no_clusters = [2, 4, 6, 8]

    for no_instance in no_instances:
        for no_feature in no_features:
            for no_cluster in no_clusters:
                print "# ",
                sys.stdout.flush()
                generate_spherical_clusters(number_of_samples=no_instance, number_of_clusters=no_cluster, n_features=no_feature)
                generate_anisotropically_clusters(number_of_samples=no_instance, number_of_clusters=no_cluster, n_features=no_feature)
                generate_varied_clusters(number_of_samples=no_instance, number_of_clusters=no_cluster, n_features=no_feature)
            print
        print


def generate_regression_datasets(n_samples, n_features, noise=25, bias=100):
    X = datasets.make_regression(n_samples=n_samples, n_features=n_features, coef=True, noise=noise, bias=bias)
    header = ["A" + str(i) for i in xrange(len(X[0][0]))] + ["Dep"]
    content = []
    for independent, dependent in zip(X[0], X[1]):
        content.append(independent.tolist() + [dependent])

    filename = "./RData/Regression_" + str(n_samples) + "_" + str(n_features) + "_" + str(noise) + "_" + str(bias) + ".csv"
    df = pd.DataFrame(content, columns=header)
    df.to_csv(filename, index=False)


def _generate_regression_datasets():
    n_samples = [100, 1000, 10000, 100000, 1000000]
    no_features = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    for n_sample in n_samples:
        for no_feature in no_features:
                print "# ",
                sys.stdout.flush()
                generate_regression_datasets(n_samples=n_sample, n_features=no_feature)
        print


def generate_classification_dataset(n_samples, n_features, n_informative, weights, n_clusters_per_class=3, n_classes=2):
    """
    Constraints:
        - Number of informative, redundant and repeated features must sum to less than the number of total features
        - n_classes * n_clusters_per_class must be smaller or equal 2 ** n_informative

    :param n_samples:
    :param n_features:
    :param n_informative:
    :param weights:
    :param n_clusters_per_class:
    :param n_classes:
    :return:
    """
    from sklearn import datasets

    X = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                     n_informative=n_informative, n_redundant=2, weights=weights)
    header = ["A" + str(i) for i in xrange(len(X[0][0]))] + ["Dep"]
    content = []
    for independent, dependent in zip(X[0], X[1]):
        content.append(independent.tolist() + [dependent])

    filename = "./CData/Classification_" + str(n_samples) + "_" + str(n_features) + "_" + str(n_classes) + "_" + str(weights[0]*100) + ".csv"
    df = pd.DataFrame(content, columns=header)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    n_samples = [100, 1000, 10000, 100000, 1000000]
    no_features = [8, 16, 32, 64, 128, 256, 512]
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    for n_sample in n_samples:
        for no_feature in no_features:
            for weight in weights:
                print "# ",
                sys.stdout.flush()
                t_w = [weight, 1-weight]
                generate_classification_dataset(n_samples=n_sample, n_features=no_feature, n_informative=2, weights=t_w)
            print
        print