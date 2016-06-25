import pandas as pd
import numpy as np
import sys, time


def test_kmeans(filename, no_clusters):
    from sklearn.cluster import KMeans
    intertias = []
    df = pd.read_csv(filename)
    h_indep = [d for d in df.columns if "features" in d]
    h_dep = [d for d in df.columns if "class" in d]
    for _ in xrange(10):
        try:
            print "- ",
            sys.stdout.flush()

            indep = df[h_indep]
            dep = df[h_dep]

            kmeans = KMeans(n_clusters =no_clusters,  max_iter=100)
            kmeans.fit(indep)
            intertias.append(kmeans.inertia_)

        except:
            import traceback
            traceback.extract_stack()

    return np.mean(intertias)


def _test_kmeans():
    folder_name = "./data/"
    from os import listdir

    files = ['./data/ionek_f_eight_c_six.csv', './data/ionek_f_fiveonetwo_c_two.csv', './data/ionek_f_four_c_four.csv',
             './data/ionek_f_sixteen_c_eight.csv', './data/ionek_f_thirty_two_c_eight.csv',
             './data/ionek_f_two_c_two.csv', './data/ionek_f_twofiftysix_c_eight.csv']

    clusters = [6,2, 4, 8, 8, 2, 8]
    for file, cluster in zip(files, clusters):
        print file, " : ",
        print test_kmeans(file, cluster)

if __name__ == "__main__":
    _test_kmeans()