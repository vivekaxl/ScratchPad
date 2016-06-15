from __future__ import division
import pandas as pd
import numpy as np
import sys, time
from sklearn.metrics import precision_score, recall_score


def test_linear_regression(filename):
    start_time = time.time()
    coeffs = []
    rss = []
    from sklearn import datasets, linear_model
    df = pd.read_csv(filename)
    h_indep = [d for d in df.columns if "A" in d]
    h_dep = [d for d in df.columns if "A" not in d]
    for _ in xrange(10):
        print "- ",
        sys.stdout.flush()
        msk = np.random.rand(len(df)) < 0.5
        train_data = df[msk]
        test_data = df[msk]

        assert(len(train_data) == len(test_data)), "Something is wrong"

        train_indep = train_data[h_indep]
        train_dep = train_data[h_dep]

        test_indep = test_data[h_indep]
        test_dep = test_data[h_dep]

        regr = linear_model.LinearRegression()
        regr.fit(train_indep, train_dep)

        coeffs.append(regr.coef_)

        rss.append(np.mean((regr.predict(test_indep) - test_dep) ** 2))

    extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    import pickle
    pickle.dump(coeffs, open("./Results_Linear_Regression/coeffs_" + extract_name, "wb"))
    pickle.dump(rss, open("./Results_Linear_Regression/rss_" + extract_name, "wb"))
    print
    print "Total Time: ", time.time() - start_time


def _test_linear_regression():
    folder_name = "./RData/"
    from os import listdir
    files = sorted([folder_name + file for file in listdir(folder_name)])
    for file in files:
        print file
        test_linear_regression(file)


def test_logistic_regression(filename):
    start_time = time.time()
    coeffs = []
    acc = []
    from sklearn import datasets, linear_model
    df = pd.read_csv(filename)
    h_indep = [d for d in df.columns if "A" in d]
    h_dep = [d for d in df.columns if "A" not in d]
    for _ in xrange(10):
        print "- ",
        sys.stdout.flush()
        msk = np.random.rand(len(df)) < 0.5
        train_data = df[msk]
        test_data = df[msk]

        assert (len(train_data) == len(test_data)), "Something is wrong"

        train_indep = train_data[h_indep]
        train_dep = train_data[h_dep]

        test_indep = test_data[h_indep]
        test_dep = test_data[h_dep]
        logistic = linear_model.LogisticRegression()
        logistic.fit(train_indep, [i[-1] for i in train_dep.values.tolist()])
        coeffs.append(logistic.coef_)
        acc.append(logistic.score(X=test_indep, y=[i[-1] for i in test_dep.values.tolist()]))

    extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    import pickle
    pickle.dump(coeffs, open("./Results_Logistic_Regression/coeffs_" + extract_name, "wb"))
    pickle.dump(acc, open("./Results_Logistic_Regression/rss_" + extract_name, "wb"))
    print
    print "Total Time: ", time.time() - start_time


def _test_logistic_regression(filename):
    folder_name = "./CData/"
    from os import listdir
    files = sorted([folder_name + file for file in listdir(folder_name)])
    for file in files:
        print file
        # file = "./CData/Classification_10000_32_2_50.0.csv"
        test_logistic_regression(file)


def test_random_forest_classification(filename):
    start_time = time.time()
    confusion_matrices = []
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(filename)
    h_indep = df.columns[:-1]
    h_dep = df.columns[-1]
    for _ in xrange(10):
            # print "- ",
            sys.stdout.flush()
            msk = np.random.rand(len(df)) < 0.5
            train_data = df[msk]
            test_data = df[msk]

            assert (len(train_data) == len(test_data)), "Something is wrong"
            train_indep = train_data[h_indep]
            train_dep = train_data[h_dep]

            test_indep = test_data[h_indep]
            test_dep = test_data[h_dep]
            rf = RandomForestClassifier(n_estimators=10)
            rf.fit(train_indep, [i for i in train_dep.values.tolist()])
            prediction = rf.predict(test_indep)
            from sklearn.metrics import confusion_matrix
            # print confusion_matrix(test_dep, prediction)

            if len(set(test_dep)) > 2:
                confusion_matrices.append([precision_score(test_dep, prediction,average='macro'),
                                       recall_score(test_dep, prediction, average='macro')])
            else:
                confusion_matrices.append([precision_score(test_dep, prediction),
                                           recall_score(test_dep, prediction)])
            # print len(confusion_matrices),

    precisions = [x[0] for x in confusion_matrices]
    recalls = [x[1] for x in confusion_matrices]

    extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    import pickle
    pickle.dump(confusion_matrices, open("./Results_RF_Classification/CM_" + extract_name, "wb"))
    print round(np.mean(precisions), 3), round(np.mean(recalls), 3), round(time.time() - start_time, 3), "sec"



def _test_random_forest_classification():
    """ Storing the confusion Matrix for classification Tasks"""
    folder_name = "."
    from os import listdir
    files = sorted([file for file in listdir(folder_name) if "csv" in file])
    for file in files:
        print file,
        test_random_forest_classification(file)


def test_kmeans(filename):
    from sklearn.cluster import KMeans
    stripped_down = filename.split("-")[-1].strip()
    contents = stripped_down.split("_")
    no_points = int(contents[1])
    no_features = int(contents[3])
    no_clusters = int(contents[5].split(".")[0])
    confusion_matrices = []
    start_time = time.time()

    df = pd.read_csv(filename)
    h_indep = [d for d in df.columns if "features" in d]
    h_dep = [d for d in df.columns if "class" in d]
    for _ in xrange(10):
        try:
            print "- ",
            sys.stdout.flush()

            indep = df[h_indep]
            dep = df[h_dep]

            kmeans = KMeans(n_clusters =no_clusters)
            kmeans.fit(indep)
            prediction = kmeans.labels_
            from sklearn.metrics import confusion_matrix
            confusion_matrices.append(confusion_matrix(dep, prediction))
        except:
            import traceback
            traceback.extract_stack()

    extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    import pickle
    pickle.dump(confusion_matrices, open("./Results_K_Means/Kmeans_" + extract_name, "wb"))
    print " Total Time: ", time.time() - start_time


def _test_kmeans():
    folder_name = "."
    from os import listdir
    files = sorted([folder_name + file for file in listdir(folder_name) if ".csv" in file])
    for file in files:
        print file, " : ",
        test_kmeans(file)

if __name__ == "__main__":
    _test_random_forest_classification()