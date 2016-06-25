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
    h_indep = df.columns[:-1]
    h_dep = df.columns[-1]
    for _ in xrange(10):
        # print "- ",
        sys.stdout.flush()
        msk = np.random.rand(len(df)) < 0.4
        train_data = df[msk]
        test_data = df[~msk]

        assert(len(train_data) + len(test_data) == len(df)), "Something is wrong"

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
    # print
    print round(np.mean(rss), 3), round(time.time() - start_time, 3)


def _test_linear_regression():
    folder_name = "."
    from os import listdir
    files = sorted([file for file in listdir(folder_name) if ".csv" in file])
    for file in files:
        print file,
        test_linear_regression(file)


def test_random_forest_regression(filename):
    start_time = time.time()
    scores = []
    from sklearn.ensemble import RandomForestRegressor
    df = pd.read_csv(filename)
    h_indep = df.columns[:-1]
    h_dep = df.columns[-1]
    for _ in xrange(10):
            # print "- ",
            sys.stdout.flush()
            msk = np.random.rand(len(df)) < 0.4
            train_data = df[msk]
            test_data = df[~msk]

            assert (len(train_data) + len(test_data) == len(df)), "Something is wrong"
            train_indep = train_data[h_indep]
            train_dep = train_data[h_dep]

            test_indep = test_data[h_indep]
            test_dep = test_data[h_dep]
            rf = RandomForestRegressor(n_estimators=10)
            rf.fit(train_indep, [i for i in train_dep.values.tolist()])
            prediction = rf.predict(test_indep)
            from sklearn.metrics import mean_absolute_error
            # print confusion_matrix(test_dep, prediction)

            scores.append(mean_absolute_error(test_dep, prediction))
            # print len(confusion_matrices),

    # extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    # import pickle
    # pickle.dump(confusion_matrices, open("./Results_RF_Classification/CM_" + extract_name, "wb"))
    print round(np.mean(scores), 3), round(time.time() - start_time, 3), "sec"


def _test_random_forest_regression():
    """ Storing the confusion Matrix for classification Tasks"""
    folder_name = "."
    from os import listdir
    files = sorted([file for file in listdir(folder_name) if "csv" in file])
    for file in files:
        print file,
        test_random_forest_regression(file)


def test_decision_tree_regression(filename):
    start_time = time.time()
    scores = []
    from sklearn.tree import DecisionTreeRegressor
    df = pd.read_csv(filename)
    h_indep = df.columns[:-1]
    h_dep = df.columns[-1]
    for _ in xrange(10):
            # print "- ",
            sys.stdout.flush()
            msk = np.random.rand(len(df)) < 0.4
            train_data = df[msk]
            test_data = df[~msk]

            # print len(train_data), len(test_data)
            assert (len(train_data) + len(test_data) == len(df)), "Something is wrong"
            train_indep = train_data[h_indep]
            train_dep = train_data[h_dep]

            test_indep = test_data[h_indep]
            test_dep = test_data[h_dep]
            dt = DecisionTreeRegressor()
            dt.fit(train_indep, [i for i in train_dep.values.tolist()])
            prediction = dt.predict(test_indep)
            from sklearn.metrics import mean_absolute_error

            scores.append(mean_absolute_error(test_dep, prediction))
            # print len(confusion_matrices),

    extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    # import pickle
    # pickle.dump(confusion_matrices, open("./Results_RF_Classification/CM_" + extract_name, "wb"))
    print round(np.mean(scores), 3), round(time.time() - start_time, 3), "sec"


def _test_decision_tree_regression():
    """ Storing the confusion Matrix for classification Tasks"""
    folder_name = "."
    from os import listdir
    files = sorted([file for file in listdir(folder_name) if "csv" in file])
    for file in files:
        print file,
        test_decision_tree_regression(file)


def test_logistic_regression(filename):
    start_time = time.time()
    coeffs = []
    acc = []
    confusion_matrices =[]
    from sklearn import datasets, linear_model
    df = pd.read_csv(filename)
    h_indep = df.columns[:-1]
    h_dep = df.columns[-1]
    for _ in xrange(10):
        # print "- ",
        sys.stdout.flush()
        msk = np.random.rand(len(df)) < 0.4
        train_data = df[msk]
        test_data = df[~msk]

        assert (len(train_data)+len(test_data) == len(df)), "Something is wrong"

        train_indep = train_data[h_indep]
        train_dep = train_data[h_dep]

        test_indep = test_data[h_indep]
        test_dep = test_data[h_dep]
        logistic = linear_model.LogisticRegression()
        logistic.fit(train_indep, [i for i in train_dep.values.tolist()])
        coeffs.append(logistic.coef_)
        prediction = logistic.predict(test_indep)

        if len(set(test_dep)) > 2:
            confusion_matrices.append([precision_score(test_dep, prediction, average='macro'),
                                       recall_score(test_dep, prediction, average='macro')])
        else:
            confusion_matrices.append([precision_score(test_dep, prediction),
                                       recall_score(test_dep, prediction)])

    extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    precisions = [x[0] for x in confusion_matrices]
    recalls = [x[1] for x in confusion_matrices]
    # import pickle
    # pickle.dump(coeffs, open("./Results_Logistic_Regression/coeffs_" + extract_name, "wb"))
    # pickle.dump(acc, open("./Results_Logistic_Regression/rss_" + extract_name, "wb"))
    # print
    print round(np.mean(precisions), 3), round(np.mean(recalls), 3), round(time.time() - start_time, 3), "sec"


def _test_logistic_regression():
    folder_name = "."
    from os import listdir
    files = sorted([file for file in listdir(folder_name) if "csv" in file])
    for file in files:
        print file,
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
            msk = np.random.rand(len(df)) < 0.4
            train_data = df[msk]
            test_data = df[~msk]

            assert (len(train_data) + len(test_data) == len(df)), "Something is wrong"
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
    # import pickle
    # pickle.dump(confusion_matrices, open("./Results_RF_Classification/CM_" + extract_name, "wb"))
    print round(np.mean(precisions), 3), round(np.mean(recalls), 3), round(time.time() - start_time, 3), "sec"


def _test_random_forest_classification():
    """ Storing the confusion Matrix for classification Tasks"""
    folder_name = "."
    from os import listdir
    files = sorted([file for file in listdir(folder_name) if "csv" in file])
    for file in files:
        print file,
        test_random_forest_classification(file)


def test_decision_tree_classification(filename):
    start_time = time.time()
    accuracy = []
    from sklearn.tree import DecisionTreeClassifier
    df = pd.read_csv(filename)
    h_indep = df.columns[:-1]
    h_dep = df.columns[-1]
    for _ in xrange(10):
            # print "- ",
            sys.stdout.flush()
            msk = np.random.rand(len(df)) < 0.4
            train_data = df[msk]
            test_data = df[~msk]

            # print len(train_data), len(test_data)
            assert (len(train_data) + len(test_data) == len(df)), "Something is wrong"
            train_indep = train_data[h_indep]
            train_dep = train_data[h_dep]

            test_indep = test_data[h_indep]
            test_dep = test_data[h_dep]
            dt = DecisionTreeClassifier()
            dt.fit(train_indep, [i for i in train_dep.values.tolist()])
            prediction = dt.predict(test_indep)
            from sklearn.metrics import accuracy_score
            # print confusion_matrix(test_dep, prediction)

            if len(set(test_dep)) > 2:
                accuracy.append(accuracy_score(test_dep, prediction))
            else:
                accuracy.append(accuracy_score(test_dep, prediction))
            # print len(confusion_matrices),



    extract_name = filename.split("/")[-1].split(".")[0] + ".p"
    # import pickle
    # pickle.dump(confusion_matrices, open("./Results_RF_Classification/CM_" + extract_name, "wb"))
    print round(np.mean(accuracy), 3)


def _test_decision_tree_classification():
    """ Storing the confusion Matrix for classification Tasks"""
    folder_name = "."
    from os import listdir
    files = sorted([file for file in listdir(folder_name) if "csv" in file])
    for file in files:
        print file,
        test_decision_tree_classification(file)


if __name__ == "__main__":
    # _test_random_forest_classification()
    # _test_logistic_regression()
    _test_decision_tree_classification()
    # print "LR "
    # _test_linear_regression()
    # print "RF "
    # _test_random_forest_regression()
    # print "DT "
    # _test_decision_tree_regression()