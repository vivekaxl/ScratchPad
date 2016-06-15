from sklearn import datasets


data = datasets.load_boston()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))

data = datasets.load_breast_cancer()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))

data = datasets.load_diabetes()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))

data = datasets.load_digits()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))



data = datasets.load_iris()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))


# data = datasets.load_linnerud()
# import pdb
# pdb.set_trace()
# print "Features: ", len(data["data"][0])
# print "Instances: ", len(data["data"])
# print len(set(data["target"]))

data = datasets.load_mlcomp()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
import pdb
pdb.set_trace()

data = datasets.load_sample_image()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))

data = datasets.load_sample_images()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))

data = datasets.load_svmlight_file()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))

data = datasets.load_svmlight_files()
print "Features: ", len(data["data"][0])
print "Instances: ", len(data["data"])
print len(set(data["target"]))