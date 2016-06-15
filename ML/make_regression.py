from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

X = datasets.make_regression(n_samples=100, n_features=1, coef=True, noise=25.0, bias=100)
header = ["A"+str(i) for i in xrange(len(X[0][0]))] + ["Dep"]
content = []
for independent, dependent in zip(X[0], X[1]):
    content.append(independent.tolist() + [dependent])

df = pd.DataFrame(content, columns=header)
df.to_csv("temp.csv", index=False)


coeffs = []
for _ in xrange(10):
    from sklearn import linear_model
    regressor = linear_model.LinearRegression()
    regressor.fit(X, y)

    coeffs.append(regressor.coef_)
from numpy import mean, std
print mean(coeffs), std(coeffs)
plt.scatter(X, y)
plt.show()
