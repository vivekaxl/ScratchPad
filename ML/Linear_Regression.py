import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd

df = pd.read_csv("./Data/data1.csv")
h_indep = [d for d in df.columns if "A" in d]
h_dep = [d for d in df.columns if "A" not in d]
data_indep = df[h_indep].values.tolist()
data_dep = df[h_dep].values.tolist()
regressor = linear_model.LinearRegression()
regressor.fit(data_indep, data_dep)
print list(regressor.coef_[-1])
