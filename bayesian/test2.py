import numpy as np
from nbmodel import load_model
import pandas as pd

path = 'iris_model.npz'
path2 = 'water.npz'

# data = np.load(path)
# c = data['c']
# C = len(c)
# out = {i: list() for i in c}
# for k in range(len(data.files) - 2):
#     get = data['type_{0}'.format(k)]
#     print(get)
#     p = get[1:C + 1, :].astype(np.float64)
#     for i in range(C):
#         gro = {k: v for k, v in zip(typeName, p[i])}
#         out[c[i]].append(gro)
# p_c = {k:v for k,v in zip(c,data['p_c'])}


m = load_model(path)

PATH = '../Iris数据集/iris.csv'
data = pd.read_csv(PATH)
XX = data.to_numpy()[:, 1: 5]
YY = data.to_numpy()[:, 5]

# PATH = "../watermelon3_0.csv"
# data = pd.read_csv(PATH)
# XX = data.to_numpy()[:, 1: 9]
# YY = data.to_numpy()[:, 9]

result = np.array([m.predict(i) for i in XX])
# print(result)
acc = (result==YY)
print(result)
print(acc)
print(acc.mean())
print(acc.sum())


