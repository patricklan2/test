import pandas as pd
from nbmodel import NB_model
import numpy as np

PATH = "../watermelon3_0.csv"
data = pd.read_csv(PATH)
XX = data.to_numpy()[:, 1: 9]
YY = data.to_numpy()[:, 9]

feature_space = [0, 0, 0, 0, 0, 0, 1, 1]# 0 : discrete   1 : continuous
data_info = {

    'Laplacian_correction' : True,
    'train' : XX,
    'label' : YY,
}

test_data = ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460]

model = NB_model(feature_space)
model.fix(**data_info)
model.save('water.npz')

result = np.array([model.predict(i) for i in XX])
acc = (result==YY)
print(result)
print(acc)
print(acc.mean())
print(acc.sum())

# get = model.P_xi_c
# get1 = np.array(list(get.keys()))
# get2 = np.array(list(get.values()))
# # print(np.array(get2))
# print(get2.shape)
# print(get2)