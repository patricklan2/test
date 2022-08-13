import pandas as pd
from nbmodel import NB_model
import numpy as np
from extendfunction import *

PATH = '../Iris数据集/iris.csv'
data = pd.read_csv(PATH)
XX = data.to_numpy()[:, 1: 5]
YY = data.to_numpy()[:, 5]
# test_x = data.to_numpy()[100:150, 1: 5]
# test_y = data.to_numpy()[100:150, 5]


feature_space = [1,1,1,1] # 0 : discrete   1 : continuous
data_info = {
    'Laplacian_correction' : True,
    'train' : XX,
    'label' : YY,
}

model = NB_model(feature_space)
model.fix(**data_info)

# result = np.array([model.predict(i) for i in XX])
# # print(result)
# acc = (result==YY)
# print(result)
# print(acc)
# print(acc.mean())
# print(acc.sum())

# def grassian(miu,sigma,x):
#     return np.exp(-1. * ((x - miu) * (x - miu)) / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma)
#
#
# for k,v in model.P_xi_c.items():
#     get = FunDraw(lambda x:grassian(v[0]['miu'],v[0]['sigma'],x),0,10)
#     get.draw()
#
# plt.show()

# model.save('iris_model.npz')

# print(model.P_xi_c)