import pandas as pd
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from extendfunction import FunDraw

# 加载数据
data = pd.read_csv('../watermelon4_0.csv')
data = data.iloc[:,1:3].values
M,N = data.shape

# model数据初始化
k=3
averages = [data[5],data[21],data[26]]
C_matrices = [np.array([[0.1,0],[0,0.1]]) for _ in range(3)]
alphas = np.array([1/3,1/3,1/3])

# 计算
def get_x_z(_x,_k):
    ave,cM = averages[_k],C_matrices[_k]
    _k = 1 / (2 * np.pi * np.linalg.det(cM) ** (1 / 2))
    _s = np.dot(_x - ave, np.linalg.inv(cM)).dot(_x     - ave)
    return _k * np.exp(-_s / 2)

def get_xz(_x,_k):
    return alphas[_k]*get_x_z(_x,_k)

def get_x(_x):
    return np.array([get_xz(_x,_k) for _k in range(k)]).sum()

def get_z_x(_x,_k):
    return get_xz(_x,_k)/get_x(_x)

R = np.array([[get_z_x(x, _k) for x in data] for _k in range(k)])


# 更新参数
def update(_k):
    SRj = R[_k].sum()
    new_alpha = SRj / 30
    new_average = R[_k].dot(data) / SRj
    fun = lambda x: np.dot((x - new_average).reshape(2, 1), (x - new_average).reshape(1, 2))
    new_C_matrix = np.zeros(4).reshape(2, 2)
    for i in range(30):
        new_C_matrix += fun(data[i]) * R[_k][i]
    new_C_matrix /= SRj
    return new_alpha,new_average,new_C_matrix

# train
times = 50
for _ in range(times):
    for i in range(k):
        Update = update(i)
        alphas[i],averages[i],C_matrices[i], = Update[0],Update[1],Update[2]
    R = np.array([[get_z_x(x, _k) for x in data] for _k in range(k)])
    print(averages)

# model
def predict(xx):
    p_list = np.array([get_x_z(xx,_k) for _k in range(k)])
    return p_list.argmax()

# 预测
col = np.array(['g','r','b'])
pre = [predict(i) for i in data]
out = col[pre]

# 画画
fig = plt.figure()
ax1,ax2,ax3 = [fig.add_subplot(i,projection = '3d')for i in [221,222,223]]
ax4 = fig.add_subplot(224)
x ,y= np.linspace(0.1,0.9,40),np.linspace(-0.1,0.6,40)
for _k,ax in enumerate([ax1,ax2,ax3]):
    ax.set_xlabel('dense')
    ax.set_ylabel('sugar content')
    Z = np.array([[get_x_z(np.array([_x,_y]),_k) for _x in x]for _y in y])
    ax.plot_surface(*np.meshgrid(x,y), Z, cmap=cm.coolwarm)
    ax.scatter(averages[_k][0], averages[_k][1], 2)
ax4.scatter(data[:,0],data[:,1],c = out)
ax4.scatter(np.array(averages)[:,0],np.array(averages)[:,1],marker = '+')
Z = col[np.array([[predict(np.array([_x,_y])) for _x in x]for _y in y])]
print(Z.shape)
ax4.scatter(*np.meshgrid(x,y),c = Z.reshape(40*40),s=0.3)

plt.show()



















