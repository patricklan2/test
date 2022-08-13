import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from model import *

data = pd.read_csv('../watermelon3_0.csv')

x_data = data.to_numpy()[:,7:9]
y_data = (data.to_numpy()[:,9]=='是')*2-1
M,N = x_data.shape
D_data = np.array([1/M]*M)
H_list =Stumps()


def drawdot():
    plt.scatter(x_data[:, 0], x_data[:, 1],
                c=['g' if i == 1 else 'r' for i in y_data],
                s=D_data * 500)

def drawline(attribute,value,**kwargs):
    if attribute == 1:
        plt.plot(range(0, 2), [value for _ in range(0, 2)])
    else:
        plt.plot([value for _ in range(0, 2)], range(0, 2))

def draw(**kwargs):
    drawdot()
    drawline(**kwargs)

def get_h(X,Y,D):
    minvalue = (0,0,0,0)
    m = len(Y)
    for n in range(2):
        sort = np.argsort(X[:, n]),
        x, d, y = X[sort], D[sort], Y[sort]
        judge_list = [(x[i-1][n]+x[i][n])/2 for i in range(1,m)]
        p_1,p_0 = (y==1)*d,(y==-1)*d
        for i in range(1,m):
            acc = p_1[0:i].sum()+p_0[i:m].sum()
            if acc<0.5:
                if 1-acc>minvalue[1]:
                    minvalue = (n,1-acc,1,judge_list[i-1])
            else:
                if acc>minvalue[1]:
                    minvalue = (n,acc,-1,judge_list[i-1])
    return {
        'attribute':minvalue[0],
        'stump':DecisionStump(minvalue[0],minvalue[3],minvalue[2]),
        'acc':minvalue[1],
        'value':minvalue[3]
    }

def get_alpha(acc,stump,**kwargs):
    return {
        'errorRate': 1 - acc,# 错误的
        'stump':stump,
        'alpha' : np.log(acc / (1 - acc)) / 2,
        'Z' : 2 * np.sqrt(acc * (1 - acc))
    }
#
def update_D(alpha,stump,Z,**kwargs):
    for i in range(M):
        D_data[i] = D_data[i] * np.exp(-alpha * y_data[i] * stump(x_data[i])) / Z

##1
for _ in range(11):
    choose = get_h(x_data,y_data,D_data)
    get = get_alpha(**choose)
    # drawline(**choose)

    H_list.append(get['stump'],get['alpha'])
    update_D(**get)
    # print(choose)
    # print(get)
    # print(([H_list(x) for x in x_data]==y_data).mean())


drawdot()

x = np.linspace(0.15,0.9,100)
y = np.linspace(0,0.5,100)

# print(x_data[0])
for i in x:
    col = ['b' if H_list(np.array([i, j])) == 1 else 'y' for j in y]
    plt.scatter([i for _ in y],y,c=col , s= 0.1)


plt.show()
