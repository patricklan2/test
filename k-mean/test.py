import pandas as pd
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from extendfunction import FunDraw

data = pd.read_csv('../watermelon4_0.csv')
data = data.iloc[:,1:3].values
M,N = data.shape

k = 3
col = ['r','g','b']
Y = np.array([col[0] for _ in range(M)])

def drawdot():
    plt.scatter(data[:, 0], data[:, 1], c=Y)
    plt.xlabel("dense")
    plt.ylabel("sugar content")

def drawmu():
    plt.scatter(mu[:, 0], mu[:, 1],marker='+')

def drawline():
    line(mu[0], mu[1])
    line(mu[0], mu[2])
    line(mu[2], mu[1])


def dist(x1,x2):
    return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)

def grope_clear():
    for C in grope:
        C.clear()

def mean(x):
    return np.mean(x,axis=0)

def line(a,b):
    w = -(b[0]-a[0])/(b[1]-a[1])
    xm,ym = (a[0]+b[0])/2,(a[1]+b[1])/2
    y = lambda x:w*(x-xm)+ym
    FunDraw(y,0.2,0.8).draw()

grope = [list() for _ in range(k)]
mu = data[[3,4,5]]

for j in range(10):
    plt.grid()
    grope_clear()
    for i in range(M):
        minIndex = argmin([dist(data[i],x) for x in mu])
        Y[i] = col[minIndex]
        grope[minIndex].append(i)
    for i in range(k):
        mu[i] = mean(data[grope[i]])
    drawdot()
    drawmu()
    drawline()
    plt.show()




