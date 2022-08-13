import pandas as pd
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

mu = np.array([0,0])
std = np.array([[0.1,0],[0,0.1]])
def gauss(_x,_y):
    _o = np.array([_x,_y])
    _k = 1/(2*np.pi*np.linalg.det(std)**(1/2))
    _s = np.dot(_o-mu,std).dot(_o-mu)
    return _k*np.exp(-_s/2)

def cal(function,_X,_Y):
    return [[function(_x,_y) for _y in _Y]for _x in _X]


plt.plot(np.linspace(0,1,100),np.linspace(1,2,100))
plt.show()

