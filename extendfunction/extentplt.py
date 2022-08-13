import matplotlib.pyplot as plt
import numpy as np

class FunDraw(object):
    x = np.linspace(0,0,0)
    y = np.array([])
    def __init__(self,fun,a,b,sep=1000):
        self.x = np.linspace(a,b,sep)
        for i in self.x:
            self.y = np.append(self.y,fun(i))

    def draw(self):
        plt.plot(self.x,self.y)


class LineDraw(object):
    def __init__(self,w1,w2,b,start,end,sep=1000):
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.f = lambda x:(-w1*x-b)/w2
        self.d = FunDraw(self.f,start,end,sep)

    def draw(self):
        self.d.draw()

class ScatterDraw(object):
    def __init__(self,data,result):
        self.x = np.array([data[i][0] for i in range(len(data))])
        self.y = np.array([data[i][1] for i in range(len(data))])
        self.result = result

    def draw(self):
        plt.scatter(self.x,self.y, c=self.result)
