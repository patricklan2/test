import numpy as np

class DecisionStump(object):
    def __init__(self,attribute,judgeNum,mode):
        self.mode = mode
        self.judge = judgeNum
        self.attribute = attribute

    def __call__(self, num):
        if num[self.attribute] > self.judge:
            return self.mode
        elif num[self.attribute] < self.judge:
            return -self.mode
        else:
            return 0

class Stumps(object):
    def __init__(self):
        self.stumps = list()
        self.alphas = list()

    def append(self,stump,alpha):
        self.stumps.append(stump)
        self.alphas.append(alpha)

    def __str__(self):
        return self.stumps.__str__() + '\n' + self.alphas.__str__()

    def __call__(self, x):
        hs = np.array([h(x) for h in self.stumps])
        return np.sign(np.dot(hs,self.alphas))



if __name__ == '__main__':
    pass