import numpy as np
from collections import Counter
from functools import reduce


class NB_model(object):
    def __init__(self,feature_space,P_c=None,P_xi_c=None):
        self.feature_space = feature_space
        self._feature_num = len(feature_space)
        self.P_c = P_c
        self.P_xi_c = P_xi_c

    def fix(self,train,label,Laplacian_correction=True):
        self.P_c = self.get_p_c(label,Laplacian_correction)
        self.P_xi_c = self.get_p_xi_c(train,label,Laplacian_correction)  # p(xi|c)  key: Tuple[str, str] (label, single feature) value : float

    def get_p_c(self, label,_correction):
        result = {}
        label_c = Counter(label)
        C = label_c.keys()

        for k, v in label_c.items():
            if _correction:
                result[k] = (v + 1) / (len(label) + len(C))  # calculate all the p(c)
            else:
                result[k] = v / len(label)
        return result

    def get_p_xi_c(self, X, Y,_correction):
        N = self._feature_num
        label_c = Counter(Y)
        C = label_c.keys()
        Y_c = {i: list() for i in C}
        for i in range(len(Y)):
            Y_c[Y[i]].append(i)

        result = {i: list() for i in C}
        for k,v in label_c.items():  # process by feature
            for i in range(N):
                x = Counter([X[j, i] for j in range(len(Y))])
                x_i = Counter([X[j, i] for j in Y_c[k]])  # collect data whose label is current k
                if self.feature_space[i] == 0:  # discrete feature
                    if _correction:
                        get = {dk : (dv + 1) / (v + len(x_i)) for dk, dv in x.items()}
                    else:
                        get = {dk : dv / v for dk, dv in x_i.items()}
                else:  # continuous feature
                    get = {
                        "miu": np.mean([X[j, i] for j in Y_c[k]]),
                        "sigma": np.std([X[j, i] for j in Y_c[k]])
                    }
                result[k].append(get)
        return result

    def predict(self, X):
        result = np.argmax([self.p_x_c(label,X) for label in self.P_c])
        return list(self.P_c.keys())[result]

    def p_x_c(self,c,x):
        p_xi_c_list = [self.p_xi_c(i, c, x) for i in range(self._feature_num)]
        return self.P_c[c] * (reduce(lambda a, b: a * b, p_xi_c_list))

    def p_xi_c(self, index, c, X):
        attribute = X[index]
        if self.feature_space[index] == 0:
            return self.P_xi_c[c][index][attribute]
        else:
            miu,sigma, x= self.P_xi_c[c][index]["miu"], self.P_xi_c[c][index]["sigma"],attribute
            return np.exp(-1. * ((x - miu) * (x - miu)) / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma)

    def save(self,path):
        N,M = self._feature_num,len(self.P_c)
        c = np.array(list(self.P_c.keys()))
        p_c = np.array(list(self.P_c.values()))
        value = list(self.P_xi_c.values())
        out = {}
        for j in range(N):
            get2 = [value[i][j] for i in range(M)]
            out['type_{0}'.format(j)] = np.array([list(get2[0].keys()), *[list(i.values()) for i in get2]])
        np.savez(path,c = c,p_c = p_c,**out)

    def imformation(self):
        for i,j in self.P_xi_c.items():
            print(i)
            for k in j:
                print(k)


def load_model(path):
    data = np.load(path)
    c = data['c']
    C = len(c)
    out = {i: list() for i in c}
    for k in range(len(data.files) - 2):
        get = data['type_{0}'.format(k)]
        typeName = get[0, :]
        p = get[1:C + 1, :].astype(np.float64)
        for i in range(C):
            gro = {k: v for k, v in zip(typeName, p[i])}
            out[c[i]].append(gro)
    p_c = {k:v for k,v in zip(c,data['p_c'])}
    feature_space = [(len(i)==2 and list(i.keys())==['miu','sigma'])*1 for i in list(out.values())[0]]
    return NB_model(feature_space,p_c,out)