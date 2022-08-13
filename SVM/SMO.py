# 加载数据
import numpy as np
PATH = "../MNIST.npz"
mnist = np.load(PATH)
images,labels = mnist["train_images"],mnist['train_label']
images = images.reshape(-1,28*28)/255
labels = (labels==0)*2-1
X,Y = images[:100],labels[:100]

# 超参数和变量
N,M = X.shape
C = 200
alpha = np.ones(N)
b = 0.0

# K函数，高斯核函数
#l2又可以写为l2 = ((X[i]-X[j])**2).sum()**0.5
def K(i,j):
  l2 = np.linalg.norm(X[i]-X[j],ord=2)
  return np.exp(-l2/200)

# 预测函数
# g(x) = sigma i=0->N[alpha_i * y_i * K(x_i,x)] + b
def g(i):
    _sum = 0
    for j in range(N):
        _sum+=alpha[j]*Y[j]*K(i,j)
    _sum += b
    return _sum

# E(x) = g(x) - y
def E(i):
    return g(i)-Y[i]

# KKT条件：
# if alpha == 0 than y * g(x) >= 1
# if alpha == C than y * g(x) <= 1
# if among than y * g(x) == 1
def kkt(i):
    if alpha[i] == 0:
        return Y[i] * g(i) >= 1
    if alpha[i] == C:
        return Y[i] * g(i) <= 1
    if 0 < alpha[i] < C:
        return Y[i] * g(i) == 1
    return False

def get_ttk_different(i):
    if kkt(i):
        return 0
    if alpha[i] == 0:
        return 1 - Y[i] * g(i)
    if alpha[i] == C:
        return Y[i] * g(i) - 1
    if 0< alpha[i] < C:
        return np.abs(1 - Y[i] * g(i))
    return 100

# 找一对 alpha
def pick_idx():
    max_kkt_different = 0
    _i = -1
    for i in range(N):
        if kkt(i):
            continue
        kkt_different = get_ttk_different(i)
        if max_kkt_different >= kkt_different:
            continue
        max_kkt_different = kkt_different
        _i = i

    max_E_different = 0
    _j = -1
    for i in range(N):
        E_different = np.abs(E(_i) - E(i))
        if max_E_different >= E_different:
            continue
        max_E_different = E_different
        _j = i
    return _i,_j

# if y[i]!=y[j] than L = max(0,alpha[j]-alpha[i]) H = min(C,C+alpha[j]-alpha[i])
# if y[i]==y[j] than L = max(0,alpha[j]+alpha[i]-C) H = min(C,alpha[j]+alpha[i])
def get_L_and_H(i,j):
    if Y[i]!=Y[j]:
        L = max(0,alpha[j] - alpha[i])
        H = min(C,C + alpha[j] - alpha[i])
    else:
        L = max(0,alpha[j] + alpha[i] - C)
        H = min(C,alpha[j] + alpha[i])
    return L,H

# 计算b
def get_b(i,j,alpha_new_i,alpha_new_j):
    b1 = 0 - E(i) - Y[i] * K(i,i) * (alpha_new_i - alpha[i]) - Y[j] * K(j,i) * (alpha_new_j - alpha[j]) + b
    b2 = 0 - E(j) - Y[i] * K(i,j) * (alpha_new_i - alpha[i]) - Y[j] * K(j,j) * (alpha_new_j - alpha[j]) + b

    if 0 < alpha_new_i < C and 0 < alpha_new_j < C:
        return b1
    else:
        return (b1+b2)/2

# 计算啊alpha
def get_new_alpha(i,j,L,H):
    alpha_new_j = alpha[j] + Y[j] * (E(i) - E(j)) / (K(i,i) + K(j,j) - 2 * K(i,j))
    if alpha_new_j > H:
        alpha_new_j = H
    if alpha_new_j < L:
        alpha_new_j = L

    alpha_new_i = alpha[i] + Y[i] * Y[j] * (alpha[j] - alpha_new_j)
    return alpha_new_i,alpha_new_j

# train
def train(times = 20):
    global alpha
    global b

    iterStep = 0
    while iterStep < times :
        iterStep +=1

        i,j = pick_idx()
        L,H = get_L_and_H(i,j)
        alpha_new_i,alpha_new_j = get_new_alpha(i,j,L,H)
        b = get_b(i,j,alpha_new_i,alpha_new_j)

        alpha[i] = alpha_new_j
        alpha[j] = alpha_new_j

def packet(path):
    alpha_y = alpha*Y
    np.savez(path, alpha_y=alpha_y,x=X,b=b)

train()
packet('./SMOmnist.npz')

# result = np.array([g(i) for i in range(100)])
# result = (result>0)*2-1
# print(result)
# acc = (result==Y)
# print(acc)
# print(acc.mean())