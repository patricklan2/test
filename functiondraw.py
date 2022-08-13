

def hinge(x):
    return max(0,1-x)

def exponential_loss(x):
    return np.exp(-x)

def logistic_loss(x):
    return np.log2(1+np.exp(-x))


A = FunDraw(hinge,-2,5)
A.draw()
B = FunDraw(exponential_loss,-2,5)
B.draw()
C = FunDraw(logistic_loss,-2,5)
C.draw()
plt.show()