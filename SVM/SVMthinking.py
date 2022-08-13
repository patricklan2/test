import numpy as np
import sympy as sp
import extendfunction as ex

X = np.array([[3,3],[4,3],[1,1]])
Y = np.array([1,1,-1])

x = sp.Matrix(X)
y = sp.Matrix(Y)

N,M = x.shape
ws,w = ex.creatVector(M,'w')
b = sp.symbols('b')
lams,lam = ex.creatVector(N,'lam')
w_b = w.row_insert(M,sp.Matrix([b]))

tip = sp.diag(*y)*x*x.T*sp.diag(*y)

goalfun = ex.sumMatrix(lam) - lam.T*tip*lam/2
print(goalfun)







import matplotlib.pyplot as plt
plt.grid(visible=True)
ex.ScatterDraw(X,Y).draw()
ex.LineDraw(0.5,0.5,-2,-5,5).draw()
ex.LineDraw(0.5,0.5,-2+1,-5,5).draw()
ex.LineDraw(0.5,0.5,-2-1,-5,5).draw()
# plt.show()