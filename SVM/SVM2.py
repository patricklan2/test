import numpy as np

x = np.array([[3,3],[4,3],[1,1]])
x_0 = np.array([[3,4,1],[3,3,1]])
y = np.array([1,1,-1])

N,M = x.shape
w = np.ones(M)
b = 0.0

import sympy as sp

w1,w2,B,lam1,lam2,lam3 = sp.symbols('w1 w2 B lam1 lam2 lam3')
f = (w1**2+w2**2)/2

st1 = 1 - (3*w1 + 3*w2 + B)
st2 = 1 - (4*w1 + 3*w2 + B)
st3 = 1 - (-1*w1 + -1*w2 + -B)


sigma = 0
lams = [lam1,lam2,lam3]
for i in range(N):
    for j in range(N):
        sigma += lams[i] * lams[j] * y[i] * y[j] * x[i].dot(x[j])

print(sigma)

f = sigma/2 - lam1 - lam2 - lam3

st = y[0]*lam1 + y[1]*lam2 + y[2]*lam3

f = f.subs(lam3,lam1+lam2)

d_lam1 = sp.diff(f,lam1)
d_lam2 = sp.diff(f,lam2)

# print(d_lam1)
# print(d_lam2)

solve = sp.solve([d_lam1,d_lam2],lam1,lam2)

solve1 = sp.solve([sp.diff(f.subs(lam1,0),lam2)],lam2)
solve1[lam1] = 0
min_subs = f.subs(solve1)
solve = solve1

solve2 = sp.solve([sp.diff(f.subs(lam2,0),lam1)],lam1)
solve2[lam2] = 0
if f.subs(solve2) < min_subs:
    solve = solve2

result = np.array([
    float(solve[lam1]),
    float(solve[lam2]),
    float(solve[lam1]+solve[lam2])
])
print(result)


w = np.dot(result*y,x)
print(w)
b = y[0] - np.dot(result*y,np.dot(x,x[0]))
print(b)
print(result*y,np.dot(x,x[0]))
print(x,x[0])