import numpy as np

x_0 = np.array([[3,3],[4,3],[1,1]])
x = np.array([[3,4,1],[3,3,1]])
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

L = f + lam1*st1 + lam2*st2 + lam3*st3

lam_st_kkts = [
    lam1 * st1,
    lam2 * st2,
    lam3 * st3,
]

lam_kkts = [
    sp.Ge(lam1,0),
    sp.Ge(lam2,0),
    sp.Ge(lam3,0),
]

st_kkts = [
    sp.Le(st1,0),
    sp.Le(st2,0),
    sp.Le(st3,0),
]

d_w1 = sp.diff(L,w1)
d_w2 = sp.diff(L,w2)
d_b = sp.diff(L,B)

solve = sp.solve([d_w1,d_w2,d_b,*lam_st_kkts],w1,w2,B,lam1,lam2,lam3)

min_solve = None
min_subs = 1e20
for i in solve:
    II = {
        'w1':i[0],
        'w2':i[1],
        'B':i[2],
        'lam1':i[3],
        'lam2': i[4],
        'lam3': i[5],
    }

    if not lam_kkts[0].subs(II):
        continue
    if not lam_kkts[1].subs(II):
        continue
    if not lam_kkts[2].subs(II):
        continue

    if st_kkts[0].subs(II) != True:
        continue
    if st_kkts[1].subs(II) != True:
        continue
    if st_kkts[2].subs(II) != True:
        continue

    subs = f.subs(II)
    if min_subs > subs:
        min_subs = subs
        min_solve = II

# print(min_solve)

x_1 = sp.symbols('x_1')
mid = (-(w1*x_1+B)/w2).subs(min_solve)
left = ((1-(w1*x_1+B))/w2).subs(min_solve)
right = ((-1-(w1*x_1+B))/w2).subs(min_solve)
f = sp.lambdify(x_1,mid,"numpy")
f_l = sp.lambdify(x_1,left,"numpy")
f_r = sp.lambdify(x_1,right,"numpy")

a = np.linspace(-5,5,1000)
result = f(a)
result_l = f_l(a)
result_r = f_r(a)

import matplotlib.pyplot as plt


plt.scatter(*x,c=y)
plt.plot(a,result)
plt.plot(a,result_l)
plt.plot(a,result_r)
plt.show()