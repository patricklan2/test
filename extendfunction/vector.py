import sympy as sp

def creatVector(num,name,mode = True):
    if isinstance(num,int) is False:
        raise Exception
    if isinstance(name,str) is False:
        raise Exception
    if mode is True:
        p = 1
    else:
        p = 0
    namelist = list()
    for i in range(num):
        namelist.append("{0}_{1}".format(name,i+p))
    get = list(sp.symbols(' '.join(namelist)))
    matrix = sp.Matrix(get)
    return get,matrix




if __name__ == '__main__':
    print("str")
    print(type('x'))
    print("{0}".format(123))
    x = creatVector(5,'x')
    print(x)