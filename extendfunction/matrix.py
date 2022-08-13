import sympy as sp

def matrixDiff(goalFunction,vector):
    if isinstance(vector,sp.matrices.dense.MutableDenseMatrix) is False:
        raise Exception
    if isinstance(goalFunction,sp.matrices.dense.MutableDenseMatrix) is False:
        raise Exception
    if vector.shape[1]!=1 or goalFunction.shape[1]!=1:
        raise Exception

    result = list()
    for xi in vector:
        result.append([sp.diff(yi,xi) for yi in goalFunction])
    return sp.Matrix(result)

def sumMatrix(matrix,mode = True):
    if isinstance(matrix, sp.matrices.dense.MutableDenseMatrix) is False:
        raise Exception
    s = 0
    for i in matrix:
        s += i
    if mode is True:
        return sp.Matrix([s])
    else:
        return s

def to_matrix(core):
    return sp.Matrix([core])

def to_expr(matrix):
    if isinstance(matrix,sp.matrices.dense.MutableDenseMatrix) is False:
        raise Exception
    if matrix.shape == (1,1):
        return matrix[0]
    else:
        return list(matrix)

def to_numpy(matrix):
    import numpy as np
    expr = to_expr(matrix)
    try:
        expr = [float(i) for i in expr]
    except Exception:
        raise Exception
    return np.array(expr).reshape(matrix.shape)

def numpy_to_matrix(array):
    import numpy as np
    if isinstance(array,np.ndarray) is False:
        raise Exception



if __name__ == '__main__':
    import numpy as np
    x = np.array([1])
    print(type(x))

    s = sp.Matrix(x)
    print(s)


