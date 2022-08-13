

'''
def Ent(Y,D):
    p_0,p_1,m = 0,0,len(Y)
    for i in range(m):
        if Y[i]==1:
            p_1+=D[i]
        else:
            p_0+=D[i]
    if p_0 == 0 or p_1 == 0:
        return 0
    return -p_0*np.log2(p_0)-p_1*np.log2(p_1)

def get_h(X,Y,D):
    minvalue,m  = (0,100,-1),len(Y)
    for n in range(2):
        sort = np.argsort(X[:,n])
        x,y,d  = X[sort],Y[sort],D[sort]
        candidates = [(x[i - 1][n] + x[i][n]) / 2 for i in range(1, m)]
        for i in range(1,m):
            p_less, p_more= d[0:i].sum(), d[i:m].sum()
            Ent_less, Ent_more = Ent(y[0:i],d[0:i]/p_less), Ent(y[i:m], d[i:m] / p_more)
            Ent_all = p_less*Ent_less+p_more*Ent_more
            if Ent_all<minvalue[1]:
                minvalue = (n,Ent_all,candidates[i-1])
    stumps =  [DecisionStump(minvalue[0],minvalue[2],-1),DecisionStump(minvalue[0],minvalue[2], 1)]
    accRates = [np.dot(np.array([S(x) for x in X])==Y,D) for S in stumps]
    _max = np.argmax(accRates)
    return {
        'attribute':minvalue[0],
        'stump':stumps[_max],
        'acc':accRates[_max],
        'Ent':minvalue[1],
        'value':minvalue[2]
    }
'''