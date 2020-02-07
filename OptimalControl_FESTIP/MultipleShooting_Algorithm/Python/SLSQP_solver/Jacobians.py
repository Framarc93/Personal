# sparseJacCost = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacCost.npz")          #############################################
# spCost = sparseJacCost.todense()                                                                                      THIS PART MUST BE WRITTEN IN THE MAIN
# sparseJacEq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacEq.npz")              #############################################
# spEq = sparseJacEq.todense()
# sparseJacIneq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacIneq.npz")
# spIneq = sparseJacIneq.todense()
# sparseJacEq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacEq.npz")
# spEq = sparseJacEq.todense()
# sparseJacIneq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacIneq.npz")
# spIneq = sparseJacIneq.todense()

def do(eps, dx, jac, x0, sp, f0, count):
    dx[count] = eps
    jac[count] = (cost_fun(x0 + dx, sp) - f0) / eps
    dx[count] = 0.0
    return dx, jac


def JacFunSave(var, sp):
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(cost_fun(x0, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = eps
        jac[i] = (cost_fun(x0 + dx, sp) - f0) / eps
        dx[i] = 0.0
    sparse = csc_matrix(jac)
    save_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacCost", sparse)
    return jac.transpose()


def JacEqSave(var, type, sp):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var,type, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (constraints(var+dx, type, sp) - f0) / epsilon
        dx[i] = 0.0
    sparse = csc_matrix(jac)
    save_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacEq", sparse)
    return jac.transpose()


def JacIneqSave(var, type, sp):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var, type, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (constraints(var+dx, type, sp) - f0) / epsilon
        dx[i] = 0.0
    sparse = csc_matrix(jac)
    save_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacIneq", sparse)
    return jac.transpose()


def JacFun(var, sp):
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(cost_fun(x0, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    row = np.shape(sp)[0]
    for i in range(row):
        if sp[i, 0] != 0:
            dx[i] = eps
            jac[i] = (cost_fun(x0 + dx, sp) - f0) / eps
            dx[i] = 0.0
    return jac.transpose()


def JacEq(var, type, sp):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var,type, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    row = np.shape(sp)[0]
    column = np.shape(sp)[1]
    for i in range(row):
        for j in range(column):
            if sp[i, j] != 0:
                dx[i] = epsilon
                jac[i] = (constraints(var+dx, type, sp) - f0) / epsilon
                dx[i] = 0.0
                break

    return jac.transpose()


def JacIneq(var, type, sp):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var, type, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))

    row = np.shape(sp)[0]
    column = np.shape(sp)[1]
    for i in range(row):
        for j in range(column):
            if sp[i, j] != 0:
                dx[i] = epsilon
                jac[i] = (constraints(var+dx, type, sp) - f0) / epsilon
                dx[i] = 0.0
                break
    return jac.transpose()