from scipy.interpolate import PchipInterpolator
import numpy as np
from functools import partial
import constraints as cons
import models as mod

def SingleShooting(states, controls, dyn, tstart, tfin, Nint, NContPoints, Nstates, cl, cd, cm, presv, spimpv, obj):
    '''this function integrates the dynamics equation over time.'''
    '''INPUT: states: states vector
              controls: controls matrix
              dyn: dynamic equations
              tstart: initial time
              tfin: final time
              Nint: number of integration steps'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''tstart and tfin are the initial and final time of the considered leg'''
    Nintlocal = Nint
    timeCont = np.linspace(tstart, tfin, NContPoints)
    x = np.zeros((Nint, Nstates))
    time_old = np.linspace(tstart, tfin, Nint)

    alfa_Int = PchipInterpolator(timeCont, controls[0, :])
    delta_Int = PchipInterpolator(timeCont, controls[1, :])
    #deltaf_Int = PchipInterpolator(timeCont, controls[2, :])
    #tau_Int = PchipInterpolator(timeCont, controls[3, :])
    #mu_Int = PchipInterpolator(timeCont, controls[4, :])


    time_int = np.linspace(tstart, tfin, Nintlocal)
    dt = (time_int[1] - time_int[0])
    t = time_int
    x[0,:] = states

    for c in range(Nint - 1):
        k1 = dt * dyn(t[c], x[c, :], alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k2 = dt * dyn(t[c] + dt / 2, x[c, :] + k1 / 2, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k3 = dt * dyn(t[c] + dt / 2, x[c, :] + k2 / 2, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k4 = dt * dyn(t[c + 1], x[c, :] + k3, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    vres = x[:, 0]  # orig interavals
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_old)
    deltares = delta_Int(time_old)
    deltafres = np.zeros(len(vres))  # deltaf_Int(time_old)
    taures = np.zeros(len(vres))  # tau_Int(time_old)
    mures = np.zeros(len(vres))  # mu_Int(time_old)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_old, alfares, deltares, deltafres, taures, mures, alfa_Int, delta_Int


def SingleShootingMulti(i, var, dyn, Nint, NContPoints, Nstates, varTot, varStates, Ncontrols, cl, cd, cm, presv, spimpv, obj):
    '''this function integrates the dynamics equation over time.'''
    '''INPUT: states: states vector
              controls: controls matrix
              dyn: dynamic equations
              tstart: initial time
              tfin: final time
              Nint: unmber of integration steps'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''tstart and tfin are the initial and final time of the considered leg'''
    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    #deltaf = np.zeros((NContPoints))
    #tau = np.zeros((NContPoints))
    #mu = np.zeros((NContPoints))
    states = var[i * Nstates:(i + 1) * Nstates]  # orig intervals
    if i ==0:
        tstart = 0
    else:
        tstart = var[varTot + i]
    tfin = tstart + var[varTot + i]

    for k in range(NContPoints):
        '''this for loop takes the controls from the optimization variable and stores them into different variables'''
        '''here controls are scaled'''
        alfa[k] = var[varStates + i * (Ncontrols * NContPoints) + Ncontrols * k]
        delta[k] = var[varStates + i * (Ncontrols * NContPoints) + 1 + Ncontrols * k]
        #deltaf[k] = var[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
        #tau[k] = var[varStates + i * (Ncontrols * NContPoints) + 3 + Ncontrols * k]
        #mu[k] = var[varStates + i * (Ncontrols * NContPoints) + 4 + Ncontrols * k]

    controls = np.vstack((alfa, delta)) #, deltaf, tau, mu))  # orig intervals

    Nintlocal = Nint
    timeCont = np.linspace(tstart, tfin, NContPoints)
    x = np.zeros((Nint, Nstates))
    time_old = np.linspace(tstart, tfin, Nint)

    alfa_Int = PchipInterpolator(timeCont, controls[0, :])
    delta_Int = PchipInterpolator(timeCont, controls[1, :])
    #deltaf_Int = PchipInterpolator(timeCont, controls[2, :])
    #tau_Int = PchipInterpolator(timeCont, controls[3, :])
    #mu_Int = PchipInterpolator(timeCont, controls[4, :])

    time_int = np.linspace(tstart, tfin, Nintlocal)
    dt = (time_int[1] - time_int[0])
    t = time_int
    x[0,:] = states

    for c in range(Nint - 1):
        k1 = dt * dyn(t[c], x[c, :], alfa_Int, delta_Int, cl, cd, cm, obj, presv,
                      spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k2 = dt * dyn(t[c] + dt / 2, x[c, :] + k1 / 2, alfa_Int, delta_Int, cl, cd, cm, obj, presv,
                      spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k3 = dt * dyn(t[c] + dt / 2, x[c, :] + k2 / 2, alfa_Int, delta_Int, cl, cd, cm, obj, presv,
                      spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        k4 = dt * dyn(t[c + 1], x[c, :] + k3, alfa_Int, delta_Int, cl, cd, cm, obj, presv,
                      spimpv)  # , deltaf_Int, tau_Int, mu_Int)
        x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    vres = x[:, 0]  # orig interavals
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_old)
    deltares = delta_Int(time_old)
    deltafres = np.zeros(len(vres)) #deltaf_Int(time_old)
    taures = np.zeros(len(vres)) #tau_Int(time_old)
    mures = np.zeros(len(vres)) #mu_Int(time_old)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_old, alfares, deltares, deltafres, taures, mures


def MultiShooting(var, dyn, obj, p, Nint, Nleg, presv, spimpv, NContPoints, Nstates, varTot, varStates, Ncontrols, cl, cd, cm, states_init, cont_init):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in radians'''
    global penalty
    penalty = False
    vres = np.zeros((0))
    chires = np.zeros((0))
    gammares = np.zeros((0))
    tetares = np.zeros((0))
    lamres = np.zeros((0))
    hres = np.zeros((0))
    mres = np.zeros((0))
    alfares = np.zeros((0))
    deltares = np.zeros((0))
    #deltafres = np.zeros((0))
    #taures = np.zeros((0))
    #mures = np.zeros((0))
    tres = np.zeros((0))

    states_atNode = np.zeros((0))
    controls_atNode = np.zeros((0))

    varD = var * (obj.UBV - obj.LBV) + obj.LBV

    res = p.map(partial(SingleShootingMulti, var=var, dyn=dyn, Nint=Nint, NContPoints=NContPoints, Nstates=Nstates, varTot=varTot, varStates=varStates, Ncontrols=Ncontrols, cl=cl, cd=cd, cm=cm, presv=presv, spimpv=spimpv, obj=obj), range(Nleg))

    for i in range(Nleg):
        leg = res[i]
        vres = np.hstack((vres, leg[0]))
        chires = np.hstack((chires, leg[1]))
        gammares = np.hstack((gammares, leg[2]))
        tetares = np.hstack((tetares, leg[3]))
        lamres = np.hstack((lamres, leg[4]))
        hres = np.hstack((hres, leg[5]))
        mres = np.hstack((mres, leg[6]))
        tres = np.hstack((tres, leg[7]))
        alfares = np.hstack((alfares, leg[8]))
        deltares = np.hstack((deltares, leg[9]))
        #deltafres = np.hstack((deltafres, leg[10]))
        #taures = np.hstack((taures, leg[11]))
        #mures = np.hstack((mures, leg[12]))
        states_atNode = np.hstack((states_atNode, ((vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]))))  # new intervals
        controls_atNode = np.hstack((controls_atNode, ((alfares[-1], deltares[-1])))) #, deltafres[-1], taures[-1], mures[-1]))))

    vrescol = np.reshape(vres, (Nint*Nleg, 1))
    chirescol = np.reshape(chires, (Nint*Nleg, 1))
    gammarescol = np.reshape(gammares, (Nint*Nleg, 1))
    tetarescol = np.reshape(tetares, (Nint*Nleg, 1))
    lamrescol = np.reshape(lamres, (Nint*Nleg, 1))
    hrescol = np.reshape(hres, (Nint*Nleg, 1))
    mrescol = np.reshape(mres, (Nint*Nleg, 1))
    alfarescol = np.reshape(alfares, (Nint*Nleg, 1))
    deltarescol = np.reshape(deltares, (Nint*Nleg, 1))
    #deltafrescol = np.reshape(deltafres, (Nint*Nleg, 1))
    #murescol = np.reshape(mures, (Nint*Nleg, 1))
    #taurescol = np.reshape(taures, (Nint*Nleg, 1))


    states_after = np.column_stack((vrescol, chirescol, gammarescol, tetarescol, lamrescol, hrescol, mrescol))
    controls_after = np.column_stack((alfarescol, deltarescol)) #, deltafrescol, taurescol, murescol))


    obj.States = states_after
    obj.Controls = controls_after

    eq_c = cons.equality(varD, states_atNode, varStates, obj, Nstates, Nleg+1, Nleg, Ncontrols, NContPoints, states_init, cont_init, cl, cd, cm, presv, spimpv)

    ineq_c = cons.inequalityAll(states_after, controls_after, Nint*Nleg, obj, cl, cd, cm, presv, spimpv)

    h = states_after[-1, 5]
    m = states_after[-1, 6]
    delta = controls_after[-1, 1]
    tau = 0.0 #controls_after[-1, 3]

    Press, rho, c = mod.isa(h, obj.psl, obj.g0, obj.Re)

    T, Deps, isp, MomT = mod.thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10, obj.lRef,
                                obj.xcgf, obj.xcg0)

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / (obj.g0 * isp))


    pen = []
    for i in range(len(vres)):
        # if vres[i] > obj.vmax:
        #    pen.append((vres[i]-obj.vmax)/obj.vmax)
        # elif vres[i] < obj.vmin:
        #    pen.append((vres[i] - obj.vmin) / obj.vmax)
        # if chires[i] > obj.chimax:
        #    pen.append((chires[i]-obj.chimax)/obj.chimax)
        # elif chires[i] < obj.chimin:
        #    pen.append((chires[i] - obj.chimin) / obj.chimax)
        # if gammares[i] > obj.gammamax:
        #    pen.append((gammares[i]-obj.gammamax)/obj.gammamax)
        # elif gammares[i] < obj.gammamin:
        #    pen.append((gammares[i] - obj.gammamin) / obj.gammamax)
        # if hres[i] > obj.hmax:
        #    pen.append((hres[i]-obj.hmax)/obj.hmax)
        if hres[i] < obj.hmin:
            pen.append((hres[i] - obj.hmin) / obj.hmax)

    cost = 1 - mf / obj.M0 + sum(abs(np.array(pen)))
    if penalty:
        cost = 1 - 0.0001 + sum(abs(np.array(pen)))
        eq_c = eq_c * 10
        ineq_c = ineq_c * 10

    obj.costOld = cost
    obj.varOld = var
    obj.ineqOld = ineq_c
    obj.eqOld = eq_c

    return eq_c, ineq_c, cost