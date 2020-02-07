import numpy as np
import sys
sys.path.insert(0, 'home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP')
import models as mod
from copy import copy
import dynamics as dyns

def equality(var, varStates, cont_init, states_array, controls_array, obj, cl, cd, cm, presv, spimpv):
    h = states_array[-2]
    lam = states_array[-3]

    vtAbs, chiass, vtAbs2 = mod.vass(states_array, controls_array, dyns.dynamicsVel, obj.omega, cl, cd, cm, obj, presv, spimpv)

    vvv = np.sqrt(obj.GMe / (obj.Re + h))

    if np.cos(obj.incl) / np.cos(lam) > 1:
        chifin = np.pi
    elif np.cos(obj.incl) / np.cos(lam) < - 1:
        chifin = 0.0
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam))

    eq_cond = np.zeros((0))
    eq_cond = np.concatenate((eq_cond, (var[varStates + 1] - cont_init[-1],)))
    eq_cond = np.concatenate((eq_cond, ((vvv - vtAbs)/obj.vmax,)))
    eq_cond = np.concatenate((eq_cond, ((chifin - chiass)/obj.chimax,)))
    eq_cond = np.concatenate((eq_cond, (states_array[-5]/obj.gammamax,)))  # final condition on gamma
    return eq_cond


def inequalityAll(states, controls, varnum, obj, cl, cd, cm, presv, spimpv):
    '''this function takes states and controls unscaled
    Equality constraint means that the constraint function result is to be zero whereas inequality means that it is to be non-negative'''
    v = copy(states[:, 0]).T
    gamma = copy(states[:, 2]).T
    h = copy(states[:, 5]).T
    m = copy(states[:, 6]).T
    alfa = copy(controls[:, 0]).T
    delta = copy(controls[:, 1]).T
    deltaf = np.zeros(len(v))
    tau = np.zeros(len(v))

    Press, rho, c = mod.isaMulti(h, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    M = v / c

    L, D, MomA = mod.aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10, obj.xcg0, obj.xcgf, obj.pref, varnum)

    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    T, Deps, isp, MomT = mod.thrustMulti(Press, m, presv, spimpv, delta, tau, varnum, obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)

    T = np.asarray(T, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)

    # accelerations

    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / np.exp((Dv1 + Dv2) / obj.gIsp)

    iC = np.hstack(((obj.MaxAx - ax)/obj.MaxAx, (obj.MaxAz - az)/obj.MaxAz, (obj.MaxQ-q)/obj.MaxQ, (mf-obj.m10)/obj.M0, (obj.M0 - mf)/obj.M0, (h[-1]-6e4)/obj.hmax, (2e5-h[-1])/obj.hmax, (obj.gammamax-gamma)/obj.gammamax, h/obj.hmax))
    #iC = (mf-obj.m10)/obj.M0
    return iC