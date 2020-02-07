import numpy as np
import dynamics as dyns
import models as mod
from copy import copy

def equality(var, conj, varStates, obj, Nstates, Nbar, Nleg, Ncontrols, NContPoints, states_init, cont_init, cl, cd, cm, presv, spimpv):
    h = conj[varStates - 2]
    lam = conj[varStates - 3]
    stat = conj[varStates - Nstates:]
    cont = obj.Controls[-1, :]

    vtAbs, chiass, vtAbs2 = mod.vass(stat, cont, dyns.dynamicsVel, obj.omega, cl, cd, cm, obj, presv, spimpv)

    vvv = np.sqrt(obj.GMe / (obj.Re + h))

    if np.cos(obj.incl) / np.cos(lam) > 1:
        chifin = np.pi
    elif np.cos(obj.incl) / np.cos(lam) < - 1:
        chifin = 0.0
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam))
    div = np.tile([100, 1, 1, 1, 1, 100, 1000], Nleg - 1)
    div2 = np.tile([np.deg2rad(40),1], Nleg - 1)
    contr_left = np.zeros((0))
    contr_right = np.zeros((0))
    for i in range(Nleg):
        contr_left = np.hstack((contr_left, var[varStates+Ncontrols*NContPoints*i:varStates+Ncontrols*(NContPoints*i+1)-1]))
        contr_right = np.hstack((contr_right, var[varStates+Ncontrols*(NContPoints*i+1):varStates+Ncontrols*(NContPoints*i+2)-1]))

    eq_cond = np.zeros((0))
    eq_cond = np.concatenate((eq_cond, (var[0] - states_init[0],), var[2:7]-states_init[2:])) # initial conditions on all states except chi
    eq_cond = np.concatenate((eq_cond, (var[Nstates:varStates] - conj[:Nstates * (Nleg - 1)])))  # knotting conditions on states
    eq_cond = np.concatenate((eq_cond, (contr_left - contr_right))) # knotting conditions on controls
    eq_cond = np.concatenate((eq_cond, var[varStates:varStates + Ncontrols] - cont_init))  # init cond on alpha
    eq_cond = np.concatenate((eq_cond, ((vvv - vtAbs)/vvv,)))  # orbital insertion velocity
    eq_cond = np.concatenate((eq_cond, ((chifin - chiass)/chifin,))) # final gamma for orbit insertion
    eq_cond = np.concatenate((eq_cond, (conj[varStates - 5],)))  # final condition on gamma
    #for j in range(len(eq_cond)):
     #   if eq_cond[j] > 1:
      #      print("eq", eq_cond[j], j)

    return eq_cond

def inequalityAll(states, controls, varnum, obj, cl, cd, cm, presv, spimpv):
    '''this function takes states and controls unscaled'''
    v = copy(states[:, 0]).T
    # gamma = copy(states[:, 2]).T
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

    L, D, MomA = mod.aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                 obj.xcg0, obj.xcgf, obj.pref, varnum)

    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    T, Deps, isp, MomT = mod.thrustMulti(Press, m, presv, spimpv, delta, tau, varnum, obj.psl, obj.M0, obj.m10, obj.lRef,
                                     obj.xcgf, obj.xcg0)

    T = np.asarray(T, dtype=np.float64)
    isp = np.asarray(isp, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)

    # accelerations

    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / np.exp((Dv1 + Dv2) / (obj.g0*isp[-1]))

    iC = np.hstack(((obj.MaxAx - ax) / obj.MaxAx, (obj.MaxAz - az) / obj.MaxAz, (obj.MaxQ - q) / obj.MaxQ,
                    (mf - obj.m10) / obj.m10))

    return iC