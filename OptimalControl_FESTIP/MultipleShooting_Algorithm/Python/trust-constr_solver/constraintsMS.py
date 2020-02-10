import numpy as np
from models import *
import dynamicsMS as dyns

def inequalityAll(states, controls, varnum, obj, cl, cd, cm, presv, spimpv):
    '''this function takes states and controls unscaled'''
    v = np.transpose(states[:, 0])
    # chi = np.transpose(states[:, 1])
    gamma = np.transpose(states[:, 2])
    # teta = np.transpose(states[:, 3])
    # lam = np.transpose(states[:, 4])
    h = np.transpose(states[:, 5])
    m = np.transpose(states[:, 6])
    alfa = np.transpose(controls[:, 0])
    delta = np.transpose(controls[:, 1])
    deltaf = np.zeros(len(v)) #np.transpose(controls[:, 2])
    tau = np.zeros(len(v)) #np.transpose(controls[:, 2])  # tau back to [-1, 1] interval
    # mu = np.transpose(controls[:, 4])

    Press, rho, c = isaMulti(h, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    M = v / c

    L, D, MomA = aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                 obj.xcg0, obj.xcgf, obj.pref, varnum)

    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    #MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrustMulti(Press, m, presv, spimpv, delta, tau, varnum, obj.psl, obj.M0, obj.m10, obj.lRef,
                                     obj.xcgf, obj.xcg0)

    T = np.asarray(T, dtype=np.float64)
    #isp = np.asarray(isp, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)
    #MomT = np.asarray(MomT, dtype=np.float64)

    #MomTot = MomA + MomT

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)

    # accelerations

    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    #r1 = h[-1] + obj.Re
    #Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    #Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    #mf = m[-1] / np.exp((Dv1 + Dv2) / obj.gIsp)

    #axnew = to_new_int(obj.MaxAx, 0.0, 1e3, 0.0, 1.0) - to_new_int(ax, 0.0, 1e3, 0.0, 1.0)
    #aznew = to_new_int(obj.MaxAz, -1e3, 1e3, 0.0, 1.0) - to_new_int(az, -1e3, 1e3, 0.0, 1.0)
    #qnew = to_new_int(obj.MaxQ / 1e3, 0, 1e3, 0.0, 1.0) - to_new_int(q / 1e3, 0, 1e3, 0.0, 1.0)
    #momnew = to_new_int(obj.k/ 1e5, -1e3, 1e3, 0.0, 1.0) - to_new_int(MomTotA / 1e5, -1e3, 1e3, 0.0, 1.0)
    #mnew = m_toNew(mf, obj) - m_toNew(obj.m10, obj)

    iC = np.hstack(((obj.MaxAx - ax)/obj.MaxAx, (obj.MaxAz - az)/obj.MaxAz, (obj.MaxQ-q)/obj.MaxQ)) #, (obj.gammamax-gamma)/obj.gammamax))

    return iC

def equality(var, conj, varStates, NContPoints, obj, Ncontrols, Nleg, cl, cd, cm, presv, spimpv):
    stat = obj.States[-1, :]
    h = stat[5]
    lam = stat[4]
    gamma = stat[2]
    cont = obj.Controls[-1, :]

    vtAbs, chiass, vtAbs2 = vass(stat, cont, dyns.dynamicsVel, obj.omega, obj, cl, cd, cm, presv, spimpv)

    vvv = np.sqrt(obj.GMe / (obj.Re + h))

    if np.cos(obj.incl) / np.cos(lam) > 1:
        chifin = np.pi
    elif np.cos(obj.incl) / np.cos(lam) < - 1:
        chifin = 0.0
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam))
    states_unit = [obj.vmax, obj.chimax, obj.gammamax, obj.tetamax, obj.lammax, obj.hmax, obj.M0]
    cont_unit = [obj.alfamax, obj.deltamax]
    eq_cond = np.zeros((0))

    eq_cond = np.concatenate((eq_cond, (var[1:varStates] - conj)/np.tile(states_unit, Nleg-1)))  # knotting conditions
    #eq_cond = np.concatenate((eq_cond, ((var[varStates + Ncontrols-1] - 1.0)/obj.deltamax,)))  # init cond on controls
    i = 1
    while i <= Nleg-1:
        eq_cond = np.concatenate((eq_cond, var[varStates+i*Ncontrols*NContPoints-1:varStates+i*Ncontrols*NContPoints] - var[varStates+i*Ncontrols*NContPoints+1:varStates+i*Ncontrols*NContPoints+Ncontrols]/cont_unit))
        i += 1
    #eq_cond = np.concatenate((eq_cond, ((vvv - vtAbs)/obj.vmax,)))
    #eq_cond = np.concatenate((eq_cond, ((chifin - chiass)/obj.chimax,)))
    #eq_cond = np.concatenate((eq_cond, (gamma/obj.gammamax,)))  # final condition on gamma
    return eq_cond
