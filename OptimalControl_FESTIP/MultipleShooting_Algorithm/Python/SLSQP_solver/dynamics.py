import numpy as np
import models as mod

def dynamicsInt(t, states, alfa_Int, delta_Int, cl, cd, cm, obj, presv, spimpv): #, deltaf_Int, tau_Int, mu_Int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    global penalty
    v = states[0]
    chi = states[1]
    gamma = states[2]
    #teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = alfa_Int(t)
    delta = delta_Int(t)
    deltaf = 0.0 #deltaf_Int(t)
    tau = 0.0 #tau_Int(t)
    mu = 0.0 #mu_Int(t)

    if h > obj.hmax or np.isinf(h):
        h = obj.hmax
        penalty = True
    elif h < obj.hmin or np.isnan(h):
        h = obj.hmin
        penalty = True
    if v > obj.vmax or np.isinf(v):
        v = obj.vmax
        penalty = True
    elif v < obj.vmin or np.isnan(v):
        v = obj.vmin
        penalty = True

    Press, rho, c = mod.isa(h, obj.psl, obj.g0, obj.Re)

    M = v / c

    L, D, MomA = mod.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10, obj.xcg0, obj.xcgf, obj.pref)

    T, Deps, isp, MomT = mod.thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    if t < obj.tvert:
        dx = np.array(((T * np.cos(eps) - D) / m - g, 0, 0, 0, 0, v, -T / (g0 * isp)))
    else:
        dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                       (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                       ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                       np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                               np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                       - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(gamma) * np.cos(  # or np.sin(lam)
                           chi),
                       ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(
                           gamma) + 2 * obj.omega \
                       * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                       (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                       -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                       np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                       v * np.sin(gamma),
                       -T / (g0 * isp)))

    return dx


def dynamicsVel(states, contr, cl, cd, cm, obj, presv, spimpv):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''

    v = states[0]
    chi = states[1]
    gamma = states[2]
    #teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = contr[0]
    delta = contr[1]
    deltaf = 0.0 #contr[2]
    tau = 0.0 #contr[2]
    mu = 0.0 #contr[4]

    '''if np.isinf(h):
        h = 2e5
    elif np.isnan(h):
        h = 0
    if np.isinf(v):
        v = 1e4
    elif np.isnan(v):
        v = 0
    if np.isinf(gamma):
        gamma = np.deg2rad(89)
    elif np.isnan(gamma):
        gamma = np.deg2rad(-89)
    if np.isinf(chi):
        chi = np.deg2rad(250)
    elif np.isnan(chi):
        chi = np.deg2rad(90)
    if np.isinf(lam):
        lam = np.deg2rad(30)
    elif np.isnan(lam):
        lam = np.deg2rad(2)'''

    Press, rho, c = mod.isa(h, obj.psl, obj.g0, obj.Re)

    M = v / c

    L, D, MomA = mod.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)

    T, Deps, isp, MomT = mod.thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = g0 * (obj.Re / (obj.Re + h)) ** 2

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                   np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                           np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(
                       chi),
                   ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(
                       gamma) + 2 * obj.omega \
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (g0 * isp)))

    return dx