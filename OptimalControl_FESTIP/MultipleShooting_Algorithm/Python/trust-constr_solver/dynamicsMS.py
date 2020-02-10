from models import *

def dynamicsInt(t, states, obj, cl, cd, cm, presv, spimpv, alfa_Int, delta_Int): #, deltaf_Int, tau_Int, mu_Int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
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

    '''if v > obj.vmax or np.isinf(v):
        pen.append((np.nan_to_num(v) - obj.vmax) / obj.vmax)
        v = obj.vmax
        penalty = True
    elif v < obj.vmin or np.isnan(v):
        pen.append((np.nan_to_num(v) - obj.vmin) / obj.vmax)
        v = obj.vmin
        penalty = True
    if chi > obj.chimax or np.isinf(chi):
        pen.append((np.nan_to_num(chi)-obj.chimax)/obj.chimax)
        chi = obj.chimax
        penalty = True
    elif chi < obj.chimin or np.isnan(chi):
        pen.append((np.nan_to_num(chi) - obj.chimin) / obj.chimax)
        chi = obj.chimin
        penalty = True
    if gamma > obj.gammamax or np.isinf(gamma):
        pen.append((np.nan_to_num(gamma) - obj.gammamax) / obj.gammamax)
        gamma = obj.gammamax
        penalty = True
    elif gamma < obj.gammamin or np.isnan(gamma):
        pen.append((np.nan_to_num(gamma) - obj.gammamin) / obj.gammamax)
        gamma = obj.gammamin
        penalty = True
    if lam > obj.lammax or np.isinf(lam):
        pen.append((np.nan_to_num(lam) - obj.lammax) / obj.lammax)
        lam = obj.lammax
        penalty = True
    elif lam < obj.lammin or np.isnan(lam):
        pen.append((np.nan_to_num(lam) - obj.lammin) / obj.lammin)
        lam = obj.lammin
        penalty = True
    if h > obj.hmax or np.isinf(h):
        pen.append((np.nan_to_num(h) - obj.hmax) / obj.hmax)
        h = obj.hmax
        penalty = True
    elif h < obj.hmin or np.isnan(h):
        pen.append((np.nan_to_num(h) - obj.hmin) / obj.hmin)
        h = obj.hmin
        penalty = True
    if m > obj.M0 or np.isinf(m):
        pen.append((np.nan_to_num(m) - obj.M0) / obj.M0)
        m = obj.M0
        penalty = True
    elif m < obj.m10 or np.isnan(m):
        pen.append((np.nan_to_num(m) - obj.m10) / obj.M0)
        m = obj.m10
        penalty = True'''

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

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

                       #((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma))
                       - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
                               np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                       - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),

                       ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(
                           gamma) + 2 * obj.omega \
                       * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                       (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),

                       -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                       np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                       v * np.sin(gamma),
                       -T / (g0 * isp)))

    return dx


def dynamicsVel(states, contr, obj, cl, cd, cm, presv, spimpv):
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

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
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
