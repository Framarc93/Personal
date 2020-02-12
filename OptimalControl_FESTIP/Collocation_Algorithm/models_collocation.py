import numpy as np
from functools import partial
from scipy.interpolate import splev, splrep

def fileReadOr(filename):
    '''function to read data from txt file'''
    with open(filename) as f:
        table = []
        for line in f:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                table.append(line)

    table = table[1:][:]
    par = [[x[1], x[2], x[3], x[4], x[5], x[6]] for x in table]
    f.close()
    return par

def to_new_int(t, a, b, c, d):
    ''' this function converts a value from an interval [a, b] to [c, d]
    t = value to be converted
    a = inf lim old interval
    b = sup lim old interval
    c = inf lim new interval
    d = sup lim new interval'''
    return c + ((d - c) / (b - a)) * (t - a)

def isa(altitude, obj, flag):
    t0 = 288.15
    p0 = obj.psl
    prevh = 0.0
    R = 287.00
    m0 = 28.9644
    Rs = 8314.32
    m0 = 28.9644
    if flag == 1:
        altitude = np.array([altitude])
    temperature = np.zeros(len(altitude))
    pressure = np.zeros(len(altitude))
    tempm = np.zeros(len(altitude))
    density = np.zeros(len(altitude))
    csound = np.zeros(len(altitude))
    k = 0

    def cal(ps, ts, av, h0, h1):
        if av != 0:
            t1 = ts + av * (h1 - h0)
            p1 = ps * (t1 / ts) ** (-obj.g0 / av / R)
        else:
            t1 = ts
            p1 = ps * np.exp(-obj.g0 / R / ts * (h1 - h0))
        return t1, p1

    def atm90(a90v, z, hi, tc1, pc, tc2, tmc):
        for num in hi:
            if z <= num:
                ind = hi.index(num)
                if ind == 0:
                    zb = hi[0]
                    b = zb - tc1[0] / a90v[0]
                    t = tc1[0] + tc2[0] * (z - zb) / 1000
                    tm = tmc[0] + a90v[0] * (z - zb) / 1000
                    add1 = 1 / ((obj.Re + b) * (obj.Re + z)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((z - b) / (z + obj.Re)))
                    add2 = 1 / ((obj.Re + b) * (obj.Re + zb)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((zb - b) / (zb + obj.Re)))
                    p = pc[0] * np.exp(-m0 / (a90v[0] * Rs) * obj.g0 * obj.Re ** 2 * (add1 - add2))
                else:
                    zb = hi[ind - 1]
                    b = zb - tc1[ind - 1] / a90v[ind - 1]
                    t = tc1[ind - 1] + (tc2[ind - 1] * (z - zb)) / 1000
                    tm = tmc[ind - 1] + a90v[ind - 1] * (z - zb) / 1000
                    add1 = 1 / ((obj.Re + b) * (obj.Re + z)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((z - b) / (z + obj.Re)))
                    add2 = 1 / ((obj.Re + b) * (obj.Re + zb)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((zb - b) / (zb + obj.Re)))
                    p = pc[ind - 1] * np.exp(-m0 / (a90v[ind - 1] * Rs) * obj.g0 * obj.Re ** 2 * (add1 - add2))
                break
        return t, p, tm

    for alt in altitude:
        if alt < 0 or np.isnan(alt):
            # print("h < 0", alt)
            t = t0
            p = p0
            d = p / (R * t)
            c = np.sqrt(1.4 * R * t)
            density[k] = d
            csound[k] = c
            # temperature[k] = t
            pressure[k] = p
            # tempm[k] = t
        elif 0 <= alt < 90000:
            for i in range(0, 8):
                if alt <= obj.hv[i]:
                    t, p = cal(p0, t0, obj.a[i], prevh, alt)
                    d = p / (R * t)
                    c = np.sqrt(1.4 * R * t)
                    density[k] = d
                    csound[k] = c
                    temperature[k] = t
                    pressure[k] = p
                    tempm[k] = t
                    t0 = 288.15
                    p0 = obj.psl
                    prevh = 0
                    break
                else:
                    t0, p0 = cal(p0, t0, obj.a[i], prevh, obj.hv[i])
                    prevh = obj.hv[i]

        elif 90000 <= alt <= 190000:
            t, p, tpm = atm90(obj.a90, alt, obj.h90, obj.tcoeff1, obj.pcoeff, obj.tcoeff2, obj.tmcoeff)
            d = p / (R * tpm)
            c = np.sqrt(1.4 * R * tpm)
            density[k] = d
            csound[k] = c
            # temperature[k] = t
            pressure[k] = p
            # tempm[k] = t
        elif alt > 190000 or np.isinf(alt):
            # print("h > 190 km", alt)
            zb = obj.h90[6]
            z = obj.h90[-1]
            b = zb - obj.tcoeff1[6] / obj.a90[6]
            t = obj.tcoeff1[6] + (obj.tcoeff2[6] * (z - zb)) / 1000
            tm = obj.tmcoeff[6] + obj.a90[6] * (z - zb) / 1000
            add1 = 1 / ((obj.Re + b) * (obj.Re + z)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((z - b) / (z + obj.Re)))
            add2 = 1 / ((obj.Re + b) * (obj.Re + zb)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((zb - b) / (zb + obj.Re)))
            p = obj.pcoeff[6] * np.exp(-m0 / (obj.a90[6] * Rs) * obj.g0 * obj.Re ** 2 * (add1 - add2))
            d = p / (R * t)
            c = np.sqrt(1.4 * R * tm)
            density[k] = d
            csound[k] = c
            # temperature[k] = t
            pressure[k] = p
            # tempm[k] = t
        k += 1
    return pressure, density, csound


def aeroForces(M, alfa, deltaf, cd, cl, cm, v, rho, mass, npoint, obj):
    alfag = np.rad2deg(alfa)
    deltafg = np.rad2deg(deltaf)
    coeffs = np.array((cl, cd, cm))
    if npoint == 1:
        M = np.array([M])
        alfag = np.array([alfag])
        deltafg = np.array([deltafg])
    cL, cD, cM1 = list(map(partial(c_eval, npoint, M, alfag, deltafg, coeffs, obj.mach, obj.angAttack, obj.bodyFlap), range(3)))

    L = 0.5 * (v ** 2) * obj.wingSurf * rho * cL
    D = 0.5 * (v ** 2) * obj.wingSurf * rho * cD
    xcg = obj.lRef * (((obj.xcgf - obj.xcg0) / (obj.m10 - obj.M0)) * (mass - obj.M0) + obj.xcg0)
    Dx = xcg - obj.pref
    cM = cM1 + cL * (Dx / obj.lRef) * np.cos(alfa) + cD * (Dx / obj.lRef) * np.sin(alfa)
    Mom = 0.5 * (v ** 2) * obj.wingSurf * obj.lRef * rho * cM

    return L, D, Mom


def thrust(presamb, mass, presv, spimpv, delta, tau, npoint, spimp_interp, obj):
    nimp = 17
    nmot = 1
    Thrust = np.zeros(npoint)
    Deps = np.zeros(npoint)
    Simp = np.zeros(npoint)
    Mom = np.zeros(npoint)
    if npoint == 1:
        delta = np.array([delta])
        tau = np.array([tau])
        mass = np.array([mass])
    for j in range(npoint):
        thrx = nmot * (5.8e6 + 14.89 * obj.psl - 11.16 * presamb[j]) * delta[j]
        if presamb[j] >= obj.psl:
            presamb[j] = obj.psl
            spimp = spimpv[-1]
        elif presamb[j] < obj.psl:
            for i in range(nimp):
                if presv[i] >= presamb[j]:
                    spimp = splev(presamb[j], spimp_interp, der=0)
                    break

        xcg = ((obj.xcgf - obj.xcg0) / (obj.m10 - obj.M0) * (mass[j] - obj.M0) + obj.xcg0) * obj.lRef
        dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * obj.psl - presamb[j])
        if tau[j] == 0:
            mommot = 0.0
            thrz = 0.0
            thrust = thrx
            deps = 0.0
        else:
            mommot = tau[j] * dthr
            thrz = -tau[j] * (2.5e6 - 22 * obj.psl + 9.92 * presamb[j])
            thrust = np.sqrt(thrx ** 2 + thrz ** 2)
            deps = np.arctan(thrz / thrx)

        Thrust[j] = thrust
        Deps[j] = deps
        Simp[j] = spimp
        Mom[j] = mommot
    return Thrust, Deps, Simp, Mom


def c_eval(n, m, a, b, cc, mach, angAttack, bodyFlap, it):
    coeff = cc[it]
    c = np.zeros(n)
    for i in range(n):
        c[i] = coefCalc(coeff, m[i], a[i], b[i], mach, angAttack, bodyFlap)
    return c


def limCalc(array, value):
    j = 0
    lim = array.__len__()
    for num in array:
        if j == lim - 1:
            sup = num
            inf = array[j - 1]
        if value < num:
            sup = num
            if j == 0:
                inf = num
            else:
                inf = array[j - 1]
            break
        j += 1
    s = np.where(array == sup)[0][0]
    i = np.where(array == inf)[0][0]
    return i, s


def coefCalc(coeff, m, alfa, deltaf, mach, angAttack, bodyFlap):
    if m > mach[-1] or np.isinf(m):
        m = mach[-1]
    elif m < mach[0] or np.isnan(m):
        m = mach[0]
    if alfa > angAttack[-1] or np.isinf(alfa):
        alfa = angAttack[-1]
    elif alfa < angAttack[0] or np.isnan(alfa):
        alfa = angAttack[0]
    if deltaf > bodyFlap[-1] or np.isinf(deltaf):
        deltaf = bodyFlap[-1]
    elif deltaf < bodyFlap[0] or np.isnan(deltaf):
        deltaf = bodyFlap[0]

    (im, sm), (ia, sa), (idf, sdf) = map(limCalc, (mach, angAttack, bodyFlap), (m, alfa, deltaf))
    cnew1 = coeff[im]  # [17 * im: 17 * im + angAttack.__len__()][:]
    cnew2 = coeff[sm]  # [17 * sm: 17 * sm + angAttack.__len__()][:]

    '''interpolation on the first table between angle of attack and deflection'''
    rowinf1 = cnew1[ia][:]
    rowsup1 = cnew1[sa][:]
    coeffinf = [rowinf1[idf], rowsup1[idf]]
    coeffsup = [rowinf1[sdf], rowsup1[sdf]]
    c1 = coeffinf[0] + (alfa - angAttack[ia]) * (
            (coeffinf[1] - coeffinf[0]) / (angAttack[sa] - angAttack[ia]))
    c2 = coeffsup[0] + (alfa - angAttack[ia]) * (
            (coeffsup[1] - coeffsup[0]) / (angAttack[sa] - angAttack[ia]))
    coeffd1 = c1 + (deltaf - bodyFlap[idf]) * ((c2 - c1) / (bodyFlap[sdf] - bodyFlap[idf]))
    '''interpolation on the first table between angle of attack and deflection'''
    rowinf2b = cnew2[ia][:]
    rowsup2b = cnew2[sa][:]
    coeffinfb = [rowinf2b[idf], rowsup2b[idf]]
    coeffsupb = [rowinf2b[sdf], rowsup2b[sdf]]
    c1b = coeffinfb[0] + (alfa - angAttack[ia]) * (
            (coeffinfb[1] - coeffinfb[0]) / (angAttack[sa] - angAttack[ia]))
    c2b = coeffsupb[0] + (alfa - angAttack[ia]) * (
            (coeffsupb[1] - coeffsupb[0]) / (angAttack[sa] - angAttack[ia]))
    coeffd2 = c1b + (deltaf - bodyFlap[idf]) * ((c2b - c1b) / (bodyFlap[sdf] - bodyFlap[idf]))
    '''interpolation on the moments to obtain final coefficient'''
    coeffFinal = coeffd1 + (m - mach[im]) * ((coeffd2 - coeffd1) / (mach[sm] - mach[im]))

    return coeffFinal

def vass(states, controls, dyn, omega, obj):
    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3]
    lam = states[4]
    h = states[5]

    vv = np.array((-v * np.cos(gamma) * np.cos(chi),
                   v * np.cos(gamma) * np.sin(chi),
                   -v * np.sin(gamma)))
    vv[0] = vv[0] + omega * np.cos(lam) * (obj.Re + h)
    vela2 = np.sqrt(vv[0] ** 2 + vv[1] ** 2 + vv[2] ** 2)
    if vv[0] <= 0.0:
        if abs(vv[0]) >= abs(vv[1]):
            chiass = np.arctan(abs(vv[1] / vv[0]))
            if vv[1] < 0.0:
                chiass = -chiass
        elif abs(vv[0]) < abs(vv[1]):
            chiass = np.pi * 0.5 - np.arctan(abs(vv[0] / vv[1]))
            if vv[1] < 0.0:
                chiass = -chiass
    elif vv[0] > 0.0:
        if abs(vv[0]) >= abs(vv[1]):
            chiass = np.pi - np.arctan((abs(vv[1] / vv[0])))
            if vv[1] < 0.0:
                chiass = - chiass
        elif abs(vv[0]) < abs(vv[1]):
            chiass = np.pi * 0.5 + np.arctan(abs(vv[0] / vv[1]))
            if vv[1] < 0.0:
                chiass = -chiass

    x = np.array(((obj.Re + h) * np.cos(lam) * np.cos(teta),
                  (obj.Re + h) * np.cos(lam) * np.sin(teta),
                  (obj.Re + h) * np.sin(lam)))

    dx = dyn(states, controls, obj)
    xp = np.array(
        (dx[5] * np.cos(lam) * np.cos(teta) - (obj.Re + h) * dx[4] * np.sin(lam) * np.cos(teta) - (obj.Re + h) * dx[3]
         * np.cos(lam) * np.sin(teta),
         dx[5] * np.cos(lam) * np.sin(teta) - (obj.Re + h) * dx[4] * np.sin(lam) * np.sin(teta) + (obj.Re + h) * dx[3]
         * np.cos(lam) * np.cos(teta),
         dx[5] * np.sin(lam) + (obj.Re + h) * dx[4] * np.cos(lam)))

    dxp = np.array((-omega * x[1],
                    omega * x[0],
                    0.0))

    vtot = np.array((xp[0] + dxp[0],
                     xp[1] + dxp[1],
                     xp[2] + dxp[2]))

    vela = np.sqrt(vtot[0] ** 2 + vtot[1] ** 2 + vtot[2] ** 2)

    return vela, chiass, vela2

