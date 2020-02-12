import numpy as np

'''function to read data from txt file'''

def fileReadMS(filename):
    with open(filename) as f:
        table = []
        for line in f:
            line = line.split()
            if line:
                line = [float(i) for i in line]
                table.append(line)

    table = table[:][:]
    par = [[x[0], x[1], x[2], x[3], x[4], x[5]] for x in table]
    f.close()
    return par

def fileReadOrMS(filename):
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

'''Atmospheric models'''
def calMS(ps, ts, av, h0, h1, g0, R):
    if av != 0:
        t1 = ts + av * (h1 - h0)
        p1 = ps * (t1 / ts) ** (-g0 / av / R)
    else:
        t1 = ts
        p1 = ps * np.exp(-g0 / R / ts * (h1 - h0))
    return t1, p1

def atm90MS(a90v, z, hi, tc1, pc, tc2, tmc, r, m0, g0, Rs):
    for num in hi:
        if z <= num:
            ind = hi.index(num)
            if ind == 0:
                zb = hi[0]
                b = zb - tc1[0] / a90v[0]
                t = tc1[0] + tc2[0] * (z - zb) / 1000
                tm = tmc[0] + a90v[0] * (z - zb) / 1000
                add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                p = pc[0] * np.exp(-m0 / (a90v[0] * Rs) * g0 * r ** 2 * (add1 - add2))
            else:
                zb = hi[ind - 1]
                b = zb - tc1[ind - 1] / a90v[ind - 1]
                t = tc1[ind - 1] + (tc2[ind - 1] * (z - zb)) / 1000
                tm = tmc[ind - 1] + a90v[ind - 1] * (z - zb) / 1000
                add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                p = pc[ind - 1] * np.exp(-m0 / (a90v[ind - 1] * Rs) * g0 * r ** 2 * (add1 - add2))
            break
    return t, p, tm

def isaMS(alt, obj):
    t0 = 288.15
    p0 = obj.psl
    prevh = 0.0
    R = 287.00
    Rs = 8314.32
    m0 = 28.9644

    if alt < 0 or np.isnan(alt):
        # print("h < 0", alt)
        t = t0
        p = p0
        d = p / (R * t)
        c = np.sqrt(1.4 * R * t)
    elif 0 <= alt < 90000:
        for i in range(0, 8):
            if alt <= obj.hv[i]:
                t, p = calMS(p0, t0, obj.a[i], prevh, alt, obj.g0, R)
                d = p / (R * t)
                c = np.sqrt(1.4 * R * t)
                break
            else:
                t0, p0 = calMS(p0, t0, obj.a[i], prevh, obj.hv[i], obj.g0, R)
                p = p0
                prevh = obj.hv[i]

    elif 90000 <= alt <= 190000:
        t, p, tpm = atm90MS(obj.a90, alt, obj.h90, obj.tcoeff1, obj.pcoeff, obj.tcoeff2, obj.tmcoeff, obj.Re, m0, obj.g0, Rs)
        d = p / (R * tpm)
        c = np.sqrt(1.4 * R * tpm)
    elif alt > 190000 or np.isinf(alt):
        zb = obj.h90[6]
        z = obj.h90[-1]
        b = zb - obj.tcoeff1[6] / obj.a90[6]
        #t = tcoeff1[6] + (tcoeff2[6] * (z - zb)) / 1000
        tm = obj.tmcoeff[6] + obj.a90[6] * (z - zb) / 1000
        add1 = 1 / ((obj.Re + b) * (obj.Re + z)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((z - b) / (z + obj.Re)))
        add2 = 1 / ((obj.Re + b) * (obj.Re + zb)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((zb - b) / (zb + obj.Re)))
        p = obj.pcoeff[6] * np.exp(-m0 / (obj.a90[6] * Rs) * obj.g0 * obj.Re ** 2 * (add1 - add2))
        d = p / (R * tm)
        c = np.sqrt(1.4 * R * tm)

    return p, d, c

def isaMultiMS(altitude, obj):
    t0 = 288.15
    p0=obj.psl
    prevh = 0.0
    R = 287.00
    m0 = 28.9644
    Rs = 8314.32
    temperature = []
    pressure = []
    tempm = []
    density = []
    csound = []

    for alt in altitude:
        if alt < 0 or np.isnan(alt):
            t = t0
            p = p0
            d = p / (R * t)
            c = np.sqrt(1.4 * R * t)
            density.append(d)
            csound.append(c)
            temperature.append(t)
            pressure.append(p)
            tempm.append(t)
        elif 0 <= alt < 90000:
            for i in range(0, 8):
                if alt <= obj.hv[i]:
                    t, p = cal(p0, t0, obj.a[i], prevh, alt, obj.g0, R)
                    d = p / (R * t)
                    c = np.sqrt(1.4 * R * t)
                    density.append(d)
                    csound.append(c)
                    temperature.append(t)
                    pressure.append(p)
                    tempm.append(t)
                    t0=288.15
                    p0=obj.psl
                    prevh=0
                    break
                else:

                    t0, p0 = cal(p0, t0, obj.a[i], prevh, obj.hv[i], obj.g0, R)
                    prevh = obj.hv[i]

        elif 90000 <= alt <= 190000:
            t, p, tpm = atm90(obj.a90, alt, obj.h90, obj.tcoeff1, obj.pcoeff, obj.tcoeff2, obj.tmcoeff, obj.Re, m0, obj.g0, Rs)
            temperature.append(t)
            pressure.append(p)
            tempm.append(tpm)
            d = p / (R * tpm)
            c = np.sqrt(1.4 * R * tpm)
            density.append(d)
            csound.append(c)
        elif alt > 190000 or np.isinf(alt):
            zb = obj.h90[6]
            z = obj.h90[-1]
            b = zb - obj.tcoeff1[6] / obj.a90[6]
            t = obj.tcoeff1[6] + (obj.tcoeff2[6] * (z - zb)) / 1000
            tm = obj.tmcoeff[6] + obj.a90[6] * (z - zb) / 1000
            add1 = 1 / ((obj.Re + b) * (obj.Re + z)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((z - b) / (z + obj.Re)))
            add2 = 1 / ((obj.Re + b) * (obj.Re + zb)) + (1 / ((obj.Re + b) ** 2)) * np.log(abs((zb - b) / (zb + obj.Re)))
            p = obj.pcoeff[6] * np.exp(-m0 / (obj.a90[6] * Rs) * obj.g0 * obj.Re ** 2 * (add1 - add2))
            temperature.append(t)
            pressure.append(p)
            tempm.append(tm)
            d = p / (R * tm)
            c = np.sqrt(1.4 * R * tm)
            density.append(d)
            csound.append(c)

    return pressure, density, csound

'''Aerodynamic models'''
def coefCalcMS(coeff, m, alfa, deltaf, mach, angAttack, bodyFlap):
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
    elif deltaf < bodyFlap[0] or np.isnan(alfa):
        deltaf = bodyFlap[0]

    im, sm = limCalc(mach, m)  # moments boundaries and determination of the 2 needed tables
    cnew1 = coeff[17 * im: 17 * im + angAttack.__len__()][:]
    cnew2 = coeff[17 * sm: 17 * sm + angAttack.__len__()][:]

    ia, sa = limCalc(angAttack, alfa)  # angle of attack boundaries

    idf, sdf = limCalc(bodyFlap, deltaf)  # deflection angle boundaries

    '''interpolation on the first table between angle of attack and deflection'''
    rowinf1 = cnew1[ia][:]
    rowsup1 = cnew1[sa][:]
    coeffinf = [rowinf1[idf], rowsup1[idf]]
    coeffsup = [rowinf1[sdf], rowsup1[sdf]]
    c1 = coeffinf[0] + (alfa - angAttack[ia]) * ((coeffinf[1] - coeffinf[0]) / (angAttack[sa] - angAttack[ia]))
    c2 = coeffsup[0] + (alfa - angAttack[ia]) * ((coeffsup[1] - coeffsup[0]) / (angAttack[sa] - angAttack[ia]))
    # c1old = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinf)
    # c2old = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsup)
    # coeffd1old = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])
    coeffd1 = c1 + (deltaf - bodyFlap[idf]) * ((c2 - c1) / (bodyFlap[sdf] - bodyFlap[idf]))
    '''interpolation on the first table between angle of attack and deflection'''
    rowinf2b = cnew2[ia][:]
    rowsup2b = cnew2[sa][:]
    coeffinfb = [rowinf2b[idf], rowsup2b[idf]]
    coeffsupb = [rowinf2b[sdf], rowsup2b[sdf]]
    # c1oldb = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinfb)
    # c2oldb = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsupb)
    c1b = coeffinfb[0] + (alfa - angAttack[ia]) * ((coeffinfb[1] - coeffinfb[0]) / (angAttack[sa] - angAttack[ia]))
    c2b = coeffsupb[0] + (alfa - angAttack[ia]) * ((coeffsupb[1] - coeffsupb[0]) / (angAttack[sa] - angAttack[ia]))
    # coeffd2old = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])
    coeffd2 = c1b + (deltaf - bodyFlap[idf]) * ((c2b - c1b) / (bodyFlap[sdf] - bodyFlap[idf]))
    '''interpolation on the moments to obtain final coefficient'''
    # coeffFinalold = np.interp(m, [mach[im], mach[sm]], [coeffd1, coeffd2])
    coeffFinal = coeffd1 + (m - mach[im]) * ((coeffd2 - coeffd1) / (mach[sm] - mach[im]))
    return coeffFinal

def limCalcMS(array, value):
    j = 0
    lim = array.__len__()
    for num in array:
        if j == lim-1:
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
    s = array.index(sup)
    i = array.index(inf)
    return i, s

def aeroForcesMS(M, alfa, deltaf, cd, cl, cm, v, rho, mass, obj):
    alfag = np.rad2deg(alfa)
    deltafg = np.rad2deg(deltaf)
    cL = coefCalc(cl, M, alfag, deltafg, obj.mach, obj.angAttack, obj.bodyFlap)
    cD = coefCalc(cd, M, alfag, deltafg, obj.mach, obj.angAttack, obj.bodyFlap)
    if v > obj.vmax:
        print("e")
    l = 0.5 * (v ** 2) * obj.wingSurf * rho * cL
    d = 0.5 * (v ** 2) * obj.wingSurf * rho * cD
    xcg = obj.lRef * (((obj.xcgf - obj.xcg0) / (obj.m10 - obj.M0)) * (mass - obj.M0) + obj.xcg0)
    Dx = xcg - obj.pref
    cM1 = coefCalc(cm, M, alfag, deltafg, obj.mach, obj.angAttack, obj.bodyFlap)
    cM = cM1 + cL * (Dx / obj.lRef) * np.cos(alfa) + cD * (Dx / obj.lRef) * np.sin(alfa)
    mom = 0.5 * (v ** 2) * obj.wingSurf * obj.lRef * rho * cM
    if l > 1e10 or np.isin(l):
        l = 1e10
    if d > 1e10 or np.isin(d):
        d = 1e10
    if mom > 1e10 or np.isin(mom):
        mom = 1e10
    return l, d, mom

def aeroForcesMultiMS(M, alfa, deltaf, cd, cl, cm, v, rho, mass, obj, npoint):
    alfag = np.rad2deg(alfa)
    deltafg = np.rad2deg(deltaf)
    L = []
    D = []
    Mom = []
    for i in range(npoint):
        cL = coefCalc(cl, M[i], alfag[i], deltafg[i], obj.mach, obj.angAttack, obj.bodyFlap)
        cD = coefCalc(cd, M[i], alfag[i], deltafg[i], obj.mach, obj.angAttack, obj.bodyFlap)
        l = 0.5 * (v[i] ** 2) * obj.wingSurf * rho[i] * cL
        d = 0.5 * (v[i] ** 2) * obj.wingSurf * rho[i] * cD
        xcg = obj.lRef * (((obj.xcgf - obj.xcg0) / (obj.m10 - obj.M0)) * (mass[i] - obj.M0) + obj.xcg0)
        Dx = xcg - obj.pref
        cM1 = coefCalc(cm, M[i], alfag[i], deltafg[i], obj.mach, obj.angAttack, obj.bodyFlap)
        cM = cM1 + cL * (Dx / obj.lRef) * np.cos(alfa[i]) + cD * (Dx / obj.lRef) * np.sin(alfa[i])
        mom = 0.5 * (v[i] ** 2) * obj.wingSurf * obj.lRef * rho[i] * cM
        if l > 1e10 or np.isin(l):
            l = 1e10
        if d > 1e10 or np.isin(d):
            d = 1e10
        if mom > 1e10 or np.isin(mom):
            mom = 1e10
        L.append(l)
        D.append(d)
        Mom.append(mom)
    return L, D, Mom

'''Populsion models'''

def thrustMS(presamb, mass, presv, spimpv, delta, tau, obj):
    nimp = 17
    nmot = 1
    thrx = nmot * (5.8e6 + 14.89 * obj.psl - 11.16 * presamb) * delta
    if presamb > obj.psl:
        presamb = obj.psl
        spimp = spimpv[-1]
    elif presamb <= obj.psl:
        for i in range(nimp):
            if presv[i] >= presamb:
                spimp = np.interp(presamb, [presv[i - 1], presv[i]], [spimpv[i - 1], spimpv[i]])
                break
    xcg = ((obj.xcgf - obj.xcg0) / (obj.m10 - obj.M0) * (mass - obj.M0) + obj.xcg0) * obj.lRef
    dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * obj.psl - presamb)
    if tau == 0:
        mommot = 0.0
        thrz = 0.0
        thrust = thrx
        deps = 0.0
    else:
        mommot = tau * dthr
        thrz = -tau * (2.5E+6 - 22 * obj.psl + 9.92 * presamb)
        thrust = np.sqrt(thrx ** 2 + thrz ** 2)
        deps = np.arctan(thrz / thrx)
    return thrust, deps, spimp, mommot

def thrustMultiMS(presamb, mass, presv, spimpv, delta, tau, npoint, obj):
    nimp = 17
    nmot = 1
    Thrust = []
    Deps = []
    Simp = []
    Mom = []
    for j in range(npoint):
        thrx = nmot * (5.8e6 + 14.89 * obj.psl - 11.16 * presamb[j]) * delta[j]
        if presamb[j] >= obj.psl:
            spimp = spimpv[-1]
            presamb[j] = obj.psl
        elif presamb[j] < obj.psl:
            for i in range(nimp):
                if presv[i] >= presamb[j]:
                    spimp = np.interp(presamb[j], [presv[i - 1], presv[i]], [spimpv[i - 1], spimpv[i]])
                    break
        xcg = ((obj.xcgf - obj.xcg0) / (obj.m10 - obj.M0) * (mass[j] - obj.M0) + obj.xcg0) * obj.lRef
        dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7*obj.psl - presamb[j])
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
        Thrust.append(thrust)
        Deps.append(deps)
        Simp.append(spimp)
        Mom.append(mommot)
    return Thrust, Deps, Simp, Mom

'''Function to evaluate the absolute velocity considering the earth rotation'''
def vassMS(states, omega):
    Re = 6371000
    v = states[0]
    chi = states[1]
    gamma = states[2]
    #teta = states[3]
    lam = states[4]
    h = states[5]
    #m = states[6]
    vv = np.array((-v * np.cos(gamma) * np.cos(chi),
                   v * np.cos(gamma) * np.sin(chi),
                   -v * np.sin(gamma)))
    vv[0] = vv[0] + omega * np.cos(lam) * (Re + h)
    vela = np.sqrt(vv[0] ** 2 + vv[1] ** 2 + vv[2] ** 2)
    if vv[0] <= 0.0 or np.isnan(vv[0]):
        if abs(vv[0]) >= abs(vv[1]):
            chiass = np.arctan(abs(vv[1] / vv[0]))
            if vv[1] < 0.0 or np.isnan(vv[1]):
                chiass = -chiass
        elif abs(vv[0]) < abs(vv[1]):
            chiass = np.pi*0.5 - np.arctan(abs(vv[0] / vv[1]))
            if vv[1] < 0.0 or np.isnan(vv[1]):
                chiass = -chiass
    elif vv[0] > 0.0 or np.isinf(vv[0]):
        if abs(vv[0]) >= abs(vv[1]):
            chiass = np.pi - np.arctan((abs(vv[1]/vv[0])))
            if vv[1] < 0.0:
                chiass = - chiass
        elif abs(vv[0]) < abs(vv[1]):
            chiass = np.pi * 0.5 + np.arctan(abs(vv[0] / vv[1]))
            if vv[1] < 0.0:
                chiass = -chiass

    '''x = np.array(((Re + h) * np.cos(lam) * np.cos(teta),
                  (Re + h) * np.cos(lam) * np.sin(teta),
                  (Re + h) * np.sin(lam)))

    dx = dyn(states, controls, obj, cl, cd, cm, presv, spimpv)
    xp = np.array((dx[5] * np.cos(lam) * np.cos(teta) - (Re + h) * dx[4] * np.sin(lam) * np.cos(teta) - (Re + h) * dx[3]
                   * np.cos(lam) * np.sin(teta),
                   dx[5] * np.cos(lam) * np.sin(teta) - (Re + h) * dx[4] * np.sin(lam) * np.sin(teta) + (Re + h) * dx[3]
                   * np.cos(lam) * np.cos(teta),
                   dx[5] * np.sin(lam) + (Re + h) * dx[4] * np.cos(lam)))

    dxp = np.array((-omega * x[1],
                    omega * x[0],
                    0.0))

    vtot = np.array((xp[0] + dxp[0],
                     xp[1] + dxp[1],
                     xp[2] + dxp[2]))

    vela = np.sqrt(vtot[0] ** 2 + vtot[1] ** 2 + vtot[2] ** 2)'''

    return vela, chiass
