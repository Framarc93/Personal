import numpy as np

def isa(alt, pstart, g0, r):

    t0 = 288.15
    p0 = pstart
    prevh = 0.0
    R = 287.00
    m0 = 28.9644
    Rs = 8314.32
    m0 = 28.9644
    a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
    a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
    hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
    h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
    tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
    pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
    tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
    tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]

    def cal(ps, ts, av, h0, h1):
        if av != 0:
            t1 = ts + av * (h1 - h0)
            p1 = ps * (t1 / ts) ** (-g0 / av / R)
        else:
            t1 = ts
            p1 = ps * np.exp(-g0 / R / ts * (h1 - h0))
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

    if alt < 0:
        # print("h < 0", alt)
        t = t0
        p = p0
        d = p / (R * t)
        c = np.sqrt(1.4 * R * t)
    elif 0 <= alt < 90000:
        for i in range(0, 8):
            if alt <= hv[i]:
                t, p = cal(p0, t0, a[i], prevh, alt)
                d = p / (R * t)
                c = np.sqrt(1.4 * R * t)
                break
            else:
                t0, p0 = cal(p0, t0, a[i], prevh, hv[i])
                prevh = hv[i]

    elif 90000 <= alt <= 190000:
        t, p, tpm = atm90(a90, alt, h90, tcoeff1, pcoeff, tcoeff2, tmcoeff)
        d = p / (R * tpm)
        c = np.sqrt(1.4 * R * tpm)
    elif alt > 190000:
        zb = h90[6]
        z = h90[-1]
        b = zb - tcoeff1[6] / a90[6]
        t = tcoeff1[6] + (tcoeff2[6] * (z - zb)) / 1000
        tm = tmcoeff[6] + a90[6] * (z - zb) / 1000
        add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
        add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
        p = pcoeff[6] * np.exp(-m0 / (a90[6] * Rs) * g0 * r ** 2 * (add1 - add2))
        d = p / (R * tm)
        c = np.sqrt(1.4 * R * tm)

    return p, d, c


def thrustSimpl(presamb, mass, presv, spimpv, delta, slpres, wlo, we, lref, xcgf, xcg0):
    nimp = 17
    nmot = 1
    # thrmax = nmot * (5.8E+6 + 14.89 * slpres - 11.16 * presamb)
    thrx = nmot * (5.8e6 + 14.89 * slpres - 11.16 * presamb) * delta
    if presamb >= slpres:
        spimp = spimpv[-1]
    elif presamb < slpres:
        for i in range(nimp):
            if presv[i] >= presamb:
                spimp = np.interp(presamb, [presv[i - 1], presv[i]], [spimpv[i - 1], spimpv[i]])
                break
    xcg = ((xcgf - xcg0) / (we - wlo) * (mass - wlo) + xcg0) * lref

    dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * slpres - presamb)

    thrust = thrx
    deps = 0.0
    mommot = 0.0


    return thrust, deps, spimp, mommot


def aeroForces(M, alfa, deltaf, cd, cl, cm, v, sup, rho, leng, mstart, mass, m10, xcg0, xcgf, pref):

    mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0]
    bodyFlap = [-20, -10, 0, 10, 20, 30]

    def limCalc(array, value):
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

    def coefCalc(coeff, m, alfa, deltaf):
        if m > mach[-1]:
            m = mach[-1]
        elif m < mach[0]:
            m = mach[0]
        if alfa > angAttack[-1]:
            alfa = angAttack[-1]
        elif alfa < angAttack[0]:
            alfa = angAttack[0]
        if deltaf > bodyFlap[-1]:
            deltaf = bodyFlap[-1]
        elif deltaf < bodyFlap[0]:
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
        c1 = coeffinf[0] + (alfa - angAttack[ia]) * ((coeffinf[1] - coeffinf[0])/(angAttack[sa] - angAttack[ia]))
        c2 = coeffsup[0] + (alfa - angAttack[ia]) * ((coeffsup[1] - coeffsup[0])/(angAttack[sa] - angAttack[ia]))
        #c1old = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinf)
        #c2old = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsup)
        #coeffd1old = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])
        coeffd1 = c1 + (deltaf - bodyFlap[idf]) * ((c2 - c1)/(bodyFlap[sdf] - bodyFlap[idf]))
        '''interpolation on the first table between angle of attack and deflection'''
        rowinf2b = cnew2[ia][:]
        rowsup2b = cnew2[sa][:]
        coeffinfb = [rowinf2b[idf], rowsup2b[idf]]
        coeffsupb = [rowinf2b[sdf], rowsup2b[sdf]]
        #c1oldb = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinfb)
        #c2oldb = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsupb)
        c1b = coeffinfb[0] + (alfa - angAttack[ia]) * ((coeffinfb[1] - coeffinfb[0]) / (angAttack[sa] - angAttack[ia]))
        c2b = coeffsupb[0] + (alfa - angAttack[ia]) * ((coeffsupb[1] - coeffsupb[0]) / (angAttack[sa] - angAttack[ia]))
        #coeffd2old = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])
        coeffd2 = c1b + (deltaf - bodyFlap[idf]) * ((c2b - c1b) / (bodyFlap[sdf] - bodyFlap[idf]))
        '''interpolation on the moments to obtain final coefficient'''
        #coeffFinalold = np.interp(m, [mach[im], mach[sm]], [coeffd1, coeffd2])
        coeffFinal = coeffd1 + (m - mach[im]) * ((coeffd2 - coeffd1) / (mach[sm] - mach[im]))
        return coeffFinal

    alfag = np.rad2deg(alfa)
    deltafg = np.rad2deg(deltaf)
    cL = coefCalc(cl, M, alfag, deltafg)
    cD = coefCalc(cd, M, alfag, deltafg)
    l = 0.5 * (v ** 2) * sup * rho * cL
    d = 0.5 * (v ** 2) * sup * rho * cD
    xcg = leng * (((xcgf - xcg0) / (m10 - mstart)) * (mass - mstart) + xcg0)
    Dx = xcg - pref
    cM1 = coefCalc(cm, M, alfag, deltafg)
    cM = cM1 + cL * (Dx / leng) * np.cos(alfa) + cD * (Dx / leng) * np.sin(alfa)
    mom = 0.5 * (v ** 2) * sup * leng * rho * cM
    if np.isnan(l) == True:
        print("L is nan")
    if np.isnan(d) == True:
        print("D is nan")
    if np.isinf(l) == True:
        print("L is inf")
        print(v, rho)
    if np.isinf(d) == True:
        print("D is inf")
        print(v, rho)
    return l, d, mom


def aeroForcesMod(M, alfa, deltaf, cd, cl, cm, v, sup, rho, leng, mstart, mass, m10, xcg0, xcgf, pref, cl_interp, cd_interp, cm_interp):

    alfag = np.rad2deg(alfa)
    deltafg = np.rad2deg(deltaf)
    if M>20.0:
        M = 20.0
    elif M<0.0:
        M=0.0
    if alfag>40.0:
        alfag = 40.0
    elif alfag<-2.0:
        alfag = -2.0
    if deltafg>30.0:
        deltafg = 30.0
    elif deltafg<-20.0:
        deltafg = 20.0
    cL = cl_interp([M, alfag, deltafg])
    cD = cd_interp([M, alfag, deltafg])

    l = 0.5 * (v ** 2) * sup * rho * cL
    d = 0.5 * (v ** 2) * sup * rho * cD
    xcg = leng * (((xcgf - xcg0) / (m10 - mstart)) * (mass - mstart) + xcg0)
    Dx = xcg - pref
    cM1 = cm_interp([M, alfag, deltafg])
    cM = cM1 + cL * (Dx / leng) * np.cos(alfa) + cD * (Dx / leng) * np.sin(alfa)
    mom = 0.5 * (v ** 2) * sup * leng * rho * cM
    #if np.isnan(l) == True:
     #   print("L is nan")
    #if np.isnan(d) == True:
     #   print("D is nan")
    #if np.isinf(l) == True:
     #   print("L is inf")
      #  print(v, rho)
    #if np.isinf(d) == True:
     #   print("D is inf")
      #  print(v, rho)
    return l, d, mom


def thrust(presamb, mass, presv, spimpv, delta, tau, slpres, wlo, we, lref, xcgf, xcg0):
    nimp = 17
    nmot = 1
    # thrmax = nmot * (5.8E+6 + 14.89 * slpres - 11.16 * presamb)
    thrx = nmot * (5.8e6 + 14.89 * slpres - 11.16 * presamb) * delta
    if presamb > slpres:
        presamb = slpres
        spimp = spimpv[-1]
    elif presamb <= slpres:
        for i in range(nimp):
            if presv[i] >= presamb:
                spimp = np.interp(presamb, [presv[i - 1], presv[i]], [spimpv[i - 1], spimpv[i]])
                break
    xcg = ((xcgf - xcg0) / (we - wlo) * (mass - wlo) + xcg0) * lref

    dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * slpres - presamb)

    mommot = tau * dthr

    thrz = -tau * (2.5E+6 - 22 * slpres + 9.92 * presamb)

    thrust = np.sqrt(thrx ** 2 + thrz ** 2)

    deps = np.arctan(thrz / thrx)

    if np.isnan(thrust) == True:
        print("T is nan")
    if np.isnan(deps) == True:
        print("deps is nan")
    if np.isinf(thrust) == True:
        print("T is inf")
    if np.isinf(deps) == True:
        print("deps is inf")

    return thrust, deps, spimp, mommot


'''function to read data from txt file'''


def fileRead(filename):
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


def fileReadOr(filename):
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


def isaMulti(altitude, pstart, g0, r):
    a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
    a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
    hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
    h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
    tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
    pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
    tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
    tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
    t0 = 288.15
    p0=pstart
    prevh = 0.0
    R = 287.00
    m0 = 28.9644
    Rs = 8314.32
    temperature = []
    pressure = []
    tempm = []
    density = []
    csound = []
    def cal(ps, ts, av, h0, h1):
        if av != 0:
            t1 = ts + av * (h1 - h0)
            p1 = ps * (t1 / ts) ** (-g0 / av / R)
        else:
            t1 = ts
            p1 = ps * np.exp(-g0 / R / ts * (h1 - h0))
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

    for alt in altitude:
        if alt < 0:
            #print("h < 0", alt)
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

                if alt <= hv[i]:
                    t, p = cal(p0, t0, a[i], prevh, alt)
                    d = p / (R * t)
                    c = np.sqrt(1.4 * R * t)
                    density.append(d)
                    csound.append(c)
                    temperature.append(t)
                    pressure.append(p)
                    tempm.append(t)
                    t0=288.15
                    p0=pstart
                    prevh=0
                    break
                else:

                    t0, p0 = cal(p0, t0, a[i], prevh, hv[i])
                    prevh = hv[i]

        elif 90000 <= alt <= 190000:
            t, p, tpm = atm90(a90, alt, h90, tcoeff1, pcoeff, tcoeff2, tmcoeff)
            temperature.append(t)
            pressure.append(p)
            tempm.append(tpm)
            d = p / (R * tpm)
            c = np.sqrt(1.4 * R * tpm)
            density.append(d)
            csound.append(c)
        elif alt > 190000:
            #print("h > 190 km", alt)
            zb = h90[6]
            z = h90[-1]
            b = zb - tcoeff1[6] / a90[6]
            t = tcoeff1[6] + (tcoeff2[6] * (z - zb)) / 1000
            tm = tmcoeff[6] + a90[6] * (z - zb) / 1000
            add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
            add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
            p = pcoeff[6] * np.exp(-m0 / (a90[6] * Rs) * g0 * r ** 2 * (add1 - add2))
            temperature.append(t)
            pressure.append(p)
            tempm.append(tm)
            d = p / (R * tm)
            c = np.sqrt(1.4 * R * tm)
            density.append(d)
            csound.append(c)

    return pressure, density, csound


def aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, sup, rho, leng, mstart, mass, m10, xcg0, xcgf, pref, npoint):

    mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0]
    bodyFlap = [-20, -10, 0, 10, 20, 30]
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
        s = array.index(sup)
        i = array.index(inf)
        return i, s

    def coefCalc(coeff, m, alfa, deltaf):
        if m > mach[-1]:
            m = mach[-1]
        elif m < mach[0]:
            m = mach[0]
        if alfa > angAttack[-1]:
            alfa = angAttack[-1]
        elif alfa < angAttack[0]:
            alfa = angAttack[0]
        if deltaf > bodyFlap[-1]:
            deltaf = bodyFlap[-1]
        elif deltaf < bodyFlap[0]:
            deltaf = bodyFlap[0]

        (im, sm), (ia, sa), (idf, sdf) = map(limCalc, (mach, angAttack, bodyFlap), (m, alfa, deltaf))
        #im, sm = limCalc(mach, m)  # moments boundaries and determination of the 2 needed tables
        # ia, sa = limCalc(angAttack, alfa)  # angle of attack boundaries
        # idf, sdf = limCalc(bodyFlap, deltaf)  # deflection angle boundaries
        cnew1 = coeff[17 * im: 17 * im + angAttack.__len__()][:]
        cnew2 = coeff[17 * sm: 17 * sm + angAttack.__len__()][:]


        '''interpolation on the first table between angle of attack and deflection'''
        rowinf1 = cnew1[ia][:]
        rowsup1 = cnew1[sa][:]
        coeffinf = [rowinf1[idf], rowsup1[idf]]
        coeffsup = [rowinf1[sdf], rowsup1[sdf]]
        c1 = coeffinf[0] + (alfa - angAttack[ia]) * ((coeffinf[1] - coeffinf[0])/(angAttack[sa] - angAttack[ia]))
        c2 = coeffsup[0] + (alfa - angAttack[ia]) * ((coeffsup[1] - coeffsup[0])/(angAttack[sa] - angAttack[ia]))
        #c1old = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinf)
        #c2old = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsup)
        #coeffd1old = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])
        coeffd1 = c1 + (deltaf - bodyFlap[idf]) * ((c2 - c1)/(bodyFlap[sdf] - bodyFlap[idf]))
        '''interpolation on the first table between angle of attack and deflection'''
        rowinf2b = cnew2[ia][:]
        rowsup2b = cnew2[sa][:]
        coeffinfb = [rowinf2b[idf], rowsup2b[idf]]
        coeffsupb = [rowinf2b[sdf], rowsup2b[sdf]]
        #c1oldb = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinfb)
        #c2oldb = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsupb)
        c1b = coeffinfb[0] + (alfa - angAttack[ia]) * ((coeffinfb[1] - coeffinfb[0]) / (angAttack[sa] - angAttack[ia]))
        c2b = coeffsupb[0] + (alfa - angAttack[ia]) * ((coeffsupb[1] - coeffsupb[0]) / (angAttack[sa] - angAttack[ia]))
        #coeffd2old = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])
        coeffd2 = c1b + (deltaf - bodyFlap[idf]) * ((c2b - c1b) / (bodyFlap[sdf] - bodyFlap[idf]))
        '''interpolation on the moments to obtain final coefficient'''
        #coeffFinalold = np.interp(m, [mach[im], mach[sm]], [coeffd1, coeffd2])
        coeffFinal = coeffd1 + (m - mach[im]) * ((coeffd2 - coeffd1) / (mach[sm] - mach[im]))
        return coeffFinal


    alfag = np.rad2deg(alfa)
    deltafg = np.rad2deg(deltaf)
    L = []
    D = []
    Mom = []
    for i in range(npoint):
        cL, cD, cM1 = map(coefCalc, (cl, cd, cm), (M[i], M[i], M[i]), (alfag[i], alfag[i], alfag[i]), (deltafg[i], deltafg[i], deltafg[i]))
        #cL = coefCalc(cl, M[i], alfag[i], deltafg[i])
        #cD = coefCalc(cd, M[i], alfag[i], deltafg[i])
        # cM1 = coefCalc(cm, M[i], alfag[i], deltafg[i])
        l = 0.5 * (v[i] ** 2) * sup * rho[i] * cL
        d = 0.5 * (v[i] ** 2) * sup * rho[i] * cD
        xcg = leng * (((xcgf - xcg0) / (m10 - mstart)) * (mass[i] - mstart) + xcg0)
        Dx = xcg - pref
        cM = cM1 + cL * (Dx / leng) * np.cos(alfa[i]) + cD * (Dx / leng) * np.sin(alfa[i])
        mom = 0.5 * (v[i] ** 2) * sup * leng * rho[i] * cM
        L.append(l)
        D.append(d)
        Mom.append(mom)
        if np.isnan(l) == True:
            print("L multi is nan")
        if np.isnan(d) == True:
            print("D multi is nan")
        if np.isinf(l) == True:
            print("L multi is inf")
        if np.isinf(d) == True:
            print("D multi is inf")
    return L, D, Mom



def thrustMulti(presamb, mass, presv, spimpv, delta, tau, npoint, slpres, wlo, we, lref, xcgf, xcg0):
    nimp = 17
    nmot = 1
    Thrust = []
    Deps = []
    Simp = []
    Mom = []
    # thrmax = nmot * (5.8E+6 + 14.89 * slpres - 11.16 * presamb)
    for j in range(npoint):
        thrx = nmot * (5.8e6 + 14.89 * slpres - 11.16 * presamb[j]) * delta[j]
        if presamb[j] >= slpres:
            spimp = spimpv[-1]
            presamb[j] = slpres
        elif presamb[j] < slpres:
            for i in range(nimp):
                if presv[i] >= presamb[j]:
                    spimp = spimpv[i-1] + (presamb[j] - presv[i - 1]) * ((spimpv[i] - spimpv[i-1])/(presv[i] - presv[i-1]))
                    break
        xcg = ((xcgf - xcg0) / (we - wlo) * (mass[j] - wlo) + xcg0) * lref

        dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7*slpres - presamb[j])

        mommot = tau[j] * dthr
        thrz = -tau[j] * (2.5e6 - 22 * slpres + 9.92 * presamb[j])
        thrust = np.sqrt(thrx ** 2 + thrz ** 2)

        deps = np.arctan(thrz / thrx)

        Thrust.append(thrust)
        Deps.append(deps)
        Simp.append(spimp)
        Mom.append(mommot)
        if np.isnan(thrust) == True:
            print("T is nan")
        if np.isnan(deps) == True:
            print("deps is nan")
        if np.isinf(thrust) == True:
            print("T is inf")
        if np.isinf(deps) == True:
            print("deps is inf")
    return Thrust, Deps, Simp, Mom


def vass(states, controls, dyn, omega):
    Re = 6371000
    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3]
    lam = states[4]
    h = states[5]

    vv = np.array((-v * np.cos(gamma) * np.cos(chi),
                   v * np.cos(gamma) * np.sin(chi),
                   -v * np.sin(gamma)))
    vv[0] = vv[0] + omega * np.cos(lam) * (Re + h)
    vela2 = np.sqrt(vv[0] ** 2 + vv[1] ** 2 + vv[2] ** 2)
    if vv[0] <= 0.0:
        if abs(vv[0]) >= abs(vv[1]):
            chiass = np.arctan(abs(vv[1] / vv[0]))
            if vv[1] < 0.0:
                chiass = -chiass
        elif abs(vv[0]) < abs(vv[1]):
            chiass = np.pi*0.5 - np.arctan(abs(vv[0] / vv[1]))
            if vv[1] < 0.0:
                chiass = -chiass
    elif vv[0] > 0.0:
        if abs(vv[0]) >= abs(vv[1]):
            chiass = np.pi - np.arctan((abs(vv[1]/vv[0])))
            if vv[1] < 0.0:
                chiass = - chiass
        elif abs(vv[0]) < abs(vv[1]):
            chiass = np.pi * 0.5 + np.arctan(abs(vv[0] / vv[1]))
            if vv[1] < 0.0:
                chiass = -chiass

    x = np.array(((Re + h) * np.cos(lam) * np.cos(teta),
                  (Re + h) * np.cos(lam) * np.sin(teta),
                  (Re + h) * np.sin(lam)))

    dx = dyn(states, controls)
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

    vela = np.sqrt(vtot[0] ** 2 + vtot[1] ** 2 + vtot[2] ** 2)

    return vela, chiass, vela2



