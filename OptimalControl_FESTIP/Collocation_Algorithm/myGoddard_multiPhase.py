from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics


class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = 5.2  # deg latitude
        self.longstart = -52.775  # deg longitude
        self.chistart = 113  # deg flight direction
        self.incl = 51.6  # deg orbit inclination
        self.gammastart = 89.9  # deg
        self.M0 = 450400  # kg  starting mass
        self.g0 = 9.80665  # m/s2
        self.gIsp = self.g0 * 455 # g0 * Isp max
        self.omega = 7.2921159e-5
        self.MaxQ = 40000  # Pa
        self.MaxAx = 30  # m/s2
        self.MaxAz = 15  # m/s2
        self.Htarget = 400000  # m target height after hohmann transfer
        self.wingSurf = 500.0  # m2
        self.lRef = 34.0  # m
        self.k = 5000  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 180000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(np.deg2rad(self.incl)) / np.cos(np.deg2rad(self.latstart)))

    @staticmethod
    def isa(altitude, pstart, g0, r):
        t0 = 288.15
        p0 = pstart
        prevh = 0.0
        R = 287.00
        m0 = 28.9644
        Rs = 8314.32
        m0 = 28.9644
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
                        t0 = 288.15
                        p0 = pstart
                        prevh = 0
                        break
                    else:

                        t0, p0 = cal(p0, t0, a[i], prevh, hv[i])
                        prevh = hv[i]

            elif 90000 <= alt <= 190000:
                t, p, tpm = atm90(a90, alt, h90, tcoeff1, pcoeff, tcoeff2, tmcoeff)
                temperature.append(t)
                pressure.append(p)
                tempm.append(tpm)
                d = p / (R * t)
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
                d = p / (R * t)
                c = np.sqrt(1.4 * R * tm)
                density.append(d)
                csound.append(c)

        return pressure, density, csound

    @staticmethod
    def aeroForces(M, alfa, deltaf, cd, cl, cm, v, sup, rho, leng, mstart, mass, m10, xcg0, xcgf, pref, npoint):
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
            c1 = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinf)
            c2 = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsup)
            coeffd1 = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])

            '''interpolation on the first table between angle of attack and deflection'''
            rowinf2 = cnew2[ia][:]
            rowsup2 = cnew2[sa][:]
            coeffinf = [rowinf2[idf], rowsup2[idf]]
            coeffsup = [rowinf2[sdf], rowsup2[sdf]]
            c1 = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffinf)
            c2 = np.interp(alfa, [angAttack[ia], angAttack[sa]], coeffsup)
            coeffd2 = np.interp(deltaf, [bodyFlap[idf], bodyFlap[sdf]], [c1, c2])

            '''interpolation on the moments to obtain final coefficient'''
            coeffFinal = np.interp(m, [mach[im], mach[sm]], [coeffd1, coeffd2])

            return coeffFinal

        alfag = np.rad2deg(alfa)
        deltafg = np.rad2deg(deltaf)
        L = []
        D = []
        Mom = []
        for i in range(npoint):
            cL = coefCalc(cl, M[i], alfag[i], deltafg[i])
            cD = coefCalc(cd, M[i], alfag[i], deltafg[i])
            l = 0.5 * (v[i] ** 2) * sup * rho[i] * cL
            d = 0.5 * (v[i] ** 2) * sup * rho[i] * cD
            xcg = leng * (((xcgf - xcg0) / (m10 - mstart)) * (mass[i] - mstart) + xcg0)
            Dx = xcg - pref
            cM1 = coefCalc(cm, M[i], alfag[i], deltafg[i])
            cM = cM1 + cL * (Dx / leng) * np.cos(alfa[i]) + cD * (Dx / leng) * np.cos(alfa[i])
            mom = 0.5 * (v[i] ** 2) * sup * leng * rho[i] * cM
            L.append(l)
            D.append(d)
            Mom.append(mom)

        return L, D, Mom

    @staticmethod
    def thrust(presamb, mass, presv, spimpv, delta, tau, npoint, slpres, wlo, we, lref, xcgf, xcg0):
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
            elif presamb[j] < slpres:
                for i in range(nimp):
                    if presv[i] >= presamb[j]:
                        spimp = np.interp(presamb[j], [presv[i - 1], presv[i]], [spimpv[i - 1], spimpv[i]])
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
        return Thrust, Deps, Simp, Mom


def dynamics(prob, obj, section):
    v = prob.states(0, section)
    chi = prob.states(1, section)
    gamma = prob.states(2, section)
    teta = prob.states(3, section)
    lam = prob.states(4, section)
    h = prob.states(5, section)
    m = prob.states(6, section)

    alfa = prob.controls(0, section)
    delta = prob.controls(1, section)
    deltaf = prob.controls(2, section)
    tau = prob.controls(3, section)
    #mu = prob.controls(4, section)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)

    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, prob.nodes[section])
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[section], obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)
    T = np.asarray(T, dtype=np.float64)
    isp = np.asarray(isp, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)
    MomT = np.asarray(MomT, dtype=np.float64)

    MomTot = MomA + MomT

    eps = Deps + alfa
    g0 = obj.g0
    g = []

    for alt in h:
        if alt == 0:
            g.append(g0)
        else:
            g.append(obj.g0 * (obj.Re / (obj.Re + alt)) ** 2)  # [m/s2]
    g = np.asarray(g, dtype=np.float64)

    dx = Dynamics(prob, section)
    dx[0] = ((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega**2) * (obj.Re + h) * np.cos(lam) * \
            (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi))

    dx[1] = ((T * np.sin(eps) + L) / (m * v * np.cos(gamma))) - np.cos(gamma) * np.cos(chi) * np.tan(lam) \
            * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam))\
            - (obj.omega ** 2) * ((obj.Re + h)/(v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi)

    dx[2] = ((T * np.sin(eps) + L) / (m * v)) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
            * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
            (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma))

    dx[3] = -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam)))
    dx[4] = np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h))
    dx[5] = v * np.sin(gamma)
    dx[6] = -T / (g0 * isp)

    return dx()


def equality(prob, obj):
    v = prob.states_all_section(0)
    chi = prob.states_all_section(1)
    gamma = prob.states_all_section(2)
    teta = prob.states_all_section(3)
    lam = prob.states_all_section(4)
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)

    alfa = prob.controls_all_section(0)
    delta = prob.controls_all_section(1)
    deltaf = prob.controls_all_section(2)
    tau = prob.controls_all_section(3)
    #mu = prob.controls_all_section(4)

    tf = prob.time_final(-1)

    # knotting condition

    v1 = prob.states(0, 0)
    v2 = prob.states(0, 1)
    #v3 = prob.states(0, 2)
    chi1 = prob.states(1, 0)
    chi2 = prob.states(1, 1)
    #chi3 = prob.states(1, 2)
    gamma1 = prob.states(2, 0)
    gamma2 = prob.states(2, 1)
    #gamma3 = prob.states(2, 2)
    teta1 = prob.states(3, 0)
    teta2 = prob.states(3, 1)
    #teta3 = prob.states(3, 2)
    lam1 = prob.states(4, 0)
    lam2 = prob.states(4, 1)
    #lam3 = prob.states(4, 2)
    h1 = prob.states(5, 0)
    h2 = prob.states(5, 1)
    #h3 = prob.states(5, 2)
    m1 = prob.states(6, 0)
    m2 = prob.states(6, 1)
    #m3 = prob.states(6, 2)

    alfa1 = prob.controls(0, 0)
    alfa2 = prob.controls(0, 1)
    #alfa3 = prob.controls(0, 2)
    delta1 = prob.controls(1, 0)
    delta2 = prob.controls(1, 1)
    #delta3 = prob.controls(1, 2)
    deltaf1 = prob.controls(2, 0)
    deltaf2 = prob.controls(2, 1)
    #deltaf3 = prob.controls(2, 2)
    tau1 = prob.controls(3, 0)
    tau2 = prob.controls(3, 1)
    #tau3 = prob.controls(3, 2)

    unit_v = prob.unit_states[0][0]
    unit_chi = prob.unit_states[0][1]
    unit_gamma = prob.unit_states[0][2]
    unit_teta = prob.unit_states[0][3]
    unit_lam = prob.unit_states[0][4]
    unit_h = prob.unit_states[0][5]
    unit_m = prob.unit_states[0][6]
    unit_alfa = prob.unit_controls[0][0]
    unit_delta = prob.unit_controls[0][1]
    unit_deltaf = prob.unit_controls[0][2]
    unit_tau = prob.unit_controls[0][3]

    vt = np.sqrt(obj.GMe / (obj.Re + h[-1]))
    Press1, rho1, c1 = obj.isa(h1, obj.psl, obj.g0, obj.Re)
    #Press2, rho2, c2 = obj.isa(h2, obj.psl, obj.g0, obj.Re)
    #Press, rho, c3 = obj.isa(h3, obj.psl, obj.g0, obj.Re)
    c1 = np.asarray(c1, dtype=np.float64)
    #c2 = np.asarray(c2, dtype=np.float64)
    #c3 = np.asarray(c3, dtype=np.float64)
    M1 = v1 / c1
    #M2 = v2 / c2
    #M3 = v3 / c3

    result = Condition()

    # event condition
    result.equal(v[0], 1e-7, unit=prob.unit_states[0][0])
    result.equal(chi[0], np.deg2rad(obj.chistart), unit=prob.unit_states[0][1])
    result.equal(gamma[0], np.deg2rad(obj.gammastart), unit=prob.unit_states[0][2])
    result.equal(teta[0], np.deg2rad(obj.longstart), unit=prob.unit_states[0][3])
    result.equal(lam[0], np.deg2rad(obj.latstart), unit=prob.unit_states[0][4])
    result.equal(h[0], 1e-7, unit=prob.unit_states[0][5])
    result.equal(m[0], obj.M0, unit=prob.unit_states[0][6])
    result.equal(alfa[0], 0.0, unit=prob.unit_controls[0][0])
    result.equal(delta[0], 1.0, unit=prob.unit_controls[0][1])
    result.equal(deltaf[0], 0.0, unit=prob.unit_controls[0][2])
    result.equal(tau[0], 0.0, unit=prob.unit_controls[0][3])
    #result.equal(mu[0], 0.0, unit=prob.unit_controls[0][4])
    result.equal(v[-1], vt, unit=prob.unit_states[0][0])
    result.equal(chi[-1], obj.chi_fin, unit=prob.unit_states[0][1])
    result.equal(gamma[-1], 0.0, unit=prob.unit_states[0][2])
    result.equal(M1[-1], 1)
    #result.equal(M2[0], 1)



    #result.equal(deltaf[-1], 0.0, unit=unit_deltaf)
    #result.equal(mu[-1], 0.0, unit=prob.unit_controls[0][4])


    result.equal(v2[0], v1[-1], unit=unit_v)
    #result.equal(v3[0], v2[-1], unit=unit_v)
    result.equal(chi2[0], chi1[-1], unit=unit_chi)
    #result.equal(chi3[0], chi2[-1], unit=unit_chi)
    result.equal(gamma2[0], gamma1[-1], unit=unit_gamma)
    #result.equal(gamma3[0], gamma2[-1], unit=unit_gamma)
    result.equal(teta2[0], teta1[-1], unit=unit_teta)
    #result.equal(teta3[0], teta2[-1], unit=unit_teta)
    result.equal(lam2[0], lam1[-1], unit=unit_lam)
    #result.equal(lam3[0], lam2[-1], unit=unit_lam)
    result.equal(h2[0], h1[-1], unit=unit_h)
    #result.equal(h3[0], h2[-1], unit=unit_h)
    result.equal(m2[0], m1[-1], unit=unit_m)
    #result.equal(m3[0], m2[-1], unit=unit_m)

    result.equal(alfa2[0], alfa1[-1], unit=unit_alfa)
    #result.equal(alfa3[0], alfa2[-1], unit=unit_alfa)
    result.equal(delta2[0], delta1[-1], unit=unit_delta)
    #result.equal(delta3[0], delta2[-1], unit=unit_delta)
    result.equal(deltaf2[0], deltaf1[-1], unit=unit_deltaf)
    #result.equal(deltaf3[0], deltaf2[-1], unit=unit_deltaf)
    result.equal(tau2[0], tau1[-1], unit=unit_tau)
    #result.equal(tau3[0], tau2[-1], unit=unit_tau)
    #result.equal(M2[0], M1[-1])
    #result.equal(M3[0], M2[-1], unit=20)

    return result()


def inequality(prob, obj):
    v = prob.states_all_section(0)
    chi = prob.states_all_section(1)
    gamma = prob.states_all_section(2)
    teta = prob.states_all_section(3)
    lam = prob.states_all_section(4)
    h = prob.states_all_section(5)
    m = prob.states_all_section(6)

    alfa = prob.controls_all_section(0)
    delta = prob.controls_all_section(1)
    deltaf = prob.controls_all_section(2)
    tau = prob.controls_all_section(3)
    #mu = prob.controls_all_section(4)

    tf = prob.time_final(-1)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    M = v / c

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, ntot)

    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, ntot, obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)
    T = np.asarray(T, dtype=np.float64)
    #isp = np.asarray(isp, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)
    MomT = np.asarray(MomT, dtype=np.float64)

    MomTot = MomA + MomT
    #MomTotA = abs(MomTot)


    # dynamic pressure

    q = 0.5 * rho * (v ** 2)
    # accelerations
    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))



    result = Condition()

    # lower bounds
    result.lower_bound(v, 1e-7, unit=prob.unit_states[0][0])
    result.lower_bound(chi, np.deg2rad(90), unit=prob.unit_states[0][1])
    result.lower_bound(lam, np.deg2rad(-obj.incl), unit=prob.unit_states[0][4])
    result.lower_bound(h, 1e-7, unit=prob.unit_states[0][5])
    result.lower_bound(m[1:], obj.m10, unit=prob.unit_states[0][6])
    result.lower_bound(h[-1], 80000, unit=prob.unit_states[0][5])
    result.lower_bound(alfa, np.deg2rad(-2), unit=prob.unit_controls[0][0])
    result.lower_bound(delta, 0.1, unit=prob.unit_controls[0][1])
    result.lower_bound(deltaf, np.deg2rad(-20), unit=prob.unit_controls[0][2])
    result.lower_bound(tau, 0.0, unit=prob.unit_controls[0][3])
    #result.lower_bound(mu, np.deg2rad(-90), unit=prob.unit_controls[0][4])
    result.lower_bound(MomTot, -obj.k)
    #result.lower_bound(q, 0.0)
    #result.lower_bound(ax, 0.0)
    #result.lower_bound(az, 0.0)
    result.lower_bound(mf, obj.m10)



    # upper bounds
    result.upper_bound(chi, np.deg2rad(270), unit=prob.unit_states[0][1])
    result.upper_bound(lam, np.deg2rad(obj.incl), unit=prob.unit_states[0][4])
    result.upper_bound(h, obj.Hini, unit=prob.unit_states[0][5])
    result.upper_bound(m, obj.M0, unit=prob.unit_states[0][6])
    result.upper_bound(alfa, np.deg2rad(40), unit=prob.unit_controls[0][0])
    result.upper_bound(delta, 1.0, unit=prob.unit_controls[0][1])
    result.upper_bound(deltaf, np.deg2rad(30), unit=prob.unit_controls[0][2])
    result.upper_bound(tau, 1, unit=prob.unit_controls[0][3])
    #result.upper_bound(mu, np.deg2rad(90), unit=prob.unit_controls[0][4])
    result.upper_bound(MomTot, obj.k)
    result.upper_bound(q, obj.MaxQ)
    result.upper_bound(ax, obj.MaxAx)
    result.upper_bound(az, obj.MaxAz)
    result.upper_bound(mf, obj.M0)


    return result()


def cost(prob, obj):
    #v = prob.states_all_section(0)
    #chi = prob.states_all_section(1)
    #gamma = prob.states_all_section(2)
    #teta = prob.states_all_section(3)
    #lam = prob.states_all_section(4)

    #alfa = prob.controls_all_section(0)
    #delta = prob.controls_all_section(1)
    #deltaf = prob.controls_all_section(2)
    #tau = prob.controls_all_section(3)
    #mu = prob.controls_all_section(4)

    h = prob.states(5, 1)
    m = prob.states(6, 1)
    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))

    return -mf[-1] / prob.unit_states[1][6]


def cost_derivative(prob, obj):
    jac = Condition(prob.number_of_variables)
    index_m0 = prob.index_states(6, 0, 0)
    index_mf = prob.index_states(6, 0, -1)
    m = prob.states_all_section(6)
    jac.change_value(index_mf, -1.0)
    return jac()

# ==============


plt.ion()

# Program starting point!!!
time_init = [0.0, 50.0, 400.0]
n = [5, 15]
num_states = [7, 7]
num_controls = [4, 4]
max_iteration = 500
ntot=sum(n)


a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
mach = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
angAttack = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0]
bodyFlap = [-20, -10, 0, 10, 20, 30]

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


cl = fileReadOr("clfile.txt")
cd = fileReadOr("cdfile.txt")
cm = fileRead("cmMFormat.txt")
cl = np.asarray(cl)
cd = np.asarray(cd)
cm = np.asarray(cm)

with open("impulse.dat") as f:
    impulse = []
    for line in f:
        line = line.split()
        if line:
            line = [float(i) for i in line]
            impulse.append(line)

f.close()

presv = []
spimpv = []

for i in range(len(impulse)):
    presv.append(impulse[i][0])
    spimpv.append(impulse[i][1])

presv = np.asarray(presv)
spimpv = np.asarray(spimpv)

flag_savefig = True
savefig_file = "Test_Plot_Goddard_multi_CmAllSmooth_"


# -------------
# set OpenGoddard class for algorithm determination
prob = Problem(time_init, n, num_states, num_controls, max_iteration)

# -------------
# create instance of operating object
obj = Spaceplane()

# max value used to divide?
unit_v = 8000
unit_chi = np.deg2rad(270)
unit_gamma = np.deg2rad(obj.gammastart)
unit_teta = np.deg2rad(obj.longstart)
unit_lam = np.deg2rad(obj.incl)
unit_h = obj.Hini
unit_m = obj.M0
unit_t = 500
unit_alfa = np.deg2rad(40)
unit_delta = 1
unit_deltaf = np.deg2rad(30)
unit_tau = 1
#unit_mu = np.deg2rad(90)
prob.set_unit_states_all_section(0, unit_v)
prob.set_unit_states_all_section(1, unit_chi)
prob.set_unit_states_all_section(2, unit_gamma)
prob.set_unit_states_all_section(3, unit_teta)
prob.set_unit_states_all_section(4, unit_lam)
prob.set_unit_states_all_section(5, unit_h)
prob.set_unit_states_all_section(6, unit_m)
prob.set_unit_controls_all_section(0, unit_alfa)
prob.set_unit_controls_all_section(1, unit_delta)
prob.set_unit_controls_all_section(2, unit_deltaf)
prob.set_unit_controls_all_section(3, unit_tau)
#prob.set_unit_controls_all_section(4, unit_mu)
prob.set_unit_time(unit_t)


# =================
# initial parameters guess
# velocity
v_init = Guess.cubic(prob.time_all_section, 1e-5, 0.0, obj.Vtarget, 0.0)
chi_init = Guess.cubic(prob.time_all_section, np.deg2rad(obj.chistart), 0.0, obj.chi_fin, 0.0)
gamma_init = Guess.cubic(prob.time_all_section, np.deg2rad(obj.gammastart), 0.0, 0.0, 0.0)
teta_init = Guess.constant(prob.time_all_section, np.deg2rad(obj.longstart))
lam_init = Guess.constant(prob.time_all_section, np.deg2rad(obj.latstart))
h_init = Guess.cubic(prob.time_all_section, 1e-5, 0.0, obj.Hini, 0.0)
m_init = Guess.cubic(prob.time_all_section, obj.M0, 0.0, obj.m10, 0.0)

alfa_init = Guess.zeros(prob.time_all_section)
delta_init = Guess.cubic(prob.time_all_section, 1.0, 0.0, 0.1, 0.0)
deltaf_init = Guess.zeros(prob.time_all_section)
tau_init = Guess.zeros(prob.time_all_section)
mu_init = Guess.zeros(prob.time_all_section)



# ===========
# Substitution initial value to parameter vector to be optimized
# non dimensional values (Divide by scale factor)

prob.set_states_all_section(0, v_init)
prob.set_states_all_section(1, chi_init)
prob.set_states_all_section(2, gamma_init)
prob.set_states_all_section(3, teta_init)
prob.set_states_all_section(4, lam_init)
prob.set_states_all_section(5, h_init)
prob.set_states_all_section(6, m_init)
prob.set_controls_all_section(0, alfa_init)
prob.set_controls_all_section(1, delta_init)
prob.set_controls_all_section(2, deltaf_init)
prob.set_controls_all_section(3, tau_init)
#prob.set_controls_all_section(4, mu_init)


# ========================
# Main Process
# Assign problem to SQP solver
prob.dynamics = [dynamics, dynamics]
prob.knot_states_smooth = [False]
prob.cost = cost
prob.equality = equality
prob.inequality = inequality
#prob.cost_derivative = cost_derivative



def display_func():
    #chi = prob.states_all_section(1)
    #gamma = prob.states_all_section(2)
    m = prob.states_all_section(6)
    h = prob.states_all_section(5)
    #alfa = prob.controls_all_section(0)
    #delta = prob.controls_all_section(1)
    deltaf = prob.controls_all_section(2)
    #tau = prob.controls_all_section(3)
    #mu = prob.controls_all_section(4)
    tf = prob.time_final(-1)

    #time = prob.time_update()

    # Hohmann transfer mass calculation
    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))

    '''
    plt.figure()
    plt.title("Body Flap deflection1")
    plt.plot(time, np.rad2deg(deltaf), marker="o", label="Delta f")
    for line in prob.time_knots():
        plt.axvline(line, color="k", alpha=0.5)
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")
    if flag_savefig:
        plt.savefig(savefig_file + "bdyFlap" + ".png")

    plt.show()
    '''

    print("m0          : {0:.5f}".format(m[0]))
    print("m before Ho : {0:.5f}".format(m[-1]))
    print("mf          : {0:.5f}".format(mf))
    print("altitude Hohmann starts: {0:.5f}".format(h[-1]))
    print("final time  : {0:.3f}".format(tf))


prob.solve(obj, display_func, ftol=1e-8, eps=1e-8, maxiter=70)
prob.plot()
# =====================
# Post Process
# ------------------------
# Convert parameter vector to variable

v = prob.states_all_section(0)
chi = prob.states_all_section(1)
gamma = prob.states_all_section(2)
teta = prob.states_all_section(3)
lam = prob.states_all_section(4)
h = prob.states_all_section(5)
m = prob.states_all_section(6)
alfa = prob.controls_all_section(0)
delta = prob.controls_all_section(1)
deltaf = prob.controls_all_section(2)
tau = prob.controls_all_section(3)
#mu = prob.controls_all_section(4)
time = prob.time_update()

Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)
Press = np.asarray(Press, dtype=np.float64)
rho = np.asarray(rho, dtype=np.float64)
c = np.asarray(c, dtype=np.float64)
M = v / c

L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10, obj.xcg0, obj.xcgf, obj.pref, ntot)
L = np.asarray(L, dtype=np.float64)
D = np.asarray(D, dtype=np.float64)
MomA = np.asarray(MomA, dtype=np.float64)

T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, ntot, obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)
T = np.asarray(T, dtype=np.float64)
Deps = np.asarray(Deps, dtype=np.float64)
MomT = np.asarray(MomT, dtype=np.float64)

MomTot = MomA + MomT

g0 = obj.g0
eps = Deps + alfa

# dynamic pressure

q = 0.5 * rho * (v ** 2)

# accelerations

ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

r1 = h + obj.Re
Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))

downrange = -(v**2)/g0 * np.sin(2 * chi)

res = open("resultsMulti.txt", "w")
res.write("v: " + str(v) + "\n" + "Chi: " + str(np.rad2deg(chi)) + "\n" + "Gamma: " + str(np.rad2deg(gamma)) + "\n"
          + "Teta: " + str(np.rad2deg(teta)) + "\n" + "Lambda: " + str(np.rad2deg(lam)) + "\n" + "Height: " + str(h) + "\n"
          + "Mass: " + str(m) + "\n" + "Alfa: " + str(np.rad2deg(alfa)) + "\n" + "Delta: " + str(delta) + "\n"
          + "Delta f: " + str(np.rad2deg(deltaf)) + "\n" + "Tau: " + str(tau) + "\n"
          + "Eps: " + str(np.rad2deg(eps)) + "\n" + "Lift: " + str(L) + "\n" + "Drag: " + str(D) + "\n"
          + "Thrust: " + str(T) + "\n" + "Spimp: " + str(isp) + "\n" + "c: " + str(c) + "\n" + "Mach: " + str(M) + "\n"
          + "Time: " + str(time) + "\n" + "Press: " + str(Press) + "\n" + "Dens: " + str(rho) + "\n" + "Mf: " + str(mf) + "\n"
          + "MomA: " + str(MomA) + "\n" + "MomT: " + str(MomT) + "\n" + "MomTot: " + str(MomTot))
res.close()


# ------------------------
# Visualization
plt.figure()
plt.title("Altitude profile")
plt.plot(time, h/1000, marker="o", label="Altitude")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
if flag_savefig:
    plt.savefig(savefig_file + "altitude" + ".png")

plt.figure()
plt.title("Velocity")
plt.plot(time, v, marker="o", label="V")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "velocity" + ".png")

plt.figure()
plt.title("Mass")
plt.plot(time, m, marker="o", label="Mass")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
if flag_savefig:
    plt.savefig(savefig_file + "mass" + ".png")

plt.figure()
plt.title("Acceleration")
plt.plot(time, ax, marker="o", label="Acc x")
plt.plot(time, az, marker="o", label="Acc z")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Acceleration [m/s2]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "acceleration" + ".png")

plt.figure()
plt.title("Throttle profile")
plt.plot(time, delta, marker="o", label="Delta")
plt.plot(time, tau, marker="o", label="Tau")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" % ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "Throttle" + ".png")

plt.figure()
plt.title("Angle of attack profile")
plt.plot(time, np.rad2deg(alfa), marker="o", label="Alfa")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "angAttack" + ".png")

plt.figure()
plt.title("Body Flap deflection profile")
plt.plot(time, np.rad2deg(deltaf), marker="o", label="Delta f")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "bdyFlap" + ".png")

plt.figure()
plt.title("Trajectory angles")
plt.plot(time, np.rad2deg(chi), marker="o", label="Chi")
plt.plot(time, np.rad2deg(gamma), marker="o", label="Gamma")
plt.plot(time, np.rad2deg(teta), marker="o", label="Teta")
plt.plot(time, np.rad2deg(lam), marker="o", label="Lambda")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "Angles" + ".png")

plt.figure()
plt.title("Dynamic pressure profile")
plt.plot(time, q/1000, marker="o", label="Q")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" kPa ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "dynPress" + ".png")

plt.figure()
plt.title("Moment")
plt.plot(time, MomTot / 1000, marker="o", label="Total Moment")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" kNm ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "MomTot" + ".png")

plt.figure()
plt.title("Trajectory")
plt.plot(downrange/1000, h/1000, marker="o", label="Trajectory")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("km")
plt.ylabel(" km ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "downrange" + ".png")

plt.figure()
plt.title("Mach profile")
plt.plot(time, M, marker="o", label="Mach")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "mach" + ".png")

plt.figure()
plt.title("Lift, Drag and Thrust profile")
plt.plot(time, L, marker="o", label="Lift")
plt.plot(time, D, marker="o", label="Drag")
plt.plot(time, T, marker="o", label="Thrust")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "LDT" + ".png")

plt.show()
