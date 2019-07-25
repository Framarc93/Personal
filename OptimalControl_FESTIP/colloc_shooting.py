from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics
import time
from scipy import integrate, interpolate
from functools import partial
from models import *
from mapping_functions import *
import os
import datetime
from scipy.integrate import solve_ivp
import sys
sys.path.insert(0, 'home/francesco/git_workspace/FESTIP_Work')

'''this script is a collocation algorithm on FESTIP model'''

'''it contains all the states and controls of the original model but the fact that the latitude is set at 5.2 deg as initial
condition, it gives problems. Without that, everything works fine, even with the bank angle. And the moment is inside 
the boundaries'''

'''No initial condition on alpha set for now'''

def to_new_int(t, a, b, c, d):
    ''' this function converts a value from an interval [a, b] to [c, d]
    t = value to be converted
    a = inf lim old interval
    b = sup lim old interval
    c = inf lim new interval
    d = sup lim new interval'''
    return c+((d-c)/(b-a))*(t-a)


def to_orig_int(ft, a, b, c, d):
    ''' this function converts a value from an interval [a, b] to [c, d]
        ft = value to be converted
        a = inf lim old interval
        b = sup lim old interval
        c = inf lim new interval
        d = sup lim new interval'''
    return a + (ft - c) * ((b-a)/(d-c))


start = time.time()
timestr = time.strftime("%Y%m%d-%H%M%S")
flag_savefig = True

class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
        self.chistart = np.deg2rad(113)  # deg flight direction
        self.incl = np.deg2rad(51.6)  # deg orbit inclination
        self.gammastart = np.deg2rad(89.9)  # deg
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
        self.k = 5e3  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 100000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart)) # value in radians

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
            cM = cM1 + cL * (Dx / leng) * np.cos(alfa[i]) + cD * (Dx / leng) * np.sin(alfa[i])
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
            if presamb[j] > slpres:
                presamb[j] = slpres
                spimp = spimpv[-1]
            elif presamb[j] <= slpres:
                for i in range(nimp):
                    if presv[i] >= presamb[j]:
                        spimp = np.interp(presamb[j], [presv[i - 1], presv[i]], [spimpv[i - 1], spimpv[i]])
                        break

            xcg = ((xcgf - xcg0) / (we - wlo) * (mass[j] - wlo) + xcg0) * lref

            dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7*slpres - presamb[j])

            mommot = tau[j] * dthr

            thrz = -tau[j] * (2.5e6 - 22 * slpres + 9.92 * presamb[j])
            thrust = np.sqrt(thrx ** 2 + thrz ** 2)

            if thrx == 0.0 and thrz == 0.0:
                deps = 0.0
            else:
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
    mu = prob.controls(4, section)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)

    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, prob.nodes[0])
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)
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

    dx[1] = ((T * np.sin(eps) + L) * np.sin(mu)) / (m*v*np.cos(gamma)) - np.cos(gamma) * np.cos(chi) * np.tan(lam) \
            * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam))\
            - (obj.omega ** 2) * ((obj.Re + h)/(v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi)

    dx[2] = ((T * np.sin(eps) + L) * np.cos(mu))/ (m * v) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
            * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
            (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma))

    dx[3] = -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam)))
    dx[4] = np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h))
    dx[5] = v * np.sin(gamma)
    dx[6] = -T / (g0 * isp)

    return dx()


def dynamicsVel(states, contr):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''

    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = contr[0]
    delta = contr[1]
    deltaf = contr[2]
    tau = contr[3]
    mu = contr[4]

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)
    # Press = np.asarray(Press, dtype=np.float64)
    # rho = np.asarray(rho, dtype=np.float64)
    # c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)
    # L = np.asarray(L, dtype=np.float64)
    # D = np.asarray(D, dtype=np.float64)
    # MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

    # T = np.asarray(T, dtype=np.float64)
    # isp = np.asarray(isp, dtype=np.float64)
    # Deps = np.asarray(Deps, dtype=np.float64)
    # MomT = np.asarray(MomT, dtype=np.float64)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   ((T * np.sin(eps) + L) * np.sin(mu)) / (m*v*np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                   np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
                   ((T * np.sin(eps) + L) * np.cos(mu))/ (m * v)  - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (g0 * isp)))

    return dx


def dynamicsInt(t, states, alfa_int, delta_int, deltaf_int, tau_int, mu_int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''

    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = alfa_int(t)
    delta = delta_int(t)
    deltaf = deltaf_int(t)
    tau = tau_int(t)
    mu = mu_int(t)

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)
    # Press = np.asarray(Press, dtype=np.float64)
    # rho = np.asarray(rho, dtype=np.float64)
    # c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)
    # L = np.asarray(L, dtype=np.float64)
    # D = np.asarray(D, dtype=np.float64)
    # MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

    # T = np.asarray(T, dtype=np.float64)
    # isp = np.asarray(isp, dtype=np.float64)
    # Deps = np.asarray(Deps, dtype=np.float64)
    # MomT = np.asarray(MomT, dtype=np.float64)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                   np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
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


def SingleShooting(states, controls, dyn, time, Nint):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''

    #Time = time

    x = np.zeros((Nint, Nstates))

    x[0,:] = states[0:Nstates]  # vector of intial states ready

    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(time, controls[0, :])
    delta_Int = interpolate.PchipInterpolator(time, controls[1, :])
    deltaf_Int = interpolate.PchipInterpolator(time, controls[2, :])
    tau_Int = interpolate.PchipInterpolator(time, controls[3, :])
    mu_Int = interpolate.PchipInterpolator(time, controls[4, :])

    time_new = np.linspace(0, time[-1], Nint)

    dt = (time_new[1] - time_new[0])

    t = time_new

    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int), t_span=[0, time[-1]], y0=x,
     #               t_eval=time_new, method='RK45')

    for i in range(Nint - 1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt * dyn(t[i], x[i, :], alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k1: ", k1)
        k2 = dt * dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k2: ", k2)
        k3 = dt * dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k3: ", k3)
        k4 = dt * dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


    #vres = sol.y[0, :]
    #chires = sol.y[1, :]
    #gammares = sol.y[2, :]
    #tetares = sol.y[3, :]
    #lamres = sol.y[4, :]
    #hres = sol.y[5, :]
    #mres = sol.y[6, :]
    vres = x[:, 0]
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_new)
    deltares = delta_Int(time_new)
    deltafres = deltaf_Int(time_new)
    taures = tau_Int(time_new)
    mures = mu_Int(time_new)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, mures


def SingleShootingMulti(states, controls, dyn, timestart, timeend, Nint):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''

    #Time = time

    x = np.zeros((Nint, Nstates))

    x[0,:] = states  # vector of intial states ready

    # now interpolation of controls


    time_new = np.linspace(timestart, timeend, Nint)

    dt = (time_new[1] - time_new[0])

    t = time_new

    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int), t_span=[0, time[-1]], y0=x,
     #               t_eval=time_new, method='RK45')

    for i in range(Nint - 1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt * dyn(x[i, :], controls)
        # print("k1: ", k1)
        k2 = dt * dyn(x[i, :] + k1 / 2, controls)
        # print("k2: ", k2)
        k3 = dt * dyn(x[i, :] + k2 / 2, controls)
        # print("k3: ", k3)
        k4 = dt * dyn(x[i, :] + k3, controls)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


    #vres = sol.y[0, :]
    #chires = sol.y[1, :]
    #gammares = sol.y[2, :]
    #tetares = sol.y[3, :]
    #lamres = sol.y[4, :]
    #hres = sol.y[5, :]
    #mres = sol.y[6, :]
    vres = x[:, 0]
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = controls[0]
    deltares = controls[1]
    deltafres = controls[2]
    taures = controls[3]
    mures = controls[4]

    return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, mures


def MultiShooting(states, controls, dyn, time):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in radians'''

    states_atNode = np.zeros((0))


    for i in range(n[0]-1):

        stat = states[:, i]  # orig intervals unscaled

        cont = controls[:, i]
        timestart = time[i]
        timeend = time[i+1]
        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures\
            = SingleShootingMulti(stat, cont, dynamicsVel, timestart, timeend, Nintm)

        states_atNode = np.concatenate((states_atNode, (
                    (vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]))))


    return states_atNode

def conversionStates(var, len):
    new_var = np.zeros((len*Nstates))
    i = 0
    for i in range(len):
        new_var[i * Nstates] = v_toNew(var[i * Nstates])
        new_var[i * Nstates + 1] = chi_toNew(var[i * Nstates + 1])
        new_var[i * Nstates + 2] = gamma_toNew(var[i * Nstates + 2])
        new_var[i * Nstates + 3] = teta_toNew(var[i * Nstates + 3])
        new_var[i * Nstates + 4] = lam_toNew(var[i * Nstates + 4], obj)
        new_var[i * Nstates + 5] = h_toNew(var[i * Nstates + 5])
        new_var[i * Nstates + 6] = m_toNew(var[i * Nstates + 6], obj)
    return new_var

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
    mu = prob.controls_all_section(4)
    tf = prob.time_update()
    s = np.vstack((v, chi, gamma, teta, lam, h, m))

    c = np.vstack((alfa, delta, deltaf, tau, mu))

    integration = MultiShooting(s, c, dynamicsInt, tf)


    States = np.array((v[-1], chi[-1], gamma[-1], teta[-1], lam[-1], h[-1], m[-1]))
    Controls = np.array((alfa[-1], delta[-1], deltaf[-1], tau[-1], mu[-1]))
    vt = np.sqrt(obj.GMe / (obj.Re + h[-1])) #- obj.omega*np.cos(lam[-1])*(obj.Re+h[-1])
    vtAbs, chiass, vtAbs2 = vass(States, Controls, dynamicsVel, obj.omega)

    if np.cos(obj.incl) > np.cos(lam[-1]):
        chifin = np.pi
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam[-1]))
    result = Condition()

    # event condition
    result.equal(to_new_int(v[0]/1e3, 0.0, 10, 0.0, 1.0), to_new_int(1.0/1e3, 0.0, 10, 0.0, 1.0), unit=1)
    result.equal(to_new_int(chi[0], np.deg2rad(90), np.deg2rad(270), 0.0, 1.0),
                 to_new_int(obj.chistart, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0), unit=1)
    result.equal(to_new_int(gamma[0], np.deg2rad(-2), np.deg2rad(90), 0.0, 1.0),
                 to_new_int(obj.gammastart, 0.0, np.deg2rad(90), 0.0, 1.0), unit=1)
    result.equal(to_new_int(teta[0], np.deg2rad(-90), 0.0, 0.0, 1.0),
                 to_new_int(obj.longstart, np.deg2rad(-90), 0.0, 0.0, 1.0), unit=1)
    result.equal(to_new_int(lam[0], -obj.incl, obj.incl, 0.0, 1.0),
                 to_new_int(obj.latstart, -obj.incl, obj.incl, 0.0, 1.0), unit=1)
    result.equal(to_new_int(h[0]/1e4, 0.0, 20, 0.0, 1.0), to_new_int(1/1e4, 0.0, 20, 0.0, 1.0), unit=1)
    result.equal(to_new_int(m[0], obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.M0, obj.m10, obj.M0, 0.0, 1.0), unit=1)
    result.equal(conversionStates(s[Nstates:], n[0]-1), conversionStates(integration[Nstates*n[0]-Nstates], n[0]-1), unit=1)
    result.equal(to_new_int(alfa[0], np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0),
                 to_new_int(0.0, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)
    result.equal(delta[0], 1.0, unit=1)
    result.equal(to_new_int(deltaf[0], np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0),
                 to_new_int(0.0, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)
    result.equal((tau[0]+1)/2, 0.5, unit=1)
    result.equal(to_new_int(mu[0], np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0),
                 to_new_int(0.0, np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0), unit=1)
    result.equal(to_new_int(vtAbs/1e3, 0.0, 10, 0.0, 1.0), to_new_int(vt/1e3, 0.0, 10, 0.0, 1.0), unit=1)
    result.equal(to_new_int(chiass, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0),
                 to_new_int(chifin, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0), unit=1)
    result.equal(to_new_int(gamma[-1], 0.0, np.deg2rad(90), 0.0, 1.0),
                 to_new_int(0.0, 0.0, np.deg2rad(90), 0.0, 1.0), unit=1)

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
    mu = prob.controls_all_section(4)

    tf = prob.time_final(-1)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    M = v / c

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, prob.nodes[0])

    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)
    T = np.asarray(T, dtype=np.float64)
    isp = np.asarray(isp, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)
    MomT = np.asarray(MomT, dtype=np.float64)

    MomTot = MomA + MomT
    MomTotA = abs(MomTot)


    # dynamic pressure

    q = 0.5 * rho * (v ** 2)
    # accelerations
    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0*isp[-1])))

    result = Condition()

    # lower bounds
    result.lower_bound(to_new_int(v/1e3, 0.0, 10, 0.0, 1.0), to_new_int(1e-5, 0.0, 10, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(chi, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0),
                       to_new_int(np.deg2rad(90), np.deg2rad(90), np.deg2rad(270), 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(gamma, np.deg2rad(-2), np.deg2rad(90), 0.0, 1.0),
                       to_new_int(0.0, 0.0, np.deg2rad(90), 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(teta, np.deg2rad(-90), 0.0, 0.0, 1.0),
                       to_new_int(np.deg2rad(-90), np.deg2rad(-90), 0.0, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(lam, -obj.incl, obj.incl, 0.0, 1.0),
                       to_new_int(-obj.incl, -obj.incl, obj.incl, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(h/1e4, 0.0, 20, 0.0, 1.0), to_new_int(1e-6, 0.0, 20, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(m[1:], obj.m10, obj.M0, 0.0, 1.0),
                       to_new_int(obj.m10, obj.m10, obj.M0, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(alfa, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0),
                       to_new_int(np.deg2rad(-2), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)

    result.lower_bound(delta, 0.001, unit=1)

    result.lower_bound(to_new_int(deltaf, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0),
                       to_new_int(np.deg2rad(-20), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)

    result.lower_bound((tau+1)/2, 0.0, unit=1)
    result.lower_bound(to_new_int(mu, np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0),
                 to_new_int(np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(mf, obj.m10, obj.M0, 0.0, 1.0),
                       to_new_int(obj.m10, obj.m10, obj.M0, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(h[-1]/1e4, 0.0, 20, 0.0, 1.0), to_new_int(9, 0.0, 20, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(MomTot/1e5, -1e2, 1e2, 0.0, 1.0), to_new_int(-obj.k/1e5, -1e2, 1e2, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(az, -1e2, 1e2, 0.0, 1.0), to_new_int(-obj.MaxAz, -1e2, 1e2, 0.0, 1.0), unit=1)





    # upper bounds
    result.upper_bound(to_new_int(v/1e3, 0.0, 10, 0.0, 1.0), to_new_int(9, 0.0, 10, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(chi, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0),
                       to_new_int(np.deg2rad(270), np.deg2rad(90), np.deg2rad(270), 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(gamma, np.deg2rad(-2), np.deg2rad(90), 0.0, 1.0),
                       to_new_int(np.deg2rad(89.9), 0.0, np.deg2rad(90), 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(teta, np.deg2rad(-90), 0.0, 0.0, 1.0),
                       to_new_int(0.0, np.deg2rad(-90), 0.0, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(lam, -obj.incl, obj.incl, 0.0, 1.0),
                       to_new_int(obj.incl, -obj.incl, obj.incl, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(h/1e4, 0.0, 20, 0.0, 1.0), to_new_int(18, 0.0, 20, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(m, obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.M0, obj.m10, obj.M0, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(alfa, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0),
                       to_new_int(np.deg2rad(40), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)

    result.upper_bound(delta, 1.0, unit=1)

    result.upper_bound(to_new_int(deltaf, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0),
                       to_new_int(np.deg2rad(30), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)

    result.upper_bound((tau+1)/2, 1.0, unit=1)

    result.upper_bound(to_new_int(mu, np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0),
                       to_new_int(np.deg2rad(90), np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(MomTot/1e5, -1e2, 1e2, 0.0, 1.0), to_new_int(obj.k/1e5, -1e2, 1e2, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(az, -1e2, 1e2, 0.0, 1.0), to_new_int(obj.MaxAz, -1e2, 1e2, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(ax, 0.0, 1e2, 0.0, 1.0), to_new_int(obj.MaxAx, 0.0, 1e2, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(q/1000, 0.0, 1e2, 0.0, 1.0), to_new_int(obj.MaxQ/1000, 0.0, 1e2, 0.0, 1.0), unit=1)

    return result()


def cost(prob, obj):

    h = prob.states_all_section(5)
    m = prob.states_all_section(6)
    delta = prob.controls_all_section(1)
    tau = prob.controls_all_section(3)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)

    isp = np.asarray(isp, dtype=np.float64)

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0*isp[-1])))

    return -mf / obj.M0


# ==============


plt.ion()

# Program starting point!!!
#magic_ratio = 52/355.09  --> 0.15
n = [10]
time_init = [0.0, 350]
num_states = [7]
num_controls = [5]
max_iteration = 30
Ncontrols = num_controls[0]
Nstates = num_states[0]
Npoints = n[0]
varStates = Nstates * Npoints
varTot = (Nstates+Ncontrols)*Npoints
Nint = 10000
maxiter = 20
ftol = 1e-10
Nintm = 150

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


cl = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/clfile.txt")
cd = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/cdfile.txt")
cm = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/cmfile.txt")
cl = np.asarray(cl)
cd = np.asarray(cd)
cm = np.asarray(cm)


with open("/home/francesco/git_workspace/FESTIP_Work/coeff_files/impulse.dat") as f:
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


savefig_file = "ResColloc_OrigCoeff_noMu_{}_{}".format(os.path.basename(__file__), timestr)


# -------------
# set OpenGoddard class for algorithm determination
prob = Problem(time_init, n, num_states, num_controls, max_iteration)

# -------------
# create instance of operating object
obj = Spaceplane()

# max value used to divide?
unit_v = 9000
unit_chi = np.deg2rad(270)
unit_gamma = obj.gammastart
unit_teta = obj.longstart
unit_lam = obj.incl
unit_h = 200000
unit_m = obj.M0
unit_t = 1000
unit_alfa = np.deg2rad(40)
unit_delta = 1
unit_deltaf = np.deg2rad(30)
unit_tau = 1
unit_mu = np.deg2rad(90)
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
prob.set_unit_controls_all_section(4, unit_mu)
prob.set_unit_time(unit_t)


# =================
# initial parameters guess
# velocity
v_init = Guess.cubic(prob.time_all_section, 1, 0.0, obj.Vtarget, 0.0)
chi_init = Guess.cubic(prob.time_all_section, obj.chistart, 0.0, obj.chi_fin, 0.0)
gamma_init = Guess.linear(prob.time_all_section, obj.gammastart, 0.0)
teta_init = Guess.constant(prob.time_all_section, obj.longstart)
lam_init = Guess.constant(prob.time_all_section, obj.latstart)
h_init = Guess.cubic(prob.time_all_section, 1, 0.0, obj.Hini, 0.0)
m_init = Guess.cubic(prob.time_all_section, obj.M0, 0.0, obj.m10, 0.0)

alfa_init = Guess.zeros(prob.time_all_section)
delta_init = Guess.cubic(prob.time_all_section, 1.0, 0.0, 0.01, 0.0)
deltaf_init = Guess.zeros(prob.time_all_section)
tau_init = Guess.constant(prob.time_all_section, 0.0)
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
prob.set_controls_all_section(4, mu_init)


# ========================
# Main Process
# Assign problem to SQP solver
prob.dynamics = [dynamics]
#prob.knot_states_smooth = []
prob.cost = cost
prob.equality = equality
prob.inequality = inequality
#prob.cost_derivative = cost_derivative


def display_func():

    m = prob.states_all_section(6)
    h = prob.states_all_section(5)
    delta = prob.controls_all_section(1)
    tau = prob.controls_all_section(3)

    tf = prob.time_final(-1)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, prob.nodes[0], obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)

    isp = np.asarray(isp, dtype=np.float64)


    # Hohmann transfer mass calculation
    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0*isp[-1])))


    print("m0          : {0:.5f}".format(m[0]))
    print("m before Ho : {0:.5f}".format(m[-1]))
    print("mf          : {0:.5f}".format(mf))
    print("altitude Hohmann starts: {0:.5f}".format(h[-1]))
    print("final time  : {0:.3f}".format(tf))


prob.solve(obj, display_func, ftol=ftol, maxiter=maxiter)

end = time.time()
time_elapsed = end - start
tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
print("Time elapsed:for total optimization ", tformat)


def dynamicsInt(t, states, alfa_int, delta_int, deltaf_int, tau_int, mu_int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''

    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = alfa_int(t)
    delta = delta_int(t)
    deltaf = deltaf_int(t)
    tau = tau_int(t)
    mu = mu_int(t)

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)
    # Press = np.asarray(Press, dtype=np.float64)
    # rho = np.asarray(rho, dtype=np.float64)
    # c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                            obj.xcg0, obj.xcgf, obj.pref)
    # L = np.asarray(L, dtype=np.float64)
    # D = np.asarray(D, dtype=np.float64)
    # MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                obj.lRef, obj.xcgf, obj.xcg0)

    # T = np.asarray(T, dtype=np.float64)
    # isp = np.asarray(isp, dtype=np.float64)
    # Deps = np.asarray(Deps, dtype=np.float64)
    # MomT = np.asarray(MomT, dtype=np.float64)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   ((T * np.sin(eps) + L) * np.sin(mu)) / (m * v * np.cos(gamma)) - np.cos(gamma) * np.cos(chi) *
                   np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (
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


def SingleShooting(states, controls, dyn, time, Nint):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''

    #Time = time

    x = np.zeros((Nint, Nstates))

    x[0,:] = states[0:Nstates]  # vector of intial states ready

    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(time, controls[0, :])
    delta_Int = interpolate.PchipInterpolator(time, controls[1, :])
    deltaf_Int = interpolate.PchipInterpolator(time, controls[2, :])
    tau_Int = interpolate.PchipInterpolator(time, controls[3, :])
    mu_Int = interpolate.PchipInterpolator(time, controls[4, :])

    time_new = np.linspace(0, time[-1], Nint)

    dt = (time_new[1] - time_new[0])

    t = time_new

    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int), t_span=[0, time[-1]], y0=x,
     #               t_eval=time_new, method='RK45')

    for i in range(Nint - 1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt * dyn(t[i], x[i, :], alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k1: ", k1)
        k2 = dt * dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k2: ", k2)
        k3 = dt * dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k3: ", k3)
        k4 = dt * dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


    #vres = sol.y[0, :]
    #chires = sol.y[1, :]
    #gammares = sol.y[2, :]
    #tetares = sol.y[3, :]
    #lamres = sol.y[4, :]
    #hres = sol.y[5, :]
    #mres = sol.y[6, :]
    vres = x[:, 0]
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_new)
    deltares = delta_Int(time_new)
    deltafres = deltaf_Int(time_new)
    taures = tau_Int(time_new)
    mures = mu_Int(time_new)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, mures


#prob.plot()

#print(prob.__repr__())

#prob.to_csv()
# =====================
# Post Process
# ------------------------
# Convert parameter vector to variable

'''integration of collocation results'''
'''here angles are in radians and not scaled'''

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
mu = prob.controls_all_section(4)
time = prob.time_update()

Uval = np.vstack((alfa, delta, deltaf, tau, mu))

Xval = np.array((v, chi, gamma, teta, lam, h, m))

X = np.zeros((0))
for i in range(Npoints):
    '''creation of vector of states initial guesses'''
    for j in range(Nstates):
        X = np.hstack((X, Xval[j][i]))

vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures = \
    SingleShooting(X, Uval, dynamicsInt, time, Nint)

Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re)
Press = np.asarray(Press, dtype=np.float64)
rho = np.asarray(rho, dtype=np.float64)
c = np.asarray(c, dtype=np.float64)
M = v / c

L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10, obj.xcg0,
                            obj.xcgf, obj.pref, n[0])
L = np.asarray(L, dtype=np.float64)
D = np.asarray(D, dtype=np.float64)
MomA = np.asarray(MomA, dtype=np.float64)

T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, n[0], obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf,
                                obj.xcg0)
T = np.asarray(T, dtype=np.float64)
Deps = np.asarray(Deps, dtype=np.float64)
MomT = np.asarray(MomT, dtype=np.float64)
isp = np.asarray(isp, dtype=np.float64)

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
mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0*isp)))

downrange = -(v**2)/g0 * np.sin(2 * chi)

res = open("resColloc_{}_{}.txt".format(os.path.basename(__file__), timestr), "w")
res.write("Max number Optimization iterations: " + str(max_iteration) + "\n" + "Number of NLP iterations: "
          + str(maxiter) + "\n" + "v: " + str(v) + "\n" + "Chi: " + str(np.rad2deg(chi)) + "\n" + "Gamma: "
          + str(np.rad2deg(gamma)) + "\n" + "Teta: " + str(np.rad2deg(teta)) + "\n" + "Lambda: "
          + str(np.rad2deg(lam)) + "\n" + "Height: " + str(h) + "\n" + "Mass: " + str(m) + "\n" + "mf: "
          + str(mf) + "\n" + "Objective Function: " + str(-mf / unit_m) + "\n" + "Alfa: " + str(np.rad2deg(alfa))
          + "\n" + "Delta: " + str(delta) + "\n" + "Delta f: " + str(np.rad2deg(deltaf)) + "\n" + "Tau: "
          + str(tau) + "\n" + "Mu: " + str(np.rad2deg(mu)) + "\n"  + "Eps: " + str(np.rad2deg(eps)) + "\n" + "Lift: "
          + str(L) + "\n" + "Drag: " + str(D) + "\n" + "Thrust: " + str(T) + "\n" + "Spimp: " + str(isp) + "\n" + "c: "
          + str(c) + "\n" + "Mach: " + str(M) + "\n" + "Time vector: " + str(time) + "\n" + "Press: " + str(Press)
          + "\n" + "Dens: " + str(rho) + "\n" + "Time elapsed for optimization: " + tformat + "\n")
res.close()

# ------------------------
# Visualization

plt.figure(0)
plt.title("Altitude profile")
plt.plot(time, h/1000, marker=".", label="Altitude")
plt.plot(tres, hres/1000, label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "altitude" + ".png")

plt.figure(1)
plt.title("Velocity")
plt.plot(time, v, marker=".", label="V")
plt.plot(tres, vres, label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "velocity" + ".png")

plt.figure(2)
plt.title("Mass")
plt.plot(time, m, marker=".", label="Mass")
plt.plot(tres, mres, label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "mass" + ".png")

plt.figure(3)
plt.title("Altitude profile")
plt.plot(time, h/1000, marker=".", label="Altitude")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "altitudeNONINT" + ".png")

plt.figure(4)
plt.title("Velocity")
plt.plot(time, v, marker=".", label="V")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "velocityNONINT" + ".png")

plt.figure(5)
plt.title("Mass")
plt.plot(time, m, marker=".", label="Mass")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "massNONINT" + ".png")

plt.figure(6)
plt.title("Acceleration")
plt.plot(time, ax, marker=".", label="Acc x")
plt.plot(time, az, marker=".", label="Acc z")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Acceleration [m/s2]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "acceleration" + ".png")

plt.figure(7)
plt.title("Throttle profile")
plt.plot(time, delta, ".", label="Delta")
plt.plot(tres, deltares, label="Interp")
plt.plot(time, tau, ".", label="Tau")
plt.plot(tres, taures, label="Interp")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" % ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "Throttle" + ".png")

plt.figure(8)
plt.title("Angle of attack profile")
plt.plot(time, np.rad2deg(alfa), ".", label="Alpha")
plt.plot(tres, np.rad2deg(alfares), label="Interp")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "angAttack" + ".png")

plt.figure(9)
plt.title("Body Flap deflection profile")
plt.plot(time, np.rad2deg(deltaf), ".", label="Delta f")
plt.plot(tres, np.rad2deg(deltafres), label="Interp")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "bdyFlap" + ".png")


plt.figure(10)
plt.title("Bank angle profile")
plt.plot(time, np.rad2deg(mu), ".", label="Mu")
plt.plot(tres, np.rad2deg(mures), label="Interp")
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "Roll" + ".png")

'''
plt.subplot(3,1,1)

plt.step(time, delta, marker="o", label="Delta")
plt.step(time, tau, marker="o", label="Tau")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" % ")
plt.legend(loc="best")


plt.subplot(3,1,2)
plt.plot(time, np.rad2deg(alfa), marker="o", label="Alfa")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.subplot(3,1,3)
plt.plot(time, np.rad2deg(deltaf), marker="o", label="Deltaf")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")


plt.subplot(2,2,4)
plt.step(time, np.rad2deg(mu), where='post', marker=".", label="Mu")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
plt.suptitle("Controls profile")
if flag_savefig:
    plt.savefig(savefig_file + "Commands" + ".png")

'''
plt.figure(11)
plt.title("Trajectory angles")
plt.plot(time, np.rad2deg(chi), marker=".", label="Chi")
plt.plot(time, np.rad2deg(gamma), marker=".", label="Gamma")
plt.plot(time, np.rad2deg(teta), marker=".", label="Theta")
plt.plot(time, np.rad2deg(lam), marker=".", label="Lambda")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "Angles" + ".png")

plt.figure(12)
plt.title("Chi")
plt.plot(time, np.rad2deg(chi), marker=".", label="Chi")
plt.plot(tres, np.rad2deg(chires), label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "chi" + ".png")

plt.figure(13)
plt.title("Gamma")
plt.plot(time, np.rad2deg(gamma), marker=".", label="Gamma")
plt.plot(tres, np.rad2deg(gammares), label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "Gamma" + ".png")

plt.figure(14)
plt.title("Teta")
plt.plot(time, np.rad2deg(teta), marker=".", label="Theta")
plt.plot(tres, np.rad2deg(tetares), label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "teta" + ".png")

plt.figure(15)
plt.title("Lambda")
plt.plot(time, np.rad2deg(lam), marker=".", label="Lambda")
plt.plot(tres, np.rad2deg(lamres), label="Integration")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" deg ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "lam" + ".png")

plt.figure(16)
plt.title("Dynamic pressure profile")
plt.plot(time, q/1000, marker=".", label="Q")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" kPa ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "dynPress" + ".png")

plt.figure(17)
plt.title("Moment")
plt.plot(time, MomTot / 1000, marker=".", label="Total Moment")
plt.axhline(obj.k/1000, 0, time[-1], color='r')
plt.axhline(-obj.k/1000, 0, time[-1], color='r')
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" kNm ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "MomTot" + ".png")

plt.figure(18)
plt.title("Trajectory")
plt.plot(downrange/1000, h/1000, marker=".", label="Trajectory")
plt.grid()
plt.xlabel("km")
plt.ylabel(" km ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "downrange" + ".png")

plt.figure(19)
plt.title("Mach profile")
plt.plot(time, M, marker=".", label="Mach")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "mach" + ".png")

plt.figure(20)
plt.title("Lift, Drag and Thrust profile")
plt.plot(time, L, marker=".", label="Lift")
plt.plot(time, D, marker=".", label="Drag")
plt.plot(time, T, marker=".", label="Thrust")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "LDT" + ".png")

plt.show()

plt.close(0)
plt.close(1)
plt.close(2)
plt.close(3)
plt.close(4)
plt.close(5)
plt.close(6)
plt.close(7)
plt.close(8)
plt.close(9)
plt.close(10)
plt.close(11)
plt.close(12)
plt.close(13)
plt.close(14)
plt.close(15)
plt.close(16)
plt.close(17)
plt.close(18)
plt.close(19)
plt.close(20)