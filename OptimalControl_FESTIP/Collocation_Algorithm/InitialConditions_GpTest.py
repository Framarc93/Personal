import operator
import random
from deap.algorithms import varOr
from deap import base, creator, tools, algorithms, gp
import multiprocessing
import numpy as np
from OpenGoddard.optimize import Guess
from scipy import interpolate
from functools import partial
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")
import GP_PrimitiveSet as gpprim


def selBest_mut(ind):
    ind_test = deepcopy(ind)
    del ind_test.fitness.values
    ind_list = []
    for _ in range(10):
        ind_list.append(deepcopy(gp.mutEphemeral(ind_test, 'all')[0]))
    ind_list.append(ind_test)
    fitnesses = toolbox.map(toolbox.evaluate, ind_list)
    for indd, fit in zip(ind_list, fitnesses):
        indd.fitness.values = fit
    best_ind = tools.selBest(ind_list, 1)[0]
    return best_ind

def eaMuPlusLambdaTol(population, toolbox, mu, lambda_, ngen, tol, stats=None, halloffame=None, verbose=__debug__):
    global cxpb, mutpb

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    min_fit = np.array(logbook.chapters["fitness"].select("min"))
    # Begin the generational process
    gen = 1
    while gen < ngen + 1 and min_fit[-1] > tol:
        # Vary the population

        ip = 0
        while ip < len(population):
            ind = deepcopy(population[ip])
            population[ip] = selBest_mut(ind)
            ip = ip + 2

        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu-1)
        population.append(tools.selBest(population + offspring, 1)[0])
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        min_fit = np.array(logbook.chapters["fitness"].select("min"))
        if verbose:
            print(logbook.stream)
        gen += 1

    return population, logbook



def mut(ind, expr, strp):

    choice = random.random()
    if choice < strp:
        indx = gp.mutUniform(ind, expr, pset=pset)
        ind = indx[0]
        return ind,
    else:
        indx = gp.mutEphemeral(ind, "all")
        ind = indx[0]
        return ind,

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

cl = np.array(fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/clfile.txt"))
cd = np.array(fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cdfile.txt"))
cm = np.array(fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cmfile.txt"))
# cl = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cl_smooth_few.npy")
# cd = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cd_smooth_few.npy")
# cm = np.load("/home/francesco/Desktop/PhD/FESTIP_Work/coeff_files/cm_smooth_few.npy")
cl = np.reshape(cl, (13, 17, 6))
# cd = np.asarray(cd)
cd = np.reshape(cd, (13, 17, 6))
# cm = np.asarray(cm)
cm = np.reshape(cm, (13, 17, 6))

with open("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/impulse.dat") as f:
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
spimp_interp = splrep(presv, spimpv, s=2)


def dynamicsInt(t, states, alfa_int, delta_int):  # , deltaf_int, tau_int, mu_int):
    # this functions receives the states and controls unscaled and calculates the dynamics

    v = states[0]
    chi = states[1]
    gamma = states[2]
    # teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = alfa_int(t)
    delta = delta_int(t)
    deltaf = 0.0  # deltaf_int(t)
    tau = 0.0  # tau_int(t)
    mu = 0.0  # mu_int(t)

    # if h<0:
    #   print("h: ", h, "gamma: ", gamma, "v: ", v)

    Press, rho, c = obj.isa(h, obj.psl, obj.g0, obj.Re, 1)

    M = v / c

    L, D, MomA = obj.aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                obj.xcg0, obj.xcgf, obj.pref, 1)

    T, Deps, isp, MomT = obj.thrust(Press, m, presv, spimpv, delta, tau, 1, obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2

    if t < obj.tvert:
        dx = np.array(((T * np.cos(eps) - D) / m - g, 0, 0, 0, 0, v, -T / (g0 * isp)))
    else:
        dx = np.array(
            (((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
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
    dx = np.reshape(dx, (7,))
    return dx


def SingleShooting(states, controls, dyn, time, Nint):
    x = np.zeros((Nint, Nstates))

    x[:Nstates] = states  # vector of intial states ready

    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(time, controls[0, :])
    delta_Int = interpolate.PchipInterpolator(time, controls[1, :])
    # deltaf_Int = interpolate.PchipInterpolator(time, controls[2, :])
    # tau_Int = interpolate.PchipInterpolator(time, controls[2, :])
    # mu_Int = interpolate.PchipInterpolator(time, controls[4, :])

    time_new = np.linspace(0, time[-1], Nint)

    dt = (time_new[1] - time_new[0])

    t = time_new

    # sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int), t_span=[0, time[-1]], y0=x,
    # t_eval=time_new, method='RK45')

    for i in range(Nint - 1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt * dyn(t[i], x[i, :], alfa_Int, delta_Int)  # , deltaf_Int, tau_Int, mu_Int)
        # print("k1: ", k1)
        k2 = dt * dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int)  # , deltaf_Int, tau_Int, mu_Int)
        # print("k2: ", k2)
        k3 = dt * dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int)  # , deltaf_Int, tau_Int, mu_Int)
        # print("k3: ", k3)
        k4 = dt * dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int)  # , deltaf_Int, tau_Int, mu_Int)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    vres = x[:, 0]
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_new)
    deltares = delta_Int(time_new)
    deltafres = np.zeros(len(time_new))  # deltaf_Int(time_new)
    taures = np.zeros(len(time_new))  # tau_Int(time_new)
    mures = np.zeros(len(time_new))  # mu_Int(time_new)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, mures


class Spaceplane():

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
        self.incl = np.deg2rad(51.6)  # deg orbit inclination
        self.gammastart = np.deg2rad(89.9)  # deg
        self.M0 = 450400  # kg  starting mass
        self.g0 = 9.80665  # m/s2
        self.gIsp = self.g0 * 455  # g0 * Isp max
        self.omega = 7.2921159e-5
        self.MaxQ = 40000  # Pa
        self.MaxAx = 30  # m/s2
        self.MaxAz = 15  # m/s2
        self.Htarget = 400000  # m target height after hohmann transfer
        self.wingSurf = 500.0  # m2
        self.lRef = 34.0  # m
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 100000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))  # value in radians
        self.chistart = np.deg2rad(125)  # deg flight direction
        self.mach = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0])
        self.angAttack = np.array(
            [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0])
        self.bodyFlap = np.array([-20, -10, 0, 10, 20, 30])
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
        self.tvert = 2

    @staticmethod
    def isa(altitude, pstart, g0, r, flag):
        t0 = 288.15
        p0 = pstart
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
                        p0 = pstart
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
            elif alt > 190000:
                # print("h > 190 km", alt)
                zb = obj.h90[6]
                z = obj.h90[-1]
                b = zb - obj.tcoeff1[6] / obj.a90[6]
                t = obj.tcoeff1[6] + (obj.tcoeff2[6] * (z - zb)) / 1000
                tm = obj.tmcoeff[6] + obj.a90[6] * (z - zb) / 1000
                add1 = 1 / ((r + b) * (r + z)) + (1 / ((r + b) ** 2)) * np.log(abs((z - b) / (z + r)))
                add2 = 1 / ((r + b) * (r + zb)) + (1 / ((r + b) ** 2)) * np.log(abs((zb - b) / (zb + r)))
                p = obj.pcoeff[6] * np.exp(-m0 / (obj.a90[6] * Rs) * g0 * r ** 2 * (add1 - add2))
                d = p / (R * t)
                c = np.sqrt(1.4 * R * tm)
                density[k] = d
                csound[k] = c
                # temperature[k] = t
                pressure[k] = p
                # tempm[k] = t
            k += 1
        return pressure, density, csound

    @staticmethod
    def aeroForces(M, alfa, deltaf, cd, cl, cm, v, sup, rho, leng, mstart, mass, m10, xcg0, xcgf, pref, npoint):
        alfag = np.rad2deg(alfa)
        deltafg = np.rad2deg(deltaf)
        coeffs = np.array((cl, cd, cm))
        if npoint == 1:
            M = np.array([M])
            alfag = np.array([alfag])
            deltafg = np.array([deltafg])
        cL, cD, cM1 = map(partial(c_eval, npoint, M, alfag, deltafg, coeffs, obj.mach, obj.angAttack, obj.bodyFlap),
                          range(3))

        L = 0.5 * (v ** 2) * sup * rho * cL
        D = 0.5 * (v ** 2) * sup * rho * cD
        xcg = leng * (((xcgf - xcg0) / (m10 - mstart)) * (mass - mstart) + xcg0)
        Dx = xcg - pref
        cM = cM1 + cL * (Dx / leng) * np.cos(alfa) + cD * (Dx / leng) * np.sin(alfa)
        Mom = 0.5 * (v ** 2) * sup * leng * rho * cM

        return L, D, Mom

    @staticmethod
    def thrust(presamb, mass, presv, spimpv, delta, tau, npoint, slpres, wlo, we, lref, xcgf, xcg0):
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
            thrx = nmot * (5.8e6 + 14.89 * slpres - 11.16 * presamb[j]) * delta[j]
            if presamb[j] >= slpres:
                presamb[j] = slpres
                spimp = spimpv[-1]
            elif presamb[j] < slpres:
                for i in range(nimp):
                    if presv[i] >= presamb[j]:
                        spimp = splev(presamb[j], spimp_interp, der=0)
                        break

            xcg = ((xcgf - xcg0) / (we - wlo) * (mass[j] - wlo) + xcg0) * lref
            dthr = 0.4224 * (36.656 - xcg) * thrx - 19.8 * (32 - xcg) * (1.7 * slpres - presamb[j])
            mommot = tau[j] * dthr
            thrz = -tau[j] * (2.5e6 - 22 * slpres + 9.92 * presamb[j])
            thrust = np.sqrt(thrx ** 2 + thrz ** 2)

            if thrx == 0.0 and thrz == 0.0:
                deps = 0.0
            else:
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


obj = Spaceplane()
Nstates = 7

def evaluate(individual):
    tfin = 400
    time = np.linspace(0, tfin, 500)
    Nint = 500
    alfa_fun = toolbox.compile(individual) #np.zeros(len(tvec))
    alfa_init = alfa_fun(time)
    if not hasattr(alfa_init, '__len__'):
        alfa_init = np.zeros(len(time))*alfa_init
    part1 = np.repeat(1.0, int(len(time)*0.4))
    part2 = Guess.linear(time[int(len(time)*0.4):], 1.0, 0.0001)
    delta_init = np.hstack((part1, part2))
    Uval = np.vstack((alfa_init, delta_init))#, deltaf, tau, mu))
    Xinit = np.array((0.001, np.deg2rad(113), np.deg2rad(89.9), np.deg2rad(-52.775), np.deg2rad(5.2), 0.001, 450400))
    vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures = SingleShooting(Xinit, Uval, dynamicsInt, time, Nint)
    Pressres, rhores, cres = obj.isa(hres, obj.psl, obj.g0, obj.Re, 0)
    Mres = vres / cres
    Lres, Dres, MomAres = obj.aeroForces(Mres, alfares, deltafres, cd, cl, cm, vres, obj.wingSurf, rhores, obj.lRef, obj.M0, mres, obj.m10,obj.xcg0, obj.xcgf, obj.pref, len(vres))
    Tres, Depsres, ispres, MomTres = obj.thrust(Pressres, mres, presv, spimpv, deltares, taures, len(vres), obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf,obj.xcg0)
    qres = 0.5 * rhores * (vres ** 2)
    ax_res = (Tres * np.cos(Depsres) - Dres * np.cos(alfares) + Lres * np.sin(alfares)) / mres
    az_res = (Tres * np.sin(Depsres) + Dres * np.sin(alfares) + Lres * np.cos(alfares)) / mres
    r1 = hres + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = mres / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * ispres)))

    penalty = []

    for i in range(len(tres)):
        if np.isnan(hres[i]):
            penalty.append(-1e2)
        elif hres[i] < 0:
            penalty.append(hres[i])
        if np.isinf(hres[i]):
            penalty.append(1e2)
        if np.isnan(vres[i]):
            penalty.append(-1e2)
        elif vres[i] < 0:
            penalty.append(vres[i])
        if np.isinf(vres[i]):
            penalty.append(1e2)
        if np.isinf(np.rad2deg(chires[i])):
            penalty.append(1e2)
        elif np.rad2deg(chires[i]) > 150 or np.rad2deg(chires[i]) < 110:
            penalty.append(np.rad2deg(chires[i])/(chires[i]*2))
        if np.isinf(np.rad2deg(gammares[i])):
            penalty.append(1e2)
        elif np.rad2deg(gammares[i]) > 90 or np.rad2deg(gammares[i]) < -90:
            penalty.append(np.rad2deg(gammares[i]))
        if np.rad2deg(lamres[i]) > 30 or np.rad2deg(lamres[i]) < 2:
            penalty.append(np.rad2deg(lamres[i]))
        if np.rad2deg(tetares[i]) < -60 or np.rad2deg(tetares[i]) > 0:
            penalty.append(np.rad2deg(tetares[i]))
        if np.rad2deg(alfares[i]) < -2 or np.rad2deg(alfares[i]) > 40:
            penalty.append(np.rad2deg(alfares[i]))
        if mres[i] < obj.m10 or mres[i] > obj.M0:
            penalty.append(mres[i])

        '''if ax_res[i] > obj.MaxAx:
            penalty.append(ax_res[i]/30)
        if az_res[i] < -obj.MaxAz or az_res[i] > obj.MaxAz:
            penalty.append(az_res[i]/15)
        if qres[i] > obj.MaxQ:
            penalty.append(qres[i]/40000)'''
    if mf[-1] < obj.m10 or mf[-1] > obj.M0:
        penalty.append(mf[-1])
    if len(penalty) == 0:
        pen = 0
    else:
        pen = np.sqrt(sum(np.array(penalty) ** 2))

    return (pen,)

def main():

    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)
    pop = toolbox.population(n=size_pop)

    print("INITIAL POP SIZE: %d" % size_pop)

    print("GEN SIZE: %d" % size_gen)

    print("\n")

    random.seed()

    history.update(pop)
    hof = tools.HallOfFame(20) ### OLD ###

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height, size=stats_size)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

    pop, log = eaMuPlusLambdaTol(pop, toolbox, Mu, Lambda, size_gen, 0.1, stats=mstats, halloffame=hof, verbose=True)
    ####################################################################################################################

    pool.close()
    pool.join()
    return pop, log, hof


limit_height = 15  # Max height (complexity) of the controller law
limit_size = 20  # Max size (complexity) of the controller law
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(gpprim.TriAdd, 3)
#pset.addPrimitive(Abs, 1)
#pset.addPrimitive(Div, 2)
pset.addPrimitive(gpprim.Sqrt, 1)
pset.addPrimitive(gpprim.Log, 1)
pset.addPrimitive(gpprim.modExp, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)
pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-5, 5), 4))
pset.addEphemeralConstant("rand102", lambda: round(random.uniform(-5, 5), 4))
pset.addEphemeralConstant("rand103", lambda: round(random.uniform(-5, 5), 4))
pset.addEphemeralConstant("rand104", lambda: round(random.uniform(-5, 5), 4))
pset.addEphemeralConstant("rand105", lambda: round(random.uniform(-5, 5), 4))
pset.renameArguments(ARG0='t')


################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0,))  # MINIMIZATION OF THE FITNESS FUNCTION

creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)   #### OLD ####

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####

toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####

toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evaluate)  ### OLD ###

toolbox.register("select", tools.selDoubleTournament, fitness_size=2, parsimony_size=1.0, fitness_first=True)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

########################################################################################################################


if __name__ == "__main__":
    mutpb = 0.1
    cxpb = 0.8
    size_pop = 200  # Pop size
    size_gen = 100  # Gen size
    Mu = int(size_pop)
    Lambda = int(size_pop * 1.5)
    nbCPU = multiprocessing.cpu_count()

    pop, log, hof = main()

    tfin = 400
    time = np.linspace(0, tfin, 500)
    Nint = 500
    alfa_fun = toolbox.compile(hof[0])  # np.zeros(len(tvec))
    alfa_init = alfa_fun(time)
    part1 = np.repeat(1.0, int(len(time) * 0.35))
    part2 = Guess.linear(time[int(len(time) * 0.35):], 1.0, 0.0001)
    delta_init = np.hstack((part1, part2))

    Uval = np.vstack((alfa_init, delta_init))  # , deltaf, tau, mu))

    Xinit = np.array((0.0001, np.deg2rad(113), np.deg2rad(89.9), np.deg2rad(-52.775), np.deg2rad(5.2), 0.0001, 450400))

    vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures = \
        SingleShooting(Xinit, Uval, dynamicsInt, time, Nint)

    Pressres, rhores, cres = obj.isa(hres, obj.psl, obj.g0, obj.Re, 0)

    Mres = vres / cres

    Lres, Dres, MomAres = obj.aeroForces(Mres, alfares, deltafres, cd, cl, cm, vres, obj.wingSurf, rhores, obj.lRef,
                                         obj.M0, mres, obj.m10,
                                         obj.xcg0, obj.xcgf, obj.pref, len(vres))

    Tres, Depsres, ispres, MomTres = obj.thrust(Pressres, mres, presv, spimpv, deltares, taures, len(vres), obj.psl,
                                                obj.M0, obj.m10, obj.lRef,
                                                obj.xcgf, obj.xcg0)

    MomTotres = MomAres + MomTres

    g0 = obj.g0
    eps = Depsres + alfares

    # dynamic pressure

    qres = 0.5 * rhores * (vres ** 2)

    # accelerations

    ax_res = (Tres * np.cos(Depsres) - Dres * np.cos(alfares) + Lres * np.sin(alfares)) / mres
    az_res = (Tres * np.sin(Depsres) + Dres * np.sin(alfares) + Lres * np.cos(alfares)) / mres

    r1 = hres + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = mres / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0 * ispres)))

    plt.figure(0)
    plt.title("Altitude profile")
    plt.plot(tres, hres / 1000, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.legend(loc="best")

    plt.figure(1)
    plt.title("Velocity")
    plt.plot(tres, vres, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend(loc="best")

    plt.figure(2)
    plt.title("Mass")
    plt.plot(tres, mres, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc="best")

    plt.figure(6)
    plt.title("Acceleration")
    plt.plot(tres, ax_res, label="Acc x Integration")
    plt.plot(tres, az_res, label="Acc z Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel("Acceleration [m/s2]")
    plt.legend(loc="best")

    plt.figure(7)
    plt.title("Throttle profile")
    plt.plot(tres, deltares, label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" % ")
    plt.legend(loc="best")

    plt.figure(8)
    plt.title("Angle of attack profile")
    plt.plot(tres, np.rad2deg(alfares), label="Interp")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")


    plt.figure(12)
    plt.title("Chi")
    plt.plot(tres, np.rad2deg(chires), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")

    plt.figure(13)
    plt.title("Gamma")
    plt.plot(tres, np.rad2deg(gammares), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")

    plt.figure(14)
    plt.title("Teta")
    plt.plot(tres, np.rad2deg(tetares), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")

    plt.figure(15)
    plt.title("Lambda")
    plt.plot(tres, np.rad2deg(lamres), label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" deg ")
    plt.legend(loc="best")

    plt.figure(16)
    plt.title("Dynamic pressure profile")
    plt.plot(tres, qres / 1000, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" kPa ")
    plt.legend(loc="best")

    plt.figure(17)
    plt.title("Moment")
    plt.plot(tres, MomTotres / 1000, label="Integration")
    plt.grid()
    plt.xlabel("time [s]")
    plt.ylabel(" kNm ")
    plt.legend(loc="best")

    plt.figure(19)
    plt.title("Mach profile")
    plt.plot(tres, Mres, marker=".", label="Mach")
    plt.grid()
    plt.xlabel("time [s]")
    plt.legend(loc="best")

    plt.figure(20)
    plt.title("Lift, Drag and Thrust profile")
    plt.plot(tres, Lres, marker=".", label="Lift")
    plt.plot(tres, Dres, marker=".", label="Drag")
    plt.plot(tres, Tres, marker=".", label="Thrust")
    plt.grid()
    plt.xlabel("time [s]")
    plt.legend(loc="best")
    plt.show(block=True)
