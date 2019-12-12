import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import Bounds
from OpenGoddard.optimize import Guess
from scipy.sparse import csc_matrix, save_npz, load_npz
from models import *
from mapping_functions import *
import time
import datetime
import os
import sys
from functools import partial
from smooth_fun import *
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D


#sys.path.insert(0, 'home/francesco/Desktop/PhD/FESTIP_Work')
#sys.path.insert(1, 'home/francesco/Desktop/PhD/FESTIP_Work/MultipleShooting_Algorithm')


'''modified initial guess for controls so it has more points than states. States points are only in the conjunction of legs while controls are also inside'''
'''simplified version of problem. All controls but equatorial trajectory and no earth rotation'''
'''definition of vehicle and mission parameters. Angles are set in degrees'''
'''maybe best to set initial values of angles in radians'''

'''DIFFERENT METHOD FOR INTEGRATION'''

'''set initial conditions constraints on all parameters'''

'''try to reduce the dynamic evaluations at one insted of two. maybe by passing arguments to the functions'''



'''vehicle parameters'''


class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
        self.chistart = np.deg2rad(113)  # deg flight direction
        self.incl = np.deg2rad(51.6)  # deg orbit inclination
        self.gammastart = np.deg2rad(89)  # deg
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
        self.k = 5e4  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 100000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))
        self.varOld = np.zeros((0))
        self.costOld = 0
        self.eqOld = np.zeros((0))
        self.ineqOld = np.zeros((0))
        self.States = np.zeros((0))
        self.Controls = np.zeros((0))
        self.time = np.zeros((0))
        self.Conj = np.zeros((0))
        self.bad = False
        self.UBV = np.zeros((0))
        self.LBV = np.zeros((0))


def dynamicsInt(t, states, alfa_Int, delta_Int): #, deltaf_Int, tau_Int, mu_Int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    v = states[0]
    chi = states[1]
    gamma = states[2]
    #teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = float(alfa_Int(t))
    delta = float(delta_Int(t))
    deltaf = 0.0 #float(deltaf_Int(t))
    tau = 0.0 #float(tau_Int(t))
    mu = 0.0 #float(mu_Int(t))
    if abs(v) > 1e10:
        if v > 0:
            v = 1e10
        else:
            v = -1e10
        obj.bad = True
    if abs(h) > 1e20:
        if h > 0:
            h = 1e20
        else:
            h = -1e20
        obj.bad = True
    if np.isnan(v):
        v = 0.1
        obj.bad = True
    if np.isnan(h):
        h = 0.1
        obj.bad = True
    if gamma >=np.deg2rad(90):
        gamma = np.deg2rad(89.9)
        obj.bad = True

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
                   np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
                   ((T * np.sin(eps) + L) * np.cos(mu)) / (m * v) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (g0 * isp)))

    return dx


def dynamicsVel(states, contr):
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
    tau = 0.0 #contr[3]
    mu = 0.0 #contr[4]
    if abs(v) > 1e10:
        if v > 0:
            v = 1e10
        else:
            v = -1e10
        obj.bad = True
    if abs(h) > 1e20:
        if h > 0:
            h = 1e20
        else:
            h = -1e20
        obj.bad = True
    if np.isnan(v):
        v = 0.1
        obj.bad = True
    if np.isnan(h):
        h = 0.1
        obj.bad = True
    if gamma >=np.deg2rad(90):
        gamma = np.deg2rad(89.9)
        obj.bad = True

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


def inequalityAll(states, controls, varnum):
    '''this function takes states and controls unscaled'''
    v = np.transpose(states[:, 0])
    #chi = np.transpose(states[:, 1])
    #gamma = np.transpose(states[:, 2])
    #teta = np.transpose(states[:, 3])
    #lam = np.transpose(states[:, 4])
    h = np.transpose(states[:, 5])
    m = np.transpose(states[:, 6])
    alfa = np.transpose(controls[:, 0])
    delta = np.transpose(controls[:, 1])
    deltaf = np.transpose(np.zeros(len(v))) #np.transpose(controls[:, 2])
    tau = np.transpose(np.zeros(len(v))) #np.transpose(controls[:, 3])  # tau back to [-1, 1] interval
    #mu = np.transpose(controls[:, 4])

    for i in range(len(v)):
        if abs(v[i]) > 1e10:
            if v[i] > 0:
                v[i] = 1e10
            else:
                v[i] = -1e10
            obj.bad = True
        elif np.isnan(v[i]):
            v[i] = 0.1
            obj.bad = True
    for i in range(len(h)):
        if abs(h[i]) > 1e20:
            if h[i] > 0:
                h[i] = 1e20
            else:
                h[i] = -1e20
            obj.bad = True
        elif np.isnan(h[i]):
            h[i] = 0.1
            obj.bad = True

    Press, rho, c = isaMulti(h, obj.psl, obj.g0, obj.Re)
    Press = np.asarray(Press, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    M = v / c

    L, D, MomA = aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                 obj.xcg0, obj.xcgf, obj.pref, varnum)

    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrustMulti(Press, m, presv, spimpv, delta, tau, varnum, obj.psl, obj.M0, obj.m10, obj.lRef,
                                     obj.xcgf, obj.xcg0)

    T = np.asarray(T, dtype=np.float64)
    isp = np.asarray(isp, dtype=np.float64)
    Deps = np.asarray(Deps, dtype=np.float64)
    MomT = np.asarray(MomT, dtype=np.float64)

    MomTot = MomA + MomT


    # dynamic pressure

    q = 0.5 * rho * (v ** 2)

    # accelerations

    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / np.exp((Dv1 + Dv2) / (obj.g0*isp[-1]))

    #axnewlim = to_new_int(obj.MaxAx/100, 0.0, 1e3, 0.0, 1.0) - to_new_int(ax/100, 0.0, 1e3, 0.0, 1.0)
    #aznewlim = to_new_int(obj.MaxAz/100, 0.0, 1e3, 0.0, 1.0) - to_new_int(abs(az)/100, 0.0, 1e3, 0.0, 1.0)
    #qnewlim = to_new_int(obj.MaxQ / 1e6, 0, 1e2, 0.0, 1.0) - to_new_int(q / 1e6, 0, 1e2, 0.0, 1.0)
    #Momnewlim = to_new_int(obj.k / 1e5, -1e3, 1e3, 0.0, 1.0) - to_new_int(MomTotA / 1e5, -1e3, 1e3, 0.0, 1.0)
    #mnew = m_toNew(mf, obj) - m_toNew(obj.m10, obj)
    #print(max(MomTotA))

    #iC = np.hstack(((v - 1.0)/unit_v, (chi - np.deg2rad(90))/unit_chi, gamma/unit_gamma, (lam - (-obj.incl))/unit_lam, (h - 1.0)/unit_h, (m - obj.m10)/unit_m,
     #               (np.deg2rad(270) - chi)/unit_chi, (np.deg2rad(89) - gamma)/unit_gamma, (obj.incl - lam)/unit_lam, (obj.M0 - m)/unit_m,
      #              (alfa - np.deg2rad(-2))/unit_alfa, delta - 0.0001, (deltaf - np.deg2rad(-20))/unit_deltaf, tau - (-1), (mu - np.deg2rad(-90))/unit_mu,
       #             (np.deg2rad(40) - alfa)/unit_alfa, 1.0 - delta, (np.deg2rad(30) - deltaf)/unit_deltaf, 1.0 - tau, (np.deg2rad(90) - mu)/unit_mu,
        #            (obj.MaxAx - ax)/obj.MaxAx, (obj.MaxAz - az)/obj.MaxAz, (obj.MaxQ - q)/obj.MaxQ, (obj.k-MomTotA)/(obj.k*1e2), (mf-obj.m10)/obj.M0))
    iC = np.hstack(((obj.MaxAx - ax) / obj.MaxAx, (obj.MaxAz - az) / obj.MaxAz, (obj.MaxQ - q) / obj.MaxQ, (mf - obj.m10) / obj.m10))
                    #(chi - np.deg2rad(90))/np.deg2rad(270), (np.deg2rad(270)-chi)/np.deg2rad(270)))
    #for j in range(len(iC)):
     #   if iC[j] > 1:
      #      print(iC[j], j)
    return iC


def SingleShooting(states, controls, dyn, tstart, tfin, Nint):
    '''this function integrates the dynamics equation over time.'''
    '''INPUT: states: states vector
              controls: controls matrix
              dyn: dynamic equations
              tstart: initial time
              tfin: final time
              Nint: number of integration steps'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''tstart and tfin are the initial and final time of the considered leg'''
    Nintlocal = Nint
    #print("single shooting")
    timeCont = np.linspace(tstart, tfin, NContPoints)
    x = np.zeros((Nint, Nstates))
      # vector of intial states ready
    time_old = np.linspace(tstart, tfin, Nint)
    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(timeCont, controls[0, :])
    delta_Int = interpolate.PchipInterpolator(timeCont, controls[1, :])
    #deltaf_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
    #tau_Int = interpolate.PchipInterpolator(timeCont, controls[3, :])
    #mu_Int = interpolate.PchipInterpolator(timeCont, controls[4, :])


    #status = 1
    #while status != 0:
    time_int = np.linspace(tstart, tfin, Nintlocal)
    dt = (time_int[1] - time_int[0])
    t = time_int
    x[0,:] = states
    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int), t_span=[tstart, tfin], y0=x, t_eval=time_int, method='RK45')

    #    status = sol.status
    #if sol.status != 0:
    #  print(sol.y)
      #      Nintlocal = Nintlocal*1.5
       #     print("new int", Nintlocal)
    for c in range(Nint-1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt*dyn(t[c], x[c, :], alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        #print("k1", k1, "k1/dt", k1/dt)
        k2 = dt*dyn(t[c] + dt / 2, x[c, :] + k1 / 2, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        #print("k2", k2, "k1/dt", k2 / dt)
        k3 = dt*dyn(t[c] + dt / 2, x[c, :] + k2 / 2, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        #print("k3", k3, "k1/dt", k3 / dt)
        k4 = dt*dyn(t[c + 1], x[c, :] + k3, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        #print("k4", k4, "k1/dt", k4 / dt)
        x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    vres = x[:, 0]  # orig interavals
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_old)
    deltares = delta_Int(time_old)
    deltafres = np.zeros(len(time_old)) #deltaf_Int(time_old)
    taures = np.zeros(len(time_old)) #tau_Int(time_old)
    mures = np.zeros(len(time_old)) #mu_Int(time_old)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_old, alfares, deltares, deltafres, taures, mures, alfa_Int, delta_Int#, deltaf_Int, tau_Int, mu_Int


def SingleShootingMulti(var, dyn, Nint, i):
    '''this function integrates the dynamics equation over time.'''
    '''INPUT: states: states vector
              controls: controls matrix
              dyn: dynamic equations
              tstart: initial time
              tfin: final time
              Nint: unmber of integration steps'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''tstart and tfin are the initial and final time of the considered leg'''
    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    #deltaf = np.zeros((NContPoints))
    #tau = np.zeros((NContPoints))
    #mu = np.zeros((NContPoints))
    states = var[i * Nstates:(i + 1) * Nstates]  # orig intervals
    if i ==0:
        tstart = 0
    else:
        tstart = var[varTot + i]
    tfin = tstart + var[varTot + i]

    for k in range(NContPoints):
        '''this for loop takes the controls from the optimization variable and stores them into different variables'''
        '''here controls are scaled'''
        alfa[k] = var[varStates + i * (Ncontrols * NContPoints) + Ncontrols * k]
        delta[k] = var[varStates + i * (Ncontrols * NContPoints) + 1 + Ncontrols * k]
        #deltaf[k] = var[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
        #tau[k] = var[varStates + i * (Ncontrols * NContPoints) + 3 + Ncontrols * k]
        #mu[k] = var[varStates + i * (Ncontrols * NContPoints) + 4 + Ncontrols * k]

    controls = np.vstack((alfa, delta))#, deltaf, tau, mu))  # orig intervals

    Nintlocal = Nint
    #print("single shooting")
    timeCont = np.linspace(tstart, tfin, NContPoints)
    x = np.zeros((Nint, Nstates))
    # vector of intial states ready
    time_old = np.linspace(tstart, tfin, Nint)
    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(timeCont, controls[0, :])
    delta_Int = interpolate.PchipInterpolator(timeCont, controls[1, :])
    #deltaf_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
    #tau_Int = interpolate.PchipInterpolator(timeCont, controls[3, :])
    #mu_Int = interpolate.PchipInterpolator(timeCont, controls[4, :])

    time_int = np.linspace(tstart, tfin, Nintlocal)
    dt = (time_int[1] - time_int[0])
    t = time_int
    x[0,:] = states

    for c in range(Nint-1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt*dyn(t[c], x[c, :], alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        #print("k1", k1, "k1/dt", k1/dt)
        k2 = dt*dyn(t[c] + dt / 2, x[c, :] + k1 / 2, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        #print("k2", k2, "k1/dt", k2 / dt)
        k3 = dt*dyn(t[c] + dt / 2, x[c, :] + k2 / 2, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        #print("k3", k3, "k1/dt", k3 / dt)
        k4 = dt*dyn(t[c + 1], x[c, :] + k3, alfa_Int, delta_Int)#, deltaf_Int, tau_Int, mu_Int)
        #print("k4", k4, "k1/dt", k4 / dt)
        x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    vres = x[:, 0]  # orig interavals
    chires = x[:, 1]
    gammares = x[:, 2]
    tetares = x[:, 3]
    lamres = x[:, 4]
    hres = x[:, 5]
    mres = x[:, 6]
    alfares = alfa_Int(time_old)
    deltares = delta_Int(time_old)
    deltafres = np.zeros(len(time_old)) #deltaf_Int(time_old)
    taures = np.zeros(len(time_old)) #tau_Int(time_old)
    mures = np.zeros(len(time_old)) #mu_Int(time_old)

    return vres, chires, gammares, tetares, lamres, hres, mres, time_old, alfares, deltares, deltafres, taures, mures


def MultiShooting(var, dyn):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in radians'''

    vres = np.zeros((0))
    chires = np.zeros((0))
    gammares = np.zeros((0))
    tetares = np.zeros((0))
    lamres = np.zeros((0))
    hres = np.zeros((0))
    mres = np.zeros((0))
    alfares = np.zeros((0))
    deltares = np.zeros((0))
    #deltafres = np.zeros((0))
    #taures = np.zeros((0))
    #mures = np.zeros((0))
    tres = np.zeros((0))

    states_atNode = np.zeros((0))
    controls_atNode = np.zeros((0))

    varD = var * (obj.UBV - obj.LBV) + obj.LBV

    res = p.map(partial(SingleShootingMulti, varD, dyn, Nint), range(Nleg))

    for i in range(Nleg):
        leg = res[i]
        vres = np.hstack((vres, leg[0]))
        chires = np.hstack((chires, leg[1]))
        gammares = np.hstack((gammares, leg[2]))
        tetares = np.hstack((tetares, leg[3]))
        lamres = np.hstack((lamres, leg[4]))
        hres = np.hstack((hres, leg[5]))
        mres = np.hstack((mres, leg[6]))
        tres = np.hstack((tres, leg[7]))
        alfares = np.hstack((alfares, leg[8]))
        deltares = np.hstack((deltares, leg[9]))
        #deltafres = np.hstack((deltafres, leg[10]))
        #taures = np.hstack((taures, leg[11]))
        #mures = np.hstack((mures, leg[12]))
        states_atNode = np.hstack((states_atNode, ((vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]))))  # new intervals
        controls_atNode = np.hstack((controls_atNode, ((alfares[-1], deltares[-1]))))#, deltafres[-1], taures[-1], mures[-1]))))

    vrescol = np.reshape(vres, (Nint*Nleg, 1))
    chirescol = np.reshape(chires, (Nint*Nleg, 1))
    gammarescol = np.reshape(gammares, (Nint*Nleg, 1))
    tetarescol = np.reshape(tetares, (Nint*Nleg, 1))
    lamrescol = np.reshape(lamres, (Nint*Nleg, 1))
    hrescol = np.reshape(hres, (Nint*Nleg, 1))
    mrescol = np.reshape(mres, (Nint*Nleg, 1))
    alfarescol = np.reshape(alfares, (Nint*Nleg, 1))
    deltarescol = np.reshape(deltares, (Nint*Nleg, 1))
    #deltafrescol = np.reshape(deltafres, (Nint*Nleg, 1))
    #murescol = np.reshape(mures, (Nint*Nleg, 1))
    #taurescol = np.reshape(taures, (Nint*Nleg, 1))


    states_after = np.column_stack((vrescol, chirescol, gammarescol, tetarescol, lamrescol, hrescol, mrescol))
    controls_after = np.column_stack((alfarescol, deltarescol))#, deltafrescol, taurescol, murescol))

    '''for i in range(Nleg):

        states = varD[i * Nstates:(i + 1) * Nstates]  # orig intervals

        timeend = timestart + varD[varTot + i]
        time_vec = np.concatenate((time_vec, (timeend,)))  # vector with time points
        obj.time = np.hstack((obj.time, timeend))

        for k in range(NContPoints):
            #this for loop takes the controls from the optimization variable and stores them into different variable
            #here controls are scaled
            alfa[k] = varD[varStates + i * (Ncontrols * NContPoints) + Ncontrols * k]
            delta[k] = varD[varStates + i * (Ncontrols * NContPoints) + 1 + Ncontrols * k]
            deltaf[k] = varD[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
            tau[k] = varD[varStates + i * (Ncontrols * NContPoints) + 3 + Ncontrols * k]
            mu[k] = varD[varStates + i * (Ncontrols * NContPoints) + 4 + Ncontrols * k]

        controls = np.vstack((alfa, delta, deltaf, tau, mu)) # orig intervals


        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures, \
        alfa_I, delta_I, deltaf_I, tau_I, mu_I = SingleShooting(states, controls, dyn, timestart, timeend, Nint)

        states_atNode = np.concatenate((states_atNode, (
            (vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]))))  # new intervals
        #res quantities are unscaled

        vres = np.reshape(vres, (Nint, 1))
        chires = np.reshape(chires, (Nint, 1))
        gammares = np.reshape(gammares, (Nint, 1))
        tetares = np.reshape(tetares, (Nint, 1))
        lamres = np.reshape(lamres, (Nint, 1))
        hres = np.reshape(hres, (Nint, 1))
        mres = np.reshape(mres, (Nint, 1))
        alfares = np.reshape(alfares, (Nint, 1))
        deltares = np.reshape(deltares, (Nint, 1))
        deltafres = np.reshape(deltafres, (Nint, 1))
        mures = np.reshape(mures, (Nint, 1))
        taures = np.reshape(taures, (Nint, 1))

        #here controls and states matrices are defined
        if i == 0:
            states_after = np.column_stack((vres, chires, gammares, tetares, lamres, hres, mres))
            controls_after = np.column_stack((alfares, deltares, deltafres, taures, mures))
        else:
            states_after = np.vstack(
                (states_after, np.column_stack((vres, chires, gammares, tetares, lamres, hres, mres))))
            controls_after = np.vstack((controls_after, np.column_stack((alfares, deltares, deltafres, taures, mures))))

        timestart = timeend'''


    obj.States = states_after
    obj.Controls = controls_after

    eq_c = equality(varD, states_atNode, controls_atNode)

    ineq_c = inequalityAll(states_after, controls_after, Nint*Nleg)

    h = states_after[-1, 5]
    m = states_after[-1, 6]
    delta = controls_after[-1, 1]
    tau = 0.0 #controls_after[-1, 3]

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10, obj.lRef,
                                obj.xcgf, obj.xcg0)

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / (obj.g0 * isp))
    cost = -mf / obj.M0

    if obj.bad:
        cost = -0.05
        ineq_c = ineq_c * 1e3
        eq_c = eq_c * 1e3
        obj.bad = False

    obj.costOld = cost
    obj.varOld = var
    obj.ineqOld = ineq_c
    obj.eqOld = eq_c

    return eq_c, ineq_c, cost


def plot(var, Nint):

    time = np.zeros((1))
    timeTotal = np.zeros((0))
    alfaCP = np.zeros((Nleg, NContPoints))
    deltaCP = np.zeros((Nleg, NContPoints))
    #deltafCP = np.zeros((Nleg, NContPoints))
    #tauCP = np.zeros((Nleg, NContPoints))
    #muCP = np.zeros((Nleg, NContPoints))
    if flag_save:
        res = open("res_{}_{}.txt".format(os.path.basename(__file__), timestr), "w")
    tCtot = np.zeros((0))
    timestart = 0.0
    varD = var * (obj.UBV - obj.LBV) + obj.LBV
    for i in range(Nleg):
        alfa = np.zeros((NContPoints))
        delta = np.zeros((NContPoints))
        #deltaf = np.zeros((NContPoints))
        #tau = np.zeros((NContPoints))
        #mu = np.zeros((NContPoints))
        states = varD[i * Nstates:(i + 1) * Nstates]  # orig intervals

        timeend = timestart + varD[i + varTot]
        timeTotal = np.linspace(timestart, timeend, Nint)
        time = np.concatenate((time, (timeend,)))
        tC = np.linspace(timestart, timeend, NContPoints)
        tCtot = np.concatenate((tCtot, tC))

        for k in range(NContPoints):
            alfa[k] = varD[varStates + i * (Ncontrols * NContPoints) + Ncontrols * k]
            alfaCP[i, k] = alfa[k]
            delta[k] = varD[varStates + i * (Ncontrols * NContPoints) + 1 + Ncontrols * k]
            deltaCP[i, k] = delta[k]
            #deltaf[k] = varD[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
            #deltafCP[i, k] = deltaf[k]
            #tau[k] = varD[varStates + i * (Ncontrols * NContPoints) + 3 + Ncontrols * k]
            #tauCP[i, k] = tau[k]
            #mu[k] = varD[varStates + i * (Ncontrols * NContPoints) + 4 + Ncontrols * k]
            #muCP[i, k] = mu[k]
        controls = np.vstack((alfa, delta))#, deltaf, tau, mu)) # orig intervals


        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures, alfa_I, delta_I \
            = SingleShooting(states, controls, dynamicsInt, timestart, timeend, Nint)

        rep = len(vres)

        Press, rho, c = isaMulti(hres, obj.psl, obj.g0, obj.Re)
        Press = np.asarray(Press, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        M = vres / c

        L, D, MomA = aeroForcesMulti(M, alfares, deltafres, cd, cl, cm, vres, obj.wingSurf, rho, obj.lRef, obj.M0, mres,
                                 obj.m10, obj.xcg0, obj.xcgf, obj.pref, rep)
        L = np.asarray(L, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        MomA = np.asarray(MomA, dtype=np.float64)

        T, Deps, isp, MomT = thrustMulti(Press, mres, presv, spimpv, deltares, taures, rep, obj.psl, obj.M0, obj.m10,
                                     obj.lRef, obj.xcgf, obj.xcg0)
        T = np.asarray(T, dtype=np.float64)
        isp = np.asarray(isp, dtype=np.float64)
        Deps = np.asarray(Deps, dtype=np.float64)
        MomT = np.asarray(MomT, dtype=np.float64)

        MomTot = MomA + MomT

        r1 = hres + obj.Re
        Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
        Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
        mf = mres / np.exp((Dv1 + Dv2) / (obj.g0*isp))

        g0 = obj.g0
        eps = Deps + alfares
        g = []
        for alt in hres:
            if alt == 0:
                g.append(g0)
            else:
                g.append(obj.g0 * (obj.Re / (obj.Re + alt)) ** 2)
        g = np.asarray(g, dtype=np.float64)
        # dynamic pressure

        q = 0.5 * rho * (vres ** 2)

        # accelerations

        ax = (T * np.cos(Deps) - D * np.cos(alfares) + L * np.sin(alfares)) / mres
        az = (T * np.sin(Deps) + D * np.sin(alfares) + L * np.cos(alfares)) / mres

        if flag_save:
            res.write("Number of leg: " + str(Nleg) + "\n" + "Max number Optimization iterations: " + str(maxIterator) + "\n"
                + "Number of NLP iterations: " + str(maxiter) + "\n" + "Leg Number:" + str(i) + "\n" + "v: " + str(
                vres) + "\n" + "Chi: " + str(np.rad2deg(chires))
                + "\n" + "Gamma: " + str(np.rad2deg(gammares)) + "\n" + "Teta: " + str(
                np.rad2deg(tetares)) + "\n" + "Lambda: "
                + str(np.rad2deg(lamres)) + "\n" + "Height: " + str(hres) + "\n" + "Mass: " + str(
                mres) + "\n" + "mf: " + str(mf) + "\n"
                + "Objective Function: " + str(-mf / obj.M0) + "\n" + "Alfa: "
                + str(np.rad2deg(alfares)) + "\n" + "Delta: " + str(deltares) + "\n" + "Delta f: " + str(
                np.rad2deg(deltafres)) + "\n"
                + "Tau: " + str(taures) + "\n" + "Eps: " + str(np.rad2deg(eps)) + "\n" + "Lift: "
                + str(L) + "\n" + "Drag: " + str(D) + "\n" + "Thrust: " + str(T) + "\n" + "Spimp: " + str(
                isp) + "\n" + "c: "
                + str(c) + "\n" + "Mach: " + str(M) + "\n" + "Time vector: " + str(timeTotal) + "\n" + "Press: " + str(
                Press) + "\n" + "Dens: " + str(rho) + "\n" + "Time elapsed during optimization: " + tformat)

        plt.figure(0)
        plt.title("Velocity")
        plt.plot(timeTotal, vres)
        plt.grid()
        plt.ylabel("m/s")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "velocity" + ".png")


        plt.figure(1)
        plt.title("Flight path angle \u03C7")
        plt.plot(timeTotal, np.rad2deg(chires))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "chi" + ".png")


        plt.figure(2)
        plt.title("Angle of climb \u03B3")
        plt.plot(timeTotal, np.rad2deg(gammares))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "gamma" + ".png")


        plt.figure(3)
        plt.title("Longitude \u03B8")
        plt.plot(timeTotal, np.rad2deg(tetares))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "theta" + ".png")


        plt.figure(4)
        plt.title("Latitude \u03BB")
        plt.plot(timeTotal, np.rad2deg(lamres))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "lambda" + ".png")


        plt.figure(5)
        plt.title("Flight angles")
        plt.plot(timeTotal, np.rad2deg(chires), color="g")
        plt.plot(timeTotal, np.rad2deg(gammares), color="b")
        plt.plot(timeTotal, np.rad2deg(tetares), color="r")
        plt.plot(timeTotal, np.rad2deg(lamres), color="k")
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Chi", "Gamma", "Theta", "Lambda"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "angles" + ".png")


        plt.figure(6)
        plt.title("Altitude")
        plt.plot(timeTotal, hres / 1000)
        plt.grid()
        plt.ylabel("km")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "altitude" + ".png")


        plt.figure(7)
        plt.title("Mass")
        plt.plot(timeTotal, mres)
        plt.grid()
        plt.ylabel("kg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "mass" + ".png")


        plt.figure(8)
        plt.title("Angle of attack \u03B1")
        plt.plot(tC, np.rad2deg(alfaCP[i, :]), 'ro')
        plt.plot(timeTotal, np.rad2deg(alfares))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "alpha" + ".png")


        plt.figure(9)
        plt.title("Throttles")
        plt.plot(timeTotal, deltares * 100, color='r')
        #plt.plot(timeTotal, taures * 100, color='k')
        plt.plot(tC, deltaCP[i, :] * 100, 'ro')
        #plt.plot(tC, tauCP[i, :] * 100, 'ro')
        plt.grid()
        plt.ylabel("%")
        plt.xlabel("time [s]")
        plt.legend(["Delta", "Tau", "Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "throttles" + ".png")


        '''plt.figure(10)
        plt.title("Body Flap deflection \u03B4")
        plt.plot(tC, np.rad2deg(deltafCP[i, :]), "ro")
        plt.plot(timeTotal, np.rad2deg(deltafres))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "deltaf" + ".png")'''

        '''plt.figure(11)
        plt.title("Bank angle profile \u03BC")
        plt.plot(tC, np.rad2deg(muCP[i, :]), "ro")
        plt.plot(timeTotal, np.rad2deg(mures))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "mu" + ".png")'''


        plt.figure(12)
        plt.title("Dynamic Pressure")
        plt.plot(timeTotal, q / 1000)
        plt.grid()
        plt.ylabel("kPa")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "dynPress" + ".png")


        plt.figure(13)
        plt.title("Accelerations")
        plt.plot(timeTotal, ax, color='b')
        plt.plot(timeTotal, az, color='r')
        plt.grid()
        plt.ylabel("m/s^2")
        plt.xlabel("time [s]")
        plt.legend(["ax", "az"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "accelerations" + ".png")

        plt.figure(15)
        plt.title("Forces")
        plt.plot(timeTotal, T / 1000, color='r')
        plt.plot(timeTotal, L / 1000, color='b')
        plt.plot(timeTotal, D / 1000, color='k')
        plt.grid()
        plt.ylabel("kN")
        plt.xlabel("time [s]")
        plt.legend(["Thrust", "Lift", "Drag"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "forces" + ".png")


        plt.figure(16)
        plt.title("Mach")
        plt.plot(timeTotal, M)
        plt.grid()
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "mach" + ".png")


        plt.figure(17)
        plt.title("Total pitching Moment")
        plt.plot(timeTotal, MomTot / 1000, color='k')
        plt.grid()
        plt.axhline(5, 0, timeTotal[-1], color='r')
        plt.axhline(-5, 0, timeTotal[-1], color='r')
        plt.ylabel("kNm")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        if flag_save:
            plt.savefig(savefig_file + "moment" + ".png")

        figs = plt.figure(18)
        ax = figs.add_subplot(111, projection='3d')
        ax.plot(np.rad2deg(tetares), np.rad2deg(lamres), hres / 1e3, color='b', label="3d Trajectory")
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        ax.set_zlabel('Altitude [km]')
        ax.legend()
        if flag_save:
            plt.savefig(savefig_file + "traj" + ".png")


        timestart = timeend


    plt.show(block=True)
    print("m before Ho : {0:.5f}".format(mres[-1]))
    print("mf          : {0:.5f}".format(mf[-1]))
    print("altitude Hohmann starts: {0:.5f}".format(hres[-1]))
    print("final time  : {}".format(time))

    if flag_save:
        res.close()

    '''plt.close(0)
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
    plt.close(15)
    plt.close(16)
    plt.close(17)
    plt.close(18)'''


def equality(var, conj, cont_conj):
    h = conj[varStates - 2]
    lam = conj[varStates - 3]
    stat = conj[varStates - Nstates:]
    cont = obj.Controls[-1, :]

    vtAbs, chiass, vtAbs2 = vass(stat, cont, dynamicsVel, obj.omega)

    vvv = np.sqrt(obj.GMe / (obj.Re + h))

    if np.cos(obj.incl) / np.cos(lam) > 1:
        chifin = np.pi
    elif np.cos(obj.incl) / np.cos(lam) < - 1:
        chifin = 0.0
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam))
    div = np.tile([100, 1, 1, 1, 1, 100, 1000], Nleg - 1)
    #div2 = np.tile([np.deg2rad(40),1,np.deg2rad(30),1,np.deg2rad(90)], Nleg - 1)
    div2 = np.tile([np.deg2rad(40), 1], Nleg - 1)
    contr_left = np.zeros((0))
    contr_right = np.zeros((0))
    for i in range(Nbar-1):
        contr_left = np.hstack((contr_left, var[varStates+Ncontrols*NContPoints*i:varStates+Ncontrols*(NContPoints*i+1)-1]))
        contr_right = np.hstack((contr_right, var[varStates+Ncontrols*(NContPoints*i+1):varStates+Ncontrols*(NContPoints*i+2)-1]))

    eq_cond = np.zeros((0))
    eq_cond = np.concatenate((eq_cond, (var[0] - states_init[0],), var[2:7]-states_init[2:]))
    eq_cond = np.concatenate((eq_cond, (var[Nstates:varStates] - conj[:Nstates * (Nleg - 1)])/div))  # knotting conditions
    #eq_cond = np.concatenate((eq_cond, (contr_left - contr_right)/div2)) # knotting condition on controls
    #eq_cond = np.concatenate((eq_cond, abs((contr_check - cont_conj[:Ncontrols * (Nleg - 1)])/div2)))  # knotting conditions
    eq_cond = np.concatenate((eq_cond, var[varStates:varStates + Ncontrols] - cont_init))  # init cond on alpha
    eq_cond = np.concatenate((eq_cond, ((vvv - vtAbs)/vvv,)))
    eq_cond = np.concatenate((eq_cond, ((chifin - chiass)/chifin,)))
    eq_cond = np.concatenate((eq_cond, (conj[varStates - 5],)))  # final condition on gamma
    #for j in range(len(eq_cond)):
     #   if eq_cond[j] > 1:
      #      print("eq", eq_cond[j], j)

    return eq_cond


def constraints(var, type):
    if (var==obj.varOld).all():
        if type == "eq":
            return obj.eqOld
        else:
            return obj.ineqOld
    else:
        eq, ineq, cost = MultiShooting(var, dynamicsInt)
        if type == "eq":
            return eq
        else:
            return ineq


def cost_fun(var):
    if (var==obj.varOld).all():
        return obj.costOld
    else:
        eq, ineq, cost = MultiShooting(var, dynamicsInt)
        return cost


def do(eps, dx, jac, x0, sp, f0, count):
    dx[count] = eps
    jac[count] = (cost_fun(x0 + dx, sp) - f0) / eps
    dx[count] = 0.0
    return dx, jac


def JacFunSave(var, sp):
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(cost_fun(x0, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = eps
        jac[i] = (cost_fun(x0 + dx, sp) - f0) / eps
        dx[i] = 0.0
    sparse = csc_matrix(jac)
    save_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacCost", sparse)
    return jac.transpose()


def JacEqSave(var, type, sp):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var,type, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (constraints(var+dx, type, sp) - f0) / epsilon
        dx[i] = 0.0
    sparse = csc_matrix(jac)
    save_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacEq", sparse)
    return jac.transpose()


def JacIneqSave(var, type, sp):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var, type, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (constraints(var+dx, type, sp) - f0) / epsilon
        dx[i] = 0.0
    sparse = csc_matrix(jac)
    save_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacIneq", sparse)
    return jac.transpose()


def JacFun(var, sp):
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(cost_fun(x0, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    row = np.shape(sp)[0]
    for i in range(row):
        if sp[i, 0] != 0:
            dx[i] = eps
            jac[i] = (cost_fun(x0 + dx, sp) - f0) / eps
            dx[i] = 0.0
    return jac.transpose()


def JacEq(var, type, sp):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var,type, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    row = np.shape(sp)[0]
    column = np.shape(sp)[1]
    for i in range(row):
        for j in range(column):
            if sp[i, j] != 0:
                dx[i] = epsilon
                jac[i] = (constraints(var+dx, type, sp) - f0) / epsilon
                dx[i] = 0.0
                break

    return jac.transpose()


def JacIneq(var, type, sp):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var, type, sp))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))

    row = np.shape(sp)[0]
    column = np.shape(sp)[1]
    for i in range(row):
        for j in range(column):
            if sp[i, j] != 0:
                dx[i] = epsilon
                jac[i] = (constraints(var+dx, type, sp) - f0) / epsilon
                dx[i] = 0.0
                break
    return jac.transpose()


cons = ({'type': 'eq',
         'fun': constraints,
         'args':("eq",)},
        {'type': 'ineq',
         'fun': constraints,
         'args': ("ineq",)})  # equality and inequality constraints

'''NLP SOLVER'''
if __name__ == '__main__':
    timestr = time.strftime("%Y%m%d-%H%M%S")
    savefig_file = "MultiShooting_{}_{}_".format(os.path.basename(__file__), timestr)
    flag_save = True
    obj = Spaceplane()
    start = time.time()

    '''reading of aerodynamic coefficients and specific impulse from file'''

    cl = np.array(fileReadOr("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/clfile.txt"))
    cd = np.array(fileReadOr("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cdfile.txt"))
    cm = np.array(fileReadOr("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cmfile.txt"))

    # cl_interp = RegularGridInterpolator((mach, angAttack, bodyFlap), cl)
    # cd_interp = RegularGridInterpolator((mach, angAttack, bodyFlap), cd)
    # cm_interp = RegularGridInterpolator((mach, angAttack, bodyFlap), cm)

    #sparseJacCost = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacCost.npz")
    #spCost = sparseJacCost.todense()
    #sparseJacEq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacEq.npz")
    #spEq = sparseJacEq.todense()
    #sparseJacIneq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacIneq.npz")
    #spIneq = sparseJacIneq.todense()
    # sparseJacEq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacEq.npz")
    # spEq = sparseJacEq.todense()
    # sparseJacIneq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacIneq.npz")
    # spIneq = sparseJacIneq.todense()

    with open("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/impulse.dat") as f:
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

    '''set problem parameters'''

    time_tot = 350  # initial time
    Nbar = 6  # number of conjunction points
    Nleg = Nbar - 1  # number of multiple shooting sub intervals
    NContPoints = 5  # number of control points for interpolation inside each interval
    Nint = 400  # number of points for each single shooting integration
    Nstates = 7  # number of states
    Ncontrols = 2  # number of controls
    varStates = Nstates * Nleg  # total number of optimization variables for states
    varControls = Ncontrols * Nleg * NContPoints  # total number of optimization variables for controls
    varTot = varStates + varControls  # total number of optimization variables for states and controls
    NineqCond = Nint
    tnew = np.linspace(0, time_tot, Nbar)  # time vector used for interpolation of states initial guess
    tcontr = np.linspace(0, time_tot,
                         int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess
    tstat = np.linspace(0, time_tot, Nleg)
    unit_t = 1000
    Nintplot = 1000

    '''NLP solver parameters'''
    maxiter = 20  # max number of iterations for nlp solver
    ftol = 1e-8  # numeric tolerance of nlp solver
    # eps = 1e-10
    eps = 1e-09  # increment of the derivative
    # u = 2.220446049250313e-16
    maxIterator = 4  # max number of optimization iterations
    if flag_save:
        os.makedirs("/home/francesco/Desktop/PhD/FESTIP_Work/MultipleShooting_Algorithm/SLSQP/Results/Res{}_leg{}_it{}x{}_{}".format(
            os.path.basename(__file__), Nleg, maxiter, maxIterator, timestr))
        savefig_file = "/home/francesco/Desktop/PhD/FESTIP_Work/MultipleShooting_Algorithm/SLSQP/Results/Res{}_leg{}_it{}x{}_{}/Res_".format(
            os.path.basename(__file__), Nleg, maxiter, maxIterator, timestr)
    '''definiton of initial conditions'''

    # set vector of initial conditions of states and controls
    X = np.zeros((0))
    U = np.zeros((0))

    '''vars = smooth_init(Nleg, NContPoints)

    v_init = vars[0]
    chi_init = vars[1]
    gamma_init = vars[2]
    teta_init = vars[3]
    lam_init = vars[4]
    h_init = vars[5]
    m_init = vars[6]
    alfa_init = vars[7]
    delta_init = vars[8]
    deltaf_init = vars[9]
    tau_init = vars[10]
    mu_init = vars[11]
    v_init[0] = 1.0
    chi_init[0] = obj.chistart
    gamma_init[0] = obj.gammastart
    teta_init[0] = obj.longstart
    lam_init[0] = obj.latstart
    h_init[0] = 1.0
    m_init[0] = obj.M0
    alfa_init[0] = 0.0
    delta_init[0] = 1.0
    deltaf_init[0] = 0.0
    tau_init[0] = 0.0
    mu_init[0] = 0.0'''
    v_init = Guess.linear(tstat, 1, obj.Vtarget)
    chi_init = Guess.linear(tstat, obj.chistart, obj.chi_fin)
    gamma_init = Guess.linear(tstat, obj.gammastart, 0.0)
    teta_init = Guess.constant(tstat, obj.longstart)
    lam_init = Guess.constant(tstat, obj.latstart)
    h_init = Guess.linear(tstat, 1, obj.Hini)
    m_init = Guess.linear(tstat, obj.M0, obj.m10)

    alfa_init = Guess.zeros(tcontr)
    delta_init = Guess.linear(tcontr, 1.0, 0.05)
    #deltaf_init = Guess.zeros(tcontr)
    #tau_init = Guess.zeros(tcontr)
    #mu_init = Guess.zeros(tcontr)

    states_init = np.array((v_init[0], chi_init[0], gamma_init[0], teta_init[0], lam_init[0], h_init[0], m_init[0]))
    cont_init = np.array((alfa_init[0], delta_init[0]))#, deltaf_init[0], tau_init[0], mu_init[0]))

    XGuess = np.array((v_init, chi_init, gamma_init, teta_init, lam_init, h_init, m_init))  # states initial guesses

    UGuess = np.array((alfa_init, delta_init))#, deltaf_init, tau_init, mu_init))  # states initial guesses

    for i in range(Nleg):
        '''creation of vector of states initial guesses'''
        for j in range(Nstates):
            X = np.hstack((X, XGuess[j][i]))

    for i in range(int(varControls / Ncontrols)):
        '''creation of vector of controls initial guesses'''
        for j in range(Ncontrols):
            U = np.hstack((U, UGuess[j][i]))

    dt = np.zeros((0))
    for i in range(len(tnew) - 1):
        '''creation of vector of time intervals'''
        dt = np.hstack((dt, tnew[i + 1] - tnew[i]))

    uplimx = np.tile([1e4, np.deg2rad(150), np.deg2rad(89.9), 0.0, np.deg2rad(25), 2e5, obj.M0], Nleg)
    inflimx = np.tile([1.0, np.deg2rad(110), np.deg2rad(-50), np.deg2rad(-60), np.deg2rad(2), 1.0, obj.m10], Nleg)
    #uplimu = np.tile([np.deg2rad(40), 1.0, np.deg2rad(30), 1.0, np.deg2rad(90)], Nleg * NContPoints)
    #inflimu = np.tile([np.deg2rad(-2), 0.0001, np.deg2rad(-20), -1.0, np.deg2rad(-90)], Nleg * NContPoints)
    uplimu = np.tile([np.deg2rad(40), 1.0], Nleg * NContPoints)
    inflimu = np.tile([np.deg2rad(-2), 0.0001], Nleg * NContPoints)

    for i in range(len(X)):
        if X[i] < inflimx[i]:
            X[i] = inflimx[i]
        if X[i] > uplimx[i]:
            X[i] = uplimx[i]
    for j in range(len(U)):
        if U[j] < inflimu[j]:
            U[j] = inflimu[j]
        if U[j] > uplimu[j]:
            U[j] = uplimu[j]
    X0d = np.hstack((X, U, dt))  # vector of initial conditions here all the angles are in degrees!!!!!
    obj.varOld = np.zeros((len(X0d)))

    #LbS, LbC, UbS, UbC = bound_def(X, U, uplimx, inflimx, uplimu, inflimu)

    LbS = [0.5, np.deg2rad(110), np.deg2rad(88), np.deg2rad(-53), np.deg2rad(4.8), 0.5, obj.M0 - 10,
           10, np.deg2rad(100), np.deg2rad(50), np.deg2rad(-60), np.deg2rad(2.0), 1e3, 2.5e5,
           100, np.deg2rad(100), np.deg2rad(0), np.deg2rad(-60), np.deg2rad(2.0), 1e4, 1.5e5,
           500, np.deg2rad(100), np.deg2rad(-50), np.deg2rad(-60), np.deg2rad(2.0), 2e4, 1e5,
           1000, np.deg2rad(100), np.deg2rad(-20), np.deg2rad(-60), np.deg2rad(2.0), 5e4, 5e4]

    UbS = [1.5, np.deg2rad(115), np.deg2rad(89.99), np.deg2rad(-51), np.deg2rad(5.8), 1.5, obj.M0,
           1500, np.deg2rad(150), np.deg2rad(70), np.deg2rad(-45), np.deg2rad(8.0), 3e4, 3.5e5,
           3500, np.deg2rad(150), np.deg2rad(30), np.deg2rad(-45), np.deg2rad(15.0), 8e4, 3e5,
           5500, np.deg2rad(150), np.deg2rad(30), np.deg2rad(-45), np.deg2rad(25.0), 1e5, 2e5,
           6500, np.deg2rad(150), np.deg2rad(30), np.deg2rad(-45), np.deg2rad(25.0), 1.2e5, 1e5]

    LbC = [np.deg2rad(-1.0), 0.9,#np.deg2rad(-20.0), -0.1, np.deg2rad(-1),
           np.deg2rad(-1.0), 0.9,#np.deg2rad(-20.0), -0.2, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -0.2, np.deg2rad(-5),
           np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -0.5, np.deg2rad(-10),
           #np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -0.5, np.deg2rad(-15),
           #np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -0.6, np.deg2rad(-20),
           np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -0.6, np.deg2rad(-25), # leg1
           np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -0.8, np.deg2rad(-30),
           #np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -1, np.deg2rad(-35),
           #np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -1, np.deg2rad(-60),
           np.deg2rad(-2.0), 0.9,#np.deg2rad(-20.0), -1, np.deg2rad(-70),
           np.deg2rad(-2.0), 0.8,#np.deg2rad(-20.0), -1, np.deg2rad(-70),
           np.deg2rad(-2.0), 0.7,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.7,#np.deg2rad(-20.0), -1, np.deg2rad(-90), # leg2
           np.deg2rad(-2.0), 0.6,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           #np.deg2rad(-2.0), 0.6,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           #np.deg2rad(-2.0), 0.5,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.5,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.4,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.4,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.3,#np.deg2rad(-20.0), -1, np.deg2rad(-90), # leg3
           np.deg2rad(-2.0), 0.3,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           #np.deg2rad(-2.0), 0.2,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           np.deg2rad(-2.0), 0.2,#np.deg2rad(-20.0), -1, np.deg2rad(-90),
           #np.deg2rad(-2.0), 0.1,#np.deg2rad(-20.0), -1, np.deg2rad(-80),
           np.deg2rad(-2.0), 0.1,#np.deg2rad(-20.0), -0.9, np.deg2rad(-50),
           np.deg2rad(-2.0), 0.05, #np.deg2rad(-20.0), -0.8, np.deg2rad(-20),
           np.deg2rad(-2.0), 0.01, #np.deg2rad(-20.0), -0.5, np.deg2rad(-10), # leg4
           np.deg2rad(-2.0), 0.001,#np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           #np.deg2rad(-2.0), 0.001,#np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           #np.deg2rad(-2.0), 0.001,#np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.001,#np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.001,#np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.001,#np.deg2rad(-20.0), -0.01, np.deg2rad(-1),
           np.deg2rad(-2.0), 0.001]#np.deg2rad(-20.0), -0.01, np.deg2rad(-1)] # leg5

    UbC = [np.deg2rad(3.0), 1.0, #np.deg2rad(30), 0.1, np.deg2rad(1),
           np.deg2rad(5.0), 1.0, #np.deg2rad(30.0), 0.2, np.deg2rad(1),
           #np.deg2rad(10.0), 1.0, #np.deg2rad(30.0), 0.2, np.deg2rad(5),
           np.deg2rad(15.0), 1.0, #np.deg2rad(30.0), 0.5, np.deg2rad(10),
           #np.deg2rad(20.0), 1.0, #np.deg2rad(30.0), 0.5, np.deg2rad(15),
           np.deg2rad(30.0), 1.0, #np.deg2rad(30.0), 0.6, np.deg2rad(20),
           np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 0.6, np.deg2rad(25), # leg1
           np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 0.8, np.deg2rad(30),
           #np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(35),
           #np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(60),
           np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(70),
           np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(70),
           np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(90),
           np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(90), # leg2
           #np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(90),
           #np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(90),
           np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 1.0, #np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.7, #np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.6, #np.deg2rad(30.0), 1, np.deg2rad(90),
           np.deg2rad(40.0), 0.6, #np.deg2rad(30.0), 1, np.deg2rad(90.0), # leg3
           #np.deg2rad(40.0), 0.5, #np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.5, #np.deg2rad(30.0), 1, np.deg2rad(90.0),
           np.deg2rad(40.0), 0.5, #np.deg2rad(30.0), 1, np.deg2rad(90.0),
           #np.deg2rad(40.0), 0.4, #np.deg2rad(30.0), 1, np.deg2rad(80.0),
           np.deg2rad(40.0), 0.4, #np.deg2rad(30.0), 0.9, np.deg2rad(50.0),
           np.deg2rad(40.0), 0.3, #np.deg2rad(30.0), 0.8, np.deg2rad(20.0),
           np.deg2rad(40.0), 0.25, #np.deg2rad(30.0), 0.5, np.deg2rad(10.0), # leg4
           np.deg2rad(40.0), 0.25, #np.deg2rad(30.0), 0.01, np.deg2rad(1),
           #np.deg2rad(40.0), 0.2,#np.deg2rad(30.0), 0.01, np.deg2rad(1),
           #np.deg2rad(40.0), 0.2,#np.deg2rad(30.0), 0.01, np.deg2rad(1),
           np.deg2rad(40.0), 0.2,#np.deg2rad(30.0), 0.01, np.deg2rad(1),
           np.deg2rad(40.0), 0.2,#np.deg2rad(30.0), 0.01, np.deg2rad(1),
           np.deg2rad(40.0), 0.2,#np.deg2rad(30.0), 0.01, np.deg2rad(1),
           np.deg2rad(40.0), 0.2]#np.deg2rad(30.0), 0.01, np.deg2rad(1)] #leg5

    Tlb = [3.0]
    Tub = [200]

    obj.LBV = np.hstack((LbS, LbC, np.repeat(Tlb, Nleg)))
    obj.UBV = np.hstack((UbS, UbC, np.repeat(Tub, Nleg)))
    X0a = (X0d - obj.LBV)/(obj.UBV - obj.LBV)
    # X0 has first all X guesses and then all U guesses
    # at this point the vectors of initial guesses for states and controls for every time interval are defined

    '''set upper and lower bounds for states, controls and time, scaled'''
    '''major issues with bounds!!!'''
    bndX = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    bndU = ((0.0, 1.0), (0.0, 1.0)) #, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    bndT = ((0.0, 1.0),)
    bnds = bndX * Nleg + bndU * Nleg * NContPoints + bndT * Nleg

    iterator = 0
    p = Pool(processes=Nleg)

    while iterator < maxIterator:
        print("---- iteration : {0} ----".format(iterator + 1))

        opt = optimize.minimize(cost_fun,
                                X0a,
                                constraints=cons,
                                bounds=bnds,
                                method='SLSQP',
                                options={"maxiter": maxiter,
                                         "ftol": ftol,
                                         "iprint":2,
                                         "disp":True})

        X0a = opt.x

        if not (opt.status):
            break
        iterator += 1

    p.close()
    p.join()

    end = time.time()
    time_elapsed = end-start
    tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
    print("Time elapsed for total optimization ", tformat)
    plot(X0a, Nintplot)
    plt.show(block=True)
    #np.save("opt", X0a)
'''
    def mask_MS_2(var):
        eq_Cond, ineq_Cond, objective = MultiShooting(var, dynamicsInt)

        try:
            eq_Cond, ineq_Cond, objective = MultiShooting(var, dynamicsInt)
            a = sum(np.maximum(ineq_Cond, np.zeros((len(ineq_Cond)))))
            b = sum(abs(eq_Cond))
            J = objective + a + b
            if np.isnan(J) or not(np.isreal(J)):
                J=1e6+random.random() * 1e6
        except:
            J=1e8+random.random() * 1e8

        return J, a, b


    X0a = np.load("opt.npy")
    Jorig, ineq, eq = mask_MS_2(X0a)
    centerA = X0a[varStates]
    #centerB = X0a[varStates+Ncontrols]
    d = 0.1
    infA = centerA - d
    supA = centerA + d
    #infB = centerB - d
    #supB = centerB + d
    if supA > 1.0:
        supA = 1.0
    if infA < 0.0:
        infA = 0.0
    #if supB>1.0:
     #   supB = 1.0
    #if infB < 0.0:
    #    infB = 0.0
    xA = infA
    #xB = infB
    vvv = np.array((xA, Jorig, ineq, eq))
    while xA <= supA: #and xB <= supB:
        X0a[varStates] = xA
        #X0a[varStates+Ncontrols] = xB
        J, ineq, eq = mask_MS_2(X0a)
        vvv = np.vstack((vvv, np.array((xA*(uplimu[0] - inflimu[0])+inflimu[0], J, ineq, eq))))
        xA = xA + 0.0001
        #xB = xB + 0.0001
    plt.figure()
    plt.grid()
    axes = plt.gca()
    plt.xlabel("alpha [rad]")
    plt.ylabel("Objective function")
    axes.set_ylim([2000, 2500.0])
    plt.plot(vvv[1:, 0], vvv[1:, 1])
    #plt.plot(vvv[1:, 1], vvv[1:, 2])
    plt.plot(centerA*(uplimu[0] - inflimu[0])+inflimu[0], Jorig, marker='.', label="Alpha 0")
    print("Ineq: {}, Eq: {}".format(max(vvv[1:,2]), max(vvv[1:,3])))
    #plt.plot(centerB*(uplimu[Ncontrols] - inflimu[Ncontrols])+inflimu[Ncontrols], Jorig, marker='.', label="Alpha 1")
    plt.savefig(savefig_file + "obj_fun_alfa" + ".png")
    plt.legend(loc="best")
    plt.show()
'''

