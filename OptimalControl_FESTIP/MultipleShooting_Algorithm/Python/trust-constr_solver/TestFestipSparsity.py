import matplotlib.pyplot as plt
from OpenGoddard.optimize import Guess
from scipy import interpolate, optimize
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint, basinhopping, shgo
from models import *
from mapping_functions import *
import time
import datetime
import os
from scipy.sparse import csc_matrix, save_npz, load_npz
import sys
from smooth_fun import *
import multiprocessing
from functools import partial
import os
sys.path.insert(0, 'home/francesco/Desktop/PhD/FESTIP_Work')


'''Old TestFestipSparsity'''
'''modified initial guess for controls so it has more points than states. States points are only in the conjunction of legs while controls are also inside'''
'''simplified version of problem. All controls but equatorial trajectory and no earth rotation'''
'''definition of vehicle and mission parameters. Angles are set in degrees'''
'''maybe best to set initial values of angles in radians'''

'''DIFFERENT METHOD FOR INTEGRATION'''

'''set initial conditions constraints on all parameters'''

'''try to reduce the dynamic evaluations at one insted of two. maybe by passing arguments to the functions'''

start = time.time()
timestr = time.strftime("%Y%m%d-%H%M%S")
flag_save = True
laptop = False
if flag_save and laptop:
    os.makedirs("/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm"
               "/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}_".format(os.path.basename(__file__), timestr))
    savefig_file = "/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}/Plot_".format(os.path.basename(__file__), timestr)
    savedata_file ="/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Personal/OptimalControl_FESTIP/MUltipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}_/Data_".format(os.path.basename(__file__), timestr)
elif flag_save and not laptop:
    os.makedirs("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}_".format(os.path.basename(__file__), timestr))
    savefig_file = "/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}/Plot_".format(os.path.basename(__file__), timestr)
    savedata_file ="/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/trust-constr_solver/Results/MultiShooting_trust-constr_{}_{}_/Data_".format(os.path.basename(__file__), timestr)

'''vehicle parameters'''


class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = np.deg2rad(5.2)  # deg latitude
        self.longstart = np.deg2rad(-52.775)  # deg longitude
        self.chistart = np.deg2rad(125)  # deg flight direction
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
        self.k = 5e3  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 100000
        self.hvert = 100
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = 0.5 * np.pi + np.arcsin(np.cos(self.incl) / np.cos(self.latstart))
        self.varOld = np.zeros((0))
        self.costOld = np.zeros((0))
        self.eqOld = np.zeros((0))
        self.ineqOld = np.zeros((0))
        self.States = np.zeros((0))
        self.Controls = np.zeros((0))




'''definition of initial and final conditions for states and controls for equality constraints'''

#init_condX = np.array((1 / unit_v, obj.chistart / unit_chi, obj.gammastart / unit_gamma, obj.longstart / unit_teta,
 #                      obj.latstart / unit_lam, 1 / unit_h, obj.M0 / unit_m))
#init_condU = np.array((0.0, 1.0, 0.0, 0.0))
#final_cond = np.array((0.0))  # final conditions on gamma

'''function definitions'''


def conversionStatesToOrig(var, len):
    new_var = np.zeros((len*Nstates))
    i = 0
    for i in range(len):
        new_var[i * Nstates] = v_toOrig(var[i * Nstates])
        new_var[i * Nstates + 1] = chi_toOrig(var[i * Nstates + 1])
        new_var[i * Nstates + 2] = gamma_toOrig(var[i * Nstates + 2])
        new_var[i * Nstates + 3] = teta_toOrig(var[i * Nstates + 3])
        new_var[i * Nstates + 4] = lam_toOrig(var[i * Nstates + 4], obj)
        new_var[i * Nstates + 5] = h_toOrig(var[i * Nstates + 5])
        new_var[i * Nstates + 6] = m_toOrig(var[i * Nstates + 6], obj)
    return new_var


def dynamicsInt(t, states, alfa_Int, delta_Int): #, deltaf_Int, tau_Int, mu_Int):
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

    if h > 1e5 or np.isinf(h):
        h = 1e5
    elif h < 1 or np.isnan(h):
        h = 1

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

    if h < obj.hvert:
        dx = np.array(((T * np.cos(eps) - D) / m - g, 0, 0, 0, 0, v, -T / (g0 * isp)))
    else:
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

    if h < obj.hvert:
        dx = np.array(((T * np.cos(eps) - D) / m - g, 0, 0, 0, 0, v, -T / (g0 * isp)))
    else:
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


def inequalityAll(states, controls, varnum):
    '''this function takes states and controls unscaled'''
    v = np.transpose(states[:, 0])
    # chi = np.transpose(states[:, 1])
    #gamma = np.transpose(states[:, 2])
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

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / np.exp((Dv1 + Dv2) / obj.gIsp)

    #axnew = to_new_int(obj.MaxAx, 0.0, 1e3, 0.0, 1.0) - to_new_int(ax, 0.0, 1e3, 0.0, 1.0)
    #aznew = to_new_int(obj.MaxAz, -1e3, 1e3, 0.0, 1.0) - to_new_int(az, -1e3, 1e3, 0.0, 1.0)
    #qnew = to_new_int(obj.MaxQ / 1e3, 0, 1e3, 0.0, 1.0) - to_new_int(q / 1e3, 0, 1e3, 0.0, 1.0)
    #momnew = to_new_int(obj.k/ 1e5, -1e3, 1e3, 0.0, 1.0) - to_new_int(MomTotA / 1e5, -1e3, 1e3, 0.0, 1.0)
    #mnew = m_toNew(mf, obj) - m_toNew(obj.m10, obj)

    iC = np.hstack(((obj.MaxAx - ax)/obj.MaxAx, (obj.MaxAz - az)/obj.MaxAz, (obj.MaxQ-q)/obj.MaxQ, (mf-obj.m10)/obj.M0))

    return iC


def SingleShootingMulti(i, var, dyn, Nint):
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
        #tau[k] = var[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
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
    #tau_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
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

    varD = var * (obj.UBV - obj.LBV) + obj.LBV

    try:
        res = p.map(partial(SingleShootingMulti, var=varD, dyn=dyn, Nint=Nint), range(Nleg))

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
            states_atNode = np.hstack((states_atNode, (
                (vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]))))  # new intervals

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

        obj.States = states_after
        obj.Controls = controls_after

        eq_c = equality(varD, states_atNode)
        obj.eqOld = eq_c

        ineq_c = inequalityAll(states_after, controls_after, Nint*Nleg)
        obj.ineqOld = ineq_c

        h = states_after[-1, 5]
        m = states_after[-1, 6]
        delta = controls_after[-1, 1]
        tau = 0.0 #controls_after[-1, 2]

        Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

        T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10, obj.lRef,
                                    obj.xcgf, obj.xcg0)

        r1 = h + obj.Re
        Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
        Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
        mf = m / np.exp((Dv1 + Dv2) / (obj.g0 * isp))

        cost = -mf / obj.M0
        obj.costOld = cost

        obj.varOld = var

        #if obj.bad == 1:
        #    eq_c = eq_c + 1000
         #   ineq_c = ineq_c - 1000
         #   cost = -obj.m10/obj.M0
         #   obj.costOld = cost
         #   obj.ineqOld = ineq_c
          #  obj.eqOld = eq_c
    except ValueError:
        eq_c = obj.eqOld*2
        ineq_c = obj.ineqOld*2
        cost = obj.costOld/2

    return eq_c, ineq_c, cost

def SingleShooting(states, controls, dyn, tstart, tfin, Nint):
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
    #tau_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
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

    return vres, chires, gammares, tetares, lamres, hres, mres, time_old, alfares, deltares, deltafres, taures, mures#, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int

def plot(var, Nint):

    time = np.zeros((1))
    #timeTotal = np.zeros((0))
    alfaCP = np.zeros((Nleg, NContPoints))
    deltaCP = np.zeros((Nleg, NContPoints))
    #deltafCP = np.zeros((Nleg, NContPoints))
    #tauCP = np.zeros((Nleg, NContPoints))
    #muCP = np.zeros((Nleg, NContPoints))

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
            #tau[k] = varD[varStates + i * (Ncontrols * NContPoints) + 2 + Ncontrols * k]
            #tauCP[i, k] = tau[k]
            #mu[k] = varD[varStates + i * (Ncontrols * NContPoints) + 4 + Ncontrols * k]
            #muCP[i, k] = mu[k]
        controls = np.vstack((alfa, delta))#, deltaf, tau, mu)) # orig intervals


        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures \
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
        #g = np.asarray(g, dtype=np.float64)
        # dynamic pressure

        q = 0.5 * rho * (vres ** 2)

        # accelerations

        ax = (T * np.cos(Deps) - D * np.cos(alfares) + L * np.sin(alfares)) / mres
        az = (T * np.sin(Deps) + D * np.sin(alfares) + L * np.cos(alfares)) / mres

        if flag_save:
            res = open(savedata_file + "res_{}_{}.txt".format(os.path.basename(__file__), timestr), "w")
            res.write("Number of leg: " + str(Nleg) + "\n"
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
            res.close()

        downrange = (vres ** 2) / g * np.sin(2 * gammares)

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
            plt.savefig(savefig_file + "deltaf" + ".png")

        plt.figure(11)
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


        plt.figure(14)
        plt.title("Downrange")
        plt.plot(downrange / 1000, hres / 1000)
        plt.grid()
        plt.ylabel("km")
        plt.xlabel("km")
        if flag_save:
            plt.savefig(savefig_file + "downrange" + ".png")


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

        timestart = timeend

    print("m before Ho : {0:.5f}".format(mres[-1]))
    print("mf          : {0:.5f}".format(mf[-1]))
    print("altitude Hohmann starts: {0:.5f}".format(hres[-1]))
    print("final time  : {}".format(time))


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


def equality(var, conj):
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
    div = np.tile([100,1,1,1,1,100,1000], Nleg - 1)

    eq_cond = np.zeros((0))
    eq_cond = np.concatenate((eq_cond, var[0:7] - states_init))
    eq_cond = np.concatenate((eq_cond, (var[Nstates:varStates] - conj[:Nstates * (Nleg - 1)])/div))  # knotting conditions
    eq_cond = np.concatenate((eq_cond, var[varStates:varStates + Ncontrols] - cont_init))  # init cond on alpha
    eq_cond = np.concatenate((eq_cond, ((vvv - vtAbs)/1e4,)))
    eq_cond = np.concatenate((eq_cond, ((chifin - chiass)/np.deg2rad(150),)))
    eq_cond = np.concatenate((eq_cond, (conj[varStates - 5]/np.deg2rad(90),)))  # final condition on gamma
    return eq_cond


def constraints(var):
    #print("constr", var - obj.varOld)
    #print("constraints", type)
    if (var==obj.varOld).all():
        return np.concatenate((obj.eqOld, obj.ineqOld))

    else:
        eq_c, ineq_c, cost = MultiShooting(var, dynamicsInt)
        return np.concatenate((eq_c, ineq_c))

def constraints_slsqp(var, type):
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

cons_slsqp = ({'type': 'eq',
         'fun': constraints_slsqp,
         'args':("eq",)},
        {'type': 'ineq',
         'fun': constraints_slsqp,
         'args': ("ineq",)})  # equality and inequality constraints

def cost_fun(var):
    #print("fun", var - obj.varOld)
    #print("cost fun")
    #print(max(var-obj.varOld), min(var-obj.varOld))
    if (var==obj.varOld).all():
        #print("costold")
        #obj.varOld = var
        return obj.costOld
    else:
        #print("costnew")
        eq, ineq, cost = MultiShooting(var, dynamicsInt)

        return cost


if __name__ == '__main__':
    obj = Spaceplane()


    '''reading of aerodynamic coefficients and specific impulse from file'''

    cl = fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/clfile.txt")
    cd = fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cdfile.txt")
    cm = fileReadOr("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/coeff_files/cmfile.txt")
    cl = np.asarray(cl)
    cd = np.asarray(cd)
    cm = np.asarray(cm)

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

    '''set problem parameters'''

    time_tot = 350  # initial time
    Nbar = 6 # number of conjunction points
    Nleg = Nbar - 1  # number of multiple shooting sub intervals
    NContPoints = 5  # number of control points for interpolation inside each interval
    Nint = 100# number of points for each single shooting integration
    Nstates = 7  # number of states
    Ncontrols = 2  # number of controls
    varStates = Nstates * Nleg  # total number of optimization variables for states
    varControls = Ncontrols * Nleg * NContPoints   # total number of optimization variables for controls
    varTot = varStates + varControls  # total number of optimization variables for states and controls
    NineqCond = Nint # Nleg * NContPoints - Nbar + 2
    tnew = np.linspace(0, time_tot, Nbar)  # time vector used for interpolation of states initial guess
    tcontr = np.linspace(0, time_tot, int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess
    unit_t = 1000

    '''NLP solver parameters'''
    general_tol = 1e-8
    tr_radius = 10
    constr_penalty = 10
    maxiter = 200 # max number of iterations for nlp solver

    '''definiton of initial conditions'''

    # set vector of initial conditions of states and controls. Angles are in radians
    X = np.zeros((0))
    U = np.zeros((0))

    vars = smooth_init(Nleg, NContPoints)

    '''v_init = vars[0]
    chi_init = vars[1]
    gamma_init = vars[2]
    teta_init = vars[3]
    lam_init = vars[4]
    h_init = vars[5]
    m_init = vars[6]
    alfa_init = vars[7]
    delta_init = vars[8]
    #deltaf_init = vars[9]
    tau_init = vars[9]
    #mu_init = vars[11]
    v_init[0] = 1.0
    chi_init[0] = obj.chistart
    gamma_init[0] = obj.gammastart
    teta_init[0] = obj.longstart
    lam_init[0] = obj.latstart
    h_init[0] = 1.0
    m_init[0] = obj.M0
    alfa_init[0] = 0.0
    delta_init[0] = 1.0
    #deltaf_init[0] = 0.0
    #tau_init[0] = 0.0
    #mu_init[0] = 0.0'''
    t_stat = np.linspace(0, time_tot, Nbar)
    t_contr = np.linspace(0, time_tot, NContPoints*Nleg)
    v_init = Guess.linear(t_stat, 1, obj.Vtarget)
    chi_init = Guess.linear(t_stat, obj.chistart, obj.chi_fin)
    gamma_init = Guess.linear(t_stat, obj.gammastart, 0.0)
    teta_init = Guess.constant(t_stat, obj.longstart)
    lam_init = Guess.constant(t_stat, obj.latstart)
    h_init = Guess.linear(t_stat, 1, obj.Hini)
    m_init = Guess.linear(t_stat, obj.M0, obj.m10)

    alfa_init = Guess.zeros(t_contr)
    part1 = np.repeat(1.0, int(len(t_contr)/3))
    part2 = Guess.linear(t_contr[int(len(t_contr)/3):], 1.0, 0.05)
    delta_init = np.hstack((part1, part2))

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

    uplimx = np.tile([1e4, np.deg2rad(270), np.deg2rad(89), 0.0, np.deg2rad(25), 1e5, obj.M0], Nleg)
    inflimx = np.tile([1.0, np.deg2rad(90), np.deg2rad(-89), np.deg2rad(-70), np.deg2rad(2), 1.0, obj.m10], Nleg)
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


    LbS, LbC, UbS, UbC = bound_def(X, U, uplimx, inflimx, uplimu, inflimu)
    Tlb = [1.0]
    Tub = [200]

    obj.LBV = np.hstack((LbS, LbC, np.repeat(Tlb, Nleg)))
    obj.UBV = np.hstack((UbS, UbC, np.repeat(Tub, Nleg)))
    X0a = (X0d - obj.LBV)/(obj.UBV - obj.LBV)

    Tlb = ([0.0]) # time lower bounds
    Tub = ([1.0]) # time upper bounds
    Xlb = ([0.0])
    Xub = ([1.0])
    Ulb = ([0.0])
    Uub = ([1.0])
    Vlb = Xlb*Nleg*Nstates + Ulb * Ncontrols * Nleg * NContPoints + Tlb*Nleg
    Vub = Xub*Nleg*Nstates + Uub * Ncontrols * Nleg * NContPoints + Tub*Nleg
    bnds = Bounds(Vlb, Vub)

    bndX_slsqp = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    bndU_slsqp = ((0.0, 1.0), (0.0, 1.0))  # , (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    bndT_slsqp = ((0.0, 1.0),)
    bnds_slsqp = bndX_slsqp * Nleg + bndU_slsqp * Nleg * NContPoints + bndT_slsqp * Nleg

    sparseJac = load_npz("/home/francesco/Desktop/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/trust-constr_solver/FestipSparsity.npz")
    sp = sparseJac.todense()
    row = np.shape(sp)[0]
    column = np.shape(sp)[1]
    for i in range(row):
        for j in range(column):
            if sp[i,j]!=0:
                sp[i,j]=1

    lbeq = ([0.0]) # lower bound for equality constraints
    ubeq = ([0.0]) # upper bound for equality constraints

    lbineq = ([0.0]) # lower bound for inequality constraints
    ubineq = ([np.inf]) # upper bound for inequality constraints

    lb = lbeq * (Nstates*Nleg + 3 + Ncontrols) + lbineq * (3 * NineqCond * Nleg + 1) # all lower bounds
    ub = ubeq * (Nstates*Nleg + 3 + Ncontrols) + ubineq * (3 * NineqCond * Nleg + 1) # all upper bounds

    cons = NonlinearConstraint(constraints, lb, ub, finite_diff_jac_sparsity=sp)
    """ Custom step-function """


    class RandomDisplacementBounds(object):
        """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
            Modified! (dropped acceptance-rejection sampling for a more specialized approach)
        """

        def __init__(self, xmin, xmax, stepsize=0.5):
            self.xmin = xmin
            self.xmax = xmax
            self.stepsize = stepsize

        def __call__(self, x):
            """take a random step but ensure the new position is within the bounds """
            min_step = np.maximum(self.xmin - x, -self.stepsize)
            max_step = np.minimum(self.xmax - x, self.stepsize)

            random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
            xnew = x + random_step

            return xnew


    bounded_step = RandomDisplacementBounds(np.array([b[0] for b in bnds_slsqp]), np.array([b[1] for b in bnds_slsqp]))
    '''NLP SOLVER'''

    iterator = 0
    tot_it = 5
    nbCPU = multiprocessing.cpu_count()
    p = multiprocessing.Pool(nbCPU)
    while iterator < tot_it:

        print("Start global search")
        minimizer_kwargs = {'method':'SLSQP', "constraints":cons_slsqp, "bounds":bnds_slsqp, "options": {"maxiter":300}}
        optb = basinhopping(cost_fun, X0a, niter=5, disp=True, minimizer_kwargs=minimizer_kwargs, take_step=bounded_step)
        #optb = shgo(cost_fun, bnds_slsqp, minimizer_kwargs=minimizer_kwargs, options={"disp": True, "maxtime":1})
        print("Done global search")
        X0a = optb.x
        opt = optimize.minimize(cost_fun,
                                X0a,
                                constraints=cons,
                                bounds=bnds,
                                method='trust-constr',
                                options={"maxiter": maxiter,
                                         "xtol":general_tol,
                                         "gtol":general_tol,
                                         "barrier_tol":general_tol,
                                         "initial_tr_radius":tr_radius,
                                         "initial_constr_penalty":constr_penalty,
                                         "verbose":2})

        print("Done local search")
        X0a = opt.x
        iterator += 1
    end = time.time()
    p.close()
    p.join()
    time_elapsed = end-start
    tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
    print("Time elapsed for total optimization ", tformat)
    #sparse = csc_matrix(opt.jac[0])
    #save_npz("/home/francesco/Desktop/PhD/Git_workspace/Personal/OptimalControl_FESTIP/MultipleShooting_Algorithm/Python/trust-constr_solver/FestipSparsity.npz", sparse)
    plot(opt.x, Nint)