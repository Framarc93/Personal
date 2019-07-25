import matplotlib.pyplot as plt
from OpenGoddard.optimize import Guess
from scipy import interpolate, optimize
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint
from models import *
import time
import datetime
import os
from scipy.sparse import csc_matrix, save_npz, load_npz

'''modified initial guess for controls so it has more points than states. States points are only in the conjunction of legs while controls are also inside'''
'''simplified version of problem. All controls but equatorial trajectory and no earth rotation'''
'''definition of vehicle and mission parameters. Angles are set in degrees'''
'''maybe best to set initial values of angles in radians'''

'''DIFFERENT METHOD FOR INTEGRATION'''

'''set initial conditions constraints on all parameters'''

'''try to reduce the dynamic evaluations at one insted of two. maybe by passing arguments to the functions'''

start = time.time()
timestr = time.strftime("%Y%m%d-%H%M%S")
savefig_file = "MultiShooting_{}_{}_".format(os.path.basename(__file__), timestr)

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
        self.k = 5e3  # [Nm] livello di precisione per trimmaggio
        self.m10 = self.M0 * 0.1
        self.xcgf = 0.37  # cg position with empty vehicle
        self.xcg0 = 0.65  # cg position at take-off
        self.pref = 21.25
        self.Hini = 190000
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


obj = Spaceplane()

'''reading of aerodynamic coefficients and specific impulse from file'''

cl = fileReadOr("clfile.txt")
cd = fileReadOr("cdfile.txt")
cm = fileReadOr("cmfile.txt")
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

'''set problem parameters'''

time_tot = 400  # initial time
Nbar = 6 # number of conjunction points
Nleg = Nbar - 1  # number of multiple shooting sub intervals
NContPoints = 9  # number of control points for interpolation inside each interval
Nint = 150 # number of points for each single shooting integration
maxiter = 500  # max number of iterations for nlp solver
general_tol = 1e-12
Nstates = 7  # number of states
Ncontrols = 4  # number of controls
#maxIterator = 1  # max number of optimization iterations
varStates = Nstates * Nleg  # total number of optimization variables for states
varControls = Ncontrols * (Nleg * NContPoints - Nbar + 2)   # total number of optimization variables for controls
varTot = varStates + varControls  # total number of optimization variables for states and controls
NineqCond = Nint # Nleg * NContPoints - Nbar + 2
tnew = np.linspace(0, time_tot, Nbar)  # time vector used for interpolation of states initial guess
tcontr = np.linspace(0, time_tot, int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess

'''definition of values used for scaling'''

unit_v = 8000
unit_chi = np.deg2rad(270)
unit_gamma = obj.gammastart
unit_teta = -obj.longstart
unit_lam = obj.incl
unit_h = obj.Hini
unit_m = obj.M0
unit_alfa = np.deg2rad(40)
unit_delta = 1
unit_deltaf = np.deg2rad(30)
unit_tau = 1
unit_t = 500
states_unit = np.array((unit_v, unit_chi, unit_gamma, unit_teta, unit_lam, unit_h, unit_m))
controls_unit = np.array((unit_alfa, unit_delta, unit_deltaf, unit_tau))

'''definiton of initial conditions'''

# set vector of initial conditions of states and controls
X = np.zeros((0))
U = np.zeros((0))

v_init = Guess.cubic(tnew, 1, 0.0, obj.Vtarget, 0.0)
chi_init = Guess.cubic(tnew, obj.chistart, 0.0, obj.chi_fin, 0.0)
gamma_init = Guess.linear(tnew, obj.gammastart, 0.0)
teta_init = Guess.constant(tnew, obj.longstart)
lam_init = Guess.constant(tnew, obj.latstart)
h_init = Guess.cubic(tnew, 1, 0.0, obj.Hini, 0.0)
m_init = Guess.cubic(tnew, obj.M0, 0.0, obj.m10, 0.0)
alfa_init = Guess.zeros(tcontr)
delta_init = Guess.cubic(tcontr, 1.0, 0.0, 0.001, 0.0)
deltaf_init = Guess.zeros(tcontr)
tau_init = Guess.zeros(tcontr)

XGuess = np.array(
    (v_init / unit_v, chi_init / unit_chi, gamma_init / unit_gamma, teta_init / unit_teta, lam_init / unit_lam,
     h_init / unit_h, m_init / unit_m))  # states initial guesses

UGuess = np.array((alfa_init / unit_alfa, delta_init / unit_delta, deltaf_init / unit_deltaf,
                   tau_init / unit_tau))  # states initial guesses

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

X0 = np.hstack((X, U, dt / unit_t))  # vector of initial conditions here all the angles are in degrees!!!!!
obj.varOld = np.zeros((len(X0)))

#obj.cost = np.zeros((0))
#objcostOld = np.zeros((0))
#obj.eq = np.zeros((0))
#obj.eqOld = np.zeros((0))
#obj.ineq = np.zeros((0))
#obj.ineqOld = np.zeros((0))
# X0 has first all X guesses and then all U guesses
# at this point the vectors of initial guesses for states and controls for every time interval are defined


'''set upper and lower bounds for states, controls and time, scaled'''
'''major issues with bounds!!!'''
#bndX = ((1e-7 / unit_v, 1.0), (np.deg2rad(90) / unit_chi, np.deg2rad(270) / unit_chi),
#        (0.0, np.deg2rad(89.9) / unit_gamma), (None, None),
#        (-obj.incl/unit_lam, obj.incl / unit_lam),
#        (1e-7 / unit_h, obj.Hini / unit_h), (obj.m10 / unit_m, obj.M0 / unit_m))
#bndU = ((np.deg2rad(-2) / unit_alfa, np.deg2rad(40) / unit_alfa), (0.001, 1.0),
#        (np.deg2rad(-20) / unit_deltaf, np.deg2rad(30) / unit_deltaf), (0.0, 1.0))
#bndT = ((1/unit_t, time_tot/unit_t),)
#
#bnds = (bndX) * Nleg + (bndU) * (Nleg * NContPoints - Nbar + 2)  + (bndT) * Nleg

Xlb = ([1e-5/unit_v, np.deg2rad(90)/unit_chi, 0.0, -np.inf, -obj.incl/unit_lam, 1e-5/unit_h, obj.m10/unit_m]) # states lower bounds
Xub = ([obj.Vtarget/unit_v, np.deg2rad(270)/unit_chi, np.deg2rad(89.9)/unit_gamma, np.inf, obj.incl/unit_lam, obj.Hini/unit_h, obj.M0/unit_m]) # states upper bounds

Ulb = ([np.deg2rad(-2)/unit_alfa, 0.001, np.deg2rad(-20)/unit_deltaf, -1.0]) # controls lower bounds
Uub = ([np.deg2rad(40)/unit_alfa, 1.0, np.deg2rad(30)/unit_deltaf, 1.0]) # controls upper bounds

Tlb = ([0.1/unit_t,]) # time lower bounds
Tub = ([time_tot/unit_t,]) # time upper bounds

lb = Xlb*Nleg + Ulb *(Nleg * NContPoints - Nbar + 2) + Tlb*Nleg
ub = Xub*Nleg + Uub *(Nleg * NContPoints - Nbar + 2) + Tub*Nleg
bnds = Bounds(lb, ub)

'''definition of initial and final conditions for states and controls for equality constraints'''

#init_condX = np.array((1 / unit_v, obj.chistart / unit_chi, obj.gammastart / unit_gamma, obj.longstart / unit_teta,
 #                      obj.latstart / unit_lam, 1 / unit_h, obj.M0 / unit_m))
#init_condU = np.array((0.0, 1.0, 0.0, 0.0))
#final_cond = np.array((0.0))  # final conditions on gamma

'''function definitions'''

def dynamicsInt(t, states, alfa_Int, delta_Int, deltaf_Int, tau_Int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = alfa_Int(t)
    delta = delta_Int(t)
    deltaf = deltaf_Int(t)
    tau = tau_Int(t)

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
        g = g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
                   ((T * np.sin(eps) + L) / (m * v)) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
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
    teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = contr[0]
    delta = contr[1]
    deltaf = contr[2]
    tau = contr[3]

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
        g = g0 * (obj.Re / (obj.Re + h)) ** 2
    # g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                   (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                   - np.cos(gamma) * np.cos(chi) * np.tan(lam) * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                   - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
                   ((T * np.sin(eps) + L) / (m * v)) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
                   * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                   (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                   -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                   np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                   v * np.sin(gamma),
                   -T / (g0 * isp)))

    return dx


def inequalityAll(states, controls, varnum):
    '''this function takes states and controls unscaled'''
    v = states[:, 0]
    chi = states[:, 1]  # it's not used for functions, can be used in degree
    gamma = states[:, 2]  # it's not used for functions, can be used in degree
    teta = states[:, 3]  # it's not used for functions, can be used in degree
    lam = states[:, 4]  # it's not used for functions, can be used in degree
    h = states[:, 5]
    m = states[:, 6]
    alfa = controls[:, 0]
    delta = controls[:, 1]
    deltaf = controls[:, 2]
    tau = controls[:, 3]
    iC = np.zeros((0))
    # print("v ", v)
    # print("chi ", chi)
    # print("gamma ", gamma)
    # print("teta ", teta)
    # print("lam ", lam)
    # print("h ", h)
    # print("m ", m)
    # print("alfa ", alfa)
    # print("delta ", delta)
    # print("deltaf ", deltaf)
    # print("tau ", tau)

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
    # isp = np.asarray(isp, dtype=np.float64)
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
    mf = m[-1] / np.exp((Dv1 + Dv2) / obj.gIsp)

    iC = np.hstack((((obj.MaxAx - ax)/(obj.MaxAx*1000), (obj.MaxAz - az)/(obj.MaxAz*1000), (obj.MaxQ - q)/(obj.MaxQ*1000),
                      (obj.k - MomTotA)/(obj.k*10000000), (mf - obj.m10)/unit_m,)))  # when with bounds

    return iC


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
    '''in this function the states are not scaled anymore'''
    '''tstart and tfin are the initial and final time of the considered leg'''
    #print("single shooting")
    Time = np.linspace(tstart, tfin, NContPoints)
    x = np.zeros((Nint, Nstates))
    x[0,:]= states * states_unit  # vector of intial states ready
    #print("Single: ", Time)
    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(Time, controls[0, :] * unit_alfa)
    delta_Int = interpolate.PchipInterpolator(Time, controls[1, :])
    deltaf_Int = interpolate.PchipInterpolator(Time, controls[2, :] * unit_deltaf)
    tau_Int = interpolate.PchipInterpolator(Time, controls[3, :])

    #VAR = np.concatenate((states[0:Nstates] * states_unit, controls[0, :] * unit_alfa, controls[1, :], controls[2, :] * unit_deltaf, controls[3, :], Time))

    time_new = np.linspace(tstart, tfin, Nint)
    #print("time: ", np.shape(time_new))
    dt = (time_new[1] - time_new[0])

    t = time_new
    #print(time_new)
    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int), t_span=[tstart, tfin], y0=x, t_eval=time_new, method='Radau')


    #print(sol)


    for i in range(Nint-1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt*dyn(t[i], x[i, :], alfa_Int, delta_Int, deltaf_Int, tau_Int)
        # print("k1: ", k1)
        k2 = dt*dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int)
        # print("k2: ", k2)
        k3 = dt*dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int)
        # print("k3: ", k3)
        k4 = dt*dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    #if len(sol.y[0,:]) != len(time_new):
     #   sol.y = obj.xold
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
    #obj.xold = sol.y
    #print(np.shape(vres))
    return vres, chires, gammares, tetares, lamres, hres, mres, time_new, alfares, deltares, deltafres, taures, \
           alfa_Int, delta_Int, deltaf_Int, tau_Int


def MultiShootingFun(var, dyn):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in degrees'''
    #print("integration")
    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    deltaf = np.zeros((NContPoints))
    tau = np.zeros((NContPoints))

    states_atNode = np.zeros((0))
    timestart = 0.0
    time_vec = np.zeros((1))

    for i in range(Nleg):

        states = var[i * Nstates:(i + 1) * Nstates] # scaled

        timeend = timestart + var[varTot + i] * unit_t
        time_vec = np.concatenate((time_vec, (timeend,))) # vector with time points

        for k in range(NContPoints):
            alfa[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + Ncontrols * k]
            delta[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 1 + Ncontrols * k]
            deltaf[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 2 + Ncontrols * k]
            tau[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 3 + Ncontrols * k]
        controls = np.vstack((alfa, delta, deltaf, tau)) # scaled

        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, alfa_I, delta_I, deltaf_I, tau_I = SingleShooting(
            states, controls, dyn, timestart, timeend, Nint)

        '''res quantities are unscaled'''

        states_atNode = np.concatenate((states_atNode, ((vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]) / states_unit)))



        timestart = timeend

    plt.show()
    h = states_atNode[varStates-2]*unit_h
    m = states_atNode[-1]*unit_m

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, deltares[-1], taures[-1], obj.psl, obj.M0, obj.m10, obj.lRef,
                                     obj.xcgf, obj.xcg0)

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / (obj.g0*isp))

    objective = 1-mf / unit_m

    obj.costOld = objective
    obj.varOld = var

    return objective


def MultiShootingCons(var, dyn):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in degrees'''
    #print("integration")
    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    deltaf = np.zeros((NContPoints))
    tau = np.zeros((NContPoints))

    vineq = np.zeros((1, Nint))  # states defined as row vectors
    chiineq = np.zeros((1, Nint))
    gammaineq = np.zeros((1, Nint))
    tetaineq = np.zeros((1, Nint))
    lamineq = np.zeros((1, Nint))
    hineq = np.zeros((1, Nint))
    mineq = np.zeros((1, Nint))

    alfaineq = np.zeros((1, Nint))
    deltaineq = np.zeros((1, Nint))
    deltafineq = np.zeros((1, Nint))
    tauineq = np.zeros((1, Nint))

    states_atNode = np.zeros((0))
    timestart = 0.0
    time_vec = np.zeros((1))

    for i in range(Nleg):

        states = var[i * Nstates:(i + 1) * Nstates] # scaled

        timeend = timestart + var[varTot + i] * unit_t
        time_vec = np.concatenate((time_vec, (timeend,))) # vector with time points

        for k in range(NContPoints):
            alfa[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + Ncontrols * k]
            delta[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 1 + Ncontrols * k]
            deltaf[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 2 + Ncontrols * k]
            tau[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 3 + Ncontrols * k]
        controls = np.vstack((alfa, delta, deltaf, tau)) # scaled

        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, alfa_I, delta_I, deltaf_I, tau_I = SingleShooting(
            states, controls, dyn, timestart, timeend, Nint)

        '''res quantities are unscaled'''

        states_atNode = np.concatenate((states_atNode, ((vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]) / states_unit)))

        # time_ineq = np.linspace(tres[0], tres[-1], NineqCond)
        # print(np.shape(tres), np.shape(vres))
        # vinterp = interpolate.PchipInterpolator(tres, vres)
        # chiinterp = interpolate.PchipInterpolator(tres, chires)
        # gammainterp = interpolate.PchipInterpolator(tres, gammares)
        # tetainterp = interpolate.PchipInterpolator(tres, tetares)
        # laminterp = interpolate.PchipInterpolator(tres, lamres)
        # hinterp = interpolate.PchipInterpolator(tres, hres)
        # minterp = interpolate.PchipInterpolator(tres, mres)

        vineq[0, :] = vres  # vinterp(time_ineq) #
        chiineq[0, :] = chires  # chiinterp(time_ineq)
        gammaineq[0, :] = gammares  # gammainterp(time_ineq)
        tetaineq[0, :] = tetares  # tetainterp(time_ineq)
        lamineq[0, :] = lamres  # laminterp(time_ineq)
        hineq[0, :] = hres  # hinterp(time_ineq)
        mineq[0, :] = mres  # minterp(time_ineq)
        alfaineq[0, :] = alfares  # alfa_I(time_ineq)
        deltaineq[0, :] = deltares  # delta_I(time_ineq)
        deltafineq[0, :] = deltafres  # deltaf_I(time_ineq)
        tauineq[0, :] = taures  # tau_I(time_ineq)

        vt = np.transpose(vineq)
        chit = np.transpose(chiineq)
        gammat = np.transpose(gammaineq)
        tetat = np.transpose(tetaineq)
        lamt = np.transpose(lamineq)
        ht = np.transpose(hineq)
        mt = np.transpose(mineq)
        alfat = np.transpose(alfaineq)
        deltat = np.transpose(deltaineq)
        deltaft = np.transpose(deltafineq)
        taut = np.transpose(tauineq)

        if i == 0:
            states_after = np.column_stack((vt, chit, gammat, tetat, lamt, ht, mt))
            controls_after = np.column_stack((alfat, deltat, deltaft, taut))
        else:
            states_after = np.vstack((states_after, np.column_stack((vt, chit, gammat, tetat, lamt, ht, mt))))
            controls_after = np.vstack((controls_after, np.column_stack((alfat, deltat, deltaft, taut))))

        timestart = timeend

    obj.States = states_after
    obj.Controls = controls_after

    ineq_Cond = inequalityAll(states_after, controls_after, len(states_after))

    eq_Cond = equality(var, states_atNode)

    obj.eqOld = eq_Cond
    obj.ineqOld = ineq_Cond
    obj.varOld = var

    return eq_Cond, ineq_Cond


def plot(var):

    time = np.zeros((1))
    timeTotal = np.zeros((0))
    alfaCP = np.zeros((Nleg, NContPoints))
    deltaCP = np.zeros((Nleg, NContPoints))
    deltafCP = np.zeros((Nleg, NContPoints))
    tauCP = np.zeros((Nleg, NContPoints))
    res = open("res_{}_{}.txt".format(os.path.basename(__file__), timestr), "w")
    tCtot = np.zeros((0))

    for i in range(Nleg):
        alfa = np.zeros((NContPoints))
        delta = np.zeros((NContPoints))
        deltaf = np.zeros((NContPoints))
        tau = np.zeros((NContPoints))

        if i == 0:
            timestart = 0
        timeend = timestart + var[i + varTot] * unit_t
        timeTotal = np.linspace(timestart, timeend, Nint)
        time = np.concatenate((time, (timeend,)))
        tC = np.linspace(timestart, timeend, NContPoints)
        tCtot = np.concatenate((tCtot, tC))

        for k in range(NContPoints):
            alfa[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + Ncontrols * k]
            alfaCP[i, k] = alfa[k] * unit_alfa
            delta[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 1 + Ncontrols * k]
            deltaCP[i, k] = delta[k]
            deltaf[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 2 + Ncontrols * k]
            deltafCP[i, k] = deltaf[k] * unit_deltaf
            tau[k] = var[varStates + i * (Ncontrols * NContPoints-Ncontrols) + 3 + Ncontrols * k]
            tauCP[i, k] = tau[k]

        timestart = timeend

        vres = obj.States[i*Nint:(i+1)*Nint,0]
        chires = obj.States[i*Nint:(i+1)*Nint, 1]
        gammares = obj.States[i*Nint:(i+1)*Nint, 2]
        tetares = obj.States[i*Nint:(i+1)*Nint, 3]
        lamres = obj.States[i*Nint:(i+1)*Nint, 4]
        hres = obj.States[i*Nint:(i+1)*Nint, 5]
        mres = obj.States[i*Nint:(i+1)*Nint, 6]

        alfares = obj.Controls[i*Nint:(i+1)*Nint, 0]
        deltares = obj.Controls[i*Nint:(i+1)*Nint, 1]
        deltafres = obj.Controls[i*Nint:(i+1)*Nint, 2]
        taures = obj.Controls[i*Nint:(i+1)*Nint, 3]

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
        Deps = np.asarray(Deps, dtype=np.float64)
        MomT = np.asarray(MomT, dtype=np.float64)

        MomTot = MomA + MomT

        r1 = hres + obj.Re
        Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
        Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
        mf = mres / np.exp((Dv1 + Dv2) / obj.gIsp)

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

        res.write("Number of leg: " + str(Nleg) + "\n" + "Max number Optimization iterations: "
                  + "Number of NLP iterations: " + str(maxiter) + "\n" + "Leg Number:" + str(i) + "\n" + "v: " + str(
            vres) + "\n" + "Chi: " + str(np.rad2deg(chires))
            + "\n" + "Gamma: " + str(np.rad2deg(gammares)) + "\n" + "Teta: " + str(
            np.rad2deg(tetares)) + "\n" + "Lambda: "
            + str(np.rad2deg(lamres)) + "\n" + "Height: " + str(hres) + "\n" + "Mass: " + str(
            mres) + "\n" + "mf: " + str(mf) + "\n"
            + "Objective Function: " + str(-mf / unit_m) + "\n" + "Alfa: "
            + str(np.rad2deg(alfares)) + "\n" + "Delta: " + str(deltares) + "\n" + "Delta f: " + str(
            np.rad2deg(deltafres)) + "\n"
            + "Tau: " + str(taures) + "\n" + "Eps: " + str(np.rad2deg(eps)) + "\n" + "Lift: "
            + str(L) + "\n" + "Drag: " + str(D) + "\n" + "Thrust: " + str(T) + "\n" + "Spimp: " + str(
            isp) + "\n" + "c: "
            + str(c) + "\n" + "Mach: " + str(M) + "\n" + "Time vector: " + str(timeTotal) + "\n" + "Press: " + str(
            Press) + "\n" + "Dens: " + str(rho) + "\n" + "Time elapsed during optimization: " + tformat)

        downrange = - (vres ** 2) / g * np.sin(2 * chires)



        plt.figure(0)
        plt.title("Velocity")
        plt.plot(timeTotal, vres)
        plt.ylabel("m/s")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "velocity" + ".png")

        plt.figure(1)
        plt.title("Flight path angle \u03C7")
        plt.plot(timeTotal, np.rad2deg(chires))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "chi" + ".png")

        plt.figure(2)
        plt.title("Angle of climb \u03B3")
        plt.plot(timeTotal, np.rad2deg(gammares))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "gamma" + ".png")

        plt.figure(3)
        plt.title("Longitude \u03B8")
        plt.plot(timeTotal, np.rad2deg(tetares))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "theta" + ".png")

        plt.figure(4)
        plt.title("Latitude \u03BB")
        plt.plot(timeTotal, np.rad2deg(lamres))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "lambda" + ".png")

        plt.figure(5)
        plt.title("Flight angles")
        plt.plot(timeTotal, np.rad2deg(chires), color="g")
        plt.plot(timeTotal, np.rad2deg(gammares), color="b")
        plt.plot(timeTotal, np.rad2deg(tetares), color="r")
        plt.plot(timeTotal, np.rad2deg(lamres), color="k")
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Chi", "Gamma", "Theta", "Lambda"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "angles" + ".png")

        plt.figure(6)
        plt.title("Altitude")
        plt.plot(timeTotal, hres / 1000)
        plt.ylabel("km")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "altitude" + ".png")

        plt.figure(7)
        plt.title("Mass")
        plt.plot(timeTotal, mres)
        plt.ylabel("kg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "mass" + ".png")

        plt.figure(8)
        plt.title("Angle of attack \u03B1")
        plt.plot(tC, np.rad2deg(alfaCP[i, :]), 'ro')
        plt.plot(timeTotal, np.rad2deg(alfares))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "alpha" + ".png")

        plt.figure(9)
        plt.title("Throttles")
        plt.plot(timeTotal, deltares * 100, color='r')
        plt.plot(timeTotal, taures * 100, color='k')
        plt.plot(tC, deltaCP[i, :] * 100, 'ro')
        plt.plot(tC, tauCP[i, :] * 100, 'ro')
        plt.ylabel("%")
        plt.xlabel("time [s]")
        plt.legend(["Delta", "Tau", "Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "throttles" + ".png")

        plt.figure(10)
        plt.title("Body Flap deflection \u03B4")
        plt.plot(tC, np.rad2deg(deltafCP[i, :]), "ro")
        plt.plot(timeTotal, np.rad2deg(deltafres))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "deltaf" + ".png")

        plt.figure(11)
        plt.title("Dynamic Pressure")
        plt.plot(timeTotal, q / 1000)
        plt.ylabel("kPa")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "dynPress" + ".png")

        plt.figure(12)
        plt.title("Accelerations")
        plt.plot(timeTotal, ax, color='b')
        plt.plot(timeTotal, az, color='r')
        plt.ylabel("m/s^2")
        plt.xlabel("time [s]")
        plt.legend(["ax", "az"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "accelerations" + ".png")

        plt.figure(13)
        plt.title("Downrange")
        plt.plot(downrange / 1000, hres / 1000)
        plt.ylabel("km")
        plt.xlabel("km")
        plt.savefig(savefig_file + "downrange" + ".png")

        plt.figure(14)
        plt.title("Forces")
        plt.plot(timeTotal, T / 1000, color='r')
        plt.plot(timeTotal, L / 1000, color='b')
        plt.plot(timeTotal, D / 1000, color='k')
        plt.ylabel("kN")
        plt.xlabel("time [s]")
        plt.legend(["Thrust", "Lift", "Drag"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "forces" + ".png")

        plt.figure(15)
        plt.title("Mach")
        plt.plot(timeTotal, M)
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "mach" + ".png")

        plt.figure(16)
        plt.title("Total pitching Moment")
        plt.plot(timeTotal, MomTot / 1000, color='k')
        plt.axhline(5, 0, timeTotal[-1], color='r')
        plt.axhline(-5, 0, timeTotal[-1], color='r')
        plt.ylabel("kNm")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "moment" + ".png")

    print("m before Ho : {0:.5f}".format(mres[-1]))
    print("mf          : {0:.5f}".format(mf[-1]))
    print("altitude Hohmann starts: {0:.5f}".format(hres[-1]))
    print("final time  : {}".format(time))

    res.close()
    plt.show()


def equality(var, conj):
    h = conj[varStates - 2] * unit_h
    lam = conj[varStates - 3] * unit_lam
    # print("eqnew")

    vtAbs, chiass, vtAbs2 = vass(conj[varStates - Nstates:varStates] * states_unit,
                                 var[varTot - Ncontrols:varTot] * controls_unit, dynamicsVel, obj.omega)
    vvv = np.sqrt(obj.GMe / (obj.Re + h))

    if np.cos(obj.incl) / np.cos(lam) > 1:
        chifin = np.pi
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam))

    eq_cond = np.zeros((0))
    #eq_cond = np.concatenate((eq_cond, abs(var[0:4] - (v_init[0] / unit_v, chi_init[0] / unit_chi, gamma_init[0] / unit_gamma, teta_init[0] / unit_teta))))
    #eq_cond = np.concatenate((eq_cond, abs(var[5:7] - (h_init[0] / unit_h, m_init[0] / unit_m))))
    eq_cond = np.concatenate((eq_cond, abs(var[0:7] - (v_init[0] / unit_v, chi_init[0] / unit_chi, gamma_init[0] / unit_gamma, teta_init[0] / unit_teta,
      lam_init[0] / unit_lam, h_init[0] / unit_h, m_init[0] / unit_m))))
    eq_cond = np.concatenate((eq_cond, abs(var[Nstates:varStates] - conj[:Nstates * (Nleg-1)])))  # knotting conditions
    eq_cond = np.concatenate((eq_cond, (abs(var[varStates]),)))  # init cond on alpha
    eq_cond = np.concatenate((eq_cond, (abs(var[varStates + 1] - 1.0),)))  # init cond on delta
    eq_cond = np.concatenate((eq_cond, (abs(var[varStates + 2]),)))  # init cond on deltaf
    eq_cond = np.concatenate((eq_cond, (abs(var[varStates + 3]),)))  # init cond on tau
    eq_cond = np.concatenate((eq_cond, (abs((vvv - vtAbs) / unit_v),)))
    eq_cond = np.concatenate((eq_cond, (abs((chifin - chiass) / unit_chi),)))
    eq_cond = np.concatenate((eq_cond, (abs(conj[varStates - 5]),)))  # final condition on gamma

    return eq_cond


def constraints(var):
    #print("constraints", type)
    if all(j<general_tol for j in abs(var - obj.varOld)):
        return np.concatenate((obj.eqOld, obj.ineqOld))

    else:
        eq_c, ineq_c = MultiShootingCons(var, dynamicsInt)
        return np.concatenate((eq_c, ineq_c))


def cost_fun(var):
    #print("cost fun")
    #print(max(var-obj.varOld), min(var-obj.varOld))
    if all(j<general_tol for j in abs(var - obj.varOld)):
        #print("costold")
        #obj.varOld = var
        return obj.costOld
    else:
        #print("costnew")
        cost = MultiShootingFun(var, dynamicsInt)

        return cost

sparseJac = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/sparse_festip_new11.npz")#
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

lb = lbeq * (Nstates+7+Nstates*(Nleg-1)) + lbineq * (4 * NineqCond * Nleg + 1) # all lower bounds
ub = ubeq * (Nstates+7+Nstates*(Nleg-1)) + ubineq * (4 * NineqCond * Nleg + 1)# all upper bounds

cons = NonlinearConstraint(constraints,lb, ub, finite_diff_jac_sparsity=sp)

#cons = ({'type': 'eq',
#         'fun': constraints,
#         'args':("eq",)},
#        {'type': 'ineq',
#         'fun': constraints,
#         'args': ("ineq",)})  # equality and inequality constraints

'''NLP SOLVER'''

iterator = 0



opt = optimize.minimize(cost_fun,
                        X0,
                        constraints=cons,
                        bounds=bnds,
                        method='trust-constr',
                        options={"maxiter": maxiter,
                                 "xtol":general_tol,
                                 "gtol":general_tol,
                                 "barrier_tol":general_tol,
                                 "initial_tr_radius":2.0,
                                 "verbose":2})

X0 = opt.x

end = time.time()
time_elapsed = end-start
tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
print("Time elapsed for total optimization ", tformat)
plot(X0)
#sparse = csc_matrix(opt.jac[0])
#save_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/sparse_festip_new11.npz", sparse)