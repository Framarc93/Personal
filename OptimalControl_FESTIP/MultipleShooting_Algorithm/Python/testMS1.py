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

sys.path.insert(0, 'home/francesco/git_workspace/FESTIP_Work')


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
        self.bad = 0


obj = Spaceplane()


'''reading of aerodynamic coefficients and specific impulse from file'''



cl = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/clfile.txt")
cd = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/cdfile.txt")
cm = fileReadOr("/home/francesco/git_workspace/FESTIP_Work/coeff_files/cmfile.txt")
cl = np.asarray(cl)
#cl = np.reshape(clOld, (13, 17, 6))
cd = np.asarray(cd)
#cd = np.reshape(cdOld, (13, 17, 6))
cm = np.asarray(cm)
#cm = np.reshape(cmOld, (13,17,6))

#cl_interp = RegularGridInterpolator((mach, angAttack, bodyFlap), cl)
#cd_interp = RegularGridInterpolator((mach, angAttack, bodyFlap), cd)
#cm_interp = RegularGridInterpolator((mach, angAttack, bodyFlap), cm)

sparseJacCost = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacCost.npz")
spCost = sparseJacCost.todense()
#sparseJacEq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacEq.npz")
#spEq = sparseJacEq.todense()
#sparseJacIneq = load_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacIneq.npz")
#spIneq = sparseJacIneq.todense()

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

'''set problem parameters'''

time_tot = 300 # initial time
Nbar = 5 # number of conjunction points
Nleg = Nbar - 1  # number of multiple shooting sub intervals
NContPoints = 5 # number of control points for interpolation inside each interval
Nint = 100 # number of points for each single shooting integration
Nstates = 7  # number of states
Ncontrols = 5  # number of controls
varStates = Nstates * Nleg  # total number of optimization variables for states
varControls = Ncontrols * (Nleg * (NContPoints - 1) + 1)   # total number of optimization variables for controls
varTot = varStates + varControls  # total number of optimization variables for states and controls
NineqCond = Nint
tnew = np.linspace(0, time_tot, Nbar)  # time vector used for interpolation of states initial guess
tcontr = np.linspace(0, time_tot, int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess
unit_t = 1000
Nintplot = 1000

'''NLP solver parameters'''
maxiter = 20 # max number of iterations for nlp solver
ftol = 1e-8  # numeric tolerance of nlp solver
#eps = 1e-10
eps = 1.4901161193847656e-08# increment of the derivative
#u = 2.220446049250313e-16
maxIterator = 20  # max number of optimization iterations


'''definiton of initial conditions'''

# set vector of initial conditions of states and controls
X = np.zeros((0))
U = np.zeros((0))

v_init = Guess.cubic(tnew, v_toNew(1.0), 0.0, v_toNew(obj.Vtarget), 0.0)
chi_init = Guess.linear(tnew, chi_toNew(obj.chistart), chi_toNew(obj.chi_fin))
gamma_init = Guess.linear(tnew, gamma_toNew(obj.gammastart), gamma_toNew(0.0))
teta_init = Guess.constant(tnew, teta_toNew(obj.longstart))
lam_init = Guess.constant(tnew, lam_toNew(obj.latstart, obj))
h_init = Guess.cubic(tnew, h_toNew(1.0), 0.0, h_toNew(obj.Hini), 0.0)
m_init = Guess.cubic(tnew, m_toNew(obj.M0, obj), 0.0, m_toNew(obj.m10, obj), 0.0)
alfa_init = Guess.constant(tcontr, alfa_toNew(0.0))
delta_init = Guess.cubic(tcontr, 1.0, 0.0, 0.001, 0.0)
deltaf_init = Guess.constant(tcontr, deltaf_toNew(0.0))
tau_init = Guess.constant(tcontr, tau_toNew(0.0))
mu_init = Guess.constant(tcontr, mu_toNew(0.0))

states_init = np.array((v_init[0], chi_init[0], gamma_init[0], teta_init[0], lam_init[0], h_init[0], m_init[0]))
cont_init =  np.array((alfa_init[0], delta_init[0], deltaf_init[0], tau_init[0], mu_init[0]))

XGuess = np.array((v_init, chi_init, gamma_init, teta_init, lam_init, h_init, m_init))  # states initial guesses

UGuess = np.array((alfa_init, delta_init, deltaf_init, tau_init, mu_init))  # states initial guesses

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

X0 = np.hstack((X, U, t_toNew(dt)))  # vector of initial conditions here all the angles are in degrees!!!!!
obj.varOld = np.zeros((len(X0)))

# X0 has first all X guesses and then all U guesses
# at this point the vectors of initial guesses for states and controls for every time interval are defined

unit_v = 10000
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
unit_mu = np.deg2rad(90)

states_unit = np.array((unit_v, unit_chi, unit_gamma, unit_teta, unit_lam, unit_h, unit_m))
controls_unit = np.array((unit_alfa, unit_delta, unit_deltaf, unit_tau, unit_mu))

'''set upper and lower bounds for states, controls and time, scaled'''
'''major issues with bounds!!!'''
bndX = ((0.0, 1.0),(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
bndU = ((0.0, 1.0),(0.0001, 1.0),(0.0, 1.0),(0.0, 1.0),(0.0, 1.0))
bndT = ((0.001, 1.0),)
#bndX = ((None, None),(None, None), (None, None), (None, None), (None, None), (None, None), (None, None))
#bndU = ((None, None),(None, None),(None, None),(None, None),(None, None))
#bndT = ((0.001, 1.0),)

#Xlb = ([v_toNew(0.001), chi_toNew(np.deg2rad(90)), gamma_toNew(0.0), teta_toNew(np.deg2rad(-90)), lam_toNew(-obj.incl, obj), h_toNew(0.001), m_toNew(obj.m10, obj)]) # states lower bounds
#Xub = ([v_toNew(1e4), chi_toNew(np.deg2rad(270)), gamma_toNew(np.deg2rad(90)), teta_toNew(0.0), lam_toNew(obj.incl, obj), h_toNew(2e5), m_toNew(obj.M0, obj)]) # states upper bounds

#Ulb = ([alfa_toNew(np.deg2rad(-2)), 0.0001, deltaf_toNew(np.deg2rad(-20)), tau_toNew(-1.0), mu_toNew(np.deg2rad(-90))]) # controls lower bounds
#Uub = ([alfa_toNew(np.deg2rad(40)), 1.0, deltaf_toNew(np.deg2rad(30)), tau_toNew(1.0), mu_toNew(np.deg2rad(90))]) # controls upper bounds

#Tlb = ([1/unit_t,]) # time lower bounds
#Tub = ([1.0,]) # time upper bounds

#lb = Xlb*Nleg + Ulb * (Nleg * NContPoints) + Tlb*Nleg
#ub = Xub*Nleg + Uub * (Nleg * NContPoints) + Tub*Nleg

#Ulb = ([0.0])
#Uub = ([1.0])

#Tlb = ([0.0]) # time lower bounds
#Tub = ([1.0]) # time upper bounds

#Vlb = Xlb*Nleg + Ulb * Ncontrols * (Nleg * NContPoints) + Tlb*Nleg
#Vub = Xub*Nleg + Uub * Ncontrols * (Nleg * NContPoints) + Tub*Nleg
#bnds = Bounds(Vlb, Vub)
bnds = bndX * Nleg + bndU * (Nleg * (NContPoints - 1) + 1) + bndT * Nleg
'''definition of initial and final conditions for states and controls for equality constraints'''

#init_condX = np.array((1 / unit_v, obj.chistart / unit_chi, obj.gammastart / unit_gamma, obj.longstart / unit_teta,
 #                      obj.latstart / unit_lam, 1 / unit_h, obj.M0 / unit_m))
#init_condU = np.array((0.0, 1.0, 0.0, 0.0))
#final_cond = np.array((0.0))  # final conditions on gamma

'''function definitions'''


def conversionStatesToOrig(var, len):
    new_var = np.zeros((len*Nstates))
    for i in range(len):
        new_var[i * Nstates] = v_toOrig(var[i * Nstates])
        new_var[i * Nstates + 1] = chi_toOrig(var[i * Nstates + 1])
        new_var[i * Nstates + 2] = gamma_toOrig(var[i * Nstates + 2])
        new_var[i * Nstates + 3] = teta_toOrig(var[i * Nstates + 3])
        new_var[i * Nstates + 4] = lam_toOrig(var[i * Nstates + 4], obj)
        new_var[i * Nstates + 5] = h_toOrig(var[i * Nstates + 5])
        new_var[i * Nstates + 6] = m_toOrig(var[i * Nstates + 6], obj)
    return new_var


def dynamicsInt(t, states, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    v = states[0]
    chi = states[1]
    gamma = states[2]
    teta = states[3]
    lam = states[4]
    h = states[5]
    m = states[6]
    alfa = float(alfa_Int(t))
    delta = float(delta_Int(t))
    deltaf = float(deltaf_Int(t))
    tau = float(tau_Int(t))
    mu = float(mu_Int(t))
    #print("states", states)
    #print("controls", alfa, delta, deltaf, tau, mu)

    if gamma == np.deg2rad(90):
        gamma = np.deg2rad(89)
    if h <= 0:
        h=1.0
    if v <= 0:
        v = 1.0

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


def inequalityAll(states, controls, varnum, time):
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
    deltaf = np.transpose(controls[:, 2])
    tau = np.transpose(controls[:, 3])  # tau back to [-1, 1] interval
    #mu = np.transpose(controls[:, 4])

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
    iC = np.hstack(((obj.MaxAx - ax)/obj.MaxAx, (obj.MaxAz - az)/obj.MaxAz, (obj.MaxQ - q)/obj.MaxQ, (obj.k-MomTot)/(obj.k), (MomTot+obj.k)/(obj.k), (mf-obj.m10)/obj.M0))
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
              Nint: unmber of integration steps'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''tstart and tfin are the initial and final time of the considered leg'''
    Nintlocal = Nint
    #print("single shooting")
    timeCont = np.linspace(tstart, tfin, NContPoints)
    #x = np.zeros((Nint, Nstates))
      # vector of intial states ready
    time_old = np.linspace(tstart, tfin, Nint)
    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(timeCont, controls[0, :])
    delta_Int = interpolate.PchipInterpolator(timeCont, controls[1, :])
    deltaf_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
    tau_Int = interpolate.PchipInterpolator(timeCont, controls[3, :])
    mu_Int = interpolate.PchipInterpolator(timeCont, controls[4, :])


    #status = 1
    #while status != 0:
    time_int = np.linspace(tstart, tfin, Nintlocal)
    #dt = (t[1] - t[0])
    x = states
    sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int), t_span=[tstart, tfin], y0=x, t_eval=time_int, method='Radau')
    '''
    if sol.status != 0:
        vres = states[0] * np.ones((Nint))
        chires = states[1] * np.ones((Nint))
        gammares = states[2] * np.ones((Nint))
        tetares = states[3] * np.ones((Nint))
        lamres = states[4] * np.ones((Nint))
        hres = states[5] * np.ones((Nint))
        mres = states[6] * np.ones((Nint))
        alfares = alfa_Int(time_old)
        deltares = delta_Int(time_old)
        deltafres = deltaf_Int(time_old)
        taures = tau_Int(time_old)
        mures = mu_Int(time_old)
        obj.bad = 1
    else:
    '''
    vres = sol.y[0, :]
    chires = sol.y[1, :]
    gammares = sol.y[2, :]
    tetares = sol.y[3, :]
    lamres = sol.y[4, :]
    hres = sol.y[5, :]
    mres = sol.y[6, :]
    alfares = alfa_Int(time_old)
    deltares = delta_Int(time_old)
    deltafres = deltaf_Int(time_old)
    taures = tau_Int(time_old)
    mures = mu_Int(time_old)
    #    status = sol.status
    #if sol.status != 0:
    #  print(sol.y)
      #      Nintlocal = Nintlocal*1.5
       #     print("new int", Nintlocal)
    #for c in range(Nint-1):
        # print(i, x[i,:])
        # print(u[i,:])
       # k1 = dt*dyn(t[c], x[c, :], alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        #print("k1", k1, "k1/dt", k1/dt)
        #k2 = dt*dyn(t[c] + dt / 2, x[c, :] + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        #print("k2", k2, "k1/dt", k2 / dt)
        #k3 = dt*dyn(t[c] + dt / 2, x[c, :] + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        #print("k3", k3, "k1/dt", k3 / dt)
        #k4 = dt*dyn(t[c + 1], x[c, :] + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        #print("k4", k4, "k1/dt", k4 / dt)
        #x[c + 1, :] = x[c, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    '''
    if Nintlocal != Nint:
        print(time_int[-1], sol.t[-1])
        print(Nint, Nintlocal)
        print(sol.message)
        print(tstart, tfin)
        #if len(sol.y[0,:]) != len(time_new):
        #   sol.y = obj.xold
        v_Int = interpolate.PchipInterpolator(sol.t, sol.y[0, :])
        chi_Int = interpolate.PchipInterpolator(sol.t, sol.y[1, :])
        gamma_Int = interpolate.PchipInterpolator(sol.t, sol.y[2, :])
        teta_Int = interpolate.PchipInterpolator(sol.t, sol.y[3, :])
        lam_Int = interpolate.PchipInterpolator(sol.t, sol.y[4, :])
        h_Int = interpolate.PchipInterpolator(sol.t, sol.y[5, :])
        m_Int = interpolate.PchipInterpolator(sol.t, sol.y[6, :])


        vres = v_Int(time_old)
        chires = chi_Int(time_old)
        gammares = gamma_Int(time_old)
        tetares = teta_Int(time_old)
        lamres = lam_Int(time_old)
        hres = h_Int(time_old)
        mres = m_Int(time_old)
    else:
        vres = sol.y[0,:]
        chires = sol.y[1,:]
        gammares = sol.y[2,:]
        tetares = sol.y[3,:]
        lamres = sol.y[4,:]
        hres = sol.y[5,:]
        mres = sol.y[6,:]
    '''
    # vres = x[:, 0]  # orig interavals
    # chires = x[:, 1]
    # gammares = x[:, 2]
    # tetares = x[:, 3]
    # lamres = x[:, 4]
    # hres = x[:, 5]
    # mres = x[:, 6]

    return vres, chires, gammares, tetares, lamres, hres, mres, time_old, alfares, deltares, deltafres, taures, mures, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int


def MultiShooting(var, dyn):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in radians'''
    obj.bad = 0
    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    deltaf = np.zeros((NContPoints))
    tau = np.zeros((NContPoints))
    mu = np.zeros((NContPoints))

    timestart = 0.0
    time_vec = np.zeros((1))
    states_atNode = np.zeros((0))
    obj.time = np.zeros((1))

    for i in range(Nleg):

        states = conversionStatesToOrig(var[i * Nstates:(i + 1) * Nstates], 1)  # orig intervals
        #print("leg: ", i, "states: ", states)
        #if states[0] <= v_init[0]+1:
         #   states[5] = h_init[0]
          #  var[i*Nstates+5] = h_toNew(h_init[0])

        timeend = timestart + t_toOrig(var[varTot + i])
        time_vec = np.concatenate((time_vec, (timeend,)))  # vector with time points
        obj.time = np.hstack((obj.time, timeend))

        for k in range(NContPoints):
            '''this for loop takes the controls from the optimization variable and stores them into different variables'''
            '''here controls are scaled'''
            alfa[k] = alfa_toOrig(var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + Ncontrols * k])
            delta[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 1 + Ncontrols * k]
            deltaf[k] = deltaf_toOrig(var[varStates + i * (Ncontrols * NContPoints- Ncontrols) + 2 + Ncontrols * k])
            tau[k] = tau_toOrig(var[varStates + i * (Ncontrols * NContPoints- Ncontrols) + 3 + Ncontrols * k])
            mu[k] = mu_toOrig(var[varStates + i * (Ncontrols * NContPoints- Ncontrols) + 4 + Ncontrols * k])

        controls = np.vstack((alfa, delta, deltaf, tau, mu)) # orig intervals

        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures, \
        alfa_I, delta_I, deltaf_I, tau_I, mu_I = SingleShooting(states, controls, dyn, timestart, timeend, Nint)

        #states_atNode = np.concatenate((states_atNode, (
        #            (v_toNew(vres[-1]), chi_toNew(chires[-1]), gamma_toNew(gammares[-1]), teta_toNew(tetares[-1]),
        #             lam_toNew(lamres[-1], obj), h_toNew(hres[-1]), m_toNew(mres[-1], obj))))) # new intervals

        states_atNode = np.concatenate((states_atNode, (
            (vres[-1], chires[-1], gammares[-1], tetares[-1], lamres[-1], hres[-1], mres[-1]))))  # new intervals

        '''res quantities are unscaled'''
        '''these are row vectors'''
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

        '''here controls and states matrices are defined'''
        if i == 0:
            states_after = np.column_stack((vres, chires, gammares, tetares, lamres, hres, mres))
            controls_after = np.column_stack((alfares, deltares, deltafres, taures, mures))
        else:
            states_after = np.vstack((states_after, np.column_stack((vres, chires, gammares, tetares, lamres, hres, mres))))
            controls_after = np.vstack((controls_after, np.column_stack((alfares, deltares, deltafres, taures, mures))))

        timestart = timeend


    obj.States = states_after
    obj.Controls = controls_after

    eq_c = equality(var, states_atNode)
    obj.eqOld = eq_c

    ineq_c = inequalityAll(states_after, controls_after, Nint*Nleg, time_vec)
    obj.ineqOld = ineq_c

    h = states_after[-1, 5]
    m = states_after[-1, 6]
    delta = controls_after[-1, 1]
    tau = controls_after[-1, 3]

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
    return eq_c, ineq_c, cost


def plot(var, Nint):

    time = np.zeros((1))
    timeTotal = np.zeros((0))
    alfaCP = np.zeros((Nleg, NContPoints))
    deltaCP = np.zeros((Nleg, NContPoints))
    deltafCP = np.zeros((Nleg, NContPoints))
    tauCP = np.zeros((Nleg, NContPoints))
    muCP = np.zeros((Nleg, NContPoints))
    res = open("res_{}_{}.txt".format(os.path.basename(__file__), timestr), "w")
    tCtot = np.zeros((0))
    timestart = 0.0
    for i in range(Nleg):
        alfa = np.zeros((NContPoints))
        delta = np.zeros((NContPoints))
        deltaf = np.zeros((NContPoints))
        tau = np.zeros((NContPoints))
        mu = np.zeros((NContPoints))
        states = conversionStatesToOrig(var[i * Nstates:(i + 1) * Nstates], 1)  # orig intervals

        timeend = timestart + t_toOrig(var[i + varTot])
        timeTotal = np.linspace(timestart, timeend, Nint)
        time = np.concatenate((time, (timeend,)))
        tC = np.linspace(timestart, timeend, NContPoints)
        tCtot = np.concatenate((tCtot, tC))

        for k in range(NContPoints):
            alfa[k] = alfa_toOrig(var[varStates + i * (Ncontrols * NContPoints- Ncontrols) + Ncontrols * k])
            alfaCP[i, k] = alfa[k]
            delta[k] = var[varStates + i * (Ncontrols * NContPoints- Ncontrols) + 1 + Ncontrols * k]
            deltaCP[i, k] = delta[k]
            deltaf[k] = deltaf_toOrig(var[varStates + i * (Ncontrols * NContPoints- Ncontrols) + 2 + Ncontrols * k])
            deltafCP[i, k] = deltaf[k]
            tau[k] = tau_toOrig(var[varStates + i * (Ncontrols * NContPoints- Ncontrols) + 3 + Ncontrols * k])
            tauCP[i, k] = tau[k]
            mu[k] = mu_toOrig(var[varStates + i * (Ncontrols * NContPoints- Ncontrols) + 4 + Ncontrols * k])
            muCP[i, k] = mu[k]
        controls = np.vstack((alfa, delta, deltaf, tau, mu)) # orig intervals


        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures, alfa_I, delta_I, deltaf_I, tau_I, mu_I \
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

        downrange = - (vres ** 2) / g * np.sin(2 * chires)

        plt.figure(0)
        plt.title("Velocity")
        plt.plot(timeTotal, vres)
        plt.grid()
        plt.ylabel("m/s")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "velocity" + ".png")


        plt.figure(1)
        plt.title("Flight path angle \u03C7")
        plt.plot(timeTotal, np.rad2deg(chires))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "chi" + ".png")


        plt.figure(2)
        plt.title("Angle of climb \u03B3")
        plt.plot(timeTotal, np.rad2deg(gammares))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "gamma" + ".png")


        plt.figure(3)
        plt.title("Longitude \u03B8")
        plt.plot(timeTotal, np.rad2deg(tetares))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "theta" + ".png")


        plt.figure(4)
        plt.title("Latitude \u03BB")
        plt.plot(timeTotal, np.rad2deg(lamres))
        plt.grid()
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
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Chi", "Gamma", "Theta", "Lambda"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "angles" + ".png")


        plt.figure(6)
        plt.title("Altitude")
        plt.plot(timeTotal, hres / 1000)
        plt.grid()
        plt.ylabel("km")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "altitude" + ".png")


        plt.figure(7)
        plt.title("Mass")
        plt.plot(timeTotal, mres)
        plt.grid()
        plt.ylabel("kg")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
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
        plt.savefig(savefig_file + "alpha" + ".png")


        plt.figure(9)
        plt.title("Throttles")
        plt.plot(timeTotal, deltares * 100, color='r')
        plt.plot(timeTotal, taures * 100, color='k')
        plt.plot(tC, deltaCP[i, :] * 100, 'ro')
        plt.plot(tC, tauCP[i, :] * 100, 'ro')
        plt.grid()
        plt.ylabel("%")
        plt.xlabel("time [s]")
        plt.legend(["Delta", "Tau", "Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "throttles" + ".png")


        plt.figure(10)
        plt.title("Body Flap deflection \u03B4")
        plt.plot(tC, np.rad2deg(deltafCP[i, :]), "ro")
        plt.plot(timeTotal, np.rad2deg(deltafres))
        plt.grid()
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
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
        plt.savefig(savefig_file + "mu" + ".png")


        plt.figure(12)
        plt.title("Dynamic Pressure")
        plt.plot(timeTotal, q / 1000)
        plt.grid()
        plt.ylabel("kPa")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
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
        plt.savefig(savefig_file + "accelerations" + ".png")


        plt.figure(14)
        plt.title("Downrange")
        plt.plot(downrange / 1000, hres / 1000)
        plt.grid()
        plt.ylabel("km")
        plt.xlabel("km")
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
        plt.savefig(savefig_file + "forces" + ".png")


        plt.figure(16)
        plt.title("Mach")
        plt.plot(timeTotal, M)
        plt.grid()
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
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
        plt.savefig(savefig_file + "moment" + ".png")

        timestart = timeend

    print("m before Ho : {0:.5f}".format(mres[-1]))
    print("mf          : {0:.5f}".format(mf[-1]))
    print("altitude Hohmann starts: {0:.5f}".format(hres[-1]))
    print("final time  : {}".format(time))

    res.close()
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
    div = abs(conj[:Nstates * (Nleg - 1)])
    if any(div) <= 0:
        div = np.ones(1, Nstates * (Nleg-1))
    eq_cond = np.zeros((0))
    eq_cond = np.concatenate((eq_cond, var[0:7] - states_init))
    eq_cond = np.concatenate((eq_cond, (conversionStatesToOrig(var[Nstates:varStates], Nleg-1) - conj[:Nstates * (Nleg - 1)]/div)))  # knotting conditions
    eq_cond = np.concatenate((eq_cond, var[varStates:varStates + Ncontrols] - cont_init))  # init cond on alpha
    eq_cond = np.concatenate((eq_cond, (vvv - vtAbs,)))
    eq_cond = np.concatenate((eq_cond, (chifin - chiass,)))
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


def display_func():

    m = obj.States[-1, 6]
    h = obj.States[-1, 5]
    delta = obj.Controls[-1, 1]
    tau = obj.Controls[-1, 3]*2-1

    tf = obj.time

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                    obj.lRef, obj.xcgf, obj.xcg0)

    # Hohmann transfer mass calculation
    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / (obj.g0*isp)))

    print("m before Ho : {0:.5f}".format(m))
    print("mf          : {0:.5f}".format(mf))
    print("altitude Hohmann starts: {0:.5f}".format(h))
    print("final time  : ", tf)


def cost_fun(var, sp):
    if (var==obj.varOld).all():
        return obj.costOld
    else:
        eq, ineq, cost = MultiShooting(var, dynamicsInt)
        return cost


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


def JacEqSave(var, type):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var,type))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (constraints(var+dx, type) - f0) / epsilon
        dx[i] = 0.0
    sparse = csc_matrix(jac)
    save_npz("/home/francesco/git_workspace/FESTIP_Work/MultipleShooting_Algorithm/jacEq", sparse)
    return jac.transpose()


def JacIneqSave(var, type):
    epsilon = eps
    x0 = np.asfarray(var)
    f0 = np.atleast_1d(constraints(var, type))
    jac = np.zeros([len(x0), len(f0)])
    dx = np.zeros(len(x0))
    for i in range(len(x0)):
        dx[i] = epsilon
        jac[i] = (constraints(var+dx, type) - f0) / epsilon
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

iterator = 0

while iterator < maxIterator:
    print("---- iteration : {0} ----".format(iterator + 1))

    opt = optimize.minimize(cost_fun,
                            X0,
                            constraints=cons,
                            bounds=bnds,
                            jac = JacFun,
                            args=(spCost,),
                            method='SLSQP',
                            options={"maxiter": maxiter,
                                     "ftol": ftol,
                                     "eps":eps,
                                     "disp": True})

    X0 = opt.x

    if not (opt.status):
        break
    iterator += 1


end = time.time()
time_elapsed = end-start
tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
print("Time elapsed for total optimization ", tformat)
plot(X0, Nintplot)
