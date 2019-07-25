import matplotlib.pyplot as plt
from scipy import interpolate, optimize
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import Bounds
from models import *
from mapping_functions import *
import time
import datetime
import os
import sys
from OpenGoddard.optimize import Guess, Condition

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


class Guess:
    """Class for initial value guess for optimization.
    Collection of class methods
    """

    @classmethod
    def zeros(cls, time):
        """ return zeros that array size is same as time length
        Args:
            time (array_like) :
        Returns:
            (N, ) ndarray
        """
        return np.zeros(len(time))

    @classmethod
    def constant(cls, time, const):
        """ return constant values that array size is same as time length
        Args:
            time (array_like) :
            const (float) : set value
        Returns:
            (N, ) ndarray
        """
        return np.ones(len(time)) * const

    @classmethod
    def linear(cls, time, y0, yf):
        """ return linear function values that array size is same as time length
        Args:
            time (array_like) : time
            y0 (float): initial value
            yf (float): final value
        Returns:
            (N, ) ndarray
        """
        x = np.array([time[0], time[-1]])
        y = np.array([y0, yf])
        f = interpolate.interp1d(x, y)
        return f(time)

    @classmethod
    def cubic(cls, time, y0, yprime0, yf, yprimef):
        """ return cubic function values that array size is same as time length
        Args:
            time (array_like) : time
            y0 (float) : initial value
            yprime0 (float) : slope of initial value
            yf (float) : final value
            yprimef (float) : slope of final value
        Returns:
            (N, ) ndarray
        """
        y = np.array([y0, yprime0, yf, yprimef])
        t0 = time[0]
        tf = time[-1]
        A = np.array([[1, t0, t0**2, t0**3], [0, 1, 2*t0, 3*t0**2],
                      [1, tf, tf**2, tf**3], [0, 1, 2*tf, 3*tf**2]])
        invA = np.linalg.inv(A)
        C = invA.dot(y)
        ys = C[0] + C[1]*time + C[2]*time**2 + C[3]*time**3
        return ys


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


obj = Spaceplane()


'''reading of aerodynamic coefficients and specific impulse from file'''

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

'''set problem parameters'''

time_tot = 300 # initial time
Nbar = 7 # number of conjunction points
Nleg = Nbar - 1  # number of multiple shooting sub intervals
NContPoints = 7  # number of control points for interpolation inside each interval
Nint = 150 # number of points for each single shooting integration
Nstates = 7  # number of states
Ncontrols = 5  # number of controls
varStates = Nstates * Nleg  # total number of optimization variables for states
varControls = Ncontrols * (Nleg * (NContPoints - 1) + 1)   # total number of optimization variables for controls
varTot = varStates + varControls  # total number of optimization variables for states and controls
NineqCond = Nint # Nleg * NContPoints - Nbar + 2
tnew = np.linspace(0, time_tot, Nbar)  # time vector used for interpolation of states initial guess
tcontr = np.linspace(0, time_tot, int(varControls / Ncontrols))  # time vector used for interpolation of controls intial guess
unit_t = 1000

'''NLP solver parameters'''
maxiter = 10 # max number of iterations for nlp solver
ftol = 1e-12  # numeric tolerance of nlp solver
eps = 1e-10
#eps = 1.4901161193847656e-08# increment of the derivative
maxIterator = 1  # max number of optimization iterations


'''definiton of initial conditions'''

# set vector of initial conditions of states and controls
X = np.zeros((0))
U = np.zeros((0))

v_init = Guess.cubic(tnew, v_toNew(1.0), 0.0, v_toNew(obj.Vtarget), 0.0)
chi_init = Guess.linear(tnew, chi_toNew(obj.chistart), chi_toNew(obj.chi_fin))
gamma_init = Guess.linear(tnew, gamma_toNew(obj.gammastart), gamma_toNew(np.deg2rad(0.0)))
teta_init = Guess.constant(tnew, teta_toNew(obj.longstart))
lam_init = Guess.constant(tnew, lam_toNew(obj.latstart, obj))
h_init = Guess.cubic(tnew, h_toNew(1.0), 0.0, h_toNew(obj.Hini), 0.0)
m_init = Guess.cubic(tnew, m_toNew(obj.M0, obj), 0.0, m_toNew(obj.m10, obj), 0.0)
alfa_init = Guess.constant(tcontr, alfa_toNew(0.0))
delta_init = Guess.cubic(tcontr, 1.0, 0.0, 0.001, 0.0)
deltaf_init = Guess.constant(tcontr, deltaf_toNew(0.0))
tau_init = Guess.constant(tcontr, tau_toNew(0.0))
mu_init = Guess.constant(tcontr, mu_toNew(0.0))

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

X0 = np.hstack((X, U, dt / unit_t))  # vector of initial conditions here all the angles are in degrees!!!!!
obj.varOld = np.zeros((len(X0)))

# X0 has first all X guesses and then all U guesses
# at this point the vectors of initial guesses for states and controls for every time interval are defined


'''set upper and lower bounds for states, controls and time, scaled'''
'''major issues with bounds!!!'''
#bndX = ((0.0, 1.0),)
#bndU = ((0.0, 1.0),)
#bndT = ((1/unit_t, time_tot/unit_t),)

Xlb = ([v_toNew(0.001), chi_toNew(np.deg2rad(90)), gamma_toNew(0.0), teta_toNew(np.deg2rad(-90)), lam_toNew(-obj.incl, obj), h_toNew(0.001), m_toNew(obj.m10, obj)]) # states lower bounds
Xub = ([v_toNew(1e4), chi_toNew(np.deg2rad(270)), gamma_toNew(np.deg2rad(90)), teta_toNew(0.0), lam_toNew(obj.incl, obj), h_toNew(2e5), m_toNew(obj.M0, obj)]) # states upper bounds

Ulb = ([alfa_toNew(np.deg2rad(-2)), 0.0001, deltaf_toNew(np.deg2rad(-20)), tau_toNew(-1.0), mu_toNew(np.deg2rad(-90))]) # controls lower bounds
Uub = ([alfa_toNew(np.deg2rad(40)), 1.0, deltaf_toNew(np.deg2rad(30)), tau_toNew(1.0), mu_toNew(np.deg2rad(90))]) # controls upper bounds

Tlb = ([1/unit_t,]) # time lower bounds
Tub = ([1.0,]) # time upper bounds

lb = Xlb*Nleg + Ulb * (Nleg * NContPoints) + Tlb*Nleg
ub = Xub*Nleg + Uub * (Nleg * NContPoints) + Tub*Nleg
bnds = Bounds(lb, ub)

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


def dynamicsInt(t, states, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int):
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
    mu = mu_Int(t)

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


def inequalityAll(states, controls, varnum):
    '''this function takes states and controls unscaled'''
    v = states[:, 0]
    chi = states[:, 1]
    gamma = states[:, 2]
    teta = states[:, 3]
    lam = states[:, 4]
    h = states[:, 5]
    m = states[:, 6]
    alfa = controls[:, 0]
    delta = controls[:, 1]
    deltaf = controls[:, 2]
    tau = controls[:, 3]  # tau back to [-1, 1] interval
    mu = controls[:, 4]

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
    MomTotA = abs(MomTot)

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)

    # accelerations

    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    r1 = h[-1] + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m[-1] / np.exp((Dv1 + Dv2) / (obj.g0*isp[-1]))

    result = Condition()

    # lower bounds
    result.lower_bound(to_new_int(v / 1e3, 0.0, 10, 0.0, 1.0), to_new_int(1e-5, 0.0, 10, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(chi, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0),
                       to_new_int(np.deg2rad(90), np.deg2rad(90), np.deg2rad(270), 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(gamma, np.deg2rad(-2), np.deg2rad(90), 0.0, 1.0),
                       to_new_int(0.0, 0.0, np.deg2rad(90), 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(teta, np.deg2rad(-90), 0.0, 0.0, 1.0),
                       to_new_int(np.deg2rad(-90), np.deg2rad(-90), 0.0, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(lam, -obj.incl, obj.incl, 0.0, 1.0),
                       to_new_int(-obj.incl, -obj.incl, obj.incl, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(h / 1e4, 0.0, 20, 0.0, 1.0), to_new_int(1e-6, 0.0, 20, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(m[1:], obj.m10, obj.M0, 0.0, 1.0),
                       to_new_int(obj.m10, obj.m10, obj.M0, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(alfa, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0),
                       to_new_int(np.deg2rad(-2), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)

    result.lower_bound(delta, 0.001, unit=1)

    result.lower_bound(to_new_int(deltaf, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0),
                       to_new_int(np.deg2rad(-20), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)

    result.lower_bound((tau + 1) / 2, 0.0, unit=1)
    result.lower_bound(to_new_int(mu, np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0),
                       to_new_int(np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(mf, obj.m10, obj.M0, 0.0, 1.0),
                       to_new_int(obj.m10, obj.m10, obj.M0, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(h[-1] / 1e4, 0.0, 20, 0.0, 1.0), to_new_int(9, 0.0, 20, 0.0, 1.0), unit=1)

    result.lower_bound(to_new_int(MomTot / 1e5, -1e2, 1e2, 0.0, 1.0), to_new_int(-obj.k / 1e5, -1e2, 1e2, 0.0, 1.0),
                       unit=1)

    result.lower_bound(to_new_int(az, -1e2, 1e2, 0.0, 1.0), to_new_int(-obj.MaxAz, -1e2, 1e2, 0.0, 1.0), unit=1)

    # upper bounds
    result.upper_bound(to_new_int(v / 1e3, 0.0, 10, 0.0, 1.0), to_new_int(9, 0.0, 10, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(chi, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0),
                       to_new_int(np.deg2rad(270), np.deg2rad(90), np.deg2rad(270), 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(gamma, np.deg2rad(-2), np.deg2rad(90), 0.0, 1.0),
                       to_new_int(np.deg2rad(89.9), 0.0, np.deg2rad(90), 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(teta, np.deg2rad(-90), 0.0, 0.0, 1.0),
                       to_new_int(0.0, np.deg2rad(-90), 0.0, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(lam, -obj.incl, obj.incl, 0.0, 1.0),
                       to_new_int(obj.incl, -obj.incl, obj.incl, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(h / 1e4, 0.0, 20, 0.0, 1.0), to_new_int(18, 0.0, 20, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(m, obj.m10, obj.M0, 0.0, 1.0), to_new_int(obj.M0, obj.m10, obj.M0, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(alfa, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0),
                       to_new_int(np.deg2rad(40), np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)

    result.upper_bound(delta, 1.0, unit=1)

    result.upper_bound(to_new_int(deltaf, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0),
                       to_new_int(np.deg2rad(30), np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)

    result.upper_bound((tau + 1) / 2, 1.0, unit=1)

    result.upper_bound(to_new_int(mu, np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0),
                       to_new_int(np.deg2rad(90), np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(MomTot / 1e5, -1e2, 1e2, 0.0, 1.0), to_new_int(obj.k / 1e5, -1e2, 1e2, 0.0, 1.0),
                       unit=1)

    result.upper_bound(to_new_int(az, -1e2, 1e2, 0.0, 1.0), to_new_int(obj.MaxAz, -1e2, 1e2, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(ax, 0.0, 1e2, 0.0, 1.0), to_new_int(obj.MaxAx, 0.0, 1e2, 0.0, 1.0), unit=1)

    result.upper_bound(to_new_int(q / 1000, 0.0, 1e2, 0.0, 1.0), to_new_int(obj.MaxQ / 1000, 0.0, 1e2, 0.0, 1.0),
                       unit=1)
    return result()


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

    #print("single shooting")
    timeCont = np.linspace(tstart, tfin, NContPoints)
    x = np.zeros((Nint, Nstates))
    x[0,:] = states  # vector of intial states ready

    #print("Single: ", Time)
    # now interpolation of controls

    alfa_Int = interpolate.PchipInterpolator(timeCont, controls[0, :])
    delta_Int = interpolate.PchipInterpolator(timeCont, controls[1, :])
    deltaf_Int = interpolate.PchipInterpolator(timeCont, controls[2, :])
    tau_Int = interpolate.PchipInterpolator(timeCont, controls[3, :])
    mu_Int = interpolate.PchipInterpolator(timeCont, controls[4, :])

    time_new = np.linspace(tstart, tfin, Nint)

    dt = (time_new[1] - time_new[0])

    t = time_new

    #sol = solve_ivp(fun=lambda t, x: dyn(t, x, alfa_Int, delta_Int, deltaf_Int, tau_Int), t_span=[tstart, tfin], y0=x, t_eval=time_new, method='RK45')

    for i in range(Nint-1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt*dyn(t[i], x[i, :], alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k1: ", k1)
        k2 = dt*dyn(t[i] + dt / 2, x[i, :] + k1 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k2: ", k2)
        k3 = dt*dyn(t[i] + dt / 2, x[i, :] + k2 / 2, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
        # print("k3: ", k3)
        k4 = dt*dyn(t[i + 1], x[i, :] + k3, alfa_Int, delta_Int, deltaf_Int, tau_Int, mu_Int)
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

    vres = x[:, 0] # orig interavals
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


def MultiShooting(var, dyn):
    '''in this function the states and controls are scaled'''
    '''this function takes the data from the optimization variable, so the angles enters in radians'''

    alfa = np.zeros((NContPoints))
    delta = np.zeros((NContPoints))
    deltaf = np.zeros((NContPoints))
    tau = np.zeros((NContPoints))
    mu = np.zeros((NContPoints))

    vineq = np.zeros((1, Nint))  # states and controls defined as row vectors
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
    muineq = np.zeros((1, Nint))

    timestart = 0.0
    time_vec = np.zeros((1))
    states_atNode = np.zeros((0))
    obj.time = np.zeros((1))

    for i in range(Nleg):

        states = conversionStatesToOrig(var[i * Nstates:(i + 1) * Nstates], 1)  # orig intervals

        timeend = timestart + var[varTot + i] * unit_t
        time_vec = np.concatenate((time_vec, (timeend,)))  # vector with time points
        obj.time = np.hstack((obj.time, timeend))

        for k in range(NContPoints):
            '''this for loop takes the controls from the optimization variable and stores them into different variables'''
            '''here controls are scaled'''
            alfa[k] = alfa_toOrig(var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + Ncontrols * k])
            delta[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 1 + Ncontrols * k]
            deltaf[k] = deltaf_toOrig(var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 2 + Ncontrols * k])
            tau[k] = tau_toOrig(var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 3 + Ncontrols * k])
            mu[k] = mu_toOrig(var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 4 + Ncontrols * k])

        controls = np.vstack((alfa, delta, deltaf, tau, mu)) # orig intervals

        vres, chires, gammares, tetares, lamres, hres, mres, tres, alfares, deltares, deltafres, taures, mures\
            = SingleShooting(states, controls, dyn, timestart, timeend, Nint)

        states_atNode = np.concatenate((states_atNode, (
                    (v_toNew(vres[-1]), chi_toNew(chires[-1]), gamma_toNew(gammares[-1]), teta_toNew(tetares[-1]),
                     lam_toNew(lamres[-1], obj), h_toNew(hres[-1]), m_toNew(mres[-1], obj))))) # new intervals

        '''res quantities are unscaled'''

        # time_ineq = np.linspace(tres[0], tres[-1], NineqCond)
        # print(np.shape(tres), np.shape(vres))
        # vinterp = interpolate.PchipInterpolator(tres, vres)
        # chiinterp = interpolate.PchipInterpolator(tres, chires)
        # gammainterp = interpolate.PchipInterpolator(tres, gammares)
        # tetainterp = interpolate.PchipInterpolator(tres, tetares)
        # laminterp = interpolate.PchipInterpolator(tres, lamres)
        # hinterp = interpolate.PchipInterpolator(tres, hres)
        # minterp = interpolate.PchipInterpolator(tres, mres)

        '''these are row vectors'''
        vineq[0, :] = vres  # vinterp(time_ineq)
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
        muineq[0, :] = mures

        '''these are column vectors'''
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
        mutt = np.transpose(muineq)

        '''here controls and states matrices are defined'''
        if i == 0:
            states_after = np.column_stack((vt, chit, gammat, tetat, lamt, ht, mt))
            controls_after = np.column_stack((alfat, deltat, deltaft, taut, mutt))
        else:
            states_after = np.vstack((states_after, np.column_stack((vt, chit, gammat, tetat, lamt, ht, mt))))
            controls_after = np.vstack((controls_after, np.column_stack((alfat, deltat, deltaft, taut, mutt))))

        timestart = timeend

    h = states_after[-1, 5]
    m = states_after[-1, 6]
    delta = controls_after[-1, 1]
    tau = controls_after[-1, 3]  # tau back in [-1,1] interval

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10, obj.lRef,
                                obj.xcgf, obj.xcg0)

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / (obj.g0 * isp))

    obj.States = states_after
    obj.Controls = controls_after

    ineq_Cond = inequalityAll(states_after, controls_after, len(states_after))  # evaluation of inequality constraints
    eq_Cond = equality(var, states_atNode)  # evaluation of equality constraints
    objective = - mf / obj.M0  # evaluation of objective function

    obj.costOld = objective
    obj.eqOld = eq_Cond
    obj.ineqOld = ineq_Cond
    obj.varOld = var

    return ineq_Cond, eq_Cond, objective


def plot(var):

    time = np.zeros((1))
    timeTotal = np.zeros((0))
    alfaCP = np.zeros((Nleg, NContPoints))
    deltaCP = np.zeros((Nleg, NContPoints))
    deltafCP = np.zeros((Nleg, NContPoints))
    tauCP = np.zeros((Nleg, NContPoints))
    muCP = np.zeros((Nleg, NContPoints))
    res = open("res_{}_{}.txt".format(os.path.basename(__file__), timestr), "w")
    tCtot = np.zeros((0))

    for i in range(Nleg):
        alfa = np.zeros((NContPoints))
        delta = np.zeros((NContPoints))
        deltaf = np.zeros((NContPoints))
        tau = np.zeros((NContPoints))
        mu = np.zeros((NContPoints))

        if i == 0:
            timestart = 0
        timeend = timestart + var[i + varTot] * unit_t
        timeTotal = np.linspace(timestart, timeend, Nint)
        time = np.concatenate((time, (timeend,)))
        tC = np.linspace(timestart, timeend, NContPoints)
        tCtot = np.concatenate((tCtot, tC))

        for k in range(NContPoints):
            alfa[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + Ncontrols * k]
            alfaCP[i, k] = alfa_toOrig(alfa[k])
            delta[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 1 + Ncontrols * k]
            deltaCP[i, k] = delta[k]
            deltaf[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 2 + Ncontrols * k]
            deltafCP[i, k] = deltaf_toOrig(deltaf[k])
            tau[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 3 + Ncontrols * k]
            tauCP[i, k] = tau_toOrig(tau[k])
            mu[k] = var[varStates + i * (Ncontrols * NContPoints - Ncontrols) + 4 + Ncontrols * k]
            muCP[i, k] = mu_toOrig(mu[k])

        timestart = timeend

        vres = obj.States[i * Nint:(i + 1) * Nint, 0]
        chires = obj.States[i * Nint:(i + 1) * Nint, 1]
        gammares = obj.States[i * Nint:(i + 1) * Nint, 2]
        tetares = obj.States[i * Nint:(i + 1) * Nint, 3]  # teta to [-90,0]
        lamres = obj.States[i * Nint:(i + 1) * Nint, 4]  # lam to [-incl, incl]
        hres = obj.States[i * Nint:(i + 1) * Nint, 5]
        mres = obj.States[i * Nint:(i + 1) * Nint, 6]

        alfares = obj.Controls[i * Nint:(i + 1) * Nint, 0]
        deltares = obj.Controls[i * Nint:(i + 1) * Nint, 1]
        deltafres = obj.Controls[i * Nint:(i + 1) * Nint, 2]
        taures = obj.Controls[i * Nint:(i + 1) * Nint, 3]  # tau to [-1,1]
        mures = obj.Controls[i * Nint:(i + 1) * Nint, 4]

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
        plt.title("Bank angle profile \u03BC")
        plt.plot(tC, np.rad2deg(muCP[i, :]), "ro")
        plt.plot(timeTotal, np.rad2deg(mures))
        plt.ylabel("deg")
        plt.xlabel("time [s]")
        plt.legend(["Control points"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "mu" + ".png")


        plt.figure(12)
        plt.title("Dynamic Pressure")
        plt.plot(timeTotal, q / 1000)
        plt.ylabel("kPa")
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "dynPress" + ".png")


        plt.figure(13)
        plt.title("Accelerations")
        plt.plot(timeTotal, ax, color='b')
        plt.plot(timeTotal, az, color='r')
        plt.ylabel("m/s^2")
        plt.xlabel("time [s]")
        plt.legend(["ax", "az"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "accelerations" + ".png")


        plt.figure(14)
        plt.title("Downrange")
        plt.plot(downrange / 1000, hres / 1000)
        plt.ylabel("km")
        plt.xlabel("km")
        plt.savefig(savefig_file + "downrange" + ".png")


        plt.figure(15)
        plt.title("Forces")
        plt.plot(timeTotal, T / 1000, color='r')
        plt.plot(timeTotal, L / 1000, color='b')
        plt.plot(timeTotal, D / 1000, color='k')
        plt.ylabel("kN")
        plt.xlabel("time [s]")
        plt.legend(["Thrust", "Lift", "Drag"], loc="best")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "forces" + ".png")


        plt.figure(16)
        plt.title("Mach")
        plt.plot(timeTotal, M)
        plt.xlabel("time [s]")
        plt.axvline(time[i], color="k", alpha=0.5)
        plt.savefig(savefig_file + "mach" + ".png")


        plt.figure(17)
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
    h = h_toOrig(conj[varStates - 2])
    lam = lam_toOrig(conj[varStates - 3], obj)
    stat = np.zeros((7))
    cont = np.zeros((5))
    stat = conj[varStates - Nstates:varStates]
    stat[0] = v_toOrig(stat[0])
    stat[1] = chi_toOrig(stat[1])
    stat[2] = gamma_toOrig(stat[2])
    stat[3] = teta_toOrig(stat[3])
    stat[4] = lam_toOrig(stat[4], obj)
    stat[5] = h_toOrig(stat[5])
    stat[6] = m_toOrig(stat[6], obj)
    #print(var[varStates:varTot])

    cont = obj.Controls[-1, :]

    vtAbs, chiass, vtAbs2 = vass(stat, cont, dynamicsVel, obj.omega)

    vvv = np.sqrt(obj.GMe / (obj.Re + h))

    if np.cos(obj.incl) / np.cos(lam) > 1:
        chifin = np.pi
    elif np.cos(obj.incl) / np.cos(lam) < - 1 :
        chifin = 0.0
    else:
        chifin = 0.5 * np.pi + np.arcsin(np.cos(obj.incl) / np.cos(lam))

    result = Condition()

    # event condition
    result.equal(var[0], to_new_int(1.0 / 1e3, 0.0, 10, 0.0, 1.0), unit=1)
    result.equal(var[1], to_new_int(obj.chistart, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0), unit=1)
    result.equal(var[2], to_new_int(obj.gammastart, 0.0, np.deg2rad(90), 0.0, 1.0), unit=1)
    result.equal(var[3], to_new_int(obj.longstart, np.deg2rad(-90), 0.0, 0.0, 1.0), unit=1)
    result.equal(var[4], to_new_int(obj.latstart, -obj.incl, obj.incl, 0.0, 1.0), unit=1)
    result.equal(var[5], to_new_int(1 / 1e4, 0.0, 20, 0.0, 1.0), unit=1)
    result.equal(var[6], to_new_int(obj.M0, obj.m10, obj.M0, 0.0, 1.0), unit=1)
    result.equal(var[Nstates:varStates], conj[:Nstates * (Nleg - 1)], unit=1)
    result.equal(var[varStates], to_new_int(0.0, np.deg2rad(-2), np.deg2rad(40), 0.0, 1.0), unit=1)
    result.equal(var[varStates + 1], 1.0, unit=1)
    result.equal(var[varStates + 2], to_new_int(0.0, np.deg2rad(-20), np.deg2rad(30), 0.0, 1.0), unit=1)
    result.equal(var[varStates + 3], 0.5, unit=1)
    result.equal(var[varStates + 4], to_new_int(0.0, np.deg2rad(-90), np.deg2rad(90), 0.0, 1.0), unit=1)
    result.equal(to_new_int(vtAbs / 1e3, 0.0, 10, 0.0, 1.0), to_new_int(vvv / 1e3, 0.0, 10, 0.0, 1.0), unit=1)
    result.equal(to_new_int(chiass, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0),
                 to_new_int(chifin, np.deg2rad(90), np.deg2rad(270), 0.0, 1.0), unit=1)
    result.equal(to_new_int(conj[varStates - 5], 0.0, np.deg2rad(90), 0.0, 1.0),
                 to_new_int(0.0, 0.0, np.deg2rad(90), 0.0, 1.0), unit=1)


    return result()


def constraints(var, type):
    #print("constraints", type)
    if all(j<eps for j in abs(var - obj.varOld)):
        #print("old cons")

        if type == "eq":
            #print("old eq", np.shape(obj.eqOld))
            #obj.varOld = var
            return obj.eqOld
        else:
            #print("old ineq", np.shape(obj.ineqOld))
            #obj.varOld = var
            return obj.ineqOld

    else:
        #print("new cons")
        ineq_c, eq_c, ob = MultiShooting(var, dynamicsInt)
        if type == "eq":
            #print("new eq", np.shape(eq_c))
            return eq_c
        else:
            #print("new ineq", np.shape(ineq_c))
            return ineq_c


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


def cost_fun(var):
    #print("cost fun")
    #print(max(var-obj.varOld), min(var-obj.varOld))
    if all(j<eps for j in abs(var - obj.varOld)):
        #print("costold")

        #obj.varOld = var
        return obj.costOld
    else:
        #print("costnew")
        ineq, eq, cost = MultiShooting(var, dynamicsInt)

        return cost


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
                            method='SLSQP',
                            options={"ftol": ftol,
                                     "eps":eps,
                                     "maxiter": maxiter,
                                     "disp":True})

    X0 = opt.x

    if not (opt.status):
        break
    iterator += 1


end = time.time()
time_elapsed = end-start
tformat = str(datetime.timedelta(seconds=int(time_elapsed)))
print("Time elapsed for total optimization ", tformat)
plot(X0)
