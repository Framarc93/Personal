from scipy import optimize
import numpy as np
from models import *
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
import random
from functools import partial
from OpenGoddard.optimize import Guess


'''definition of vehicle and mission parameters. Angles are set in degrees'''

class Spaceplane:

    def __init__(self):
        self.GMe = 3.986004418e14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371000  # Earth Radius [m]
        self.psl = 101325  # ambient pressure at sea level [Pa]
        self.latstart = 5.2  # deg latitude
        self.longstart = -52.775  # deg longitude
        self.chistart = 118  # deg flight direction
        self.incl = 51.6  # deg orbit inclination
        self.gammastart = 89.9  # deg
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
        self.Hini = 180000
        self.r2 = self.Re + self.Htarget
        self.Rtarget = self.Re + self.Hini  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s forse da modificare con velocita' assoluta
        self.chi_fin = np.rad2deg(0.5 * np.pi + np.arcsin(np.cos(np.deg2rad(self.incl)) / np.cos(np.deg2rad(self.latstart))))



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

time_tot = 300  # s
Npoints = 15  # number of multiple shooting points

Nint = 10  # number of points for initial guess from OpenGoddard
tsec = np.linspace(0, time_tot, Nint) # time vector used to interpolate data from OpenGoddard optimization (10 points)

maxiter = 100  # max number of iterations for nlp solver
ftol = 1e-8 # numeric tolerance of nlp solver
eps = 1e-10
Nstates = 7  # number of states
Ncontrols = 4  # number of controls
maxIterator = 5 # max number of optimization iterations
varStates = Nstates*Npoints  # total number of states variables
varTot = (Nstates+Ncontrols)*Npoints  # total number of variables, controls and states

'''values for reduction to unit'''

unit_v = 8000
unit_chi = 270
unit_gamma = obj.gammastart
unit_teta = abs(obj.longstart)
unit_lam = obj.incl
unit_h = obj.Hini
unit_m = obj.M0
unit_alfa = 40
unit_delta = 1
unit_deltaf = 30
unit_tau = 1
states_unit = np.array((unit_v, unit_chi, unit_gamma, unit_teta, unit_lam, unit_h, unit_m))
controls_unit = np.array((unit_alfa, unit_delta, unit_deltaf, unit_tau))

# set vector of initial conditions of states and controls
X = np.zeros((0))
U = np.zeros((0))

# set vector of initial random guesses for state and controls in all intervals taken from OpenGoddard
'''!!!!careful, angles still in radians!!!!!'''

v_guess =np.array(([1e-3, 3.35466889e+01, 3.55270005e+02, 1.26169097e+03, 2.94231538e+03, 4.94757217e+03, 6.13062730e+03, 6.78802329e+03, 7.46895240e+03, 7.85070852e+03]))
chi_guess = np.deg2rad([113.0, 116.30423696, 116.55208878, 117.92354784, 121.41884321, 124.0532628, 126.4974523, 128.1499793, 128.58329859, 128.58791495])
gamma_guess = np.deg2rad([89.9, 83.78757497, 81.15171249, 69.06418848, 47.98137708, 28.98511628, 13.56998971, 3.28760848, 0.16768323, 0.0])
teta_guess = np.deg2rad([-52.775, -52.77498353, -52.77713044, -52.78909518, -52.85679498, -52.99701637, -53.04747186, -52.95654205, -52.89033306, -52.88833992])
lam_guess = np.deg2rad([5.2, 5.20007366, 5.20362965, 5.22300301, 5.30003444, 5.43300977, 5.44236503, 5.29294487, 5.1991067, 5.19589304])
h_guess = np.array(([1e-3, 6.91699584e+02, 6.88379667e+03, 2.35289454e+04, 4.60283074e+04, 6.68505779e+04, 8.31630775e+04, 9.23002789e+04, 9.57397295e+04, 9.62557344e+04]))
m_guess = np.array(([450400.0, 448303.64036542, 429026.36297693, 371743.54107237, 274693.22046435, 176257.04622121, 112582.72040783, 73532.87311595, 52919.0970599, 47340.03235158]))
alfa_guess = np.deg2rad([0.0, 3.2622134, 0.20332326, -1.29089827, -1.98241146, 5.64311896, -0.94849787, -1.99314296, 7.58186611, -1.89409715])
delta_guess = np.array(([1.0, 0.99999786, 0.99963553, 0.99947708, 0.85892564, 0.42352122, 0.21, 0.19525226, 0.193, 0.18952055]))
deltaf_guess = np.deg2rad([0.0, 5.58416932e+00, 5.28258790e+00, -1.67103287e-02, -1.86223484e+01, -1.97922547e+01, -1.99774787e+01, -1.88513373e+01, -1.97803298e+01, -1.63827991e+01])
tau_guess = np.array(([0.0, 2.98802897e-04, 1.95047783e-01, 1.40234173e-01, 2.60378792e-04, 8.77981866e-04, 5.27933621e-07, 9.01132288e-05, 3.08648807e-06, 8.09500173e-05]))

'''interpolation over OpenGoddard data to use more than 10 points'''

tnew = np.linspace(0, time_tot, Npoints)

v_interp = interpolate.interp1d(tsec, v_guess)
chi_interp = interpolate.interp1d(tsec, chi_guess)
gamma_interp = interpolate.interp1d(tsec, gamma_guess)
teta_interp = interpolate.interp1d(tsec, teta_guess)
lam_interp = interpolate.interp1d(tsec, lam_guess)
h_interp = interpolate.interp1d(tsec, h_guess)
m_interp = interpolate.interp1d(tsec, m_guess)
alfa_interp = interpolate.interp1d(tsec, alfa_guess)
delta_interp = interpolate.interp1d(tsec, delta_guess)
deltaf_interp = interpolate.interp1d(tsec, deltaf_guess)
tau_interp = interpolate.interp1d(tsec, tau_guess)

XGuessOG = np.array((v_interp(tnew)/ unit_v, chi_interp(tnew)/ unit_chi, gamma_interp(tnew)/ unit_gamma,  # states initial guesses from OpenGoddard data
                   teta_interp(tnew)/ unit_teta, lam_interp(tnew)/ unit_lam, h_interp(tnew)/ unit_h, m_interp(tnew)/ unit_m))

UGuessOG = np.array((alfa_interp(tnew)/ unit_alfa, delta_interp(tnew)/ unit_delta, deltaf_interp(tnew)/ unit_deltaf,
                   tau_interp(tnew)/ unit_tau))  # controls initial guesses from OpenGoddard data

'''definition of another set of initial guesses, more random but reasonable'''

v_init = Guess.cubic(tnew, 1.0, 0.0, obj.Vtarget, 0.0)
chi_init = Guess.cubic(tnew, obj.chistart, 0.0, obj.chi_fin, 0.0)
gamma_init = Guess.cubic(tnew, obj.gammastart, 0.0, 0.0, 0.0)
teta_init = Guess.constant(tnew, obj.longstart)
lam_init = Guess.constant(tnew, obj.latstart)
h_init = Guess.cubic(tnew, 1.0, 0.0, obj.Hini, 0.0)
m_init = Guess.cubic(tnew, obj.M0, 0.0, obj.m10, 0.0)
alfa_init = Guess.zeros(tnew)
delta_init = Guess.cubic(tnew, 1.0, 0.0, 0.1, 0.0)
deltaf_init = Guess.zeros(tnew)
tau_init = Guess.zeros(tnew)

XGuess = np.array((v_init/ unit_v, chi_init/ unit_chi, gamma_init/ unit_gamma, teta_init/ unit_teta, lam_init/ unit_lam,
                     h_init/ unit_h, m_init/ unit_m))  # states initial guesses

UGuess = np.array((alfa_init/ unit_alfa, delta_init/ unit_delta, deltaf_init/ unit_deltaf, tau_init/ unit_tau)) # states initial guesses

for i in range(Npoints):
    '''creation of vector of states initial guesses'''
    for j in range(Nstates):
        X = np.hstack((X, XGuess[j][i]))

for i in range(Npoints):
    '''creation of vector of controls initial guesses'''
    for j in range(Ncontrols):
        U = np.hstack((U, UGuess[j][i]))

dt = np.zeros((0))
for i in range(len(tnew)-1):
    '''creation of vector of time intervals'''
    dt = np.hstack((dt, tnew[i+1] - tnew[i]))


X0 = np.hstack((X, U, dt/time_tot))  # vector of initial conditions


# X0 has first all X guesses and then all U guesses
# at this point the vectors of initial guesses for states and controls for every time interval are defined

'''set system of ode'''


def dynamics(t, states, controls):
    '''this functions receives the states and controls unscaled and calculates the dynamics'''
    v = states[0]
    chi = np.deg2rad(states[1])
    gamma = np.deg2rad(states[2])
    teta = np.deg2rad(states[3])
    lam = np.deg2rad(states[4])
    h = states[5]
    m = states[6]
    alfa = np.deg2rad(controls[0])
    delta = controls[1]
    deltaf = np.deg2rad(controls[2])
    tau = controls[3]

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)
    #Press = np.asarray(Press, dtype=np.float64)
    #rho = np.asarray(rho, dtype=np.float64)
    #c = np.asarray(c, dtype=np.float64)

    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                 obj.xcg0, obj.xcgf, obj.pref)
    #L = np.asarray(L, dtype=np.float64)
    #D = np.asarray(D, dtype=np.float64)
    #MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10,
                                     obj.lRef, obj.xcgf, obj.xcg0)

    #T = np.asarray(T, dtype=np.float64)
    #isp = np.asarray(isp, dtype=np.float64)
    #Deps = np.asarray(Deps, dtype=np.float64)
    #MomT = np.asarray(MomT, dtype=np.float64)

    eps = Deps + alfa
    g0 = obj.g0

    if h == 0:
        g = g0
    else:
        g = obj.g0 * (obj.Re / (obj.Re + h)) ** 2
    g = np.asarray(g, dtype=np.float64)

    dx = np.array((((T * np.cos(eps) - D) / m) - g * np.sin(gamma) + (obj.omega ** 2) * (obj.Re + h) * np.cos(lam) * \
                (np.cos(lam) * np.sin(gamma) - np.sin(lam) * np.cos(gamma) * np.sin(chi)),
                - np.cos(gamma) * np.cos(chi) * np.tan(lam) \
                * (v / (obj.Re + h)) + 2 * obj.omega * (np.cos(lam) * np.tan(gamma) * np.sin(chi) - np.sin(lam)) \
                - (obj.omega ** 2) * ((obj.Re + h) / (v * np.cos(gamma))) * np.cos(lam) * np.sin(lam) * np.cos(chi),
                ((T * np.sin(eps) + L) / (m * v)) - (g / v - v / (obj.Re + h)) * np.cos(gamma) + 2 * obj.omega \
                * np.cos(lam) * np.cos(chi) + (obj.omega ** 2) * ((obj.Re + h) / v) * np.cos(lam) * \
                (np.sin(lam) * np.sin(gamma) * np.sin(chi) + np.cos(lam) * np.cos(gamma)),
                -np.cos(gamma) * np.cos(chi) * (v / ((obj.Re + h) * np.cos(lam))),
                np.cos(gamma) * np.sin(chi) * (v / (obj.Re + h)),
                v * np.sin(gamma),
                -T / (g0 * isp)))
    return dx


'''set upper and lower bounds for states, controls and time, scaled'''
'''major issues with bounds!!!'''
bndX = ((0.001/unit_v, np.inf), (90/unit_chi, 270/unit_chi), (-np.inf, np.inf), (-np.inf, np.inf), (-obj.incl/unit_lam, obj.incl/unit_lam),
        (0.001/unit_h, 190000/unit_h), (obj.m10/unit_m, (obj.M0)/unit_m))
bndU = ((-2/unit_alfa, 40/unit_alfa), (0.01, 1.0), (-20/unit_deltaf, 30/unit_deltaf), (0.01, 1.0))
bndT = ((0.01, 0.3),)

bnds = (bndX)*Npoints + (bndU)*Npoints + (bndT)*(Npoints-1)
#print(bnds)
#for i in bnds:
 #   if i[1]-i[0]<0:
  #      print("error")
'''definition of initial and final conditions for states and controls for equality constraints'''

init_condX = np.array((1/unit_v, obj.chistart/unit_chi, obj.gammastart/unit_gamma, obj.longstart/unit_teta,
                       obj.latstart/unit_lam, 1/unit_h, obj.M0/unit_m))
init_condU = np.array((0.0, 1.0, 0.0, 0.0))
final_cond = np.array((obj.chi_fin/unit_chi, 0.0)) # final conditions on chi and gamma

'''set inequality constraints'''


def inequality(states, controls):
    '''this function takes states and controls unscaled and multiplies them for their units'''
    v = states[0]*states_unit[0]
    chi = states[1]*states_unit[1]  # it's not used for functions, can be used in degree
    gamma = states[2] * states_unit[2]  # it's not used for functions, can be used in degree
    teta = states[3] * states_unit[3]  # it's not used for functions, can be used in degree
    lam = states[4] * states_unit[4]  # it's not used for functions, can be used in degree
    h = states[5]*states_unit[5]
    m = states[6]*states_unit[6]
    alfa = np.deg2rad(controls[0]*controls_unit[0])
    delta = controls[1]*controls_unit[1]
    deltaf = np.deg2rad(controls[2]*controls_unit[2])
    tau = controls[3]*controls_unit[3]

    Press, rho, c = isa(h, obj.psl, obj.g0, obj.Re)
    #Press = np.asarray(Press, dtype=np.float64)
    #rho = np.asarray(rho, dtype=np.float64)
    #c = np.asarray(c, dtype=np.float64)
    M = v / c

    L, D, MomA = aeroForces(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10,
                                 obj.xcg0, obj.xcgf, obj.pref)

    #L = np.asarray(L, dtype=np.float64)
    #D = np.asarray(D, dtype=np.float64)
    #MomA = np.asarray(MomA, dtype=np.float64)

    T, Deps, isp, MomT = thrust(Press, m, presv, spimpv, delta, tau, obj.psl, obj.M0, obj.m10, obj.lRef,
                                     obj.xcgf, obj.xcg0)

    #T = np.asarray(T, dtype=np.float64)
    #isp = np.asarray(isp, dtype=np.float64)
    #Deps = np.asarray(Deps, dtype=np.float64)
    #MomT = np.asarray(MomT, dtype=np.float64)

    MomTot = MomA + MomT
    MomTotA = abs(MomTot)

    # dynamic pressure

    q = 0.5 * rho * (v ** 2)
    #if q > 40e3:
     #   print(q, rho, v, h)
    # accelerations
    ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
    az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

    # calculation of Hohmann transfer mass
    #r1 = h + obj.Re
    #Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    #Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    #mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))

    return [(obj.MaxAx-ax)/obj.MaxAx, (obj.MaxAz-az)/obj.MaxAz, (obj.MaxQ-q)/obj.MaxQ, (obj.k-MomTotA)/obj.k]


def ineqCond(var):
    '''this functions repeats the inequality conditions for every time step'''
    h = var[varStates - 2] * states_unit[5]
    m = var[varStates - 1] * states_unit[6]
    time = var[varTot:]

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / (np.exp(Dv1 / obj.gIsp) * np.exp(Dv2 / obj.gIsp))
    ineq_cond = np.zeros((0))
    for i in range(Npoints):
        ineq_cond = np.concatenate((ineq_cond, inequality(var[i*Nstates:(i+1)*Nstates], var[varStates+(i*Ncontrols):varStates+(i+1)*Ncontrols])))
    ineq_cond = np.concatenate((ineq_cond, ((mf-obj.m10)/unit_m,)))
    #ineq_cond = np.concatenate((ineq_cond, ((h-80000)/unit_h,)))
    #j=0
    #print(ineq_cond)

    #for i in ineq_cond:
     #   if i < 0:
      #      print("non compatible!", i, j)

       # j+=1
    return ineq_cond


'''set equality constraints'''


def propagation(var, dynamics):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    states_atNodes = np.zeros((varStates))  # this is the output of the states on the nodes.
    states_atNodes[0:Nstates] = init_condX  # The first node is initialized with the initial conditions scaled
    #plt.figure()

    Time = np.zeros((1))
    for i in range(Npoints-1):
        '''a vector of time deltas is created'''
        new = Time[-1]+var[i+varTot]*time_tot
        Time = np.hstack((Time, new))

    for i in range(Npoints-1):
        '''integration'''

        f = partial(dynamics, controls=var[varStates+(i*Ncontrols):varStates+((i+1)*Ncontrols)]*controls_unit)
        prop = integrate.solve_ivp(f, (Time[i], Time[i+1]), var[i*Nstates:(i+1)*Nstates]*states_unit,
                                   method='RK45')

        states_atNodes[((i+1)*Nstates):(Nstates*(i+2))] = prop.y[:, -1]/states_unit  # the states at nodes vector is filled with scaled values

     #     plt.plot(prop.t, prop.y[6, :])
      #  plt.axvline(prop.t[-1], color="k", alpha=0.5)


    #plt.show()

    return states_atNodes

def propagation2(var, dynamics):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    states_atNodes = np.zeros((varStates))  # this is the output of the states on the nodes.
    states_atNodes[0:Nstates] = init_condX  # The first node is initialized with the initial conditions scaled


    Time = np.zeros((1))
    for i in range(Npoints-1):
        '''a vector of time deltas is created'''
        new = Time[-1]+var[i+varTot]*time_tot
        Time = np.hstack((Time, new))

    for i in range(Npoints-1):
        '''integration'''

        f = partial(dynamics, controls=var[varStates+(i*Ncontrols):varStates+((i+1)*Ncontrols)]*controls_unit)
        prop = integrate.solve_ivp(f, (Time[i], Time[i+1]), var[i*Nstates:(i+1)*Nstates]*states_unit,
                                   method='RK45')

        states_atNodes[((i+1)*Nstates):(Nstates*(i+2))] = prop.y[:, -1]/states_unit  # the states at nodes vector is filled with scaled values
        plt.figure(0)
        plt.title("V")
        plt.plot(prop.t, prop.y[0, :])
        plt.axvline(prop.t[-1], color="k", alpha=0.5)

        plt.figure(1)
        plt.title("Chi")
        plt.plot(prop.t, prop.y[1, :])
        plt.axvline(prop.t[-1], color="k", alpha=0.5)

        plt.figure(2)
        plt.title("Gamma")
        plt.plot(prop.t, prop.y[2, :])
        plt.axvline(prop.t[-1], color="k", alpha=0.5)

        plt.figure(3)
        plt.title("Teta")
        plt.plot(prop.t, prop.y[3, :])
        plt.axvline(prop.t[-1], color="k", alpha=0.5)

        plt.figure(4)
        plt.title("Lambda")
        plt.plot(prop.t, prop.y[4, :])
        plt.axvline(prop.t[-1], color="k", alpha=0.5)

        plt.figure(5)
        plt.title("H")
        plt.plot(prop.t, prop.y[5, :])
        plt.axvline(prop.t[-1], color="k", alpha=0.5)

        plt.figure(6)
        plt.title("M")
        plt.plot(prop.t, prop.y[6, :])
        plt.axvline(prop.t[-1], color="k", alpha=0.5)




    plt.show()

    return states_atNodes


def equality(var):
    '''this functions applies the equality conditions, in the knotting points, plus final states
    conditions and controls initial conditions'''
    conj = propagation(var, dynamics)
    vt = np.sqrt(obj.GMe / (obj.Re + var[varStates-2]))
    #var[varStates - Nstates:varStates] = conj[varStates - Nstates:varStates] # the states at the last point are placed in the variables vector
    eq_cond = np.zeros((0))
    eq_cond = np.concatenate((eq_cond, var[0:varStates] - conj))
    #eq_cond = np.concatenate((eq_cond, var[varStates-6:varStates-4] - final_cond))
    #eq_cond = np.concatenate((eq_cond, var[varStates:varStates+Ncontrols] - init_condU))
    #eq_cond = np.concatenate((eq_cond, ((var[varStates-7] - vt)/unit_v,)))  # equality condition on orbital velocity for insertion

    #eq_cond = var[0:varStates]-conj
    #print(eq_cond)
    return eq_cond


cons = ({'type': 'eq',
           'fun': equality},
        {'type': 'ineq',
           'fun': ineqCond})
cons2 = {'type': 'eq',
           'fun': equality}

def cost_fun(var):
    '''this is the cost function of the problem, which is the propellant mass maximization'''
    h = var[varStates - 2] * states_unit[5]
    m = var[varStates-1]*states_unit[6]

    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / obj.gIsp)

    return 1 - mf / unit_m


def display_func(var):
    '''function to display data at the end of each iteration'''
    m = var[varStates-1]*states_unit[6]
    h = var[varStates-2]*states_unit[5]
    time_vec = np.zeros((1))
    for i in range(Npoints - 1):
        new = time_vec[-1] + var[i + varTot] * time_tot
        time_vec = np.hstack((time_vec, new))

    # Hohmann transfer mass calculation
    r1 = h + obj.Re
    Dv1 = np.sqrt(obj.GMe / r1) * (np.sqrt((2 * obj.r2) / (r1 + obj.r2)) - 1)
    Dv2 = np.sqrt(obj.GMe / obj.r2) * (1 - np.sqrt((2 * r1) / (r1 + obj.r2)))
    mf = m / np.exp((Dv1 + Dv2) / obj.gIsp)

    print("m before Ho : {0:.5f}".format(m))
    print("mf          : {0:.5f}".format(mf))
    print("altitude Hohmann starts: {0:.5f}".format(h))
    print("final time  : {}".format(time_vec))



'''NLP SOLVER'''

iterator = 0

while iterator < maxIterator:
    print("---- iteration : {0} ----".format(iterator + 1))
    #j = 0
    #print(bnds)

    #for i in X0[:varStates]:

      #  if i <= bnds[j][0] or i >= bnds[j][1]:
     #       print("non compatible!", i, j)

    #j += 1
    opt = optimize.minimize(cost_fun,
                            X0,
                            constraints=cons,
                            bounds=bnds,
                            method='SLSQP',
                            options={"ftol": ftol,
                                     "maxiter":maxiter,
                                     "disp":True})



    X0 = opt.x
    display_func(X0)
    if not (opt.status):
        break
    iterator += 1

res = propagation2(X0, dynamics)



'''post process - plot of results'''

v = np.zeros((0))
chi = np.zeros((0))
gamma = np.zeros((0))
teta = np.zeros((0))
lam = np.zeros((0))
h = np.zeros((0))
m = np.zeros((0))
alfa = np.zeros((0))
delta = np.zeros((0))
deltaf = np.zeros((0))
tau = np.zeros((0))
time = np.zeros((1))



for i in range(Npoints):
    v = np.hstack((v, X0[i*Nstates]*states_unit[0]))
    chi = np.hstack((chi, np.deg2rad(X0[i*Nstates+1]*states_unit[1])))
    gamma = np.hstack((gamma, np.deg2rad(X0[i*Nstates+2]*states_unit[2])))
    teta = np.hstack((teta, np.deg2rad(X0[i*Nstates+3]*states_unit[3])))
    lam = np.hstack((lam, np.deg2rad(X0[i*Nstates+4]*states_unit[4])))
    h = np.hstack((h, X0[i*Nstates+5]*states_unit[5]))
    m = np.hstack((m, X0[i*Nstates+6]*states_unit[6]))
    alfa = np.hstack((alfa, np.deg2rad(X0[varStates+i*Ncontrols]*controls_unit[0])))
    delta = np.hstack((delta, X0[varStates+1+i*Ncontrols]*controls_unit[1]))
    deltaf = np.hstack((deltaf, np.deg2rad(X0[varStates+2+i*Ncontrols]*controls_unit[2])))
    tau = np.hstack((tau, X0[varStates+3+i*Ncontrols]*controls_unit[3]))



for i in range(Npoints-1):
    new = time[-1]+X0[i+varTot]*time_tot
    time = np.hstack((time, new))


Press, rho, c = isaMulti(h, obj.psl, obj.g0, obj.Re)
Press = np.asarray(Press, dtype=np.float64)
rho = np.asarray(rho, dtype=np.float64)
c = np.asarray(c, dtype=np.float64)
M = v / c

L, D, MomA = aeroForcesMulti(M, alfa, deltaf, cd, cl, cm, v, obj.wingSurf, rho, obj.lRef, obj.M0, m, obj.m10, obj.xcg0, obj.xcgf, obj.pref, Npoints)
L = np.asarray(L, dtype=np.float64)
D = np.asarray(D, dtype=np.float64)
MomA = np.asarray(MomA, dtype=np.float64)

T, Deps, isp, MomT = thrustMulti(Press, m, presv, spimpv, delta, tau, Npoints, obj.psl, obj.M0, obj.m10, obj.lRef, obj.xcgf, obj.xcg0)
T = np.asarray(T, dtype=np.float64)
Deps = np.asarray(Deps, dtype=np.float64)
MomT = np.asarray(MomT, dtype=np.float64)

MomTot = MomA + MomT

g0 = obj.g0
eps = Deps + alfa
g  = []
for alt in h:
    if alt == 0:
        g.append(g0)
    else:
        g.append(obj.g0 * (obj.Re / (obj.Re + alt)) ** 2)
g = np.asarray(g, dtype=np.float64)
# dynamic pressure

q = 0.5 * rho * (v ** 2)

# accelerations

ax = (T * np.cos(Deps) - D * np.cos(alfa) + L * np.sin(alfa)) / m
az = (T * np.sin(Deps) + D * np.sin(alfa) + L * np.cos(alfa)) / m

downrange = -(v**2)/g * np.sin(2 * chi)

# ------------------------
# Visualization
flag_savefig = True
savefig_file = "Test_MultiShooting_"

plt.figure()
plt.title("Altitude profile")
plt.plot(time, h/1000, marker="o", label="Altitude")
#plt.plot(time, hr/1000, marker="o", label="AltitudeREs")
for line in time:
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
if flag_savefig:
    plt.savefig(savefig_file + "altitude" + ".png")

plt.figure()
plt.title("Velocity")
plt.plot(time, v, marker="o", label="V")
for line in time:
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
for line in time:
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
for line in time:
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
for line in time:
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
for line in time:
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
for line in time:
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
for line in time:
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
for line in time:
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.ylabel(" kPa ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "dynPress" + ".png")

plt.figure()
plt.title("Moment")
plt.plot(time, MomTot / 1e3, marker="o", label="Moment")
for line in time:
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
plt.grid()
plt.xlabel("km")
plt.ylabel(" km ")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "downrange" + ".png")

plt.figure()
plt.title("Mach profile")
plt.plot(time, M, marker="o", label="Mach")
for line in time:
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
for line in time:
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time [s]")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "LDT" + ".png")

plt.show()
