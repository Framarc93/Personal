from scipy import optimize
import numpy as np
from models import *
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
import random
from functools import partial
from OpenGoddard.optimize import Guess

'''This script is the multiple shooting algorithm I have written applied on the open goddard SSTO rocket example'''
'''This multiple shoting alforithm has the controls defined in NContPoints points which are the optimization variable
then a linear interpolation is done among these points to obtain a continuous profile of the controls.
For the states, only the ones in the conjunction points are used as optimization variables and then a single shooting is done in each leg.
The inequality conditions are applied along all the trajectory.'''

class Rocket:

    def __init__(self):
        self.GMe = 3.986004418 * 10 ** 14  # Earth gravitational constant [m^3/s^2]
        self.Re = 6371.0 * 1000  # Earth Radius [m]
        self.Vr = np.sqrt(self.GMe / self.Re)  # m/s
        self.H0 = 10.0  # m
        self.V0 = 0.0
        self.M0 = 100000.0  # kg
        self.Mp = self.M0 * 0.99
        self.Cd = 0.6
        self.A = 4.0  # m2
        self.Isp = 300.0  # s
        self.g0 = 9.80665  # m/s2
        self.Tmax = self.M0 * self.g0 * 1.5
        self.MaxQ = 14000.0  # Pa
        self.MaxG = 8.0  # G
        self.Htarget = 400.0 * 1000  # m
        self.Rtarget = self.Re + self.Htarget  # m/s
        self.Vtarget = np.sqrt(self.GMe / self.Rtarget)  # m/s

    @staticmethod
    def air_density(h):
        beta = 1/8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0*np.exp(-beta*h)

obj = Rocket()


'''set problem parameters'''
savefig_file = "MultiTest_GoddardEx_"
time_tot = 200  # s
Nbar = 6
Nleg = Nbar - 1  # number of multiple shooting points
NContPoints = 5  # number of control points for interpolation inside each interval
Nint = 150  # number of points for each single shooting integration
NineqCond = Nint  # number of points of which check the inequality conditions


maxiter = 50 # max number of iterations for nlp solver
ftol = 1e-8  # numeric tolerance of nlp solver
eps = 1e-10
Nstates = 5  # number of states
Ncontrols = 2  # number of controls
maxIterator = 30  # max number of optimization iterations
varStates = Nstates * Nleg
varControls = Ncontrols * Nleg * NContPoints # Ncontrols * ((Nbar - 1) * NContPoints - Nbar + 2)  # total number of states variables

varTot = varStates + varControls  # total number of variables, controls and states
tnew = np.linspace(0, time_tot, Nbar)

tcontr = np.linspace(0, time_tot, int(varControls / Ncontrols))

'''values for reduction to unit'''

unit_R = obj.Re
unit_theta = 1
unit_V = np.sqrt(obj.GMe / obj.Re)
unit_m = obj.M0
unit_t = unit_R / unit_V
unit_T = unit_m * unit_R / unit_t ** 2
states_unit = np.array((unit_R, unit_theta, unit_V, unit_V, unit_m))
controls_unit = np.array((unit_T, unit_T))
# ========================
# Initial parameter guess

# altitude profile
R_init = Guess.cubic(tnew, obj.Re, 0.0, obj.Rtarget, 0.0)

theta_init = Guess.cubic(tnew, 0.0, 0.0, np.deg2rad(25.0), 0.0)

# velocity
Vr_init = Guess.linear(tnew, 0.0, 0.0)
Vt_init = Guess.linear(tnew, 0.0, obj.Vtarget)

# mass profile
M_init = Guess.cubic(tnew, obj.M0, -0.6, obj.M0-obj.Mp, 0.0)

# thrust profile
Tr_init = Guess.cubic(tcontr, obj.Tmax/2, 0.0, 0.0, 0.0)
Tt_init = Guess.cubic(tcontr, obj.Tmax/2, 0.0, 0.0, 0.0)

XGuess = np.array((R_init/ unit_R, theta_init/ unit_theta, Vr_init/ unit_V, Vt_init/ unit_V, M_init/ unit_m))  # states initial guesses

UGuess = np.array((Tr_init/ unit_T, Tt_init/ unit_T)) # states initial guesses

X = np.zeros((0))
U = np.zeros((0))

for i in range(Nleg):
    '''creation of vector of states initial guesses'''
    for j in range(Nstates):
        X = np.hstack((X, XGuess[j][i]))

for i in range(int(varControls/Ncontrols)):
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

def dynamics(t, states, Tr_int, Tt_int):
    R     = states[0]
    theta = states[1]
    Vr    = states[2]
    Vt    = states[3]
    m     = states[4]
    Tr    = Tr_int(t)
    Tt    = Tt_int(t)

    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R)**2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dx = np.array((Vr,
                   Vt / R,
                   Tr / m - Dr / m - g + Vt**2 / R,
                   Tt / m - Dt / m - (Vr * Vt) / R,
                   - np.sqrt(Tr**2 + Tt**2) / g0 / Isp))

    return dx


'''set upper and lower bounds for states, controls and time, scaled'''
'''major issues with bounds!!!'''
bndX = ((obj.Re/unit_R, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), ((obj.M0-obj.Mp)/unit_m, obj.M0/unit_m))
bndU = ((0.0, obj.Tmax/unit_T), (0.0, obj.Tmax/unit_T))
bndT = ((0.1, 1.0),)
bnds = (bndX) * Nleg + (bndU) * Nleg * NContPoints + (bndT) * Nleg

init_condX = np.array((obj.Re/unit_R, 0.0, 0.0, 0.0, obj.M0/unit_m))

final_cond = np.array((obj.Rtarget/unit_R, 0.0, obj.Vtarget/unit_V)) # final conditions on chi and gamma


def inequality(states, controls):
    R     = states[0]
    theta = states[1]
    Vr    = states[2]
    Vt    = states[3]
    m     = states[4]
    Tr    = controls[0]
    Tt    = controls[1]


    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) \
        * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R)**2  # [m/s2]

    # dynamic pressure
    q = 0.5 * rho * (Vr**2 + Vt**2)  # [Pa]
    # accelaration
    a_r = (Tr - Dr) / m
    a_t = (Tt - Dt) / m
    a_mag = np.sqrt(a_r**2 + a_t**2)  # [m/s2]
    # Thrust
    T = np.sqrt(Tr**2 + Tt**2)

    return np.hstack(((obj.MaxG * obj.g0-a_mag)/(obj.MaxG * obj.g0), (obj.Tmax-T)/obj.Tmax))


def ineqCond(var):

    ineq_cond = MultiShootIneq(var, dynamics)

    return ineq_cond


'''set equality constraints'''


def SingleShooting(states, controls, dyn, tstart, tfin, Nint):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    '''states and controls must be given with real values!!! not scaled!!!'''
    '''needed a fixed step integrator'''
    '''in this function the states are not scaled anymore'''
    Time = np.linspace(tstart, tfin, NContPoints)
    tres = np.zeros((0))

    Rres = np.zeros((0))
    tetares = np.zeros((0))
    Vrres = np.zeros((0))
    Vtres = np.zeros((0))
    mres = np.zeros((0))

    x = np.zeros((Nint, Nstates))

    x[0, :] = states[0:Nstates] * states_unit  # vector of intial states ready

    # now interpolation of controls

    Tr_interp = interpolate.PchipInterpolator(Time, controls[0, :]*unit_T)
    Tt_interp = interpolate.PchipInterpolator(Time, controls[1, :]*unit_T)


    time_new = np.linspace(tstart, tfin, Nint)
    dt = (time_new[1] - time_new[0])

    t = time_new

    for i in range(Nint - 1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt*dyn(t[i], x[i, :], Tr_interp, Tt_interp)
        # print("k1: ", k1)
        k2 = dt*dyn(t[i] + dt / 2, x[i, :] + k1 / 2, Tr_interp, Tt_interp)
        # print("k2: ", k2)
        k3 = dt*dyn(t[i] + dt / 2, x[i, :] + k2 / 2, Tr_interp, Tt_interp)
        # print("k3: ", k3)
        k4 = dt*dyn(t[i + 1], x[i, :] + k3, Tr_interp, Tt_interp)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # the output of the integration is in degrees?


    Rres = x[:, 0]
    tetares = x[:, 1]
    Vrres = x[:, 2]
    Vtres = x[:, 3]
    mres = x[:, 4]

    Trres = Tr_interp(time_new)
    Ttres = Tt_interp(time_new)

    return Rres, tetares, Vrres, Vtres, mres, time_new, Trres, Ttres, Tr_interp, Tt_interp


def MultiShooting(var, dynamics):
    '''in this function the states and controls are scaled'''
    Tr = np.zeros((NContPoints))
    Tt = np.zeros((NContPoints))

    states_atNode = np.zeros((0))
    tres = np.zeros((0))

    Rres = np.zeros((0))
    tetares = np.zeros((0))
    Vrres = np.zeros((0))
    Vtres = np.zeros((0))
    mres = np.zeros((0))

    time = np.zeros((1))
    timestart = 0
    for i in range(Nleg):
        controls = np.zeros((NContPoints,))
        # print("Multiple shooting Leg: ", i)
        states = var[i * Nstates:(i + 1) * Nstates]

        if i == 0:
            timestart = 0.0
        timeend = timestart + var[i + varTot] * time_tot

        for k in range(NContPoints):
            Tr[k] = var[varStates + i * (Ncontrols * NContPoints) + Ncontrols * k]
            Tt[k] = var[varStates + i * (Ncontrols * NContPoints) + 1 + Ncontrols * k]

        controls = np.vstack((Tr, Tt))


        Rres, tetares, Vrres, Vtres, mres, tres, Trres, Ttres, Tr_interp, Tt_interp = SingleShooting(states, controls, dynamics, timestart, timeend, Nint)

        states_atNode = np.concatenate((states_atNode, (
                    (Rres[-1], tetares[-1], Vrres[-1], Vtres[-1], mres[-1]) / states_unit)))
        timestart = timeend
    return states_atNode


def MultiShootIneq(var, dynamics):
    '''in this function the states and controls are scaled'''
    '''in this function the states and controls are scaled'''
    Tr = np.zeros((NContPoints))
    Tt = np.zeros((NContPoints))

    states_atNode = np.zeros((0))
    tres = np.zeros((0))

    Rres = np.zeros((0))
    tetares = np.zeros((0))
    Vrres = np.zeros((0))
    Vtres = np.zeros((0))
    mres = np.zeros((0))

    time = np.zeros((1))
    timestart = 0
    for i in range(Nleg):
        controls = np.zeros((NContPoints,))
        # print("Multiple shooting Leg: ", i)
        states = var[i * Nstates:(i + 1) * Nstates]

        if i == 0:
            timestart = 0.0
        timeend = timestart + var[i + varTot] * time_tot

        for k in range(NContPoints):
            Tr[k] = var[varStates + i * (Ncontrols * NContPoints) + Ncontrols * k]
            Tt[k] = var[varStates + i * (Ncontrols * NContPoints) + 1 + Ncontrols * k]

        controls = np.vstack((Tr, Tt))

        Rres, tetares, Vrres, Vtres, mres, tres, Trres, Ttres, Tr_interp, Tt_interp = SingleShooting(states, controls, dynamics, timestart,
                                                                                timeend, Nint)
        states_after = np.vstack((Rres, tetares, Vrres, Vtres, mres))
        controls_after = np.vstack((Trres, Ttres))

        ineq_cond = inequality(states_after, controls_after)
        timestart = timeend
        '''the values output of this function are not scaled'''

    return ineq_cond


def MultiPlot(var, dynamics):
    Tr = np.zeros((NContPoints))
    Tt = np.zeros((NContPoints))


    tres = np.zeros((0))
    Rres = np.zeros((0))
    tetares = np.zeros((0))
    Vrres = np.zeros((0))
    Vtres = np.zeros((0))
    mres = np.zeros((0))
    time = np.zeros((1))
    TrContPoint = np.zeros((Nleg, NContPoints))
    TtContPoint = np.zeros((Nleg, NContPoints))

    for i in range(Nleg):
        Tr = np.zeros((NContPoints))
        Tt = np.zeros((NContPoints))

        states = var[i * Nstates:(i + 1) * Nstates]
        if i==0:
            timestart = 0
        timeend = timestart + var[i + varTot] * time_tot
        time = np.concatenate((time, (timeend,)))
        tC = np.linspace(timestart, timeend, NContPoints)

        for k in range(NContPoints):
            Tr[k] = var[varStates + i * (Ncontrols * NContPoints) + Ncontrols * k]
            TrContPoint[i,k] = Tr[k] * unit_T
            Tt[k] = var[varStates + i * (Ncontrols * NContPoints) + 1 + Ncontrols * k]
            TtContPoint[i,k] = Tt[k] * unit_T

        controls = np.vstack((Tr, Tt))

        Rres, tetares, Vrres, Vtres, mres, tres, Trres, Ttres , Trinterp, Ttinterp= SingleShooting(
            states, controls, dynamics, timestart, timeend, Nint)

        rho = obj.air_density(Rres - obj.Re)
        Dr = 0.5 * rho * Vrres * np.sqrt(Vrres ** 2 + Vtres ** 2) \
             * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vtres * np.sqrt(Vrres ** 2 + Vtres ** 2) \
             * obj.Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / Rres) ** 2  # [m/s2]

        # dynamic pressure
        q = 0.5 * rho * (Vrres ** 2 + Vtres ** 2)  # [Pa]
        # accelaration
        a_r = (Trres - Dr) / mres
        a_t = (Ttres - Dt) / mres
        a_mag = np.sqrt(a_r ** 2 + a_t ** 2)  # [m/s2]
        # Thrust
        T = np.sqrt(Trres ** 2 + Ttres ** 2)
        timestart = timeend
        plt.figure(0)
        plt.title("R")
        plt.plot(tres, (Rres - obj.Re)/1000)
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "altitude" + ".png")

        plt.figure(1)
        plt.title("teta")
        plt.plot(tres, tetares)
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "theta" + ".png")

        plt.figure(2)
        plt.title("V")
        plt.plot(tres, Vrres, color='r')
        plt.plot(tres, Vtres, color='b')
        plt.legend(['Vr', 'Vt'], loc="best")
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "V" + ".png")


        plt.figure(3)
        plt.title("M")
        plt.plot(tres, mres)
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "M" + ".png")

        plt.figure(4)
        plt.title("Tr")
        plt.plot(tres, Trres)
        plt.plot(tC, TrContPoint[i, :], 'ro')
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "Tr" + ".png")

        plt.figure(5)
        plt.title("Tt")
        plt.plot(tres, Ttres)
        plt.plot(tC, TtContPoint[i, :], 'ro')
        plt.axvline(tres[-1], color="k", alpha=0.5)
        plt.savefig(savefig_file + "Tt" + ".png")

        plt.figure(6)
        plt.title("Acceleration")

        plt.plot(tres, a_r, color='b')
        plt.plot(tres, a_t, color="g")
        plt.plot(tres, a_mag, color="r")
        plt.grid()
        plt.xlabel("time [s]")
        plt.ylabel("Acceleration [m/s2]")
        plt.legend(["Acc r", "Acc t", "Acc"], loc="best")
        plt.savefig(savefig_file + "acceleration" + ".png")


        plt.figure(7)
        plt.title("Thrust profile")

        plt.plot(tres, Trres / 1000, color="r")
        plt.plot(tres, Ttres / 1000, color="b")
        plt.plot(tres, T / 1000, color="g")
        plt.plot(tres, Dr / 1000, color="k")
        plt.plot(tres, Dt / 1000, color="c")
        plt.plot(tres, mres * g / 1000, color="m")

        plt.grid()
        plt.xlabel("time [s]")
        plt.ylabel("Thrust [kN]")
        plt.legend(["Tr", "Tt", "Thrust", "Dr", "Dt", "Gravity"], loc="best")
        plt.savefig(savefig_file + "force" + ".png")


    plt.show()


def equality(var):
    '''this functions applies the equality conditions, in the knotting points, plus final states
    conditions and controls initial conditions'''
    conj = MultiShooting(var, dynamics)

    eq_cond = np.zeros((0))
    eq_cond = np.concatenate((eq_cond, var[0:Nstates] - init_condX))
    eq_cond = np.concatenate((eq_cond, var[Nstates:varStates] - conj[:Nstates*(Nbar-2)]))
    eq_cond = np.concatenate((eq_cond, (var[varStates - 5] - final_cond[0],)))
    eq_cond = np.concatenate((eq_cond, (var[varStates - 3] - final_cond[1],)))
    eq_cond = np.concatenate((eq_cond, (var[varStates - 2] - final_cond[2],)))
    return eq_cond


cons = ({'type': 'eq',
         'fun': equality},
        {'type': 'ineq',
         'fun': ineqCond})
cons2 = {'type': 'eq',
         'fun': equality}


def cost_fun(var):
    m = var[varStates-1]*unit_m
    # return -m[-1]
    # ==== Caution ====
    # cost function should be near 1.0
    return -m / unit_m


def display_func(var):
    R = var[varStates-5]*unit_R
    m = var[varStates-1]*unit_m
    tf = np.zeros((1))
    for i in range(Nleg):
        new = tf[-1] + var[i + varTot] * time_tot
        tf = np.hstack((tf, new))

    print("mf          : {0:.5f}".format(m))
    print("max altitude: {0:.5f}".format(R))
    print("final time  : {}".format(tf))


'''NLP SOLVER'''

iterator = 0

while iterator < maxIterator:
    print("---- iteration : {0} ----".format(iterator + 1))
    # j = 0
    # print(bnds)

    # for i in X0[:varStates]:

    #  if i <= bnds[j][0] or i >= bnds[j][1]:
    #       print("non compatible!", i, j)

    # j += 1
    opt = optimize.minimize(cost_fun,
                            X0,
                            constraints=cons,
                            bounds=bnds,
                            method='SLSQP',
                            options={"ftol": ftol,
                                     "maxiter": maxiter,
                                     "disp": True})

    X0 = opt.x

    display_func(X0)
    if not (opt.status):
        break
    iterator += 1

MultiPlot(X0, dynamics)



