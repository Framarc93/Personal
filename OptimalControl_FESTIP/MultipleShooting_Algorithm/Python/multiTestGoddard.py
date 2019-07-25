from scipy import optimize
import numpy as np
from models import *
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
import random
from functools import partial
from OpenGoddard.optimize import Guess

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

time_tot = 200  # s
Npoints = 15  # number of multiple shooting points

maxiter = 10  # max number of iterations for nlp solver
ftol = 1e-8 # numeric tolerance of nlp solver
eps = 1e-12
Nstates = 5  # number of states
Ncontrols = 2  # number of controls
maxIterator = 5 # max number of optimization iterations
varStates = Nstates*Npoints  # total number of states variables
varTot = (Nstates+Ncontrols)*Npoints  # total number of variables, controls and states
tnew = np.linspace(0, time_tot, Npoints)

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
Tr_init = Guess.cubic(tnew, obj.Tmax/2, 0.0, 0.0, 0.0)
Tt_init = Guess.cubic(tnew, obj.Tmax/2, 0.0, 0.0, 0.0)

XGuess = np.array((R_init/ unit_R, theta_init/ unit_theta, Vr_init/ unit_V, Vt_init/ unit_V, M_init/ unit_m))  # states initial guesses

UGuess = np.array((Tr_init/ unit_T, Tt_init/ unit_T)) # states initial guesses
X = np.zeros((0))
U = np.zeros((0))
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

def dynamics(t, states, controls):
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
bnds = (bndX)*Npoints + (bndU)*Npoints + (bndT)*(Npoints-1)

init_condX = np.array((obj.Re/unit_R, 0.0, 0.0, 0.0, obj.M0/unit_m))

final_cond = np.array((obj.Rtarget/unit_R, 0.0, obj.Vtarget/unit_V)) # final conditions on chi and gamma

'''set inequality constraints'''

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

        #plt.plot(prop.t, prop.y[6, :])
        #plt.axvline(prop.t[-1], color="k", alpha=0.5)


    #plt.show()

    return states_atNodes

def propagation2(var, dynamics):
    '''this function integrates the dynamics equation over time. It takes as input the vector of variables and the dynamics equation set'''
    states_atNodes = np.zeros((varStates))  # this is the output of the states on the nodes.
    states_atNodes[0:Nstates] = init_condX  # The first node is initialized with the initial conditions scaled
    plt.figure()

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
        plt.plot(prop.t, prop.y[0, :])
        plt.figure(1)
        plt.plot(prop.t, prop.y[1, :])
        plt.figure(2)
        plt.plot(prop.t, prop.y[2, :])
        plt.figure(3)
        plt.plot(prop.t, prop.y[3, :])
        plt.figure(4)
        plt.plot(prop.t, prop.y[4, :])



        plt.axvline(prop.t[-1], color="k", alpha=0.5)


    plt.show()

    return states_atNodes

def equality(var):
    '''this functions applies the equality conditions, in the knotting points, plus final states
        conditions and controls initial conditions'''
    conj = propagation(var, dynamics)


    eq_cond = np.zeros((0))
    eq_cond = np.concatenate((eq_cond, var[0:varStates] - conj))
    # eq_cond = np.concatenate((eq_cond, (var[0:varStates]-conj)))
    eq_cond = np.concatenate((eq_cond, (var[varStates - 5] - final_cond[0],)))
    eq_cond = np.concatenate((eq_cond, (var[varStates - 3] - final_cond[1],)))
    eq_cond = np.concatenate((eq_cond, (var[varStates - 2] - final_cond[2],)))


    return eq_cond


def inequality(states, controls):
    R     = states[0]*unit_R
    theta = states[1]*unit_theta
    Vr    = states[2]*unit_V
    Vt    = states[3]*unit_V
    m     = states[4]*unit_m
    Tr    = controls[0]*unit_T
    Tt    = controls[1]*unit_T


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

    return [(obj.MaxG * obj.g0-a_mag)/(obj.MaxG * obj.g0), (obj.Tmax-T)/obj.Tmax]

def ineqCond(var):
    '''this functions repeats the inequality conditions for every time step'''

    ineq_cond = np.zeros((0))
    for i in range(Npoints):
        ineq_cond = np.concatenate((ineq_cond, inequality(var[i*Nstates:(i+1)*Nstates], var[varStates+(i*Ncontrols):varStates+(i+1)*Ncontrols])))
    return ineq_cond

cons = ({'type': 'eq',
           'fun': equality},
        {'type': 'ineq',
           'fun': ineqCond})

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
    for i in range(Npoints - 1):
        new = tf[-1] + var[i + varTot] * time_tot
        tf = np.hstack((tf, new))

    print("mf          : {0:.5f}".format(m))
    print("max altitude: {0:.5f}".format(R))
    print("final time  : {}".format(tf))


# NLP solver
res=propagation2(X0, dynamics)
iterator = 0

while iterator < maxIterator:
    print("---- iteration : {0} ----".format(iterator + 1))

    opt = optimize.minimize(cost_fun,
                            X0,
                            constraints=cons,
                            bounds=bnds,
                            method='SLSQP',
                            options={"disp": True,
                                     "maxiter": maxiter,
                                     "ftol": ftol})


    X0 = opt.x
    display_func(X0)
    if not (opt.status):
        break
    iterator += 1

res = propagation2(X0, dynamics)

'''post process - plot of results'''

R = np.zeros((0))
theta = np.zeros((0))
Vr = np.zeros((0))
Vt = np.zeros((0))
m = np.zeros((0))
Tr = np.zeros((0))
Tt = np.zeros((0))
time = np.zeros((1))
for i in range(Npoints):
    R = np.hstack((R, X0[i*Nstates]*states_unit[0]))
    theta = np.hstack((theta, np.deg2rad(X0[i*Nstates+1]*states_unit[1])))
    Vr = np.hstack((Vr, X0[i*Nstates+2]*states_unit[2]))
    Vt = np.hstack((Vt, X0[i*Nstates+3]*states_unit[3]))
    m = np.hstack((m, X0[i*Nstates+4]*states_unit[4]))
    Tr = np.hstack((Tr, X0[varStates+i*Ncontrols]*controls_unit[0]))
    Tt = np.hstack((Tt, X0[varStates+1+i*Ncontrols]*controls_unit[1]))

for i in range(Npoints-1):
    new = time[-1]+X0[i+varTot]*time_tot
    time = np.hstack((time, new))



# ------------------------
# Calculate necessary variables
rho = obj.air_density(R - obj.Re)
Dr  = 0.5 * rho * Vr * np.sqrt(Vr**2 + Vt**2) \
    * obj.Cd * obj.A  # [N]
Dt  = 0.5 * rho * Vt * np.sqrt(Vr**2 + Vt**2) \
    * obj.Cd * obj.A  # [N]
g   = obj.g0 * (obj.Re / R)**2  # [m/s2]

# dynamic pressure
q   = 0.5 * rho * (Vr**2 + Vt**2)  # [Pa]
# accelaration
a_r = (Tr - Dr) / m
a_t = (Tt - Dt) / m
a_mag = np.sqrt(a_r**2 + a_t**2)  # [m/s2]
# Thrust
T   = np.sqrt(Tr**2 + Tt**2)

# ------------------------
# Visualizetion
plt.figure()
plt.title("Altitude profile")
plt.plot(time, (R - obj.Re)/1000, marker="o", label="Altitude")

plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc="best")

plt.figure()
plt.title("Velocity")
plt.plot(time, Vr, marker="o", label="Vr")
plt.plot(time, Vt, marker="o", label="Vt")

plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend(loc="best")


plt.figure()
plt.title("Mass")
plt.plot(time, m, marker="o", label="Mass")

plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc="best")

plt.figure()
plt.title("Acceleration")
plt.plot(time, a_r, marker="o", label="Acc r")
plt.plot(time, a_t, marker="o", label="Acc t")
plt.plot(time, a_mag, marker="o", label="Acc")

plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Acceleration [m/s2]")
plt.legend(loc="best")

plt.figure()
plt.title("Thrust profile")
plt.plot(time, Tr / 1000, marker="o", label="Tr")
plt.plot(time, Tt / 1000, marker="o", label="Tt")
plt.plot(time, T / 1000, marker="o", label="Thrust")
plt.plot(time, Dr / 1000, marker="o", label="Dr")
plt.plot(time, Dt / 1000, marker="o", label="Dt")
plt.plot(time, m * g / 1000, marker="o", label="Gravity")

plt.grid()
plt.xlabel("time [s]")
plt.ylabel("Thrust [kN]")
plt.legend(loc="best")


plt.show()