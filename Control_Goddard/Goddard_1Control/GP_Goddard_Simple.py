from scipy.integrate import solve_ivp
import numpy as np
import operator
import pygraphviz as pgv
import random
from deap import gp
import matplotlib.pyplot as plt
import sys
import timeit
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
from scipy.interpolate import PchipInterpolator
import matplotlib.animation as animation
from matplotlib import style
import datetime
from time import time
import _pickle as cPickle
import pickle

def TriAdd(x, y, z):
    return x + y + z


def Abs(x):
    return abs(x)


def Div(left, right):
    try:
        x = left / right
        return x
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
        return 0.0


def Mul(left, right):
    try:
        #np.seterr(invalid='raise')
        return left * right
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return min(left, right)


def Sqrt(x):
    try:
        if x > 0:
            return np.sqrt(x)
        else:
            return abs(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def Log(x):
    try:
        if x > 0:
            return np.log(x)
        else:
            return abs(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def Exp(x):
    try:
        return np.exp(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def Sin(x):
    try:
        return np.sin(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def Cos(x):
    try:
        return np.cos(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


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


def integration(t, xini, dyn):
    dt = t[1] - t[0]
    Nint = len(t)
    x = np.zeros((Nint, len(xini)))
    x[0, :] = xini
    for i in range(Nint - 1):
        # print(i, x[i,:])
        # print(u[i,:])
        k1 = dt * dyn(t[i], x[i, :])
        # print("k1: ", k1)
        k2 = dt * dyn(t[i] + dt / 2, x[i, :] + k1 / 2)
        # print("k2: ", k2)
        k3 = dt * dyn(t[i] + dt / 2, x[i, :] + k2 / 2)
        # print("k3: ", k3)
        k4 = dt * dyn(t[i + 1], x[i, :] + k3)
        # print("k4: ", k4)
        x[i + 1, :] = x[i, :] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x

start = timeit.default_timer()

###############################  S Y S T E M - P A R A M E T E R S  ####################################################

class Rocket:
    GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
    Re = 6371.0 * 1000  # Earth Radius [m]
    g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]

    def __init__(self):
        self.M0 = 5000  # Initial total mass [kg]
        self.Mc = 0.4  # Initial Propellant mass over total mass
        self.Cd = 0.2  # Drag Coefficient [-]
        self.area = 10  # area [m2]
        self.Isp = 300.0  # Isp [s]
        self.max_thrust = 2  # maximum thrust to initial weight ratio
        self.Tmax = self.max_thrust * self.M0 * self.g0

    def air_density(self, h):
        beta = 1/8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0*np.exp(-beta*h)

obj = Rocket()
Nstates = 3
Ncontrols = 1


mutpb = 0.3
cxpb = 0.6
change_time = 100
size_pop = 150 # Pop size
size_gen = 50  # Gen size
Mu = int(size_pop)
Lambda = int(size_pop * 1.4)

limit_height = 30  # Max height (complexity) of the controller law
limit_size = 400  # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()

tref = np.load("time.npy")
total_time_simulation = tref[-1]
del tref
flag_seed_populations = False
flag_offdesign = False
flag_seed_populations1 = False
################################# M A I N ###############################################


def main():
    global size_gen, size_pop, Mu, Lambda, flag_seed_populations, flag_offdesign, flag_seed_populations1
    global Rfun, Vfun, mfun, mutpb, cxpb
    global tfin, flag, pas, fitness_old1, fitness_old2, fitness_old3

    flag = False
    pas = False

    fitness_old1 = 1e5
    fitness_old2 = 1e5
    fitness_old3 = 1e5

    Rref = np.load("R.npy")
    Vref = np.load("V.npy")
    mref = np.load("m.npy")
    tref = np.load("time.npy")
    tfin = tref[-1]


    Rfun = PchipInterpolator(tref, Rref)
    Vfun = PchipInterpolator(tref, Vref)
    mfun = PchipInterpolator(tref, mref)

    del Rref, Vref, mref, tref

    pool = multiprocessing.Pool(nbCPU)
    if flag_offdesign is True:
        Mu = int(size_pop)
        Lambda = int(size_pop * 1.4)
        popx = toolx.population(n=size_pop)
        history.update(popx)
        toolx.register("map", pool.map)

    toolbox.register("map", pool.map)
    print("INITIAL POP SIZE: %d" % size_pop)

    print("GEN SIZE: %d" % size_gen)

    print("\n")

    random.seed()

    pop = toolbox.population(n=size_pop)
    history.update(pop)
    hof = tools.HallOfFame(size_gen) ### OLD ###

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height, size=stats_size)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

    if flag_offdesign is True:
        pop, log = algorithms.eaMuPlusLambda(popx, toolx, Mu, Lambda, mutpb, cxpb, size_gen, stats=mstats, halloffame=hof, verbose=True)  ### OLD ###
    else:
        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, mutpb, cxpb, size_gen, stats=mstats, halloffame=hof, verbose=True)
    ####################################################################################################################

    pool.close()
    pool.join()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################


def evaluate(individual):
    global flag, flag_offdesign
    global Rfun, Vfun, mfun, flag_t
    global tfin, penalty

    flag = False
    flag_t = False
    penalty = 0
    # Transform the tree expression in a callable function
    if flag_offdesign is True:
        fT = toolx.compile(expr=individual)
    else:
        fT = toolbox.compile(expr=individual)

    x_ini = [obj.Re, 0.0, obj.M0]  # initial conditions

    def sys(t, x):
        global flag, flag_t, penalty
        # State Variables
        R = x[0]
        V = x[1]
        m = x[2]

        if R < obj.Re or np.isnan(R):
            penalty += abs(R - obj.Re)*1.2
            R = obj.Re
            flag = True
        if np.isinf(R) or R > obj.Re+80e3:
            penalty += abs(R - obj.Re) * 0.2
            R = obj.Re + 80e3
            flag = True
        if m < obj.M0*obj.Mc or np.isnan(m):
            penalty += abs(m - obj.Mc*obj.M0) * 0.5
            m = obj.M0*obj.Mc
            flag = True
        elif m > obj.M0 or np.isinf(m):
            penalty += abs(m - obj.M0) * 0.5
            m = obj.M0
            flag = True
        if abs(V) > 1e3 or np.isinf(V):
            penalty += abs(V - 1e3) * 0.5
            if V > 0:
                V = 1e3
                flag = True
            else:
                V = -1e3
                flag = True


        r = Rfun(t)
        v = Vfun(t)
        mf = mfun(t)

        er = r - R
        ev = v - V
        em = mf - m

        rho = obj.air_density(R - obj.Re)

        drag = 0.5 * rho * V ** 2 * obj.Cd * obj.area
        g = obj.GMe / R ** 2
        g0 = obj.g0
        Isp = obj.Isp
        T = fT(er, ev, em)


        if abs(T) > obj.Tmax or np.isinf(T):
            penalty += abs(T - obj.Tmax)
            T = obj.Tmax
            flag_t = True
            flag = True

        elif T < 0.0 or np.isnan(T):
            penalty += abs(T)
            T = 0.0
            flag = True
            flag_t = True

        dxdt = np.array((V, (T - drag) / m - g, - T / g0 / Isp))
        return dxdt

    tin = 0.0
    teval = np.linspace(0, tfin, int(tfin*2))

    if flag_offdesign is True:
        x_ini = xnew_ini
        tin = change_time
        teval = t_evals2
    sol = solve_ivp(sys, [tin, tfin], x_ini, t_eval=teval)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y3 = sol.y[2, :]
    tt = sol.t
    if  tt[-1] != tfin:
        penalty += abs(tt[-1] - tfin)*1.2
        flag_t = True
        flag = True

    r = Rfun(tt)
    v = Vfun(tt)
    m = mfun(tt)

    err1 = (r - y1)/(obj.Re+50*1e3)
    err2 = (v - y2)/np.sqrt(obj.GMe/obj.Re)
    err3 = (m - y3)/obj.M0

    # STEP TIME SIZE
    i = 0
    pp = 1
    step = np.zeros(len(y1), dtype='float')
    step[0] = tt[1] - tt[0]
    while i < len(tt) - 1:
        step[pp] = tt[i + 1] - tt[i]
        i = i + 1
        pp = pp + 1
    stepdiff = np.diff(tt)
    #print(step-stepdiff)
    # INTEGRAL OF ABSOLUTE ERROR (PERFORMANCE INDEX)
    IAE = np.zeros((2, len(err1)))
    j = 0
    for a, b, c, n in zip(err1, err2, err3, step):
       IAE[0][j] = n * abs(a)
       IAE[1][j] = n * abs(b)
       #IAE[2][j] = n * abs(c)
       j = j + 1

    # PENALIZING INDIVIDUALs
    # For the stats if the multiprocessing is used, there could be problems to print the correct values (parallel process(?))
    fitness1 = sum(IAE[0])
    fitness2 = sum(IAE[1])
    #fitness3 = sum(IAE[2])
    if flag is True:
        if flag_t is True:
            x = [fitness1*penalty, fitness2*penalty]
        else:
            x = [fitness1*penalty, fitness2*penalty]
    else:
        fitness = [fitness1,
                   fitness2]
                   #fitness3]

    return x if flag is True else fitness


####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(Mul, 2)
pset.addPrimitive(TriAdd, 3)
pset.addPrimitive(Abs, 1)
#pset.addPrimitive(Div, 2)
pset.addPrimitive(Sqrt, 1)
pset.addPrimitive(Log, 1)
#pset.addPrimitive(Exp, 1)
pset.addPrimitive(Sin, 1)
pset.addPrimitive(Cos, 1)
pset.addTerminal(np.pi, "pi")
pset.addTerminal(np.e, name="nap")  # e Napier constant number
pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-100, 100), 4))
pset.addEphemeralConstant("rand102", lambda: round(random.uniform(-100, 100), 4))
pset.addEphemeralConstant("rand103", lambda: round(random.uniform(-100, 100), 4))
pset.addEphemeralConstant("rand104", lambda: round(random.uniform(-100, 100), 4))
pset.addEphemeralConstant("rand105", lambda: round(random.uniform(-100, 100), 4))
pset.renameArguments(ARG0='errR')
pset.renameArguments(ARG1='errV')
pset.renameArguments(ARG2='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))  # MINIMIZATION OF THE FITNESS FUNCTION

creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)   #### OLD ####

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####

toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####

toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evaluate)  ### OLD ###

toolbox.register("select", tools.selNSGA2)

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
    obj = Rocket()
    pop, log, hof = main()

Cd_old = 0.2


print("\n ADD SOME ALTERATIONS TO PHYSICAL COMPONENTS OF THE PLANT AT %.2f [s]" % change_time)
obj.Cd = float(input("CHANGE VALUE OF THE DRAG COEFFICIENT (ORIGINAL 0.2): "))

x_ini = [obj.Re, 0.0, obj.M0]  # initial conditions
Tplot = np.zeros((0))
tplot = np.zeros((0))
def sys2GP(t, x):
    global Cd_old, Tplot, tplot
    fT = toolbox.compile(hof[0])
    R = x[0]
    V = x[1]
    m = x[2]

    '''if R < 0 or np.isnan(R):
        R = obj.Re
    if np.isinf(R) or R > obj.Re + 80e3:
        R = obj.Re + 80e3
    if m < obj.M0 * obj.Mc or np.isnan(m):
        m = obj.M0 * obj.Mc
    elif m > obj.M0 or np.isinf(m):
        m = obj.M0
    if abs(V) > 1e3 or np.isinf(V):
        if V > 0:
            V = 1e3
        else:
            V = -1e3'''

    r = Rfun(t)
    v = Vfun(t)
    mf = mfun(t)

    er = r - R
    ev = v - V
    em = mf - m

    rho = obj.air_density(R - obj.Re)
    drag = 0.5 * rho * V ** 2 * Cd_old * obj.area

    g = obj.GMe / R ** 2
    g0 = obj.g0
    Isp = obj.Isp
    T = fT(er, ev, em)
    if abs(T) > obj.Tmax or np.isinf(T):
        T = obj.Tmax
    elif T < 0.0 or np.isnan(T):
        T = 0.0

    Tplot = np.hstack((Tplot, T))
    tplot = np.hstack((tplot, t))
    dxdt = np.array((V, (T-drag)/m-g, - T / g0 / Isp))

    return dxdt

passint = tfin
tevals = np.linspace(0.0, tfin, int(passint*2))

solgp = solve_ivp(sys2GP, [0.0, tfin], x_ini, t_eval=tevals)

rout = solgp.y[0, :]
vout = solgp.y[1, :]
mout = solgp.y[2, :]
ttgp = solgp.t

rR = Rfun(ttgp)
vR = Vfun(ttgp)
mR = mfun(ttgp)


plt.ion()
plt.figure(1)
plt.plot(ttgp, (rR - obj.Re) / 1e3, 'r--', label="SET POINT")
plt.axhline(0, 0, ttgp[-1], color='r')
animated_plot = plt.plot(ttgp, (rout - obj.Re) / 1e3, marker='.', color = 'k', label="ON DESIGN")[0]
plt.figure(2)
plt.plot(ttgp, vR, 'r--', label="SET POINT")
animated_plot2 = plt.plot(ttgp, vout, marker='.', color = 'k', label="ON DESIGN")[0]
plt.figure(3)
plt.plot(ttgp, mR, 'r--', label="SET POINT")
plt.axhline(obj.M0*obj.Mc, 0, ttgp[-1], color='r')
animated_plot3 = plt.plot(ttgp, mout, marker='.', color = 'k', label="ON DESIGN")[0]
plt.figure(4)
plt.axhline(obj.Tmax, 0, tplot[-1], color='r')
plt.axhline(0, 0, tplot[-1], color='r')
animated_plot4 = plt.plot(tplot, Tplot, marker='.', color = 'k', label="ON DESIGN")[0]



#######             GRAFICO PER TEMPO DI IN FASE DI DESIGN      #####
i = 0
for items in ttgp:
    if items > change_time:
        index, = np.where(ttgp == items)
        break
    plt.figure(1)
    animated_plot.set_xdata(ttgp[0:i])
    animated_plot.set_ydata((rout[0:i]-obj.Re)/1e3)
    plt.pause(0.00000001)

    plt.figure(2)
    animated_plot2.set_xdata(ttgp[0:i])
    animated_plot2.set_ydata(vout[0:i])
    plt.pause(0.00000001)

    plt.figure(3)
    animated_plot3.set_xdata(ttgp[0:i])
    animated_plot3.set_ydata(mout[0:i])
    plt.draw()
    plt.pause(0.00000001)
    i = i + 1
for i in range(len(tplot)):
    plt.figure(4)
    animated_plot4.set_xdata(tplot[0:i])
    animated_plot4.set_ydata(Tplot[0:i])
    plt.pause(0.00000001)
    plt.draw()


u_design = hof[0]
print(u_design)
output = open("hof_GoddardSimple.pkl", "wb")
cPickle.dump(hof, output, -1)
output.close()

objects = []
with (open("hof_GoddardSimple.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
#####################################################################################################################

start = time()
if __name__ == "__main__":
    obj = Rocket()
    def initPOP1():
        global objects
        return objects[0]

    toolx = base.Toolbox()
    toolx.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)  #### OLD ####

    toolx.register("individual", tools.initIterate, creator.Individual, toolx.expr)  #### OLD ####

    toolx.register("population", tools.initRepeat, list, toolx.individual)  #### OLD ####

    toolx.register("compile", gp.compile, pset=pset)

    toolx.register("evaluate", evaluate)  ### OLD ###

    toolx.register("select", tools.selNSGA2)

    toolx.register("mate", gp.cxOnePoint)
    toolx.register("mutate", gp.mutUniform, expr=toolx.expr, pset=pset)

    toolx.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolx.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

    toolx.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolx.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

    history = tools.History()
    toolx.decorate("mate", history.decorator)
    toolx.decorate("mutate", history.decorator)
    xnew_ini = [float(rout[index]), float(vout[index]), float(mout[index])]
    t_evals2 = np.linspace(change_time, tfin, int(tfin*2))
    flag_seed_populations = True
    flag_offdesign = True
    size_pop, size_gen, cxpb, mutpb = 100, 5, 0.6, 0.2
    pop, log, hof = main()
end = time()
t_offdesign = end - start  # CALCOLO TEMPO IMPIEGATO DAL GENETIC PROGRAMMING
flag_prop = False


#########################################################################################################################
Tplot = np.zeros((0))
tplot = np.zeros((0))
def sys2GP_c(t, x):
    global u_design, flag_prop, Rfun, Vfun, mfun, Tplot, tplot
    fT = toolbox.compile(u_design)
    R = x[0]
    V = x[1]
    m = x[2]


    r = Rfun(t)
    v = Vfun(t)
    mf = mfun(t)

    er = r - R
    ev = v - V
    em = mf - m

    T = fT(er, ev, em)
    rho = obj.air_density(R - obj.Re)
    drag = 0.5 * rho * V ** 2 * obj.Cd * obj.area
    g = obj.GMe / R ** 2
    g0 = obj.g0
    Isp = obj.Isp

    '''if flag_prop is True:
        T = 0.0'''
    if abs(T) > obj.Tmax or np.isinf(T):
        T = obj.Tmax
    elif T < 0.0 or np.isnan(T):
        T = 0.0
    if m <= obj.M0*obj.Mc:
        T = 0.0
        m = obj.M0*obj.Mc
    Tplot = np.hstack((Tplot, T))
    tplot = np.hstack((tplot, t))
    dxdt = np.array((V, (T - drag) / m - g, - T / g0 / Isp))

    return dxdt


passint_c = (change_time + t_offdesign - (change_time)) * 2
tevals_c = np.linspace(change_time, change_time + t_offdesign, int(passint_c))
xnew_ini = [float(rout[index]), float(vout[index]), float(mout[index])]                              ################Mi servono questi PENSARE

solgp_c = solve_ivp(sys2GP_c, [change_time, change_time + t_offdesign], xnew_ini, t_eval=tevals_c)

rout_c = solgp_c.y[0, :]
vout_c = solgp_c.y[1, :]
mout_c = solgp_c.y[2, :]
for i in range(len(mout_c)):
    if mout_c[i] < obj.M0*obj.Mc:
        mout_c[i] = obj.M0*obj.Mc
ttgp_c = solgp_c.t

for tempi in ttgp_c:
    if tempi > t_offdesign:
        index_c, = np.where(ttgp_c == tempi)

plt.ion()
plt.figure(1)
animated_plot_c = plt.plot(ttgp_c, (rout_c - obj.Re) / 1e3, marker='.', color = 'b', label="OFF DESIGN")[0]
plt.figure(2)
animated_plot_c2 = plt.plot(ttgp_c, vout_c, marker='.', color = 'b', label="OFF DESIGN")[0]
plt.figure(3)
animated_plot_c3 = plt.plot(ttgp_c, mout_c, marker='.', color = 'b',  label="OFF DESIGN")[0]
plt.figure(4)
animated_plot_c4 = plt.plot(tplot, Tplot, marker='.', color = 'b',  label="OFF DESIGN")[0]

for i in range(len(ttgp_c)):
    plt.figure(1)
    animated_plot_c.set_xdata(ttgp_c[0:i])
    animated_plot_c.set_ydata((rout_c[0:i]-obj.Re)/1e3)
    plt.pause(0.00000001)

    plt.figure(2)
    animated_plot_c2.set_xdata(ttgp_c[0:i])
    animated_plot_c2.set_ydata(vout_c[0:i])
    #plt.draw()
    plt.pause(0.00000001)

    plt.figure(3)
    animated_plot_c3.set_xdata(ttgp_c[0:i])
    animated_plot_c3.set_ydata(mout_c[0:i])
    plt.pause(0.00000001)
    plt.draw()
for i in range(len(tplot)):
    plt.figure(4)
    animated_plot_c4.set_xdata(tplot[0:i])
    animated_plot_c4.set_ydata(Tplot[0:i])
    plt.draw()
    plt.pause(0.00000001)

##################################################################################################################


# Simulazione per TEMPO CON NUOVA LEGGE creata dal GENETIC PROGRAMMING

passint_gp = (total_time_simulation - (change_time + t_offdesign)) * 2
tevals_gp = np.linspace(change_time + t_offdesign, total_time_simulation, int(passint_gp))
xnew_ini_gp = [float(rout_c[index_c]), float(vout_c[index_c]), float(mout_c[index_c])]
Tplot = np.zeros((0))
tplot = np.zeros((0))
def sys2GP_gp(t, x):
    global flag_prop, Rfun, Vfun, mfun, Tplot, tplot
    fT = toolx.compile(hof[0])
    R = x[0]
    V = x[1]
    m = x[2]

    '''if R < 0 or np.isnan(R):
        R = obj.Re
    if np.isinf(R) or R > obj.Re + 80e3:
        R = obj.Re + 80e3
    if m < obj.M0 * obj.Mc or np.isnan(m):
        m = obj.M0 * obj.Mc-1
    elif m > obj.M0 or np.isinf(m):
        m = obj.M0
    if abs(V) > 1e3 or np.isinf(V):
        if V > 0:
            V = 1e3
        else:
            V = -1e3'''

    r = Rfun(t)
    v = Vfun(t)
    mf = mfun(t)

    er = r - R
    ev = v - V
    em = mf - m

    T = fT(er, ev, em)
    rho = obj.air_density(R - obj.Re)
    drag = 0.5 * rho * V ** 2 * obj.Cd * obj.area
    g = obj.GMe / R ** 2
    g0 = obj.g0
    Isp = obj.Isp
    if abs(T) > obj.Tmax or np.isinf(T):
        T = obj.Tmax
    elif T < 0.0 or np.isnan(T):
        T = 0.0
    if m <= obj.M0 * obj.Mc:
        T = 0.0
        m = obj.M0 * obj.Mc
    Tplot = np.hstack((Tplot, T))
    tplot = np.hstack((tplot, t))
    dxdt= np.array((V, (T - drag) / m - g, - T / g0 / Isp))

    return dxdt


solgp_gp = solve_ivp(sys2GP_gp, [change_time + t_offdesign, total_time_simulation], xnew_ini_gp, t_eval=tevals_gp)

rout_gp = solgp_gp.y[0, :]
vout_gp = solgp_gp.y[1, :]
mout_gp = solgp_gp.y[2, :]
for i in range(len(mout_gp)):
    if mout_gp[i] < obj.M0*obj.Mc:
        mout_gp[i] = obj.M0*obj.Mc
ttgp_gp = solgp_gp.t


plt.ion()
plt.figure(1)
animated_plot_gp = plt.plot(ttgp_gp, (rout_gp - obj.Re) / 1e3, marker='.', color='g',  label="ONLINE CONTROL")[0]
plt.figure(2)
animated_plot_gp2 = plt.plot(ttgp_gp, vout_gp, marker='.', color='g',  label="ONLINE CONTROL")[0]
plt.figure(3)
animated_plot_gp3 = plt.plot(ttgp_gp, mout_gp, marker='.', color='g',  label="ONLINE CONTROL")[0]
plt.figure(4)
animated_plot_gp4 = plt.plot(tplot, Tplot, marker='.', color='g',  label="ONLINE CONTROL")[0]

for i in range(len(ttgp_gp)):
    plt.figure(1)
    animated_plot_gp.set_xdata(ttgp_gp[0:i])
    animated_plot_gp.set_ydata((rout_gp[0:i]-obj.Re)/1e3)
    plt.pause(0.00000001)

    plt.figure(2)
    animated_plot_gp2.set_xdata(ttgp_gp[0:i])
    animated_plot_gp2.set_ydata(vout_gp[0:i])
    plt.pause(0.00000001)

    plt.figure(3)
    animated_plot_gp3.set_xdata(ttgp_gp[0:i])
    animated_plot_gp3.set_ydata(mout_gp[0:i])
    plt.draw()
    plt.pause(0.00000001)

for i in range(len(tplot)):
    plt.figure(4)
    animated_plot_gp4.set_xdata(tplot[0:i])
    animated_plot_gp4.set_ydata(Tplot[0:i])
    plt.pause(0.00000001)
    plt.draw()

print("\n")
print(u_design)
print(hof[0])
plt.show(block=True)