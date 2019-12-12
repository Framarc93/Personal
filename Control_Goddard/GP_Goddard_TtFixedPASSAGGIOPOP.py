from scipy.integrate import solve_ivp
import _pickle as cPickle
import pickle
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


def Div(left, right):
    try:
        x = left / right
        return x
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
        return 0.0

def TriAdd(x, y, z):
    return x + y + z

def Abs(x):
    return abs(x)

def Mul(left, right):
    try:
        #np.seterr(invalid='raise')
        return left * right
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return left


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


start = timeit.default_timer()

###############################  S Y S T E M - P A R A M E T E R S  ####################################################

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
        global flag
        beta = 1 / 8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        try:
            return rho0 * np.exp(-beta * h)
        except RuntimeWarning:
            flag = True
            return rho0 * np.exp(-beta * obj.Rtarget)
Nstates = 5
Ncontrols = 2
obj = Rocket()

mutpb = 0.4
cxpb = 0.5
change_time = 200
size_pop = 100 # Pop size
size_gen = 20  # Gen size
Mu = int(size_pop)
Lambda = int(size_pop * 1.5)

limit_height = 17  # Max height (complexity) of the controller law
limit_size = 400  # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()

tref = np.load("timeT.npy")
total_time_simulation = tref[-1]
del tref
flag_seed_populations = False
flag_offdesign = False
flag_seed_populations1 = False
flag = False
flagDeath = False
pas = False
fitness_old1 = 1e5
fitness_old2 = 1e5
fitness_old5 = 1e5
################################# M A I N ###############################################


def main():
    global flag_seed_populations, flag_offdesign, flag_seed_populations1
    global tfin, flag, pas, fitness_old1, fitness_old2, fitness_old3, fitness_old5
    global size_gen, size_pop, Mu, Lambda, mutpb, cxpb
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Ttfun
    global tfin, g1, g2, g3, flag, pas, flagDeath

    flag = False
    pas = False
    flagDeath = False

    fitness_old1 = 1e5
    fitness_old2 = 1e5
    fitness_old5 = 1e5

    Rref = np.load("RT.npy")
    Thetaref = np.load("ThetaT.npy")
    Vrref = np.load("VrT.npy")
    Vtref = np.load("VtT.npy")
    mref = np.load("mT.npy")
    tref = np.load("timeT.npy")
    Ttref = np.load("TtT.npy")
    tfin = tref[-1]

    Rfun = PchipInterpolator(tref, Rref)
    Thetafun = PchipInterpolator(tref, Thetaref)
    Vrfun = PchipInterpolator(tref, Vrref)
    Vtfun = PchipInterpolator(tref, Vtref)
    mfun = PchipInterpolator(tref, mref)
    Ttfun = PchipInterpolator(tref, Ttref)

    del Rref, Thetaref, Vrref, Vtref, mref, Ttref, tref

    pool = multiprocessing.Pool(nbCPU)


    if flag_offdesign is True:
        toolx.register("map", pool.map)
        pop = toolx.population()
        hof = tools.HallOfFame(size_gen)
    else:
        toolbox.register("map", pool.map)
        pop = toolbox.population(n=size_pop)
        hof = tools.HallOfFame(size_gen)

    print("INITIAL POP SIZE: %d" % size_pop)

    print("GEN SIZE: %d" % size_gen)

    print("\n")

    random.seed()

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height, size=stats_size)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

    if flag_offdesign is True:
        pop, log = algorithms.eaMuPlusLambda(pop, toolx, Mu, Lambda, mutpb, cxpb, size_gen, stats=mstats, halloffame=hof, verbose=True)  ### OLD ###
    else:
        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, mutpb, cxpb, size_gen, stats=mstats, halloffame=hof, verbose=True)
    ####################################################################################################################

    stop = timeit.default_timer()
    total_time = stop - start

    pool.close()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################


def evaluate(individual):
    global flag, flag_offdesign, flagDeath
    global pas
    global fitnnesoldvalue, fitness_old1, fitness_old2, fitness_old5
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Ttfun
    global tfin, g1, g2, g3

    flagDeath = False
    flag = False
    pas = False

    # Transform the tree expression in a callable function
    if flag_offdesign is True:
        fTr = toolx.compile(expr=individual)
    else:
        fTr = toolbox.compile(expr=individual)

    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

    def sys(t, x):

        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if np.isnan(theta) or np.isinf(theta):
            np.nan_to_num(theta)

        if R < obj.Re or np.isnan(R):
            R = obj.Re
            flag = True
        if R > obj.Rtarget + 1e3 or np.isinf(R):
            R = obj.Rtarget
            flag = True
        if m < obj.M0 - obj.Mp or np.isnan(m):
            m = obj.M0 - obj.Mp
            flag = True
        elif m > obj.M0 or np.isinf(m):
            m = obj.M0
            flag = True
        if abs(Vr) > 1e4 or np.isinf(Vr):
            if Vr > 0:
                Vr = 1e4
                flag = True
            else:
                Vr = -1e4
                flag = True
        if abs(Vt) > 1e4 or np.isinf(Vt):
            if Vt > 0:
                Vt = 1e4
                flag = True
            else:
                Vt = -1e4
                flag = True

        r = Rfun(t)
        th = Thetafun(t)
        vr = Vrfun(t)
        vt = Vtfun(t)
        mf = mfun(t)
        Tt = Ttfun(t)

        er = r - R
        et = th - theta
        evr = vr - Vr
        evt = vt - Vt
        em = mf - m
        dxdt = np.zeros(Nstates)
        # print("Ft: ", fTt(er, et, evr, evt, em), obj.Tmax)
        # print("Fr: ", fTr(er, et, evr, evt, em))

        rho = obj.air_density(R - obj.Re)
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        Tr = fTr(er, et, evr, evt, em)

        if abs(fTr(er, et, evr, evt, em)) > obj.Tmax or np.isinf(fTr(er, et, evr, evt, em)):
            Tr = obj.Tmax
            flag = True

        elif fTr(er, et, evr, evt, em) < 0.0 or np.isnan(fTr(er, et, evr, evt, em)):
            Tr = 0.0
            flag = True

        dxdt[0] = Vr
        dxdt[1] = Vt / R
        dxdt[2] = Tr / m - Dr / m - g + Vt ** 2 / R
        dxdt[3] = Tt / m - Dt / m - (Vr * Vt) / R
        dxdt[4] = -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp
        return dxdt

    tin = 0.0
    if flag_offdesign is True:
        x_ini = xnew_ini
        tin = change_time
    sol = solve_ivp(sys, [tin, tfin], x_ini)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y5 = sol.y[4, :]
    tt = sol.t

    if sol.t[-1] != tfin:
        flagDeath = True

    pp = 0
    r = np.zeros(len(tt), dtype='float')
    theta = np.zeros(len(tt), dtype='float')
    vr = np.zeros(len(tt), dtype='float')
    vt = np.zeros(len(tt), dtype='float')
    m = np.zeros(len(tt), dtype='float')
    for i in tt:
        r[pp] = Rfun(i)
        theta[pp] = Thetafun(i)
        vr[pp] = Vrfun(i)
        vt[pp] = Vtfun(i)
        m[pp] = mfun(i)
        pp += 1

    err1 = (r - y1) / obj.Htarget
    err2 = np.rad2deg(theta - y2) / 60
    # err3 = vr - y3
    # err4 = (vt - y4)/obj.Vtarget
    err5 = (m - y5) / obj.M0

    # STEP TIME SIZE
    i = 0
    pp = 1
    step = np.zeros(len(y1), dtype='float')
    step[0] = tt[1] - tt[0]
    while i < len(tt) - 1:
        step[pp] = tt[i + 1] - tt[i]
        i = i + 1
        pp = pp + 1

    # INTEGRAL OF ABSOLUTE ERROR (PERFORMANCE INDEX)
    IAE = np.zeros((3, len(err1)))
    j = 0
    for a, b, c, n in zip(err2, err1, err5, step):
        IAE[0][j] = n * abs(a)
        IAE[1][j] = n * abs(b)
        IAE[2][j] = n * abs(c)  # + alpha * abs(m))
        j = j + 1

    if flagDeath is True:
        y = [1e5, 1e5, 1e5]
        return y

    elif flag is True:
        x = [np.random.uniform(fitness_old1 * 1.5, fitness_old1 * 1.6),
             np.random.uniform(fitness_old2 * 1.5, fitness_old2 * 1.6),
             np.random.uniform(fitness_old5 * 1.5, fitness_old5 * 1.6)]
        return x

    else:
        fitness1 = sum(IAE[0])
        fitness2 = sum(IAE[1])
        fitness5 = sum(IAE[2])
        if fitness1 < fitness_old1:
            fitness_old1 = fitness1
        if fitness2 < fitness_old2:
            fitness_old2 = fitness2
        if fitness5 < fitness_old5:
            fitness_old5 = fitness5
        fitness = [fitness1,
                   fitness2,
                   fitness5]
        return fitness



####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("MAIN", 5)
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
pset.renameArguments(ARG1='errTheta')
pset.renameArguments(ARG2='errVr')
pset.renameArguments(ARG3='errVt')
pset.renameArguments(ARG4='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0, -0.5, -0.5))  # MINIMIZATION OF THE FITNESS FUNCTION

creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=5)   #### OLD ####

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

Cd_old = obj.Cd # original = 0.6


print("\n ADD SOME ALTERATIONS TO PHYSICAL COMPONENTS OF THE PLANT AT %.2f [s]" % change_time)
print("WRITE THE NUMBER FOR THE PARAMETER THAT YOU WANT CHANGE: Cd ( 1 )")
flag = int(input())
if flag == 1:
    obj.Cd = float(input("CHANGE VALUE OF THE DRAG COEFFICIENT: "))


x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions


def sys2GP(t, x):
    global Cd_old
    fTr = toolbox.compile(hof[0])
    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)
    mf = mfun(t)
    Tt = Ttfun(t)

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt
    em = mf - m
    dxdt = np.zeros(Nstates)

    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) \
         * Cd_old * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) \
         * Cd_old * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt[0] = Vr
    dxdt[1] = Vt / R
    dxdt[2] = fTr(er, et, evr, evt, em) / m - Dr / m - g + Vt ** 2 / R
    dxdt[3] = Tt / m - Dt / m - (Vr * Vt) / R
    dxdt[4] = - np.sqrt(Tt ** 2 + fTr(er, et, evr, evt, em) ** 2) / g0 / Isp

    return dxdt

passint = tfin*3
tevals = np.linspace(0.0, tfin, int(passint))

solgp = solve_ivp(sys2GP, [0.0, tfin], x_ini, t_eval=tevals)
rout = solgp.y[0, :]
thetaout = solgp.y[1, :]
vrout = solgp.y[2, :]
vtout = solgp.y[3, :]
mout = solgp.y[4, :]
ttgp = solgp.t
rR = np.zeros(len(ttgp), dtype='float')
tR = np.zeros(len(ttgp), dtype='float')
vrR = np.zeros(len(ttgp), dtype='float')
vtR = np.zeros(len(ttgp), dtype='float')
mR = np.zeros(len(ttgp), dtype='float')
TtR = np.zeros(len(ttgp), dtype='float')

ii = 0
for i in ttgp:
    rR[ii] = Rfun(i)
    tR[ii] = Thetafun(i)
    vrR[ii] = Vrfun(i)
    vtR[ii] = Vtfun(i)
    mR[ii] = mfun(i)
    TtR[ii] = Ttfun(i)
    ii = ii + 1


plt.ion()
plt.figure(1)
plt.plot(ttgp, (rR - obj.Re) / 1e3, 'r--', label="SET POINT")
animated_plot = plt.plot(ttgp, (rout - obj.Re) / 1e3, 'ro', label="ON DESIGN")[0]
plt.figure(2)
plt.plot(ttgp, vtR, 'r--', label="SET POINT")
animated_plot2 = plt.plot(ttgp, vtout, 'ro', label="ON DESIGN")[0]

'''fig6, ax6 = plt.subplots()
ax6.set_xlabel("time [s]")
ax6.set_ylabel("mass [kg]")
plt.plot(tgp, mout, label="GENETIC PROGRAMMING")
plt.plot(tgp, mR, 'r--', label="SET POINT")
plt.legend(loc="lower right")
plt.savefig('mass plot.png')
plt.show()'''

#######             GRAFICO PER TEMPO DI IN FASE DI DESIGN      #####
i = 0
for items in ttgp:
    plt.figure(1)
    #plt.ylim(bottom_end_stop - 1, top_end_stop + 1)
    #plt.xlim(0, total_time_simulation)


    if items > change_time:
        index, = np.where(ttgp == items)
        break
    animated_plot.set_xdata(ttgp[0:i])
    animated_plot.set_ydata((rout[0:i]-obj.Re)/1e3)
    plt.draw()
    #plt.pause(0.1)
    plt.pause(0.00000001)
    plt.figure(2)

    animated_plot2.set_xdata(ttgp[0:i])
    animated_plot2.set_ydata(vtout[0:i])
    plt.draw()
    # plt.pause(0.1)
    plt.pause(0.00000001)
    i = i + 1

u_design = hof[0]
print("\n ON DESIGN CONTROLLER")
print(u_design)

output = open("hof_10.pkl", "wb")
cPickle.dump(hof, output, -1)
output.close()

objects = []
with (open("hof_10.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break



#####################################################################################################################

start = time()
flag_gpnew = True  # POSSO METTERE I PARAMETRI CHE MI PIACCIONO SELEZIONANDOLI CON UN FLAG
if __name__ == "__main__":

    def initPOP1():
        global objects
        return objects[0]

    toolx = base.Toolbox()

    toolx.register("population", tools.initIterate, list, initPOP1)

    toolx.register("compile", gp.compile, pset=pset)

    toolx.register("evaluate", evaluate)  ### OLD ###

    toolx.register("select", tools.selNSGA2)

    toolx.register("mate", gp.cxOnePoint)
    toolx.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

    toolx.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolx.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

    toolx.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolx.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

    xnew_ini = [float(rout[index]), float(thetaout[index]), float(vrout[index]), float(vtout[index]), float(mout[index])]
    flag_seed_populations = True
    flag_offdesign = True
    size_pop, size_gen, cxpb, mutpb = size_gen, 5, 0.6, 0.35
    Mu = int(size_pop)
    Lambda = int(size_pop * 1.4)
    pop, log, hof = main()
end = time()
t_offdesign = end - start  # CALCOLO TEMPO IMPIEGATO DAL GENETIC PROGRAMMING



#########################################################################################################################
print("\n NEW INTELLIGENT CONTROL LAW:")
print(hof[0])
print("\n")

def sys2GP_c(t, x):
    global u_design
    fTr = toolbox.compile(u_design)
    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)
    mf = mfun(t)
    Tt = Ttfun(t)

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt
    em = mf - m
    dxdt = np.zeros(Nstates)

    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) \
         * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) \
         * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt[0] = Vr
    dxdt[1] = Vt / R
    dxdt[2] = fTr(er, et, evr, evt, em) / m - Dr / m - g + Vt ** 2 / R
    dxdt[3] = Tt / m - Dt / m - (Vr * Vt) / R
    dxdt[4] = - np.sqrt(Tt ** 2 + fTr(er, et, evr, evt, em) ** 2) / g0 / Isp

    return dxdt


passint_c = (change_time + t_offdesign - (change_time)) * 3
tevals_c = np.linspace(change_time, change_time + t_offdesign, int(passint_c))
xnew_ini = [float(rout[index]), float(thetaout[index]), float(vrout[index]), float(vtout[index]), float(mout[index])]

solgp_c = solve_ivp(sys2GP_c, [change_time, change_time + t_offdesign], xnew_ini, t_eval=tevals_c)

rout_c = solgp_c.y[0, :]
thetaout_c = solgp_c.y[1, :]
vrout_c = solgp_c.y[2, :]
vtout_c = solgp_c.y[3, :]
mout_c = solgp_c.y[4, :]
ttgp_c = solgp_c.t

rrr_c = np.zeros(len(ttgp_c), dtype='float')
ii = 0

errgp_c = rrr_c - rout_c

for tempi in ttgp_c:
    if tempi > t_offdesign:
        index_c, = np.where(ttgp_c == tempi)

plt.ion()
plt.figure(1)
animated_plot_c = plt.plot(ttgp_c, (rout_c - obj.Re) / 1e3, 'bo', label="OFF DESIGN")[0]
plt.figure(2)
animated_plot_c2 = plt.plot(ttgp_c, vtout_c, 'bo', label="OFF DESIGN")[0]

i = 0
for items in ttgp_c:
    plt.figure(1)
    animated_plot_c.set_xdata(ttgp_c[0:i])
    animated_plot_c.set_ydata((rout_c[0:i]-obj.Re)/1e3)
    #plt.draw()
    #plt.pause(0.1)
    plt.pause(0.00000001)
    plt.figure(2)
    animated_plot_c2.set_xdata(ttgp_c[0:i])
    animated_plot_c2.set_ydata(vtout_c[0:i])
    plt.draw()
    # plt.pause(0.1)
    plt.pause(0.00000001)
    i = i + 1

##################################################################################################################


# Simulazione per TEMPO CON NUOVA LEGGE creata dal GENETIC PROGRAMMING

passint_gp = (total_time_simulation - (change_time + t_offdesign)) * 3
tevals_gp = np.linspace(change_time + t_offdesign, total_time_simulation, int(passint_gp))
xnew_ini_gp = [float(rout_c[index_c]), float(thetaout_c[index_c]), float(vrout_c[index_c]), float(vtout_c[index_c]), float(mout_c[index_c])]

def sys2GP_gp(t, x):
    fTr = toolx.compile(hof[0])
    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)
    mf = mfun(t)
    Tt = Ttfun(t)

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt
    em = mf - m
    dxdt = np.zeros(Nstates)

    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) \
         * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) \
         * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt[0] = Vr
    dxdt[1] = Vt / R
    dxdt[2] = fTr(er, et, evr, evt, em) / m - Dr / m - g + Vt ** 2 / R
    dxdt[3] = Tt / m - Dt / m - (Vr * Vt) / R
    dxdt[4] = - np.sqrt(Tt ** 2 + fTr(er, et, evr, evt, em) ** 2) / g0 / Isp

    return dxdt


solgp_gp = solve_ivp(sys2GP_gp, [change_time + t_offdesign, total_time_simulation], xnew_ini_gp, t_eval=tevals_gp)

rout_gp = solgp_gp.y[0, :]
thetaout_gp = solgp_gp.y[1, :]
vrout_gp = solgp_gp.y[2, :]
vtout_gp = solgp_gp.y[3, :]
mout_gp = solgp_gp.y[4, :]
ttgp_gp = solgp_gp.t

plt.ion()
plt.figure(1)
animated_plot_gp = plt.plot(ttgp_gp, (rout_gp - obj.Re) / 1e3, 'go', label="ONLINE CONTROL")[0]
plt.figure(2)
animated_plot_gp2 = plt.plot(ttgp_gp, vtout_gp, 'go', label="ONLINE CONTROL")[0]

i = 0
for items in ttgp_gp:
    plt.figure(1)
    animated_plot_gp.set_xdata(ttgp_gp[0:i])
    animated_plot_gp.set_ydata((rout_gp[0:i]-obj.Re)/1e3)
    #if items==change_time+t_offdesign:
     #   plt.legend(loc='best')
    #plt.draw()
    plt.pause(0.00000001)

    plt.figure(2)
    animated_plot_gp2.set_xdata(ttgp_gp[0:i])
    animated_plot_gp2.set_ydata(vtout_gp[0:i])
    #if items==change_time+t_offdesign:
     #   plt.legend(loc='best')
    plt.draw()
    plt.pause(0.00000001)

    i = i + 1


plt.show(block=True)