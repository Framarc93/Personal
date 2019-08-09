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
from functools import partial


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


def xmate(ind1, ind2):
    i1 = random.randrange(len(ind1))
    i2 = random.randrange(len(ind2))
    ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    return ind1, ind2


def xmut(ind, expr, strp):

    i1 = random.randrange(len(ind))
    i2 = random.randrange(len(ind[i1]))
    choice = random.random()
    if choice < strp:
        indx = gp.mutUniform(ind[i1], expr, pset=pset)
        ind[i1] = indx[0]
        return ind,
    else:
        '''this part execute the mutation on a random constant'''
        indx = gp.mutEphemeral(ind[i1], "one")
        ind[i1] = indx[0]
        return ind,


# Direct copy from tools - modified for individuals with GP trees in an array
def xselDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first):
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            lind1 = sum([len(gpt) for gpt in ind1])
            lind2 = sum([len(gpt) for gpt in ind2])
            if lind1 > lind2:
                ind1, ind2 = ind2, ind1
            elif lind1 == lind2:
                # random selection in case of a tie
                prob = 0.5

            # Since size1 <= size2 then ind1 is selected
            # with a probability prob
            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=operator.attrgetter("fitness")))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)


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

mutpb = 0.25
cxpb = 0.55
change_time = 200
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
    global flag_seed_populations, flag_offdesign, flag_seed_populations1
    global tfin, flag
    global size_gen, size_pop, Mu, Lambda, mutpb, cxpb
    global Rfun, Thetafun, Vrfun, Vtfun, mfun

    flag = False

    Rref = np.load("R.npy")
    Thetaref = np.load("Theta.npy")
    Vrref = np.load("Vr.npy")
    Vtref = np.load("Vt.npy")
    mref = np.load("m.npy")
    tref = np.load("time.npy")
    tfin = tref[-1]

    Rfun = PchipInterpolator(tref, Rref)
    Thetafun = PchipInterpolator(tref, Thetaref)
    Vrfun = PchipInterpolator(tref, Vrref)
    Vtfun = PchipInterpolator(tref, Vtref)
    mfun = PchipInterpolator(tref, mref)

    del Rref, Thetaref, Vrref, Vtref, mref, tref

    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)
    toolx.register("map", pool.map)

    print("INITIAL POP SIZE: %d" % size_pop)

    print("GEN SIZE: %d" % size_gen)

    print("\n")

    random.seed()

    pop = toolbox.population(n=size_pop)
    popx = toolx.population(n=size_pop)
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
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################


def evaluate(individual):
    global flag, flag_offdesign
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun
    global tfin, t_eval2

    flag = False

    # Transform the tree expression in a callable function
    if flag_offdesign is True:
        fTr = toolx.compile(expr=individual[0])
        fTt = toolx.compile(expr=individual[1])
    else:
        fTr = toolbox.compile(expr=individual[0])
        fTt = toolbox.compile(expr=individual[1])

    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

    def sys(t, x):
        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if np.isnan(theta) or np.isinf(theta):
            theta = np.nan_to_num(theta)

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

        er = r - R
        et = th - theta
        evr = vr - Vr
        evt = vt - Vt
        em = mf - m

        rho = obj.air_density(R - obj.Re)
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        Tr = fTr(er, et, evr, evt, em)
        Tt = fTt(er, et, evr, evt, em)

        if fTr(er, et, evr, evt, em) < 0.0 or np.isnan(fTr(er, et, evr, evt, em)):
            Tr = 0.0
            flag = True

        elif fTr(er, et, evr, evt, em) > obj.Tmax or np.isinf(fTr(er, et, evr, evt, em)):
            Tr = obj.Tmax
            flag = True

        if fTt(er, et, evr, evt, em) < 0.0 or np.isnan(fTt(er, et, evr, evt, em)):
            Tt = 0.0
            flag = True

        elif fTt(er, et, evr, evt, em) > obj.Tmax or np.isinf(fTt(er, et, evr, evt, em)):
            Tt = obj.Tmax
            flag = True

        dxdt = np. array((Vr,
                          Vt / R,
                          Tr / m - Dr / m - g + Vt ** 2 / R,
                          Tt / m - Dt / m - (Vr * Vt) / R,
                          -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))
        return dxdt

    tin = 0.0
    teval = np.linspace(0.0, tfin, int(tfin*3))
    if flag_offdesign is True:
        x_ini = xnew_ini
        tin = change_time
        teval = t_eval2

    sol = solve_ivp(sys, [tin, tfin], x_ini, t_eval=teval, dense_output=True)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y5 = sol.y[4, :]
    tt = sol.t

    if sol.t[-1] != tfin:
        flag = True

    r = Rfun(tt)
    theta = Thetafun(tt)
    vr = Vrfun(tt)
    vt = Vtfun(tt)
    m = mfun(tt)

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


    if flag is True:
        x = [1e5, 1e5, 1e5]

    else:
        fitness1 = sum(IAE[0])
        fitness2 = sum(IAE[1])
        fitness5 = sum(IAE[2])
        fitness = [fitness1,
                   fitness2,
                   fitness5]

    return x if flag is True else fitness



####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2, name="Mul")
pset.addPrimitive(TriAdd, 3)
#pset.addPrimitive(operator.truediv, 2, name="Div")
#pset.addPrimitive(operator.pow, 2, name="Pow")
#pset.addPrimitive(Mul, 2)
pset.addPrimitive(Abs, 1)
#pset.addPrimitive(Div, 2)                      #rallentamento per gli ndarray utilizzati
pset.addPrimitive(Sqrt, 1)
pset.addPrimitive(Log, 1)
#pset.addPrimitive(Exp, 1)
pset.addPrimitive(Sin, 1)
pset.addPrimitive(Cos, 1)
pset.addTerminal(np.pi, "pi")
pset.addTerminal(np.e, name="nap")                   #e Napier constant number
#pset.addTerminal(2)
pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-100, 100), 4))
pset.addEphemeralConstant("rand102", lambda: round(random.uniform(-100, 100), 4))
pset.addEphemeralConstant("rand103", lambda: round(random.uniform(-100, 100), 4))
pset.addEphemeralConstant("rand104", lambda: round(random.uniform(-100, 100), 4))
pset.addEphemeralConstant("rand105", lambda: round(random.uniform(-100, 100), 4))
#pset.addADF(pset)
pset.renameArguments(ARG0='errR')
pset.renameArguments(ARG1='errTheta')
pset.renameArguments(ARG2='errVr')
pset.renameArguments(ARG3='errVt')
pset.renameArguments(ARG4='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -0.8))    # MINIMIZATION OF THE FITNESS FUNCTION

creator.create("Individual", list, fitness=creator.Fitness, height=1)

creator.create("SubIndividual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
# toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)   #### OLD ####
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, type_=pset.ret, min_=2, max_=5)  ### NEW ###

toolbox.register("leg", tools.initIterate, creator.SubIndividual, toolbox.expr)  ### NEW ###
toolbox.register("legs", tools.initRepeat, list, toolbox.leg, n=Ncontrols)  ### NEW ###

#toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)  ### NEW ###

# toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###

#toolbox.register("lambdify", gp.compile, pset=pset) ### NEW ###
#toolbox.register("stringify", gp.compile, pset=pset) ### NEW ###

toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evaluate) ### OLD ###
#toolbox.register('evaluate', evaluate, toolbox=toolbox, sourceData=data, minTrades=minTrades, log=False) ###NEW ###

# toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1, fitness_first=True) ### OLD ###
toolbox.register("select", xselDoubleTournament, fitness_size=6, parsimony_size=1.4, fitness_first=True) ### NEW ###

toolbox.register("mate", xmate) ### NEW ###
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=2, max_=5) ### NEW ###
toolbox.register("mutate", xmut, expr=toolbox.expr_mut, strp=0.7) ### NEW ###

# toolbox.register("mate", gp.cxOnePointLeafBiased,termpb=0.1) ### OLD ###
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) ### OLD ###

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

toolbox.decorate("mate", gp.staticLimit(key=len , max_value=limit_size))
#toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

toolx = base.Toolbox()

toolx.register("expr", gp.genHalfAndHalf, pset=pset, type_=pset.ret, min_=2, max_=5)  ### NEW ###

toolx.register("leg", tools.initIterate, creator.SubIndividual, toolx.expr)  ### NEW ###
toolx.register("legs", tools.initRepeat, list, toolbox.leg, n=Ncontrols)  ### NEW ###

toolx.register("individual", tools.initIterate, creator.Individual, toolx.legs)  ### NEW ###

toolx.register("population", tools.initRepeat, list, toolx.individual)  ### NEW ###

toolx.register("compile", gp.compile, pset=pset)

toolx.register("evaluate", evaluate) ### OLD ###
#toolbox.register('evaluate', evaluate, toolbox=toolbox, sourceData=data, minTrades=minTrades, log=False) ###NEW ###

toolx.register("select", xselDoubleTournament, fitness_size=6, parsimony_size=1.4, fitness_first=True) ### NEW ###

toolx.register("mate", xmate) ### NEW ###
toolx.register("expr_mut", gp.genHalfAndHalf, min_=2, max_=5) ### NEW ###
toolx.register("mutate", xmut, expr=toolx.expr_mut, strp=0.4) ### NEW ###

toolx.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

toolx.decorate("mate", gp.staticLimit(key=len , max_value=limit_size))
#toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

########################################################################################################################


if __name__ == "__main__":
    obj = Rocket()
    pop, log, hof = main()

Cd_old = obj.Cd  # original = 0.6

print("\n ADD SOME ALTERATIONS TO PHYSICAL COMPONENTS OF THE PLANT AT %.2f [s]" % change_time)
obj.Cd = float(input("CHANGE VALUE OF THE DRAG COEFFICIENT (ORIGINAL 0.6): "))

x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

def sys2GP(t, x):
    global Cd_old
    fTr = toolbox.compile(hof[0][0])
    fTt = toolbox.compile(hof[0][1])
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

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt
    em = mf - m

    Tt = fTt(er, et, evr, evt, em)
    Tr = fTr(er, et, evr, evt, em)
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd_old * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd_old * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt


tevals = np.linspace(0.0, tfin, int(tfin*4))

solgp = solve_ivp(sys2GP, [0.0, tfin], x_ini, t_eval=tevals, dense_output=True)
print(solgp.message)
rout = solgp.y[0, :]
thetaout = solgp.y[1, :]
vrout = solgp.y[2, :]
vtout = solgp.y[3, :]
mout = solgp.y[4, :]
ttgp = solgp.t

if ttgp[-1] != tfin:
    print("first integration stopped prematurly")

rR = Rfun(tevals)
tR = Thetafun(tevals)
vrR = Vrfun(tevals)
vtR = Vtfun(tevals)
mR = mfun(tevals)

plt.ion()
plt.figure(1)
plt.plot(tevals, (rR - obj.Re) / 1e3, 'r--', label="SET POINT")
animated_plot = plt.plot(ttgp, (rout - obj.Re) / 1e3, marker='.', color='k', label="ON DESIGN")[0]
plt.figure(2)
plt.plot(tevals, vtR, 'r--', label="SET POINT")
animated_plot2 = plt.plot(ttgp, vtout, marker='.', color='k', label="ON DESIGN")[0]
plt.figure(3)
plt.plot(tevals, mR, 'r--', label="SET POINT")
plt.axhline(obj.M0-obj.Mp, 0, ttgp[-1], color='r')
animated_plot3 = plt.plot(ttgp, mout, marker='.', color='k', label="ON DESIGN")[0]
plt.figure(4)
plt.plot(tevals, vrR, 'r--', label="SET POINT")
animated_plot4 = plt.plot(ttgp, vrout, marker='.', color='k', label="ON DESIGN")[0]


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
    animated_plot2.set_ydata(vtout[0:i])
    plt.pause(0.00000001)

    plt.figure(3)
    animated_plot3.set_xdata(ttgp[0:i])
    animated_plot3.set_ydata(mout[0:i])
    plt.pause(0.00000001)

    plt.figure(4)
    animated_plot4.set_xdata(ttgp[0:i])
    animated_plot4.set_ydata(vrout[0:i])
    plt.draw()
    plt.pause(0.00000001)

    i = i + 1

Tr_old = hof[0][0]
Tt_old = hof[0][1]
print(Tr_old)
print(Tt_old)
#####################################################################################################################

start = time()
if __name__ == "__main__":
    obj = Rocket()
    xnew_ini = [float(rout[index]), float(thetaout[index]), float(vrout[index]), float(vtout[index]), float(mout[index])]
    t_eval2 = np.linspace(change_time, tfin, int(tfin*3))
    flag_seed_populations = True
    flag_offdesign = True
    flag_prop = False
    size_pop, size_gen, cxpb, mutpb = 100, 30, 0.55, 0.25
    Mu = int(size_pop)
    Lambda = int(size_pop * 1.4)
    pop, log, hof = main()
end = time()
t_offdesign = end - start  # CALCOLO TEMPO IMPIEGATO DAL GENETIC PROGRAMMING


#######################################################################################################################

def sys2GP_c(t, x):
    global Tr_old, Tt_old
    fTr = toolbox.compile(Tr_old)
    fTt = toolbox.compile(Tt_old)
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

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt
    em = mf - m
    Tt = fTt(er, et, evr, evt, em)
    Tr = fTr(er, et, evr, evt, em)
    if m <= obj.M0-obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp

    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt


passint_c = (change_time + t_offdesign - (change_time)) * 3
tevals_c = np.linspace(change_time, change_time + t_offdesign, 500)
xnew_ini = [float(rout[index]), float(thetaout[index]), float(vrout[index]), float(vtout[index]), float(mout[index])]

solgp_c = solve_ivp(sys2GP_c, [change_time, change_time + t_offdesign], xnew_ini, t_eval=tevals_c)

rout_c = solgp_c.y[0, :]
thetaout_c = solgp_c.y[1, :]
vrout_c = solgp_c.y[2, :]
vtout_c = solgp_c.y[3, :]
mout_c = solgp_c.y[4, :]
for i in range(len(mout_c)):
    if mout_c[i] < obj.M0-obj.Mp:
        mout_c[i] = obj.M0-obj.Mp
ttgp_c = solgp_c.t

if ttgp_c[-1] != (change_time + t_offdesign):
    print("second integration stopped prematurly")

for tempi in ttgp_c:
    if tempi > t_offdesign:
        index_c, = np.where(ttgp_c == tempi)

plt.ion()
plt.figure(1)
animated_plot_c = plt.plot(ttgp_c, (rout_c - obj.Re) / 1e3, marker='.', color='b', label="OFF DESIGN")[0]
plt.figure(2)
animated_plot_c2 = plt.plot(ttgp_c, vtout_c, marker='.', color='b', label="OFF DESIGN")[0]
plt.figure(3)
animated_plot_c3 = plt.plot(ttgp_c, mout_c, marker='.', color='b', label="OFF DESIGN")[0]
plt.figure(4)
animated_plot_c4 = plt.plot(ttgp_c, vrout_c, marker='.', color='b', label="OFF DESIGN")[0]


for i in range(len(ttgp_c)):
    plt.figure(1)
    animated_plot_c.set_xdata(ttgp_c[0:i])
    animated_plot_c.set_ydata((rout_c[0:i]-obj.Re)/1e3)
    plt.pause(0.00000001)

    plt.figure(2)
    animated_plot_c2.set_xdata(ttgp_c[0:i])
    animated_plot_c2.set_ydata(vtout_c[0:i])
    plt.pause(0.00000001)

    plt.figure(3)
    animated_plot_c3.set_xdata(ttgp_c[0:i])
    animated_plot_c3.set_ydata(mout_c[0:i])
    plt.pause(0.00000001)

    plt.figure(4)
    animated_plot_c4.set_xdata(ttgp_c[0:i])
    animated_plot_c4.set_ydata(vrout_c[0:i])
    plt.draw()
    plt.pause(0.00000001)

##################################################################################################################


# Simulazione per TEMPO CON NUOVA LEGGE creata dal GENETIC PROGRAMMING

passint_gp = (total_time_simulation - (change_time + t_offdesign)) * 4
tevals_gp = np.linspace(change_time + t_offdesign, tfin, 500)
xnew_ini_gp = [float(rout_c[index_c]), float(thetaout_c[index_c]), float(vrout_c[index_c]), float(vtout_c[index_c]), float(mout_c[index_c])]

def sys2GP_gp(t, x):
    global flag_prop
    fTr = toolbox.compile(hof[0][0])
    fTt = toolbox.compile(hof[0][1])
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

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt
    em = mf - m
    Tt = fTt(er, et, evr, evt, em)
    Tr = fTr(er, et, evr, evt, em)
    if m <= obj.M0 - obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp

    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt

flag_prop = False
solgp_gp = solve_ivp(sys2GP_gp, [change_time + t_offdesign, tfin], xnew_ini_gp, t_eval=tevals_gp)

rout_gp = solgp_gp.y[0, :]
thetaout_gp = solgp_gp.y[1, :]
vrout_gp = solgp_gp.y[2, :]
vtout_gp = solgp_gp.y[3, :]
mout_gp = solgp_gp.y[4, :]
for i in range(len(mout_gp)):
    if mout_gp[i] < obj.M0 - obj.Mp:
        mout_gp[i] = obj.M0 - obj.Mp
ttgp_gp = solgp_gp.t
if ttgp_gp[-1] != (change_time + t_offdesign):
    print("third integration stopped prematurly")
plt.ion()
plt.figure(1)
animated_plot_gp = plt.plot(ttgp_gp, (rout_gp - obj.Re) / 1e3, marker='.', color='g', label="ONLINE CONTROL")[0]
plt.figure(2)
animated_plot_gp2 = plt.plot(ttgp_gp, vtout_gp, marker='.', color='g', label="ONLINE CONTROL")[0]
plt.figure(3)
animated_plot_gp3 = plt.plot(ttgp_gp, mout_gp, marker='.', color='g', label="ONLINE CONTROL")[0]
plt.figure(4)
animated_plot_gp4 = plt.plot(ttgp_gp, vrout_gp, marker='.', color='g', label="ONLINE CONTROL")[0]
i = 0
for items in ttgp_gp:
    plt.figure(1)
    animated_plot_gp.set_xdata(ttgp_gp[0:i])
    animated_plot_gp.set_ydata((rout_gp[0:i]-obj.Re)/1e3)
    plt.pause(0.00000001)

    plt.figure(2)
    animated_plot_gp2.set_xdata(ttgp_gp[0:i])
    animated_plot_gp2.set_ydata(vtout_gp[0:i])
    plt.pause(0.00000001)

    plt.figure(3)
    animated_plot_gp3.set_xdata(ttgp_gp[0:i])
    animated_plot_gp3.set_ydata(mout_gp[0:i])
    plt.pause(0.00000001)

    plt.figure(4)
    animated_plot_gp4.set_xdata(ttgp_gp[0:i])
    animated_plot_gp4.set_ydata(vrout_gp[0:i])
    plt.draw()
    plt.pause(0.00000001)

    i = i + 1

print("\n")
print("Tr old: ", Tr_old)
print("Tt old: ", Tt_old)
print("Tr new: ", hof[0][0])
print("Tt new: ", hof[0][1])
plt.show(block=True)