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
import datetime
import math

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


Nstates = 3
Ncontrols = 1


size_pop = 150 # Pop size
size_gen = 50  # Gen size
Mu = int(size_pop)
Lambda = int(size_pop * 1.4)

limit_height = 30  # Max height (complexity) of the controller law
limit_size = 400  # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()


################################# M A I N ###############################################


def main():
    global size_gen, size_pop, Mu, Lambda
    global Rfun, Vfun, mfun
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

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, 0.75, 0.2, size_gen, stats=mstats, halloffame=hof, verbose=True)  ### OLD ###

    ####################################################################################################################

    stop = timeit.default_timer()
    total_time = stop - start
    tformat = str(datetime.timedelta(seconds=int(total_time)))

    gen = log.select("gen")
    fit_avg = log.chapters["fitness"].select('min')

    perform1 = []
    perform2 = []
    perform3 = []
    p = 0
    for items in fit_avg:
        perform1.append(fit_avg[p][0])
        perform2.append(fit_avg[p][1])
        perform3.append(fit_avg[p][2])
        p = p + 1

    # size_avgs = log.chapters["size"].select("avg")
    fig, ax1 = plt.subplots()
    ax1.plot(gen[1:], perform1[1:], "b-", label="Min Position Fitness Performance")
    ax1.plot(gen[1:], perform2[1:], "r-", label="Min Speed Fitness Performance")
    ax1.plot(gen[1:], perform3[1:], "g-", label="Min Mass Fitness Performance")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    ax1.legend(loc="best")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    textstr = ('Total Running Time: {}'.format(tformat))
    ax1.text(0.65, 0.9, textstr, transform=ax1.transAxes, fontsize=10,
             horizontalalignment='right')

    plt.savefig('Stats')
    plt.show()

    '''print("\n")
    print("THE BEST VALUES ARE:")
    print(hof.items[0][0])
    print(hof.items[0][1])
    print("\n")
    print("THE HEIGHT OF THE BEST INDIVIDUALS ARE:")
    print(hof.items[0][0].height)
    print(hof.items[0][1].height)
    print("\n")
    print("THE SIZE OF THE BEST INDIVIDUALS ARE:")
    print(len(hof.items[0][0]))
    print(len(hof.items[0][1]))


    value = toolbox.evaluate(hof[0])
    print("THE EVALUATION OF THE BEST INDIVIDUAL IS:")
    print(value)
    print("\n")'''

    expr1 = hof[0]

    nodes1, edges1, labels1 = gp.graph(expr1)

    g1 = pgv.AGraph()
    g1.add_nodes_from(nodes1)
    g1.add_edges_from(edges1)
    g1.layout(prog="dot")
    for i in nodes1:
        n = g1.get_node(i)
        n.attr["label"] = labels1[i]
    g1.draw("tree1.png")
    image1 = plt.imread('tree1.png')
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(image1)
    ax1.axis('off')

    plt.show()

    sys.stdout.write("TOTAL RUNNING TIME: {} \n".format(tformat))

    #################################### P O S T - P R O C E S S I N G #################################################

    fT = toolbox.compile(hof[0])
    x_ini = [obj.Re, 0.0, obj.M0*obj.Mc]  # initial conditions

    def sys2GP(t, x):
        R = x[0]
        V = x[1]
        m = x[2]

        r = Rfun(t)
        v = Vfun(t)
        mf = mfun(t)

        er = r - R
        ev = v - V
        em = mf - m
        dxdt = np.zeros(Nstates)

        rho = obj.air_density(R - obj.Re)
        drag = 0.5 * rho * V ** 2 * obj.Cd * obj.area
        g = obj.GMe / R ** 2
        g0 = obj.g0
        Isp = obj.Isp
        T = fT(er, ev, em)

        dxdt[0] = V
        dxdt[1] = (T-drag)/m-g
        dxdt[2] = - T / g0 / Isp

        return dxdt

    # tevals = np.linspace(0.0, tfin, 1000)

    solgp = solve_ivp(sys2GP, [0.0, tfin], x_ini)
    rout = solgp.y[0, :]
    vout = solgp.y[1, :]
    mout = solgp.y[2, :]
    tgp = solgp.t
    rR = np.zeros(len(tgp), dtype='float')
    vR = np.zeros(len(tgp), dtype='float')
    mR = np.zeros(len(tgp), dtype='float')

    ii = 0
    for i in tgp:
        rR[ii] = Rfun(i)
        vR[ii] = Vfun(i)
        mR[ii] = mfun(i)
        ii = ii + 1

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position [km]")
    plt.plot(tgp, (rout - obj.Re) / 1e3, label="GENETIC PROGRAMMING")
    plt.plot(tgp, (rR - obj.Re) / 1e3, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('Height plot.png')

    fig4, ax4 = plt.subplots()
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("speed [m/s]")
    plt.plot(tgp, vout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, vR, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('Vr plot.png')

    fig6, ax6 = plt.subplots()
    ax6.set_xlabel("time [s]")
    ax6.set_ylabel("mass [kg]")
    plt.plot(tgp, mout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, mR, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('mass plot.png')

    pool.close()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################


def evaluate(individual):
    global flag
    global pas
    global fitnnesoldvalue, fitness_old1, fitness_old2, fitness_old3
    global Rfun, Vfun, mfun
    global tfin

    flag = False
    pas = False

    # Transform the tree expression in a callable function

    fT = toolbox.compile(expr=individual)

    x_ini = [obj.Re, 0.0, obj.M0*obj.Mc]  # initial conditions

    def sys(t, x):
        #print("------------------------iter-------------------------")
        global flag, pas

        # State Variables
        R = x[0]
        V = x[1]
        m = x[2]

        if R < 0 or np.isnan(R):
            R = obj.Re
            flag = True
        if np.isinf(R):
            R = obj.Re + 50*1e3
            flag = True
        if m < 0 or np.isnan(m):
            m = obj.M0*obj.Mc
            flag = True
        elif m > obj.M0 or np.isinf(m):
            m = obj.M0
            flag = True
        if abs(V) > 1e4 or np.isinf(V):
            if V > 0:
                V = 1e4
                flag = True
            else:
                V = -1e4
                flag = True


        r = Rfun(t)
        v = Vfun(t)
        mf = mfun(t)

        er = r - R
        ev = v - V
        em = mf - m
        dxdt = np.zeros(Nstates)
        T = fT(er, ev, em)
        # print("Fr: ", fTr(er, et, evr, evt, em))

        rho = obj.air_density(R - obj.Re)
        drag = 0.5 * rho * V ** 2 * obj.Cd * obj.area
        g = obj.GMe / R ** 2
        g0 = obj.g0
        Isp = obj.Isp
        T = fT(er, ev, em)


        if abs(fT(er, ev, em)) > obj.Tmax or np.isinf(fT(er, ev, em)):
            T = obj.Tmax
            flag = True

        elif fT(er, ev, em) < 0.0 or np.isnan(fT(er, ev, em)):
            T = 0.0
            flag = True

        dxdt[0] = V
        dxdt[1] = (T - drag) / m - g
        dxdt[2] = - T / g0 / Isp
        return dxdt

    t_eval = np.linspace(0, tfin, 1000)
    sol = solve_ivp(sys, [0.0, tfin], x_ini, t_eval=t_eval)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y3 = sol.y[2, :]
    tt = sol.t
    if sol.t[-1] != tfin:
        flag = True
    pp = 0
    r = np.zeros(len(tt), dtype='float')
    v = np.zeros(len(tt), dtype='float')
    m = np.zeros(len(tt), dtype='float')
    for i in tt:
        r[pp] = Rfun(i)
        v[pp] = Vfun(i)
        m[pp] = mfun(i)
        pp += 1

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

    # INTEGRAL OF ABSOLUTE ERROR (PERFORMANCE INDEX)
    IAE = np.zeros((3, len(err1)))
    j = 0
    for a, b, c, n in zip(err1, err2, err3, step):
       IAE[0][j] = n * abs(a)
       IAE[1][j] = n * abs(b)
       IAE[2][j] = n * abs(c)
       j = j + 1

    # PENALIZING INDIVIDUALs
    # For the stats if the multiprocessing is used, there could be problems to print the correct values (parallel process(?))

    if flag is True:
        pas = True
        x = [np.random.uniform(fitness_old1 * 1.5, fitness_old1 * 1.6),
             np.random.uniform(fitness_old2 * 1.5, fitness_old2 * 1.6),
             np.random.uniform(fitness_old3* 1.5, fitness_old3 * 1.6)]

    if flag is False:
        fitness1 = sum(IAE[0])
        fitness2 = sum(IAE[1])
        fitness3 = sum(IAE[2])
        if fitness1 < fitness_old1:
            fitness_old1 = fitness1
        if fitness2 < fitness_old2:
            fitness_old2 = fitness2
        if fitness3 < fitness_old3:
            fitness_old3 = fitness3
        fitness = [fitness1,
                   fitness2,
                   fitness3]

    return x if pas is True else fitness


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

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0))  # MINIMIZATION OF THE FITNESS FUNCTION

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

