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


def Div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def Mul(left, right):
    try:
        np.seterr(invalid='raise')
        return left * right
    except (
            RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return left


def Sqrt(x):
    try:
        if x > 0:
            return math.sqrt(x)
        else:
            return abs(x)
    except (
            RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError,
            ValueError):
        return 0


def Log(x):
    try:
        if x > 0:
            return math.log(x)
        else:
            return abs(x)
    except (
            RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError,
            ValueError):
        return 0


def Exp(x):
    try:
        return math.exp(x)
    except (
            RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError,
            ValueError):
        return 0


def Sin(x):
    try:
        return math.sin(x)
    except (
            RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError,
            ValueError):
        return 0


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
        beta = 1 / 8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0 * np.exp(-beta * h)


Nstates = 5
Ncontrols = 2

old = 0

size_pop = 100 # Pop size
size_gen = 300  # Gen size
Mu = int(size_pop)
Lambda = int(size_pop * 1.4)

limit_height = 30  # Max height (complexity) of the controller law
limit_size = 400  # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()


################################# M A I N ###############################################


def main():
    global size_gen, size_pop, Mu, Lambda
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun
    global tfin

    Rref = np.load("R.npy")
    Thetaref = np.load("Theta.npy")
    Vrref = np.load("Vr.npy")
    Vtref = np.load("Vt.npy")
    mref = np.load("m.npy")
    tref = np.load("time.npy")
    Trref = np.load("Tr.npy")
    tfin = tref[-1]


    Rfun = PchipInterpolator(tref, Rref)
    Thetafun = PchipInterpolator(tref, Thetaref)
    Vrfun = PchipInterpolator(tref, Vrref)
    Vtfun = PchipInterpolator(tref, Vtref)
    mfun = PchipInterpolator(tref, mref)
    Trfun = PchipInterpolator(tref, Trref)

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

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, 0.6, 0.1, size_gen, stats=mstats, halloffame=hof, verbose=True)  ### OLD ###

    ####################################################################################################################

    stop = timeit.default_timer()
    total_time = stop - start
    tformat = str(datetime.timedelta(seconds=int(total_time)))

    gen = log.select("gen")
    fit_avg = log.chapters["fitness"].select('avg')

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
    ax1.plot(gen[1:], perform1[1:], "b-", label="Average Position Fitness Performance")
    ax1.plot(gen[1:], perform2[1:], "r-", label="Average Speed Fitness Performance")
    ax1.plot(gen[1:], perform3[1:], "g-", label="Average Mass Fitness Performance")
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

    fTt = toolbox.compile(hof[0])
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

    def sys2GP(t, x):
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
        Tr = Trfun(t)

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
        dxdt[2] = Tr/m - Dr/m - g + Vt ** 2 / R
        dxdt[3] = fTt(er, et, evr, evt, em) / m - Dt / m - (Vr * Vt) / R
        dxdt[4] = - np.sqrt(fTt(er, et, evr, evt, em)**2 + Tr**2) / g0 / Isp

        return dxdt

    # tevals = np.linspace(0.0, tfin, 1000)

    solgp = solve_ivp(sys2GP, [0.0, tfin], x_ini)
    rout = solgp.y[0, :]
    thetaout = solgp.y[1, :]
    vrout = solgp.y[2, :]
    vtout = solgp.y[3, :]
    mout = solgp.y[4, :]
    tgp = solgp.t
    rR = np.zeros(len(tgp), dtype='float')
    tR = np.zeros(len(tgp), dtype='float')
    vrR = np.zeros(len(tgp), dtype='float')
    vtR = np.zeros(len(tgp), dtype='float')
    mR = np.zeros(len(tgp), dtype='float')
    TrR = np.zeros(len(tgp), dtype='float')

    ii = 0
    for i in tgp:
        rR[ii] = Rfun(i)
        tR[ii] = Thetafun(i)
        vrR[ii] = Vrfun(i)
        vtR[ii] = Vtfun(i)
        mR[ii] = mfun(i)
        TrR[ii] = Trfun(i)
        ii = ii + 1

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position [km]")
    plt.plot(tgp, (rout - obj.Re) / 1e3, label="GENETIC PROGRAMMING")
    plt.plot(tgp, (rR - obj.Re) / 1e3, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('Height plot.png')

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("angle [deg]")
    plt.plot(tgp, thetaout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, tR, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('Angle plot.png')

    fig4, ax4 = plt.subplots()
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("speed [m/s]")
    plt.plot(tgp, vrout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, vrR, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('Vr plot.png')

    fig5, ax5 = plt.subplots()
    ax5.set_xlabel("time [s]")
    ax5.set_ylabel("speed [m/s]")
    plt.plot(tgp, vtout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, vtR, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('Speed plot.png')

    fig6, ax6 = plt.subplots()
    ax6.set_xlabel("time [s]")
    ax6.set_ylabel("mass [kg]")
    plt.plot(tgp, mout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, mR, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('mass plot.png')

    fig7, ax7 = plt.subplots()
    ax7.set_xlabel("time [s]")
    ax7.set_ylabel("Thrust (Tr) [N]")
    plt.plot(tgp, TrR, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('thrust plot.png')
    plt.show()

    pool.close()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################
flag = False
pas = False

fitness_old1 = 1e5
fitness_old4 = 1e5
fitness_old5 = 1e5

def evaluate(individual):
    global flag
    global pas
    global fitnnesoldvalue, fitness_old1, fitness_old2, fitness_old3, fitness_old4, fitness_old5
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun
    global tfin

    flag = False
    pas = False

    # Transform the tree expression in a callable function

    fTt = toolbox.compile(expr=individual)
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

    def sys(t, x):
        #print("------------------------iter-------------------------")
        global flag, pas

        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if R < obj.Re:
            R = obj.Re
            #flag = True
            #print("R < 0")
        if np.isinf(R):
            #print("R inf")
            R = 1e10
            #flag = True
        if m < 0:
            m = obj.M0 - obj.Mp
            #flag = True
            #print("m < 0")
        elif m > obj.M0:
            m = obj.M0
            #flag = True
            #print("m > max")
        if abs(Vr) > np.sqrt(np.nan_to_num(np.inf))/1e150:
            #print("Vr inf")
            if Vr > 0:
                Vr = 1e4
                #flag = True
            else:
                Vr = -1e4
                #flag = True
        if abs(Vt) > np.sqrt(np.nan_to_num(np.inf))/1e150:
            #print("Vt inf")
            if Vt > 0:
                Vt = 1e4
                #flag = True
            else:
                Vt = -1e4
                #flag = True


        r = Rfun(t)
        th = Thetafun(t)
        vr = Vrfun(t)
        vt = Vtfun(t)
        mf = mfun(t)
        Tr = Trfun(t)

        er = r - R
        et = th - theta
        evr = vr - Vr
        evt = vt - Vt
        em = mf - m
        dxdt = np.zeros(Nstates)
        #print("Ft: ", fTt(er, et, evr, evt, em), obj.Tmax)
        # print("Fr: ", fTr(er, et, evr, evt, em))

        rho = obj.air_density(R - obj.Re)
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        dxdt[0] = Vr
        dxdt[1] = Vt / R
        dxdt[2] = Tr / m - Dr / m - g + Vt ** 2 / R

        if abs(fTt(er, et, evr, evt, em)) > obj.Tmax and not (np.isinf(fTt(er, et, evr, evt, em)) or np.isnan(fTt(er, et, evr, evt, em)) or np.iscomplex(fTt(er, et, evr, evt, em))):

            dxdt[3] = obj.Tmax / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(obj.Tmax**2 + Tr**2) / g0 / Isp
            flag = True
            #print("Tt > Tmax")

        elif fTt(er, et, evr, evt, em) < 0.0 and not (np.isinf(fTt(er, et, evr, evt, em)) or np.isnan(fTt(er, et, evr, evt, em)) or np.iscomplex(fTt(er, et, evr, evt, em))):

            dxdt[3] = - Dt / m - (Vr * Vt) / R
            dxdt[4] = - Tr/g0/Isp
            flag = True
            #print("Tt < 0")

        elif np.isinf(fTt(er, et, evr, evt, em)) or np.isnan(fTt(er, et, evr, evt, em)) or np.iscomplex(fTt(er, et, evr, evt, em)):

            dxdt[3] = np.nan_to_num(fTt(er, et, evr, evt, em)) / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(np.nan_to_num(fTt(er, et, evr, evt, em))**2 + Tr**2) / g0 / Isp
            flag = True
            #print("Tt inf or nan or complex")

        else:
            dxdt[3] = fTt(er, et, evr, evt, em) / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = -np.sqrt(fTt(er, et, evr, evt, em)**2 + Tr**2) / g0 / Isp

        return dxdt

    sol = solve_ivp(sys, [0.0, tfin], x_ini)
    y1 = sol.y[0, :]
    y4 = sol.y[3, :]
    y5 = sol.y[4, :]
    tt = sol.t
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

    err1 = (r - y1)/obj.Htarget
    #err2 = theta - y2
    #err3 = vr - y3
    err4 = (vt - y4)/obj.Vtarget
    err5 = (m - y5)/obj.M0

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
    alpha = 0.1
    for a, b, c, n in zip(err1, err4, err5, step):
       IAE[0][j] = n * abs(a)
       IAE[1][j] = n * abs(b)
       IAE[2][j] = n * abs(c) # + alpha * abs(m))
       j = j + 1
    #IAE = np.array([[np.sqrt(sum(err1**2)/len(err1))],
                    #[np.sqrt(sum(err2**2)/len(err2))],
                    #[np.sqrt(sum(err3**2)/len(err3))],
       #             [np.sqrt(sum(err4**2)/len(err4))],
        #            [np.sqrt(sum(err5**2)/len(err5))]])


    # PENALIZING INDIVIDUALs
    # For the stats if the multiprocessing is used, there could be problems to print the correct values (parallel process(?))

    if flag is True:
        pas = True
        x = [np.random.uniform(fitness_old1 * 1.5, fitness_old1 * 1.6),
             np.random.uniform(fitness_old4 * 1.5, fitness_old4 * 1.6),
             np.random.uniform(fitness_old5 * 1.5, fitness_old5 * 1.6)]

    if flag is False:
        fitness1 = sum(IAE[0])
        fitness4 = sum(IAE[1])
        fitness5 = sum(IAE[2])
        if fitness1 < fitness_old1:
            fitness_old1 = fitness1
        if fitness4 < fitness_old4:
            fitness_old4 = fitness4
        if fitness5 < fitness_old5:
            fitness_old5 = fitness5
        fitness = [fitness1,
                   fitness4,
                   fitness5]

    return x if pas is True else fitness


####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(Mul, 2)
pset.addPrimitive(Div, 2)  # rallentamento per gli ndarray utilizzati
pset.addPrimitive(Sqrt, 1)
pset.addPrimitive(Log, 1)
pset.addPrimitive(Exp, 1)
pset.addPrimitive(Sin, 1)
pset.addTerminal(np.pi, "pi")
pset.addTerminal(np.e, name="nap")  # e Napier constant number
# pset.addTerminal(2)
pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-10, 10), 2))
pset.renameArguments(ARG0='errR')
pset.renameArguments(ARG1='errTheta')
pset.renameArguments(ARG2='errVr')
pset.renameArguments(ARG3='errVt')
pset.renameArguments(ARG4='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0))  # MINIMIZATION OF THE FITNESS FUNCTION

creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=4)   #### OLD ####

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####

toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####

toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evaluate)  ### OLD ###

toolbox.register("select", tools.selNSGA2) ### OLD ###

toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1) ### OLD ###
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) ### OLD ###

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

