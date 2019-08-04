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
from functools import partial
import datetime
import math

'''References:
    [1] - Automatic creation of human competitive Programs and Controllers by Means of Genetic Programming. Koza, Keane, Yu, Bennet, Mydlowec. 2000'''

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
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
        return left


def Sqrt(x):
    try:
        if x > 0:
            return np.sqrt(x)
        else:
            return abs(x)
    except (
    RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def Log(x):
    try:
        if x > 0:
            return np.log(x)
        else:
            return abs(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return np.e


def Exp(x):
    try:
        return np.exp(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 1


def Sin(x):
    try:
        return np.sin(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0

def Cos(x):
    try:
        return np.cos(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 1


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
        try:
            val = float(ind[i1][i2].value)
            new_val = round(random.uniform(-10, 10), 4)
            #new_val_gauss = np.random.normal(ind[i1][i2].value, 1.0)  # new value of the constant determined by gaussian distribution suggested by Koza in [1]
            ind[i1][i2].value = new_val
            ind[i1][i2].name = "{}".format(new_val)
            return ind,
        except (ValueError, AttributeError):
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

            lind1 = sum([len(gpt) for gpt in ind1]) #extra
            lind2 = sum([len(gpt) for gpt in ind2]) #extra
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


def hash_fun(self):
    if self.height == np.nan:
        return hash(tuple(self))


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
        self.cosOld = 0
        self.eqOld = np.zeros((0))
        self.ineqOld = np.zeros((0))
        self.varOld = np.zeros((0))

    @staticmethod
    def air_density(h):
        beta = 1/8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0*np.exp(-beta*h)

Nstates = 5
Ncontrols = 2

old = 0

size_pop = 100# Pop size
size_gen = 200                                                                         # Gen size
Mu = int(size_pop)
Lambda = int(size_pop*1.5)

limit_height = 20                                                                   # Max height (complexity) of the controller law
limit_size = 400                                                                    # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()

################################# M A I N ###############################################

def main():
    global size_gen, size_pop, Mu, Lambda
    global size_pop
    global a0, a1, a2, rise_time, vr, yr, ar
    global stat_evoo
    global Rfun, Thetafun, Vrfun, Vtfun, mfun
    global tfin
    global probcx, probmut, counter, fit_min_old, fit_min, fit_current
    counter = 0
    probcx = 0.6
    probmut = 0.35
    fit_min_old = sum([600, 300, 60, 700, 800])
    fit_current = []
    fit_min = fit_min_old


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

    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)

    print("INITIAL POP SIZE: %d" % size_pop)

    print("GEN SIZE: %d" % size_gen)

    print("\n")

    random.seed()

    pop = toolbox.population(n=size_pop)
    history.update(pop)
    # hof = tools.HallOfFame(size_gen) ### OLD ###
    hof = tools.HallOfFame(1) ### NEW ###
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    #stats_size = tools.Statistics(len)
    #stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

    # pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, 0.6, 0.2, size_gen, stats=mstats, halloffame=hof,
                                         # verbose=True)  ### OLD ###
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=Mu, lambda_=Lambda, cxpb=probcx, mutpb=probmut, ngen=size_gen,
                              stats=mstats, halloffame=hof, verbose=True)  ### NEW ###

    ####################################################################################################################

    stop = timeit.default_timer()
    total_time = stop - start
    tformat = str(datetime.timedelta(seconds=int(total_time)))

    gen = log.select("gen")
    fit_avg = log.chapters["fitness"].select('min')

    perform = []
    perform2 = []
    perform3 = []
    perform4 = []
    perform5 = []
    p = 0
    for items in fit_avg:
        perform.append(fit_avg[p][0])
        perform2.append(fit_avg[p][1])
        perform3.append(fit_avg[p][2])
        perform4.append(fit_avg[p][3])
        perform5.append(fit_avg[p][4])
        p = p + 1

    #size_avgs = log.chapters["size"].select("avg")
    fig, ax1 = plt.subplots()
    ax1.plot(gen[1:], perform[1:], "b-", label="Minimum Position Fitness Performance")
    ax1.plot(gen[1:], perform2[1:], "r-", label="Minimum Theta Fitness Performance")
    ax1.plot(gen[1:], perform3[1:], "g-", label="Minimum Vr Fitness Performance")
    ax1.plot(gen[1:], perform4[1:], "k-", label="Minumum Vt Fitness Performance")
    ax1.plot(gen[1:], perform5[1:], "m-", label="Minimum m Fitness Performance")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    ax1.legend(loc="best")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    #ax2 = ax1.twinx()
    #line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    #ax2.set_ylabel("Size", color="r")
    #for tl in ax2.get_yticklabels():
    #    tl.set_color("r")

    #lns = line1 + line2
    #labs = [l.get_label() for l in lns]
    #ax1.legend(lns, labs, loc="center right")
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
    expr1 = hof[0][0]
    expr2 = hof[0][1]

    nodes1, edges1, labels1 = gp.graph(expr1)
    nodes2, edges2, labels2 = gp.graph(expr2)
    g1 = pgv.AGraph()
    g1.add_nodes_from(nodes1)
    g1.add_edges_from(edges1)
    g1.layout(prog="dot")
    for i in nodes1:
        n = g1.get_node(i)
        n.attr["label"] = labels1[i]
    g1.draw("tree1.png")

    g2 = pgv.AGraph()
    g2.add_nodes_from(nodes2)
    g2.add_edges_from(edges2)
    g2.layout(prog="dot")
    for i in nodes2:
        n = g2.get_node(i)
        n.attr["label"] = labels2[i]
    g2.draw("tree2.png")

    image1 = plt.imread('tree1.png')
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(image1)
    ax1.axis('off')
    image2 = plt.imread('tree2.png')
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(image2)
    ax2.axis('off')
    plt.show()

    sys.stdout.write("TOTAL RUNNING TIME: {} \n".format(tformat))

    #################################### P O S T - P R O C E S S I N G #################################################

    fTr = toolbox.compile(expr=expr2)
    fTt = toolbox.compile(expr=expr1)
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
        dxdt[3] = fTt(er, et, evr, evt, em) / m - Dt / m - (Vr * Vt) / R
        dxdt[4] = - np.sqrt(fTr(er, et, evr, evt, em) ** 2 + fTt(er, et, evr, evt, em) ** 2) / g0 / Isp

        return dxdt

    #tevals = np.linspace(0.0, tfin, 1000)

    solgp = solve_ivp(sys2GP, [0.0, tfin], x_ini)
    rout = solgp.y[0, :]
    thetaout = solgp.y[1, :]
    vrout = solgp.y[2, :]
    vtout = solgp.y[3, :]
    mout = solgp.y[4, :]
    tgp = solgp.t
    if tgp[-1] != tfin:
        print("integration stopped prematurely")
    rR = np.zeros(len(tgp), dtype='float')
    tR = np.zeros(len(tgp), dtype='float')
    vrR = np.zeros(len(tgp), dtype='float')
    vtR = np.zeros(len(tgp), dtype='float')
    mR = np.zeros(len(tgp), dtype='float')

    ii = 0
    for i in tgp:
        rR[ii] = Rfun(i)
        tR[ii] = Thetafun(i)
        vrR[ii] = Vrfun(i)
        vtR[ii] = Vtfun(i)
        mR[ii] = mfun(i)
        ii = ii + 1

    errgp = rR - rout  # Error system with genetic programming

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position [km]")
    plt.plot(tgp, (rout-obj.Re)/1e3, label="GENETIC PROGRAMMING")
    plt.plot(tgp, (rR-obj.Re)/1e3, 'r--', label="SET POINT")
    plt.legend(loc="best")
    plt.savefig('Height plot.png')

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("angle [deg]")
    plt.plot(tgp, thetaout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, tR, 'r--', label="SET POINT")
    plt.legend(loc="best")
    plt.savefig('Angle plot.png')

    fig4, ax4 = plt.subplots()
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("speed [m/s]")
    plt.plot(tgp, vrout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, vrR, 'r--', label="SET POINT")
    plt.legend(loc="best")
    plt.savefig('Vr plot.png')

    fig5, ax5 = plt.subplots()
    ax5.set_xlabel("time [s]")
    ax5.set_ylabel("speed [m/s]")
    plt.plot(tgp, vtout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, vtR, 'r--', label="SET POINT")
    plt.legend(loc="best")
    plt.savefig('Speed plot.png')

    fig6, ax6 = plt.subplots()
    ax6.set_xlabel("time [s]")
    ax6.set_ylabel("mass [kg]")
    plt.plot(tgp, mout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, mR, 'r--', label="SET POINT")
    plt.legend(loc="best")
    plt.savefig('mass plot.png')
    plt.show()

    pool.close()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################
flag = False
pas = False
#stat_evoo = [0, 0, 0, 0]

fitnnesoldvalue = 0
fitness_old1 = 1e5
fitness_old3 = 1e5
fitness_old2 = 1e5
fitness_old4 = 1e5
fitness_old5 = 1e5

def evaluate(individual):
    global flag, pas
    global fitness_old1, fitness_old2, fitness_old3, fitness_old4, fitness_old5
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun, tfin
    global fit_min_old, fit_min, fit_current, probmut, probcx
    global counter

    flag = False
    pas = False

    # Transform the tree expression in a callable function

    fTr = toolbox.compile(expr=individual[1])
    fTt = toolbox.compile(expr=individual[0])
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

    def sys(t, x):
        global fitnesshistory, fitness_mean, fitness_mean_history, probmut, probcx
        global counter
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if R<0 or np.isnan(R):
            R = obj.Re
            flag = True
        if np.isinf(R) or R > obj.Rtarget:
            R = obj.Rtarget
            flag = True
        if m<(obj.M0 - obj.Mp) or np.isnan(m):
            m = obj.M0-obj.Mp
            flag = True
        elif m>obj.M0 or np.isinf(m):
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
        dxdt = np.zeros(Nstates)
        #print("Ft: ", fTt(er, et, evr, evt, em))
        #print("Fr: ", fTr(er, et, evr, evt, em))
        rho = obj.air_density(R - obj.Re)
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        g0 = obj.g0
        g = g0 * (obj.Re / R) ** 2  # [m/s2]
        Isp = obj.Isp

        Tr = fTr(er, et, evr, evt, em)
        Tt = fTt(er, et, evr, evt, em)

        if abs(fTr(er, et, evr, evt, em)) > obj.Tmax or np.isinf(fTr(er, et, evr, evt, em)):
            Tr = obj.Tmax
            flag = True

        elif fTr(er, et, evr, evt, em) < 0.0 or np.isnan(fTr(er, et, evr, evt, em)):
            Tr = 0.0
            flag = True

        if abs(fTt(er, et, evr, evt, em)) > obj.Tmax or np.isinf(fTt(er, et, evr, evt, em)):
            Tt = obj.Tmax
            flag = True

        elif fTt(er, et, evr, evt, em) < 0.0 or np.isnan(fTt(er, et, evr, evt, em)):
            Tt = 0.0
            flag = True

        dxdt[0] = Vr
        dxdt[1] = Vt / R
        dxdt[2] = Tr / m - Dr / m - g + Vt ** 2 / R
        dxdt[3] = Tt / m - Dt / m - (Vr * Vt) / R
        dxdt[4] = - np.sqrt(Tr ** 2 + Tt ** 2) / g0 / Isp

        return dxdt

    sol = solve_ivp(sys, [0.0, tfin], x_ini, first_step=0.0001)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y3 = sol.y[2, :]
    y4 = sol.y[3, :]
    y5 = sol.y[4, :]
    tt = sol.t
    if sol.t[-1] != tfin:
        flag = True
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
    err2 = np.rad2deg(theta - y2)/60
    err3 = (vr - y3) / obj.Vtarget
    err4 = (vt - y4) / obj.Vtarget
    err5 = (m - y5) / obj.M0

    # STEP TIME SIZE
    i = 0
    pp = 1
    step = np.zeros(len(y1), dtype='float')
    step[0] = 0.0001
    while i < len(tt) - 1:
        step[pp] = tt[i + 1] - tt[i]
        i = i + 1
        pp = pp + 1

    # INTEGRAL OF ABSOLUTE ERROR (PERFORMANCE INDEX)
    IAE = np.zeros((5, len(err1)))
    j = 0
    alpha = 0.1
    for a, b, c, d, e, n in zip(err1, err2, err3, err4, err5, step):
        IAE[0][j] = n * abs(a)
        IAE[1][j] = n * abs(b)
        IAE[2][j] = n * abs(c)  # + alpha * abs(m))
        IAE[3][j] = n * abs(d)
        IAE[4][j] = n * abs(e)
        j = j + 1

    # PENALIZING INDIVIDUALs
    #For the stats if the multiprocessing is used, there could be problems to print the correct values (parallel process(?))

    if flag is True:
        pas = True
        x = [np.random.uniform(fitness_old1 * 1.5, fitness_old1 * 1.6),
         np.random.uniform(fitness_old2 * 1.5, fitness_old2 * 1.6),
         np.random.uniform(fitness_old3 * 1.5, fitness_old3 * 1.6),
         np.random.uniform(fitness_old4 * 1.5, fitness_old4 * 1.6),
         np.random.uniform(fitness_old5 * 1.5, fitness_old5 * 1.6)]

    if flag is False:
        fitness1 = sum(IAE[0])
        fitness2 = sum(IAE[1])
        fitness3 = sum(IAE[2])
        fitness4 = sum(IAE[3])
        fitness5 = sum(IAE[4])
        if fitness1 < fitness_old1:
            fitness_old1 = fitness1
        if fitness2 < fitness_old2:
            fitness_old2 = fitness2
        if fitness3 < fitness_old3:
            fitness_old3 = fitness3
        if fitness4 < fitness_old4:
            fitness_old4 = fitness4
        if fitness5 < fitness_old5:
            fitness_old5 = fitness5
        fitness = [fitness1,
                   fitness2,
                   fitness3,
                   fitness4,
                   fitness5]
    if counter > 1000:
        if pas:
            fit_current = sum(x)
            if fit_current < fit_min * 1.1 and fit_current > fit_min* 0.9:
                probmut = 0.6
                probcx = 0.4
                counter += 1
                print(probmut)
            else:
                probmut = 0.35
                probcx = 0.6
                counter += 1
            fit_min = min(fit_current, fit_min_old)
            fit_min_old = fit_min
            return x
        else:
            fit_current = sum(fitness)
            if fit_current < fit_min * 1.1 and fit_current > fit_min* 0.9:
                probmut = 0.6
                probcx = 0.4
                counter += 1
                print(probmut)
            else:
                probmut = 0.35
                probcx = 0.6
                counter += 1
            fit_min = min(fit_current, fit_min_old)
            fit_min_old = fit_min
            return fitness
    else:
        fit_min_old = sum(x if pas is True else fitness)
        counter += 1
        return x if pas is True else fitness

####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2, name="Mul")
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
pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-5, 5), 4))
pset.addEphemeralConstant("rand102", lambda: round(random.uniform(-10, 10), 4))
pset.addEphemeralConstant("rand103", lambda: round(random.uniform(-15, 15), 4))
pset.addEphemeralConstant("rand104", lambda: round(random.uniform(-20, 20), 4))
#pset.addADF(pset)
pset.renameArguments(ARG0='errR')
pset.renameArguments(ARG1='errTheta')
pset.renameArguments(ARG2='errVr')
pset.renameArguments(ARG3='errVt')
pset.renameArguments(ARG4='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-0.8, -0.8, -0.8, -1.0, -0.8))    # MINIMIZATION OF THE FITNESS FUNCTION

creator.create("Individual", list, fitness=creator.Fitness, height=1)

creator.create("SubIndividual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()
# toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)   #### OLD ####
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, type_=pset.ret, min_=1, max_=3)  ### NEW ###

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
toolbox.register("select", xselDoubleTournament, fitness_size=4, parsimony_size=1.4, fitness_first=True) ### NEW ###

toolbox.register("mate", xmate) ### NEW ###
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4) ### NEW ###
toolbox.register("mutate", xmut, expr=toolbox.expr_mut, strp=0.6) ### NEW ###

# toolbox.register("mate", gp.cxOnePointLeafBiased,termpb=0.1) ### OLD ###
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) ### OLD ###

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

toolbox.decorate("mate", gp.staticLimit(key=len , max_value=limit_size))
#toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)


########################################################################################################################


if __name__ == "__main__":
    obj = Rocket()
    pop, log, hof = main()

