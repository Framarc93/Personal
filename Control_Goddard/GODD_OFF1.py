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
from deap.algorithms import varOr
import multiprocessing
from scipy.interpolate import PchipInterpolator
import matplotlib.animation as animation
from matplotlib import style
import datetime
from time import time
from functools import partial
import _pickle as cPickle
import pickle

def eaMuPlusLambdaTol(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, tol, stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.

    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    min_fit = np.array(logbook.chapters["fitness"].select("min"))
    # Begin the generational process
    gen = 1
    while gen < ngen + 1 and not (min_fit[-1] <= tol).all():
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        #min_fit = np.array(logbook.chapters["fitness"].select("min"))
        #if (np.array(min_fit[-1] < 250)).all():
         #   mutpb = 0.1
          #  cxpb = 0.7
        #else:
         #   mutpb = 0.25
          #  cxpb = 0.6

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        gen += 1
    return population, logbook

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

def Tanh(a):
    return np.tanh(a)

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


def xmut(ind, expr):

    i1 = random.randrange(len(ind))
    i2 = random.randrange(len(ind[i1]))
    choice = random.random()
    #if choice < strp:
    indx = gp.mutUniform(ind[i1], expr, pset=pset)
    ind[i1] = indx[0]
    return ind,
    #else:
     #   '''this part execute the mutation on a random constant'''
      #  indx = gp.mutEphemeral(ind[i1], "one")
       # ind[i1] = indx[0]
        #return ind,


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
nEph = 2
mutpb = 0.25
cxpb = 0.6
change_time = 200

size_pop = 500 # Pop size
size_gen = 500  # Gen size
Mu = int(size_pop)
Lambda = int(size_pop * 1.4)

limit_height = 17  # Max height (complexity) of the controller law
limit_size = 400  # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()

tref = np.load("time.npy")
total_time_simulation = tref[-1]
del tref
flag_seed_populations = False
flag_offdesign = False
flag_seed_populations1 = False
flag=False
flag_time=False
flag_notgood=False
feasible=True
################################# M A I N ###############################################


def main():
    global flag_seed_populations, flag_offdesign, flag_seed_populations1
    global tfin, flag
    global size_gen, size_pop, Mu, Lambda, mutpb, cxpb
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, count_fit, count_old, count_mut, fit_old, count_ind

    flag = False
    count_fit = 0
    count_ind = 0
    count_mut = 0
    fit_old = [1e5, 1e5, 1e5, 1e5, 1e5]
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
    pop = toolbox.population(n=size_pop)

    print("INITIAL POP SIZE: %d" % size_pop)

    print("GEN SIZE: %d" % size_gen)

    print("\n")

    random.seed()
    history.update(pop)

    hof= tools.ParetoFront()
    #hof = tools.HallOfFame(size_gen) ### OLD ###

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height, size=stats_size)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################


    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, mutpb, cxpb, size_gen, stats=mstats, halloffame=hof, verbose=True)
    ####################################################################################################################
    stop = timeit.default_timer()
    total_time = stop - start
    tformat = str(datetime.timedelta(seconds=int(total_time)))

    res = open("HallOfFame_2Cont", "w")
    for i in range(len(hof)):
        res.write("{}: 1".format(i) + str(hof[i][0]) + "\n" + "{}: 2".format(i) + str(hof[i][1]) + "\n")
    res.close()

    gen = log.select("gen")
    fit_avg = log.chapters["fitness"].select('min')

    perform = []
    perform2 = []
    #perform3 = []
    #perform4 = []
    perform5 = []
    p = 0
    for items in fit_avg:
        perform.append(fit_avg[p][0])
        perform2.append(fit_avg[p][1])
        #perform3.append(fit_avg[p][2])
        #perform4.append(fit_avg[p][3])
        perform5.append(fit_avg[p][2])
        p = p + 1

    #size_avgs = log.chapters["size"].select("avg")
    fig, ax1 = plt.subplots()
    ax1.plot(gen[1:], perform[1:], "b-", label="Minimum Theta Fitness Performance")
    ax1.plot(gen[1:], perform2[1:], "r-", label="Minimum Position Fitness Performance")
    #ax1.plot(gen[1:], perform3[1:], "g-", label="Minimum Vr Fitness Performance")
    #ax1.plot(gen[1:], perform4[1:], "k-", label="Minumum Vt Fitness Performance")
    ax1.plot(gen[1:], perform5[1:], "m-", label="Minimum Mass Fitness Performance")
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

    fTr = toolbox.compile(expr=expr1)
    fTt = toolbox.compile(expr=expr2)
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
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
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

    if tgp[-1] < tfin:
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
    plt.plot(tgp, np.rad2deg(thetaout), label="GENETIC PROGRAMMING")
    plt.plot(tgp, np.rad2deg(tR), 'r--', label="SET POINT")
    plt.legend(loc="best")
    plt.savefig('Angle plot.png')

    fig4, ax4 = plt.subplots()
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("radial speed [m/s]")
    plt.plot(tgp, vrout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, vrR, 'r--', label="SET POINT")
    plt.legend(loc="best")
    plt.savefig('radial plot.png')

    fig5, ax5 = plt.subplots()
    ax5.set_xlabel("time [s]")
    ax5.set_ylabel("tangent speed [m/s]")
    plt.plot(tgp, vtout, label="GENETIC PROGRAMMING")
    plt.plot(tgp, vtR, 'r--', label="SET POINT")
    plt.legend(loc="best")
    plt.savefig('tangent plot.png')

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


##################################  F I T N E S S    F U N C T I O N    ##############################################



'''def feasible(individual):
    global flag, flag_offdesign
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun
    global tfin, t_eval2, penalty, count_fit, count_ind, count_mut, fit_old, mutpb, cxpb
    count_fit += 1
    count_ind += 1
    penalty = np.zeros((13))
    flag = False
    feasible=True

    # Transform the tree expression in a callable function
    fTr = toolbox.compile(expr=individual[0])
    fTt = toolbox.compile(expr=individual[1])

    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

    def sys(t, x):
        global penalty, flag,feasible
        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]


        #if np.isnan(theta) or np.isinf(theta):
         #   theta = np.nan_to_num(theta)

        if R < obj.Re or np.isnan(R):
            penalty[0] = penalty[0] + abs(R - obj.Re)/obj.Htarget
            R = obj.Re
            flag = True
        if R > obj.Rtarget or np.isinf(R):
            penalty[1] = penalty[1] + abs(R - obj.Rtarget)/obj.Htarget
            R = obj.Rtarget
            flag = True
        if m < obj.M0 - obj.Mp or np.isnan(m):
            penalty[2] = penalty[2] + abs(m - (obj.M0 - obj.Mp))/obj.M0
            m = obj.M0 - obj.Mp
            flag = True
        elif m > obj.M0 or np.isinf(m):
            penalty[3] = penalty[3] + abs(m - obj.M0)/obj.M0
            m = obj.M0
            flag = True

        if abs(Vr) > 1e4 or np.isinf(Vr):
            penalty[4] = penalty[4] + abs(Vr - 1e4)/obj.Vtarget
            if Vr > 0:
                Vr = 1e4
            else:
                Vr = -1e4
            flag = True

        if abs(Vt) > 1e4 or np.isinf(Vt):
            penalty[5] = penalty[5] + abs(Vt - 1e4)/obj.Vtarget
            if Vt > 0:
                Vt = 1e4
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
            penalty[6] = penalty[6] + abs(Tr)/obj.Tmax
            Tr = 0.0
            flag = True


        elif fTr(er, et, evr, evt, em) > obj.Tmax or np.isinf(fTr(er, et, evr, evt, em)):
            penalty[7] = penalty[7] + abs(Tr - obj.Tmax)/obj.Tmax
            Tr = obj.Tmax
            flag = True


        if fTt(er, et, evr, evt, em) < 0.0 or np.isnan(fTt(er, et, evr, evt, em)):
            penalty[8] = penalty[8] + abs(Tt)/obj.Tmax
            Tt = 0.0
            flag = True


        elif fTt(er, et, evr, evt, em) > obj.Tmax or np.isinf(fTt(er, et, evr, evt, em)):
            penalty[9] = penalty[9] + abs(Tt - obj.Tmax)/obj.Tmax
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

    sol = solve_ivp(sys, [tin, tfin], x_ini, t_eval=teval)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y3 = sol.y[2, :]
    y4 = sol.y[3, :]
    y5 = sol.y[4, :]
    tt = sol.t

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
        try:
            rho = obj.air_density(R - obj.Re)
            Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
            Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * obj.Cd * obj.A  # [N]
            g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
            g0 = obj.g0
            Isp = obj.Isp

            dxdt[0] = Vr
            dxdt[1] = Vt / R
            dxdt[2] = fTr(er, et, evr, evt, em) / m - Dr / m - g + Vt ** 2 / R
            dxdt[3] = fTt(er, et, evr, evt, em) / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(fTr(er, et, evr, evt, em) ** 2 + fTt(er, et, evr, evt, em) ** 2) / g0 / Isp
        except (OverflowError,ArithmeticError):
            return False
        return dxdt

    soll = solve_ivp(sys2GP, [tin, tfin], x_ini, t_eval=teval)
    y11 = sol.y[0, :]
    y22 = sol.y[1, :]
    y33 = sol.y[2, :]
    y44 = sol.y[3, :]
    y55 = sol.y[4, :]
    ttt = sol.t



    if ttt[-1] < tfin:
        return False

    return True'''

def evaluate(individual):
    global flag, flag_offdesign,flag_notgood
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun
    global tfin, t_eval2, penalty, count_fit, count_ind, count_mut, fit_old, mutpb, cxpb
    count_fit += 1
    count_ind += 1
    penalty = np.zeros((13))
    flag = False
    flag_notgood=False

    # Transform the tree expression in a callable function
    fTr = toolbox.compile(expr=individual[0])
    fTt = toolbox.compile(expr=individual[1])

    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions

    def sys(t, x):
        global penalty, flag,flag_notgood
        # State Variables
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

        if R < obj.Re:
            penalty[0] = penalty[0] + abs(R - r)/obj.Htarget
            R = obj.Re
            flag = True
        if R > obj.Rtarget:
            penalty[1] = penalty[1] + abs(R - r)/obj.Htarget
            R = obj.Rtarget
            flag = True
        if m < obj.M0 - obj.Mp:
            penalty[2] = penalty[2] + abs(m - mf)/obj.M0
            m = obj.M0 - obj.Mp
            flag = True
        elif m > obj.M0:
            penalty[3] = penalty[3] + abs(m - mf)/obj.M0
            m = obj.M0
            flag = True

        if abs(Vr) > 8000:
            penalty[4] = penalty[4] + abs(Vr - vr)/obj.Vtarget
            if Vr > 0:
                Vr = 8000
            else:
                Vr = -8000
            flag = True

        if abs(Vt) > 8000:
            penalty[5] = penalty[5] + abs(Vt - vt)/obj.Vtarget
            if Vt > 0:
                Vt = 8000
            else:
                Vt = -8000
            flag = True

        if abs(theta) > 0.8:
            penalty[12]=penalty[12] + abs(theta - th)/0.8
            if theta>0:
                theta=0.8
            else:
                theta=-0.8
            flag=True



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

        '''if fTr(er, et, evr, evt, em) < 0.0 or np.isnan(fTr(er, et, evr, evt, em)):
            penalty[6] = penalty[6] + abs(Tr)/obj.Tmax
            Tr = 0.0
            flag = True


        elif fTr(er, et, evr, evt, em) > obj.Tmax:
            penalty[7] = penalty[7] + abs(Tr - obj.Tmax)/obj.Tmax
            Tr = obj.Tmax
            flag = True


        if fTt(er, et, evr, evt, em) < 0.0 or np.isnan(fTt(er, et, evr, evt, em)):
            penalty[8] = penalty[8] + abs(Tt)/obj.Tmax
            Tt = 0.0
            flag = True


        elif fTt(er, et, evr, evt, em) > obj.Tmax:
            penalty[9] = penalty[9] + abs(Tt - obj.Tmax)/obj.Tmax
            Tt = obj.Tmax
            flag = True'''

        if (fTr(er, et, evr, evt, em)>0 and fTr(er, et, evr, evt, em)<obj.Tmax) and (fTt(er, et, evr, evt, em)>0 and fTt(er, et, evr, evt, em)<obj.Tmax):
            dxdt = np. array((Vr,
                            Vt / R,
                             Tr / m - Dr / m - g + Vt ** 2 / R,
                             Tt / m - Dt / m - (Vr * Vt) / R,
                             -np.sqrt(Tt ** 2 + Tr ** 2) / (g0 * Isp)))
        else:
            dxdt = np. array((Vr,
                            Vt / R,
                             0 / m - Dr / m - g + Vt ** 2 / R,
                             0 / m - Dt / m - (Vr * Vt) / R,
                             -np.sqrt(0 ** 2 + 0 ** 2) / (g0 * Isp)))

            flag_notgood=True
        return dxdt

      #errore lo devo mettere dopo solve_ivp

    tin = 0.0
    teval = np.linspace(0.0, tfin, int(tfin*3))
    if flag_offdesign is True:
        x_ini = xnew_ini
        tin = change_time
        teval = t_eval2

    sol = solve_ivp(sys, [tin, tfin], x_ini, t_eval=teval)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y3 = sol.y[2, :]
    y4 = sol.y[3, :]
    y5 = sol.y[4, :]
    tt = sol.t

    if tt[-1] < tfin:
        flag = True
        penalty[10] = abs(tt[-1] - tfin)


    r = Rfun(tt)
    theta = Thetafun(tt)
    vr = Vrfun(tt)
    vt = Vtfun(tt)
    m = mfun(tt)

    #penalty[11] = abs(y1[-1] - r[-1]) / obj.Htarget
    #penalty[12] = penalty[12] + (abs(y2[-1] - theta[-1]))/0.8


    err1 = (r - y1) / obj.Htarget
    err2 = (theta - y2) / 0.8
    err3 = (vr - y3)/obj.Vtarget
    err4 = (vt - y4)/obj.Vtarget
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
    IAE = np.zeros((5, len(err1)))
    j = 0
    for a, b, c, d, e, n in zip(err1, err2, err3, err4, err5, step):
        IAE[0][j] = n * abs(a)
        IAE[1][j] = n * abs(b)
        IAE[2][j] = n * abs(c)
        IAE[3][j] = n * abs(d)
        IAE[4][j] = n * abs(e)
        j = j + 1

    fitness = [sum(IAE[0]), sum(IAE[1]), sum(IAE[2]), sum(IAE[3]), sum(IAE[4])]
    if flag_notgood is True:
        return [10000,10000,10000,10000,10000]

    if flag is True:
        x = np.array([fitness[0] + (penalty[0] + penalty[1] + penalty[4])/fitness[0],
             #fitness[1] + ((max(penalty[6:11])) + penalty[-1]),
             fitness[1] + (penalty[-1] + penalty[0] + penalty[1] + penalty[5])/fitness[1],
             fitness[2] + (penalty[4] + penalty[0] + penalty[1] + penalty[5] + penalty[2]+ penalty[3])/fitness[2],
             #fitness[3] + max(penalty[6:11]) + penalty[5],
             fitness[3]  + (penalty[5] + penalty[0]+ penalty[1] + penalty[4] + penalty[2] + penalty[3])/fitness[3],
             fitness[4] + (penalty[2]+ penalty[3])/fitness[4]])

    return x if flag is True else fitness



####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2, name="Mul")
pset.addPrimitive(TriAdd, 3)
#pset.addPrimitive(operator.truediv, 2, name="Div")
#pset.addPrimitive(operator.pow, 2, name="Pow")
pset.addPrimitive(Abs, 1)
pset.addPrimitive(Tanh, 1)
#pset.addPrimitive(Div, 2)                      #rallentamento per gli ndarray utilizzati
pset.addPrimitive(Sqrt, 1)
#pset.addPrimitive(Log, 1)
#pset.addPrimitive(Exp, 1)
#pset.addPrimitive(Sin, 1)
#pset.addPrimitive(Cos, 1)
#pset.addTerminal(np.pi, "pi")
#pset.addTerminal(np.e, name="nap")                   #e Napier constant number
#pset.addTerminal(2)
for i in range(nEph):
    pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-500, 500), 6))

pset.renameArguments(ARG0='errR')
pset.renameArguments(ARG1='errTheta')
pset.renameArguments(ARG2='errVr')
pset.renameArguments(ARG3='errVt')
pset.renameArguments(ARG4='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-0.5, -0.2, -15.0, -0.08, -2.0))    # MINIMIZATION OF THE FITNESS FUNCTION
creator.create("Individual", list, fitness=creator.Fitness, height=1)
creator.create("SubIndividual", gp.PrimitiveTree, fitness=creator.Fitness)
toolbox = base.Toolbox()
# toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)   #### OLD ####
toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=2)  ### NEW ###
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
#toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 10e20))
#toolbox.register('evaluate', evaluate, toolbox=toolbox, sourceData=data, minTrades=minTrades, log=False) ###NEW ###

# toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1, fitness_first=True) ### OLD ###
#toolbox.register("select", xselDoubleTournament, fitness_size=2, parsimony_size=1.8, fitness_first=True) ### NEW ###
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", xmate) ### NEW ###
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3) ### NEW ###
toolbox.register("mutate", xmut, expr=toolbox.expr_mut) ### NEW ###
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