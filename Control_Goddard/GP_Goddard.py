import tkinter as tk
import time
from scipy.integrate import solve_ivp
import numpy as np
#import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
import operator
import pygraphviz as pgv
import random
from deap import gp
import matplotlib.pyplot as plt
import sys
import timeit
import pandas as pd
from functools import reduce
from operator import add, itemgetter
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
from scipy.interpolate import PchipInterpolator
from functools import partial

def Div(left, right):
    with np.errstate(divide='ignore',invalid='ignore'):
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
    RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
        return left


def Sqrt(x):
    try:
        if x > 0:
            return math.sqrt(x)
        else:
            return abs(x)
    except (
    RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def Log(x):
    try:
        if x > 0:
            return math.log(x)
        else:
            return abs(x)
    except (
    RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def Exp(x):
    try:
        return math.exp(x)
    except (
    RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def Sin(x):
    try:
        return math.sin(x)
    except (
    RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0


def xmate(ind1, ind2):
    i1 = random.randrange(len(ind1))
    i2 = random.randrange(len(ind2))
    print(ind1[i1], ind2[i2])
    ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    return ind1, ind2


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset=pset)
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


# SET IN THIS FUNCTION THE FORM OF DESIDERED SETPOINT (COMMAND)

def setpoint(t,stat):

     if setpoint_type=='Constant':
        r = 2
     elif setpoint_type=='Sinusoidal':
         r = np.sin(t)
     elif setpoint_type == 'Square':
        if (t > 4 and t < 8) or (t > 12 and t < 16):
            r = 8  # SQUARE WAVE
        else:
            r = 3
     elif setpoint_type == 'Custom':
          r = 3*abs(np.sin(t))

     return r


start = timeit.default_timer()

###############################  S I S T E M - P A R A M E T E R S  ####################################################

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


rise_time = 0.1                                                                             # rising time [s]

# FIND MAX VALUE OF POSITION SET POINT

tempo = np.linspace(1e-05, 400, 1e3)
maxsetpoint = np.zeros(len(tempo), dtype='float')
ii = 0

'''for i in tempo:
    maxsetpoint[ii] = setpoint(i,setpoint_type)
    ii = ii + 1
r_max = np.amax(maxsetpoint)'''


old = 0

size_pop = 200                                                                             # Pop size
size_gen = 5                                                                              # Gen size
Mu = int(size_pop)
Lambda = int(size_pop * 1.5)

limit_height = 17                                                                   # Max height (complexity) of the controller law
limit_size = 400                                                                    # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()

def info_sys(a0, a1, a2, rise_time):
    print("\n")

    print("M = %.2f [Kg]\nC = %.2f [Ns/m]\nK = %.2f [N/m]" % (a0, a1, a2))
    sn = np.sqrt(a2 / a0)  # NON DUMPING NATURAL PULSATION
    print("s_n = %.2f [rad/s]" % sn)

    C_cr = 2 * np.sqrt(a2 * a0)  # CRITICAL DAMPING RATIO
    zeta = a1 / C_cr  # ADIMENSIONAL DAMPING RATIO

    if zeta < 1:
        print("zeta = %.2f:  Under-damped system" % zeta)
    elif zeta > 1:
        print("zeta = %.2f:  Over-damped system" % zeta)
    elif zeta == 1:
        print("zeta = %.2f:  Critically damped system" % zeta)

    print("\n")

    print("Rising time defined by user: %.2f [s]" % rise_time)

    return "\n"

################################# M A I N     D E L    P R O G R A M M A ###############################################

def main():
    global size_gen, size_pop, Mu, Lambda
    global size_pop
    global a0, a1, a2, rise_time, vr, yr, ar
    global stat_evoo
    global Rfun, Thetafun, Vrfun, Vtfun, mfun
    global tfin

    Rref = np.load("R.npy")
    Thetaref = np.load("Theta.npy")
    Vrref = np.load("Vr.npy")
    Vtref = np.load("Vt.npy")
    mref = np.load("m.npy")
    tref = np.load("time.npy")
    tfin = tref[-1]

    for i in range(len(Rref)-4):
        if tref[i] == tref[i-1]:
            Rref = np.delete(Rref, i)
            Thetaref = np.delete(Thetaref, i)
            Vrref = np.delete(Vrref, i)
            Vtref = np.delete(Vtref, i)
            mref = np.delete(mref, i)
            tref = np.delete(tref, i)


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
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height, size=stats_size)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

    # pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, 0.6, 0.2, size_gen, stats=mstats, halloffame=hof,
                                         # verbose=True)  ### OLD ###
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=Mu, lambda_=Lambda, cxpb=0.6, mutpb=0.2, ngen=size_gen,
                              stats=mstats, halloffame=hof, verbose=True)  ### NEW ###

    ####################################################################################################################

    '''stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)

    gen = log.select("gen")
    fit_max = log.chapters["fitness"].select('max')

    perform = []
    p = 0
    for items in fit_max:
        perform.append(fit_max[p][0])
        p = p + 1

    size_avgs = log.chapters["size"].select("avg")
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, perform, "b-", label="Maximum Fitness Performance")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    textstr = ('Total Running Time:\n  %dh %dm %.3fs' % (hours, mins, secs))
    ax1.text(0.65, 0.9, textstr, transform=ax1.transAxes, fontsize=10,
             horizontalalignment='right')

    plt.savefig('Stats')
    plt.show()

    print("\n")
    print("THE BEST VALUE IS:")
    print(hof[0])
    print("\n")
    print("THE HEIGHT OF THE BEST INDIVIDUAL IS:")
    print(hof[0].height)
    print("\n")
    print("THE SIZE OF THE BEST INDIVIDUAL IS:")
    print(len(hof[0]))

   
    value = toolbox.evaluate(hof[0])
    print("THE EVALUATION OF THE BEST INDIVIDUAL IS:")
    print(value)
    print("\n")

    expr = hof[0]
    nodes, edges, labels = gp.graph(expr)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw("tree.png")

    image = plt.imread('tree.png')
    fig, ax = plt.subplots()
    im = ax.imshow(image)
    ax.axis('off')
    plt.show()

    sys.stdout.write("TOTAL RUNNING TIME:\n %d (h):%d (m):%.3f (s) \n" % (hours, mins, secs))'''

    #################################### P O S T - P R O C E S S I N G #################################################

    x_ini = [1e-05, 1e-05, 1e-05]  # initial conditions

    acc = []
    time_acc = []
    controller_value = []

    def sys2GP(t, x):
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]
        Tr = setpoint(t, setpoint_type)
        Tt = setpoint(t, setpoint_type)

        e = r - R

        rho = obj.air_density(R - obj.Re)
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) \
             * obj.Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) \
             * obj.Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp
        dxdt = np.zeros(Nstates)
        dxdt[0] = Vr
        dxdt[1] = Vt / R
        dxdt[2] = Tr / m - Dr / m - g + Vt ** 2 / R
        dxdt[3] = Tt / m - Dt / m - (Vr * Vt) / R
        dxdt[4] = - np.sqrt(Tr ** 2 + Tt ** 2) / g0 / Isp

        return [dxdt[0], dxdt[1], dxdt[2], dxdt[3], dxdt[4]]

    tevals = np.linspace(1e-05, 20, 100000)

    #solgp = solve_ivp(sys2GP, [1e-05, 20], x_ini, first_step=0.0001, t_eval=tevals)
    #ygp = solgp.y[0, :]
    #dyy = solgp.y[1, :]
    #ttgp = solgp.t
    #acc_gp = np.array(acc)
    #tt_gp = np.array(time_acc)
    #controller_value = np.array(controller_value)

    #rrr = np.zeros(len(ttgp), dtype='float')

    #ii = 0
    #for i in ttgp:
    #    rrr[ii] = setpoint(i,setpoint_type)
    #    ii = ii + 1

    #errgp = rrr - ygp  # Error system with genetic programming

    '''fig, ax2 = plt.subplots()
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position [m]")
    plt.plot(ttgp, ygp, label="GENETIC PROGRAMMING")
    plt.plot(ttgp, rrr, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('Position plot.png')
    plt.show()'''

    # PRINT PLOT WITH POSITION, VELOCITY, ACCELERATION, ERROR INFO OF THE CONTROLLED SYSTEM BY GENETIC PROGRAMMING CONTROLLER LAW

    '''fig1, ax3 = plt.subplots()
    plt.ylim(-r_max, 2 * r_max)
    ax3.set_xlabel("time [s]")
    plt.plot(ttgp, ygp, label="POSITION [m]")
    plt.plot(ttgp, errgp, label="ERROR [m]")
    plt.plot(ttgp, dyy, label="VELOCITY [m/s]")
    #plt.plot(tt_gp, acc_gp, label="ACCELERATION [m/s^2]")
    plt.plot(ttgp, rrr, 'r--', label="SET POINT [m]")
    plt.legend(loc="lower right")
    plt.savefig('Motion graphs.png')
    plt.show()'''

    # Backup Python Console data in a Excel file

    '''chapter_keys = log.chapters.keys()
    sub_chaper_keys = [c[0].keys() for c in log.chapters.values()]

    data = [list(map(itemgetter(*skey), chapter)) for skey, chapter
            in zip(sub_chaper_keys, log.chapters.values())]
    data = np.array([[*a, *b, *d] for a, b, d in zip(*data)])

    columns = reduce(add, [["_".join([x, y]) for y in s]
                           for x, s in zip(chapter_keys, sub_chaper_keys)])
    df = pd.DataFrame(data, columns=columns)

    keys = log[0].keys()
    data = [[d[k] for d in log] for k in keys]
    for d, k in zip(data, keys):
        df[k] = d
    df.to_csv('Evolution process.csv', encoding='utf-8', index=False)'''

    # HISTORY
    # CAUTION: if the size of pop or gen is high it can take so much time !

    # plt.figure(figsize=(18,18),dpi=500)
    # graph = nx.DiGraph(history.genealogy_tree)
    # graph = graph.reverse()
    # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    # positions = graphviz_layout(graph, prog="dot")
    # nx.draw_networkx_labels(graph, positions)
    # nx.draw_networkx_nodes(graph, positions, node_color=colors)
    # nx.draw_networkx_edges(graph, positions)
    # plt.savefig('HISTORY.png',dpi=500)
    # plt.show()
    '''print("\n")
    print(
        "During the evolution, the number of: \n - not feasibile individuals were:      %d, %d percent\n - individuals who exceed the constrain for all simulation time were:         %d, %d percent\n - individuals who exceed the constrain only for a part of simulation time were:         %d, %d percent \n - correct individuals were:       %d, %d percent \n" % (
        stat_evoo[0], (stat_evoo[0] / sum(stat_evoo)) * 100, stat_evoo[1], (stat_evoo[1] / sum(stat_evoo)) * 100,
        stat_evoo[2], (stat_evoo[2] / sum(stat_evoo)) * 100, stat_evoo[3], (stat_evoo[3] / sum(stat_evoo)) * 100))'''


    pool.close()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################
flag = False
pas = False
stat_evoo = [0, 0, 0, 0]
#max_force = 0
fitnnesoldvalue = 0
fitness_old=1e10

def evaluate(individual):
    global flag
    global pas
    global old
    global zeta
    global fitnnesoldvalue,fitness_old
    global rise_time
    global ar, vr, yr
    global stat_evoo
    global Rfun, Thetafun, Vrfun, Vtfun, mfun
    global tfin

    old = 0

    flag = False
    pas = False

    # Transform the tree expression in a callable function

    fTr= toolbox.compile(expr=individual[0])
    fTt = toolbox.compile(expr=individual[1])
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]  # initial conditions
    #max_force = 0
    #force_constraint = a1 * vr + a2 * yr + a0 * ar

    def sys(t, x):

        global oldTr, oldTt
        global flag

        # State Variables

        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]
        if R<0:
            R = obj.Re
            flag = True
        if m<0:
            m = obj.M0-obj.Mp
            flag = True
        elif m>obj.M0:
            m = obj.M0
            flag = True
        if abs(Vr) > 1e4:
            if Vr > 0:
                Vr = 1e4
                flag = True
            else:
                Vr = -1e4
                flag = True
        if abs(Vt) > 1e4:
            if Vt > 0:
                Vt = 1e4
                flag = True
            else:
                Vt = -1e4
                flag = True
        if np.isinf(R):
            R = np.nan_to_num(R)/1e100
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
        #print(R, Vr, Vt, m)

        if abs(fTr(er, et, evr, evt, em)) > obj.Tmax and not (np.isinf(fTr(er, et, evr, evt, em)) or np.isnan(fTr(er, et, evr, evt, em)) or np.iscomplex(fTr(er, et, evr, evt, em))):
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
            dxdt[2] = obj.Tmax / m - Dr / m - g + Vt ** 2 / R
            dxdt[3] = fTt(er, et, evr, evt, em) / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(obj.Tmax ** 2 + fTt(er, et, evr, evt, em) ** 2) / g0 / Isp
            flag = True

        elif abs(fTt(er, et, evr, evt, em)) > obj.Tmax and not (np.isinf(fTt(er, et, evr, evt, em)) or np.isnan(fTt(er, et, evr, evt, em)) or np.iscomplex(fTt(er, et, evr, evt, em))):
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
            dxdt[3] = obj.Tmax / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(fTr(er, et, evr, evt, em) ** 2 + obj.Tmax ** 2) / g0 / Isp
            flag = True

        elif np.isinf(fTr(er, et, evr, evt, em)) or np.isnan(fTr(er, et, evr, evt, em)) or np.iscomplex(fTr(er, et, evr, evt, em)):
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
            dxdt[2] = np.nan_to_num(fTr(er, et, evr, evt, em)) / m - Dr / m - g + Vt ** 2 / R
            dxdt[3] = fTt(er, et, evr, evt, em) / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(1 ** 2 + fTt(er, et, evr, evt, em) ** 2) / g0 / Isp
            flag = True

        elif np.isinf(fTt(er, et, evr, evt, em)) or np.isnan(fTt(er, et, evr, evt, em)) or np.iscomplex(fTt(er, et, evr, evt, em)):
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
            dxdt[3] = np.nan_to_num(fTt(er, et, evr, evt, em)) / m - Dt / m - (Vr * Vt) / R
            dxdt[4] = - np.sqrt(fTr(er, et, evr, evt, em) ** 2 + 1 ** 2) / g0 / Isp
            flag = True

        else:
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
            oldTr = fTr(er, et, evr, evt, em)
            oldTt = fTt(er, et, evr, evt, em)

        return [dxdt[0], dxdt[1], dxdt[2], dxdt[3], dxdt[4]]

    sol = solve_ivp(sys, [0.0, tfin], x_ini, first_step=0.0001)
    yy = sol.y[0, :]
    #dyy = sol.y[1, :]
    # yi = sol.y[2,:]
    tt = sol.t
    pp = 0
    r = np.zeros(len(tt), dtype='float')
    for i in tt:
        r[pp] = Rfun(i)
        pp += 1

    err1 = r - yy

    # STEP TIME SIZE
    i = 0
    pp = 1
    step = np.zeros(len(yy), dtype='float')
    step[0] = 0.0001
    while i < len(tt) - 1:
        step[pp] = tt[i + 1] - tt[i]
        i = i + 1
        pp = pp + 1

    # INTEGRAL OF ABSOLUTE ERROR (PERFORMANCE INDEX)
    IAE = np.empty(len(yy), dtype='float')
    j = 0
    alpha = 0.1
    for p, n in zip(err1, step):
        IAE[j] = n * abs(p)# + alpha * abs(m))
        j = j + 1


    # PENALIZING INDIVIDUALs
    #For the stats if the multiprocessing is used, there could be problems to print the correct values (parallel process(?))

    if flag is True:
        pas = True
        x = np.random.uniform(fitness_old * 1.5, fitness_old * 1.6)
        stat_evoo[0] += 1

    if flag is False:
        stat_evoo[3] += 1
        fitness = np.sum(IAE)
        if fitness < fitness_old:
            fitness_old=fitness

    return (x,) if pas is True else (fitness,)

####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2,name="Add")
pset.addPrimitive(operator.sub, 2,name="Sub")
pset.addPrimitive(Mul, 2)
#pset.addPrimitive(Div, 2)                      #rallentamento per gli ndarray utilizzati
pset.addPrimitive(Sqrt, 1)
pset.addPrimitive(Log, 1)
pset.addPrimitive(Exp, 1)
pset.addPrimitive(Sin, 1)
pset.addTerminal(np.pi,"pi")
pset.addTerminal(np.e,name="nap")                   #e Napier constant number
pset.addTerminal(2)
pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-10, 10), 1))
#pset.addEphemeralConstant("rand101",lambda: random.uniform(-5, 5))
pset.renameArguments(ARG0='errR')
pset.renameArguments(ARG1='errTheta')
pset.renameArguments(ARG2='errVr')
pset.renameArguments(ARG3='errVt')
pset.renameArguments(ARG4='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0,))    # MINIMIZATION OF THE FITNESS FUNCTION

creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

creator.create("SubIndividual", gp.PrimitiveTree, fitness=creator.Fitness, arity=5)

toolbox = base.Toolbox()
# toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)   #### OLD ####
toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=4)  ### NEW ###

toolbox.register("leg", tools.initIterate, creator.SubIndividual, toolbox.expr)  ### NEW ###
toolbox.register("legs", tools.initRepeat, list, toolbox.leg, n=Ncontrols)  ### NEW ###

#toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)  ### NEW ###

# toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###

toolbox.register("lambdify", gp.compile, pset=pset) ### NEW ###
toolbox.register("stringify", gp.compile, pset=pset) ### NEW ###

toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evaluate) ### OLD ###
#toolbox.register('evaluate', evaluate, toolbox=toolbox, sourceData=data, minTrades=minTrades, log=False) ###NEW ###

# toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1, fitness_first=True) ### OLD ###
toolbox.register("select", xselDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=True) ### NEW ###

toolbox.register("mate", xmate) ### NEW ###
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4) ### NEW ###
toolbox.register("mutate", xmut, expr=toolbox.expr_mut) ### NEW ###

# toolbox.register("mate", gp.cxOnePointLeafBiased,termpb=0.1) ### OLD ###
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset) ### OLD ###

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
    print(hof.items[0][0])
    print(hof.items[0][1])
