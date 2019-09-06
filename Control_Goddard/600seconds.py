#import tkinter as tk
from scipy.integrate import solve_ivp
import numpy as np
from numpy import *
#import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
#import pygraphviz as pgv
import random
from deap import gp
import matplotlib.pyplot as plt
import sys
import timeit
import time
import pandas as pd
from functools import reduce
from operator import *
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
import matplotlib.animation as animation
from matplotlib import style
import itertools
from functools import wraps
from time import time
#from numba import jit

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def Div(a, b):
    try:
        return truediv(a, b)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return a


def Pow(a, b):
    try:
        return pow(a, b)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return a


def Sqrt(a):
    try:
        if a > 0:
            return sqrt(a)
        else:
            return abs(a)
    except (
            RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return 1


def Log(a):
    try:
        if a > 0:
            return np.log(a)
        else:
            return abs(a)
    except (
            RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return 1


def Exp(a):
    try:
        return exp(a)
    except (
            RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return 1


def Sin(a):
    try:
        return sin(a)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError,
            FloatingPointError, OverflowError):
        return 1


# SET IN THIS FUNCTION THE FORM OF DESIDERED SETPOINT (COMMAND)
def top_endstop(t,top_end_stop):
    return top_end_stop  # STOP RUNNING [m]
def bot_endstop(t,bottom_end_stop):
    return bottom_end_stop
def setpoint(t, setpoint_type):
    ##SET HERE THE VALUES OF POSITIONS DESIDERED IN [m]

    # NOT EXCEED (-5 [m], +9 [m]) in position command for a good 3D-animation

    if setpoint_type == 'Constant':
        r = 4
    elif setpoint_type == 'Sinusoidal':
        r = np.sin(t)
    elif setpoint_type == 'Square':
        if (t > 4 and t < 8) or (t > 12 and t < 16):
            r = 4  # SQUARE WAVE
        else:
            r = 2
    elif setpoint_type == 'Custom':
        r = 3 * abs(np.sin(t))
    elif setpoint_type == "Harmonic":
        F, omega = 1.5, 1.5  # set a costant value for amplitude and frequency
        r = F * np.sin(omega * t)
    elif setpoint_type == "Multistep":
        if (t > 0 and t < 60):
            r = 1
        elif (t > 60 and t < 150):
            r = 2
        elif (t > 150 and t < 200):
            r = 4
        elif (t > 200 and t < 260):
            r = 8
        elif (t > 260 and t < 350):
            r = 3
        elif (t > 350 and t < 450):
            r = 5
        elif (t > 450 and t < 601):
            r = 9

        else:
            r = 0

    return r


start = timeit.default_timer()
###############################  S I S T E M - P A R A M E T E R S  ####################################################
a0 = 1  # MASS      M  [Kg]
a1 = 10  # DAMPING   C  [Ns/m]
a2 = 20  # STIFNESS  K  [N/m]

top_end_stop=10
bottom_end_stop = -1

setpoint_type = 'Multistep'  # Costant, Square, Sinusoidal, Harmonic, Custom,Multistep

rise_time = 0.1  # rising time [s]

total_time_simulation = 600  # total time simulation expressed in seconds [s]

# FIND MAX VALUE OF POSITION SET POINT

tempo = np.linspace(1e-05, total_time_simulation, 1000)

set_point = np.zeros(len(tempo), dtype='float')
ii = 0

for i in tempo:
    set_point[ii] = setpoint(i, setpoint_type)
    ii = ii + 1
r_max = np.amax(set_point)
r_min = np.amin(set_point)

if r_max > top_end_stop:
    print("MAXIMUM POSITION CONSENTED: %d [m]" % top_end_stop)
    print("COMMAND NOT GRANTED by the presence of the end stop")
if r_min < bottom_end_stop:
    print("MINIMUM POSITION CONSENTED: %d [m]" % bottom_end_stop)
    print("COMMAND NOT GRANTED by the presence of the end stop")
    sys.exit(0)

# CALCULATE THE MAX VALUES OF VELOCITY, ACCELERATION AND POSITION OF PLANT TO FIND THE MAX VALUE OF FORCE DEVELOPED BY CONTROLLER (LINEAR MOTION)
ar = 2 * r_max / (rise_time ** 2)
vr = ar * rise_time
yr = 0.9 * r_max

old = 0

size_pop = 30  # Pop size
size_gen = 1  # Gen size
Mu = int(size_pop)
Lambda = int(size_pop * 1.5)
pb_cross = 0.5  # crossover_probability
pb_mut = 0.4  # mutation_probability
limit_height = 17  # Max height (complexity) of the controller law
limit_size = 400  # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()


# CHARACTERISTIC OF THE PLANT WITHOUT CONTROLLER ACTION (u(t)=0)
def info_sys(a0, a1, a2, rise_time, setpoint_type):
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

    print("\n")

    print("Setpoint Type: %s " % setpoint_type)

    return "\n"


flag_seed_populations = False
flag_seed_populations1 = False
flag_offdesign = False


################################# M A I N     D E L    P R O G R A M M A ###############################################
def main():

    pool = multiprocessing.Pool(nbCPU)
    global size_gen, size_pop, Mu, Lambda, pb_mut, pb_cross, flag_seed_populations, flag_seed_populations1, flag_offdesign
    global a0, a1, a2, rise_time, vr, yr, ar
    global setpoint_type


    toolbox.register("map", pool.map)
    toolx.register("map", pool.map)

    print(info_sys(a0, a1, a2, rise_time, setpoint_type))

    print("Maximum force developed by controller: %.2f [N]" % (a1 * vr + a2 * yr + a0 * ar))
    print("\n")

    print("INITIAL POP SIZE: %d" % size_pop)

    print("GEN SIZE: %d" % size_gen)

    print("\n")

    pop = toolbox.population(n=size_pop)
    history.update(pop)
    hof = tools.HallOfFame(size_gen)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height, size=stats_size)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################
    if flag_seed_populations is True:
        pop[1] = u_design
        flag_seed_populations = False
    if flag_offdesign is True:
        pop, log = algorithms.eaMuPlusLambda(pop, toolx, Mu, Lambda, pb_cross, pb_mut, size_gen, stats=mstats, halloffame=hof, verbose=True)
    else:
        #pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, pb_cross, pb_mut, size_gen, stats=mstats, halloffame=hof, verbose=True)
        pop, log = algorithms.eaSimple(pop, toolbox, pb_cross, pb_mut, size_gen, stats=mstats, halloffame=hof, verbose=True)

    ####################################################################################################################
    if flag_seed_populations1 is True:
        pop[1] = u_design
        flag_seed_populations1 = False

    '''stop = timeit.default_timer()
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours, mins = divmod(mins, 60)

    gen = log.select("gen")
    fit_min = log.chapters["fitness"].select('min')

    perform = []
    p = 0
    for items in fit_min:
        perform.append(fit_min[p][0])
        p = p + 1

    height_avgs= log.chapters["height"].select("avg")
    size_avgs = log.chapters["size"].select("avg")
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, perform, "b-", label="Minimum Fitness Performance")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    line3 = ax2.plot(gen, height_avgs, "g-", label="Average Height")
    ax2.set_ylabel("Size, Height", color="y")
    for tl in ax2.get_yticklabels():
        tl.set_color("y")

    lns = line1 + line2 + line3
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

    with open("Control Law.txt", "w") as file:
        file.write(str(hof[0]))

    with open("Best individuals.txt", "w") as bestever:
        for i in range(size_gen):
            bestever.write(str(hof[i]))
            bestever.write("\n \n")

    print("\n")

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

    '''
    fig, ax2 = plt.subplots()
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position [m]")
    plt.plot(ttgp, ygp, label="GENETIC PROGRAMMING")
    plt.plot(ttgp, rrr, 'r--', label="SET POINT")
    plt.legend(loc="lower right")
    plt.savefig('Position plot.png')
    plt.show()

    # PRINT PLOT WITH POSITION, VELOCITY, ACCELERATION, ERROR INFO OF THE CONTROLLED SYSTEM BY GENETIC PROGRAMMING CONTROLLER LAW

    fig1, ax3 = plt.subplots()
    plt.ylim(-r_max, 2 * r_max)
    ax3.set_xlabel("time [s]")
    plt.plot(ttgp, ygp, label="POSITION [m]")
    plt.plot(ttgp, errgp, label="ERROR [m]")
    plt.plot(ttgp, dyy, label="VELOCITY [m/s]")
    #plt.plot(tt_gp, acc_gp, label="ACCELERATION [m/s^2]")
    plt.plot(ttgp, rrr, 'r--', label="SET POINT [m]")
    plt.legend(loc="lower right")
    plt.savefig('Motion graphs.png')
    plt.show()

    # Backup Python Console data in a Excel file

    chapter_keys = log.chapters.keys()
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
    df.to_csv('Evolution process.csv', encoding='utf-8', index=False)

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
    print("\n")
    print(
        "During the evolution, the number of: \n - not feasibile individuals were:      %d, %d percent\n - individuals who exceed the constrain for all simulation time were:         %d, %d percent\n - individuals who exceed the constrain only for a part of simulation time were:         %d, %d percent \n - correct individuals were:       %d, %d percent \n" % (
        stat_evoo[0], (stat_evoo[0] / sum(stat_evoo)) * 100, stat_evoo[1], (stat_evoo[1] / sum(stat_evoo)) * 100,
        stat_evoo[2], (stat_evoo[2] / sum(stat_evoo)) * 100, stat_evoo[3], (stat_evoo[3] / sum(stat_evoo)) * 100))

#######################################################################################################################

   # with open('Control Law.txt') as f:
    #    eqn = f.read().strip()
     #   s, e, t = symbols("s e t")
      #  err, dot_err, int_err = symbols("err dot_err int_err")

       # der = Function("diff")(e, t)
        #int = Function("integrate")(e, t)

        #eqn1 = eval(eqn)

        #a = eqn1.subs(err, e)
        #b = a.subs(dot_err, der)
        #final = b.subs(int_err, int)

    #with open('a.txt', "w") as fff:
     #   fff.write(str(final))
    #with open('a.txt') as ff:
       # finals = ff.read()
      #  e = Function('e')(t)
     #   laplaceq = sympify(finals)

    #print("USE THIS EQUATION TO CALCULATE LAPLACE FORM (second script LAPLACE FORM):")
    #print(laplaceq)
'''

    #######################################################################################################################

    pool.close()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################
flag = False
flag2 = False
flag3 = False
pas = False

fitnnesoldvalue = 0
fitness_old = 10e10


def evaluate(individual):
    global flag
    global flag2
    global flag3
    global pas
    global old
    global zeta
    global fitnnesoldvalue, fitness_old
    global rise_time, total_time_simulation
    global ar, vr, yr

    old = 0

    flag = False
    flag2 = False
    flag3 = False
    pas = False

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    x_ini = [1e-05, 1e-05, 1e-05]  # initial conditions
    force_constraint = a1 * vr + a2 * yr + a0 * ar

    def sys(t, x):

        global a2, a0, a1
        global old
        global flag
        global flag2
        global flag3
        global rise_time

        # State Variables

        y = x[0]  # x1 POSITION
        dydt = x[1]  # x2 VELOCITY
        dyi = x[2]  # x3 ERROR

        dxdt = np.asarray([1e-05, 1e-05, 1e-05], dtype="float")

        fix = old

        r = setpoint(t, setpoint_type)

        e = r - y

        # Save max value force controller during all time simultation for the whole evolution
        # if func(e,-dydt) > max_force:
        #   max_force = func(e,-dydt)

        if abs(func(e, -dydt, dyi)) > abs(force_constraint) and not (
                np.isinf(func(e, -dydt, dyi)) or np.isnan(func(e, -dydt, dyi)) or np.iscomplex(func(e, -dydt, dyi))):
            dxdt[0] = dydt
            dxdt[1] = (- a1 * dydt - a2 * y + fix) / a0
            dxdt[2] = e

            flag2 = True

        elif np.isinf(func(e, -dydt, dyi)) or np.isnan(func(e, -dydt, dyi)) or np.iscomplex(func(e, -dydt, dyi)):
            dxdt[0] = dydt
            dxdt[1] = (- a1 * dydt - a2 * y + fix) / a0
            dxdt[2] = e

            flag = True


        else:
            dxdt[0] = dydt
            dxdt[1] = (- a1 * dydt - a2 * y + func(e, -dydt, dyi)) / a0
            dxdt[2] = e
            old = func(e, -dydt, dyi)

            flag3 = True

        return [dxdt[0], dxdt[1], dxdt[2]]

    passintt = (total_time_simulation) * 4
    tev = np.linspace(1e-05, total_time_simulation, passintt)
    sol = solve_ivp(sys, [1e-05, total_time_simulation], x_ini, first_step=0.001, t_eval=tev)
    yy = sol.y[0, :]
    dyy = sol.y[1, :]
    # yi = sol.y[2,:]
    tt = sol.t
    pp = 0
    r = np.zeros(len(tt), dtype='float')

    for i in tt:
        r[pp] = setpoint(i, setpoint_type)
        pp = pp + 1

    err1 = r - yy

    # STEP TIME SIZE
    i = 0
    pp = 1
    step = np.zeros(len(yy), dtype='float')
    step[0] = 0.001
    while i < len(tt) - 1:
        step[pp] = tt[i + 1] - tt[i]
        i = i + 1
        pp = pp + 1

    # INTEGRAL OF ABSOLUTE ERROR (PERFORMANCE INDEX)
    IAE = np.empty(len(yy), dtype='float')
    j = 0
    alpha = 0.1
    for p, m, n in zip(err1, dyy, step):
        IAE[j] = n * (abs(p) + alpha * abs(m))
        j = j + 1

    # PENALIZING INDIVIDUALs
    # For the stats if the multiprocessing is used, there could be problems to print the correct values (parallel process(?))

    if flag is True:
        pas = True
        x = +1000

    if flag2 is True and flag3 is False:
        pas = True
        x = np.random.uniform(fitness_old * 1.5, fitness_old * 1.6)

    if flag3 is True and flag2 is True:
        pas = True
        x = np.random.uniform(fitness_old * 1.2, fitness_old * 1.3)

    if flag3 is True and flag2 is False:
        fitness = np.sum(IAE)
        if fitness < fitness_old:
            fitness_old = fitness

    return (x,) if pas is True else (fitness,)


####################################    P R I M I T I V E  -  S E T     ################################################
pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)
pset.addPrimitive(abs, 1)
# pset.addPrimitive(Div, 2)
# pset.addPrimitive(Pow, 2)
# pset.addPrimitive(Log, 1)
# pset.addPrimitive(Sqrt, 1)
# pset.addPrimitive(Sin, 1)
# pset.addPrimitive(tanh,1)
# pset.addPrimitive(sqrt,1)-
pset.addTerminal(np.pi, name="pi")
pset.addTerminal(np.e, name="e")  # Napier constant number
pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-5, 5), 1))
# pset.addEphemeralConstant("rand102", lambda: random.randint(5, 5))
# pset.addEphemeralConstant("rand103", lambda: random.randint(10, 20))
# pset.addEphemeralConstant("rand104", lambda: random.randint(50, 100)
pset.renameArguments(ARG0='err')
pset.renameArguments(ARG1='d_err')
pset.renameArguments(ARG2='i_err')
################################################## TOOLBOX #############################################################
creator.create("Fitness", base.Fitness, weights=(-1.0,))  # MINIMIZATION THE FITNESS FUNCTION
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1, fitness_first=True)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))
history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

toolx = base.Toolbox()
toolx.register("expr", gp.genFull, pset=pset, min_=2, max_=5)
toolx.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolx.register("population", tools.initRepeat, list, toolbox.individual)
toolx.register("compile", gp.compile, pset=pset)
toolx.register("evaluate", evaluate)
toolx.register("select", tools.selTournament, tournsize=2)
toolx.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolx.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolx.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=limit_height))
toolx.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=limit_height))
toolx.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
toolx.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))
history = tools.History()
toolx.decorate("mate", history.decorator)
toolx.decorate("mutate", history.decorator)

########################################################################################################################
########################################################################################################################
########################################################################################################################

if __name__ == "__main__":
    pop, log, hof = main()
a0_old=a0
a1_old=a1
a2_old=a2

change_time = 90  # [s]
print("\n ADD SOME ALTERATIONS TO PHYSICAL COMPONENTS OF THE PLANT AT %.2f [s]" % change_time)
print("WRITE THE NUMBER FOR THE PARAMETER THAT YOU WANT CHANGE: MASS ( 1 ), SPRING ( 2 ), DAMPER ( 3 )")
flag = int(input())
if flag == 1:
    a0 = float(input("CHANGE VALUE OF THE MASS in [Kg] : "))
elif flag == 2:
    a1 = float(input("CHANGE VALUE OF THE SPRING in [N/m] : "))
elif flag == 3:
    a2 = float(input("CHANGE VALUE OF THE DAMPER in [Ns/m] : "))
else:
    print("ERROR, please digit 1 for mass, 2 for spring or 3 for damper")
    sys.exit[0]

########################################################################################################################

x_ini = [1e-05, 1e-05, 1e-05]  # initial conditions


def sys2GP(t, x):
    global a0_old, a1_old, a2_old
    M, C, K = a0_old, a1_old, a2_old
    # State Variables
    y = x[0]  # x1 POSITION
    dydt = x[1]  # x2 VELOCITY
    ei = x[2]  # x3 ACCELERATION
    dxdt = [1e-05, 1e-05, 1e-05]

    r = setpoint(t, setpoint_type)
    e = r - y

    ctrl = toolbox.compile(hof[0])
    u = ctrl(e, -dydt, ei)

    dxdt[1] = (- C * dydt - K * y + u) / M
    dxdt[0] = dydt
    dxdt[2] = e

    return [dxdt[0], dxdt[1], dxdt[2]]


passint = (total_time_simulation) * 4
tevals = np.linspace(1e-05, total_time_simulation, passint)

solgp = solve_ivp(sys2GP, [1e-05, total_time_simulation], x_ini, first_step=0.001, t_eval=tevals)
# solgp = solve_ivp(sys2GP, [1e-05, 300], x_ini)
ygp = solgp.y[0, :]
dyy = solgp.y[1, :]
ttgp = solgp.t

rrr = np.zeros(len(ttgp), dtype='float')

ii = 0
for i in ttgp:
    rrr[ii] = setpoint(i, setpoint_type)
    ii = ii + 1

tes = np.zeros(len(ttgp), dtype='float')                        #TOP END STOP

ii = 0
for i in ttgp:
    tes[ii] = top_endstop(i, top_end_stop)
    ii = ii + 1
errgp = rrr - ygp  # Error system with genetic programming

bes = np.zeros(len(ttgp), dtype='float')                        #BOTTOM END STOP

ii = 0
for i in ttgp:
    bes[ii] = bot_endstop(i, bottom_end_stop)
    ii = ii + 1



plt.ion()
plt.plot(ttgp, rrr, 'r--', label="SET POINT [m]")
plt.plot(ttgp, tes, 'k', label="TOP END STOP  [m]")
plt.plot(ttgp, bes, 'k', label="BOTTOM END STOP  [m]")
animated_plot = plt.plot(ttgp, ygp, 'ro',label="ON DESIGN")[0]
#######             GRAFICO PER TEMPO DI IN FASE DI DESIGN      #####
i = 0
for items in ttgp:
    plt.figure(1)
    plt.ylim(bottom_end_stop - 1, top_end_stop +1)
    plt.xlim(0, total_time_simulation)

   # if items<=0.1:
    #    plt.legend(loc='best')
    if items > change_time:
        index, = np.where(ttgp == items)
        break
    animated_plot.set_xdata(ttgp[0:i])
    animated_plot.set_ydata(ygp[0:i])
    plt.draw()
    #plt.pause(0.1)
    plt.pause(0.00000001)
    i = i + 1

u_design = hof[0]
print(u_design)
#####################################################################################################################

start = time()
flag_gpnew = True  # POSSO METTERE I PARAMETRI CHE MI PIACCIONO SELEZIONANDOLI CON UN FLAG
if __name__ == "__main__":
    flag_seed_populations = True
    flag_seed_populations1 = True
    flag_offdesign = True
    size_pop, size_gen, pb_mut, pb_cross =100, 20, 0.6, 0.1
    Mu = int(size_pop)
    Lambda = int(size_pop * 1.5)
    pop, log, hof = main()
end = time()
t_offdesign = end - start  # CALCOLO TEMPO IMPIEGATO DAL GENETIC PROGRAMMING


#########################################################################################################################

def sys2GP_c(t, x):
    global a0, a1, a2, u_design
    M, C, K = a0, a1, a2
    # State Variables
    y = x[0]  # x1 POSITION

    #        with open("example", "a") as file:
    #           file.write("%f , %f \n" %(t,y))

    dydt = x[1]  # x2 VELOCITY
    ei = x[2]  # x3 ACCELERATION
    dxdt = [1e-05, 1e-05, 1e-05]

    r = setpoint(t, setpoint_type)
    e = r - y

    ctrl = toolbox.compile(u_design)
    u = ctrl(e, -dydt, ei)

    dxdt[1] = (- C * dydt - K * y + u) / M
    dxdt[0] = dydt
    dxdt[2] = e

    return [dxdt[0], dxdt[1], dxdt[2]]


passint_c = (change_time + t_offdesign - (change_time)) * 4
tevals_c = np.linspace(change_time, change_time + t_offdesign, passint_c)
xnew_ini = [float(ygp[index]), float(dyy[index]), float(errgp[index])]                              ################Mi servono questi PENSARE

solgp_c = solve_ivp(sys2GP_c, [change_time, change_time + t_offdesign], xnew_ini, first_step=0.001, t_eval=tevals_c)
# solgp_c = solve_ivp(sys2GP_c, [change_time, change_time+t_offdesign], xnew_ini)
ygp_c = solgp_c.y[0, :]
dyy_c = solgp_c.y[1, :]
ttgp_c = solgp_c.t

rrr_c = np.zeros(len(ttgp_c), dtype='float')
ii = 0
for i in ttgp_c:
    rrr_c[ii] = setpoint(i, setpoint_type)
    ii = ii + 1

errgp_c = rrr_c - ygp_c

for tempi in ttgp_c:
    if tempi > t_offdesign:
        index_c, = np.where(ttgp_c == tempi)

plt.ion()
animated_plot_c = plt.plot(ttgp_c, ygp_c, 'bo',label="OFF DESIGN")[0]

i = 0
for items in ttgp_c:
    plt.figure(1)
    animated_plot_c.set_xdata(ttgp_c[0:i])
    animated_plot_c.set_ydata(ygp_c[0:i])
    plt.draw()
    #plt.pause(0.1)
    plt.pause(0.00000001)
    i = i + 1

##################################################################################################################


# Simulazione per TEMPO CON NUOVA LEGGE creata dal GENETIC PROGRAMMING

passint_gp = (total_time_simulation - (change_time + t_offdesign)) * 4
tevals_gp = np.linspace(change_time + t_offdesign, total_time_simulation, passint_gp)
xnew_ini_gp = [float(ygp_c[index_c]), float(dyy_c[index_c]), float(errgp_c[index_c])]

def sys2GP_gp(t, x):
    global a0, a1, a2
    M, C, K = a0, a1, a2
    # State Variables
    y = x[0]  # x1 POSITION
    dydt = x[1]  # x2 VELOCITY
    ei = x[2]  # x3 ACCELERATION
    dxdt = [1e-05, 1e-05, 1e-05]

    r = setpoint(t, setpoint_type)
    e = r - y

    ctrl = toolbox.compile(hof[0])
    u = ctrl(e, -dydt, ei)

    dxdt[1] = (- C * dydt - K * y + u) / M
    dxdt[0] = dydt
    dxdt[2] = e

    return [dxdt[0], dxdt[1], dxdt[2]]
solgp_gp = solve_ivp(sys2GP, [change_time + t_offdesign, total_time_simulation], xnew_ini_gp, first_step=0.001,
                     t_eval=tevals_gp)
# solgp_gp = solve_ivp(sys2GP, [change_time+t_offdesign, total_time_simulation], xnew_ini_gp)
ygp_gp = solgp_gp.y[0, :]
dyy_gp = solgp_gp.y[1, :]
ttgp_gp = solgp_gp.t

plt.ion()
animated_plot_gp = plt.plot(ttgp_gp, ygp_gp, 'go',label="ONLINE CONTROL")[0]

i = 0
for items in ttgp_gp:
    plt.figure(1)
    if items==change_time+t_offdesign:
        plt.legend(loc='best')
    animated_plot_gp.set_xdata(ttgp_gp[0:i])
    animated_plot_gp.set_ydata(ygp_gp[0:i])
    plt.draw()
    #plt.pause(0.1)
    plt.pause(0.00000001)
    i = i + 1

print("\n")
print(u_design)
print(hof[0])

