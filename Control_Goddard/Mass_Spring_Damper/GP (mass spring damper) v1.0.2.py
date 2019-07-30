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


# SET IN THIS FUNCTION THE FORM OF DESIDERED SETPOINT (COMMAND)

def setpoint(t,setpoint_type):

    ##SET HERE THE VALUES OF POSITIONS DESIDERED IN [m]

    #NOT EXCEED THIS RANGE (-5 , +9 [m]) in position command for a good 3D-animation of the system

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
a0 = 1  # MASS      M  [Kg]
a1 = 10  # DAMPING   C  [Ns/m]
a2 = 20  # STIFNESS  K  [N/m]


setpoint_type = 'Custom'                                                                #Costant, Square, Sinusoidal, Custom

rise_time = 0.1                                                                             # rising time [s]

# FIND MAX VALUE OF POSITION SET POINT

tempo = np.linspace(1e-05, 20, 100000)
maxsetpoint = np.zeros(len(tempo), dtype='float')
ii = 0

for i in tempo:
    maxsetpoint[ii] = setpoint(i,setpoint_type)
    ii = ii + 1
r_max = np.amax(maxsetpoint)

# CALCULATE THE MAX VALUES OF VELOCITY, ACCELERATION AND POSITION OF SYSTEM TO FIND THE MAX VALUE OF FORCE DEVELOPED BY CONTROLLER (LINEAR MOTION)
ar = 2 * r_max / (rise_time ** 2)
vr = ar * rise_time
yr = 0.9 * r_max

old = 0

size_pop = 10                                                                             # Pop size
size_gen = 10                                                                              # Gen size
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

    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)

    print(info_sys(a0, a1, a2, rise_time))

    print("Maximum force developed by controller: %.2f [N]" % (a1 * vr + a2 * yr + a0 * ar))
    print("\n")

    print("INITIAL POP SIZE: %d" % size_pop)

    print("GEN SIZE: %d" % size_gen)

    print("\n")

    random.seed()

    pop = toolbox.population(n=size_pop)
    history.update(pop)
    hof = tools.HallOfFame(size_gen)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit, height=stats_height, size=stats_size)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, 0.6, 0.2 , size_gen, stats=mstats, halloffame=hof,
                                         verbose=True)

    ####################################################################################################################

    stop = timeit.default_timer()
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

    sys.stdout.write("TOTAL RUNNING TIME:\n %d (h):%d (m):%.3f (s) \n" % (hours, mins, secs))

    #################################### P O S T - P R O C E S S I N G #################################################

    x_ini = [1e-05, 1e-05, 1e-05]  # initial conditions

    acc = []
    time_acc = []
    controller_value = []

    def sys2GP(t, x):
        global a0, a1, a2
        M, C, K = a0, a1, a2
        # State Variables
        y = x[0]  # x1 POSITION
        dydt = x[1]  # x2 VELOCITY
        ei = x[2]  # x3 ACCELERATION
        dxdt = [1e-05, 1e-05, 1e-05]

        r = setpoint(t,setpoint_type)
        e = r - y

        ctrl = toolbox.compile(hof[0])
        u = ctrl(e, -dydt)

        dxdt[1] = (- C * dydt - K * y + u) / M
        dxdt[0] = dydt
        dxdt[2] = e

        acc.append(dxdt[1])
        time_acc.append(t)
        controller_value.append(u)

        return [dxdt[0], dxdt[1], dxdt[2]]

    tevals = np.linspace(1e-05, 20, 100000)

    solgp = solve_ivp(sys2GP, [1e-05, 20], x_ini, first_step=0.0001, t_eval=tevals)
    ygp = solgp.y[0, :]
    dyy = solgp.y[1, :]
    ttgp = solgp.t
    acc_gp = np.array(acc)
    tt_gp = np.array(time_acc)
    controller_value = np.array(controller_value)

    rrr = np.zeros(len(ttgp), dtype='float')

    ii = 0
    for i in ttgp:
        rrr[ii] = setpoint(i,setpoint_type)
        ii = ii + 1

    errgp = rrr - ygp  # Error system with genetic programming

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


    #################################### 3 D - A N I M A T I O N ###########################################################

    master = tk.Tk()
    master.title("GP Mass-Spring-Damper")

    w_width = 1000
    w_height = 500

    init_pos = [500, 250]
    w = tk.Canvas(master, width=w_width, height=w_height)
    w.pack()

    x0 = init_pos[0]
    y0 = init_pos[1]
    block = w.create_rectangle(x0 - 50, y0 - 50, x0 + 50, y0 + 50, fill="black")
    floor = 350

    ceiling = 50
    w.create_line(50, ceiling, 50, floor, width=3, fill="brown")
    w.create_line(50, floor, w_width - 50, floor, width=3)

    w.create_line(50, y0 + 20, 70, y0 + 20, width=2, fill="red")
    damper2 = w.create_line(70, y0 + 20, 100, y0 + 20, width=20, fill="red")
    damper1 = w.create_line(100, y0 + 20, x0, y0 + 20, width=4)
    pos_old = 0

    base_spring = w.create_line(50, y0 - 20, 70, y0 - 20, width=2, fill="green")
    final_spring = w.create_line(150, y0 - 20, x0, y0 - 20, width=3)
    spring_1 = w.create_line(70, y0 - 20, 250, y0 + 5, width=3)
    spring_2 = w.create_line(250, y0 + 5, 300, y0 - 5, width=3)
    spring_3 = w.create_line(200, y0 - 20, 199, 100, width=3)

    wheel_1 = w.create_oval(0, 0, 0, 0)
    wheel_2 = w.create_oval(0, 0, 0, 0)

    for j in dyy:
        w.move(block, j / 100, 0)
        pos = w.coords(block)

        if pos[2] > pos_old or pos[2] == pos_old:
            w.delete(damper1)
            w.delete(final_spring)
            w.delete(spring_1)
            w.delete(spring_2)
            w.delete(spring_3)
            w.delete(wheel_1)
            w.delete(wheel_2)
            damper1 = w.create_line(100, y0 + 20, pos[0], y0 + 20, width=5, fill="red")
            final_spring = w.create_line(abs(pos[0] - 300), y0 - 20, pos[0], y0 - 20, width=2, fill="green")
            pos_final_spring = w.coords(final_spring)
            wheel_1 = w.create_oval(pos[0], floor - 50, pos[0] + 25, floor, fill="blue")
            wheel_2 = w.create_oval(pos[2] - 25, floor - 50, pos[2], floor, fill="blue")
            spring_1 = w.create_line(pos_final_spring[0] - 10, pos_final_spring[1] + 20, pos_final_spring[0],
                                     pos_final_spring[1], width=3, fill="green")
            pos_s1 = w.coords(spring_1)
            spring_3 = w.create_line(70, y0 - 20, 90, y0 - 40, width=3, fill="green")
            pos_s3 = w.coords(spring_3)
            spring_2 = w.create_line(pos_s3[2], pos_s3[3], pos_s1[0], pos_s1[1], width=3, fill="green")
            pos_s2 = w.coords(spring_2)
            pos_old = pos[2]
        else:
            w.delete(damper1)
            w.delete(final_spring)
            w.delete(spring_1)
            w.delete(spring_2)
            w.delete(spring_3)
            w.delete(wheel_1)
            w.delete(wheel_2)
            final_spring = w.create_line(abs(pos[0] - 300), y0 - 20, pos[0], y0 - 20, width=2, fill="green")
            pos_final_spring = w.coords(final_spring)
            damper1 = w.create_line(100, y0 + 20, pos[0], y0 + 20, width=5, fill="red")
            wheel_1 = w.create_oval(pos[0], floor - 50, pos[0] + 25, floor, fill="blue")
            wheel_2 = w.create_oval(pos[2] - 25, floor - 50, pos[2], floor, fill="blue")
            spring_1 = w.create_line(pos_final_spring[0] - 10, pos_final_spring[1] + 20, pos_final_spring[0],
                                     pos_final_spring[1], width=3, fill="green")
            pos_s1 = w.coords(spring_1)
            spring_3 = w.create_line(70, y0 - 20, 90, y0 - 40, width=3, fill="green")
            pos_s3 = w.coords(spring_3)
            spring_2 = w.create_line(pos_s3[2], pos_s3[3], pos_s1[0], pos_s1[1], width=3, fill="green")
            pos_s2 = w.coords(spring_2)

        master.update()
        time.sleep(.0001)

    pool.close()
    return pop, log, hof, errgp, ygp, dyy, acc_gp, ttgp, controller_value,rrr


##################################  F I T N E S S    F U N C T I O N    ################################################
flag = False
flag2 = False
flag3 = False
pas = False
stat_evoo = [0, 0, 0, 0]
#max_force = 0
fitnnesoldvalue = 0
fitness_old=10e10

def evaluate(individual):
    global flag
    global flag2
    global flag3
    global pas
    global old
    global zeta
    global fitnnesoldvalue,fitness_old
    global rise_time
    global ar, vr, yr
    global stat_evoo
    #global max_force

    old = 0

    flag = False
    flag2 = False
    flag3 = False
    pas = False

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    x_ini = [1e-05, 1e-05, 1e-05]  # initial conditions
    #max_force = 0
    force_constraint = a1 * vr + a2 * yr + a0 * ar #throttle constraints

    def sys(t, x):

        global a2, a0, a1
        global old
        global flag
        global flag2
        global flag3
        global rise_time
        #global max_force

        # State Variables

        y = x[0]  # x1 POSITION
        dydt = x[1]  # x2 VELOCITY
        dyi = x[2]  # x3 ERROR

        dxdt = np.asarray([1e-05, 1e-05, 1e-05], dtype="float")

        fix = old

        r = setpoint(t,setpoint_type) # one for each state

        e = r - y

        #Save max value force controller during all time simultation for the whole evolution
        #if func(e,-dydt) > max_force:
         #   max_force = func(e,-dydt)


        if abs(func(e, -dydt)) > abs(force_constraint) and not (
                np.isinf(func(e, -dydt)) or np.isnan(func(e, -dydt)) or np.iscomplex(func(e, -dydt))):
            dxdt[0] = dydt
            dxdt[1] = (- a1 * dydt - a2 * y + fix)/ a0
            dxdt[2] = e

            flag2 = True

        elif np.isinf(func(e, -dydt)) or np.isnan(func(e, -dydt)) or np.iscomplex(func(e, -dydt)):
            dxdt[0] = dydt
            dxdt[1] = (- a1 * dydt - a2 * y + fix)/ a0
            dxdt[2] = e

            flag = True


        else:
            dxdt[0] = dydt
            dxdt[1] = (- a1 * dydt - a2 * y + func(e, -dydt))/ a0
            dxdt[2] = e
            old = func(e, -dydt)

            flag3 = True

        return [dxdt[0], dxdt[1], dxdt[2]]

    sol = solve_ivp(sys, [1e-05, 20], x_ini, first_step=0.0001)
    yy = sol.y[0, :]
    dyy = sol.y[1, :]
    # yi = sol.y[2,:]
    tt = sol.t
    pp = 0
    r = np.zeros(len(tt), dtype='float')
    for i in tt:
        r[pp] = setpoint(i,setpoint_type)
        pp = pp + 1

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
    for p, m, n in zip(err1, dyy, step):
        IAE[j] = n * (abs(p) + alpha * abs(m))
        j = j + 1


    # PENALIZING INDIVIDUALs
    #For the stats if the multiprocessing is used, there could be problems to print the correct values (parallel process(?))

    if flag is True:
        pas = True
        x = +1000
        stat_evoo[0] += 1

    if flag2 is True and flag3 is False:
        pas = True
        x = np.random.uniform(fitness_old * 1.5, fitness_old * 1.6)
        stat_evoo[1] += 1

    if flag3 is True and flag2 is True:
        pas = True
        x = np.random.uniform(fitness_old * 1.2, fitness_old * 1.3)
        stat_evoo[2] += 1

    if flag3 is True and flag2 is False:
        stat_evoo[3] += 1
        fitness = np.sum(IAE)
        if fitness < fitness_old:
            fitness_old=fitness



    return (x,) if pas is True else (fitness,)

####################################    P R I M I T I V E  -  S E T     ################################################

pset = gp.PrimitiveSet("MAIN", 2)
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
pset.addEphemeralConstant("rand101", lambda: round(random.uniform(-5, 5), 1))
#pset.addEphemeralConstant("rand101",lambda: random.uniform(-5, 5))
pset.renameArguments(ARG0='err')
pset.renameArguments(ARG1='dot_err')

################################################## TOOLBOX #############################################################



creator.create("Fitness", base.Fitness, weights=(-1.0,))    # MINIMIZATION OF THE FITNESS FUNCTION

creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()

# >>>> program part example
toolbox.register("map", map)
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selDoubleTournament, fitness_size=3,parsimony_size=1, fitness_first=True)
toolbox.register("mate", gp.cxOnePointLeafBiased,termpb=0.1)
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
    pop, log, hof, errorGP, position_GP, velocity_GP, acceleration_GP, time_GP, force_controller,setpoint = main()
