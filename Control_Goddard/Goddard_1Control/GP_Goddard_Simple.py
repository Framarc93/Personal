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

def top_endstop(t,top_end_stop):
    return top_end_stop  # STOP RUNNING [m]
def bot_endstop(t,bottom_end_stop):
    return bottom_end_stop

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

top_end_stop = 80  # [km]
bottom_end_stop = 0.0  # [km]
mutpb = 0.3
cxpb = 0.6
change_time = 100
size_pop = 70 # Pop size
size_gen = 15  # Gen size
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

    toolbox.register("map", pool.map)
    toolx.register("map", pool.map)

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
        pop, log = algorithms.eaMuPlusLambda(pop, toolx, Mu, Lambda, mutpb, cxpb, size_gen, stats=mstats, halloffame=hof, verbose=True)  ### OLD ###
    else:
        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, Mu, Lambda, mutpb, cxpb, size_gen, stats=mstats, halloffame=hof, verbose=True)
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
    '''fig, ax1 = plt.subplots()
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
    plt.show()'''

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

    '''expr1 = hof[0]

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
    x_ini = [obj.Re, 0.0, obj.M0]  # initial conditions

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
    plt.show()'''

    pool.close()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ################################################


def evaluate(individual):
    global flag, flag_offdesign
    global pas
    global fitnnesoldvalue, fitness_old1, fitness_old2, fitness_old3
    global Rfun, Vfun, mfun
    global tfin
    flagDeath = False
    flag = False
    pas = False

    # Transform the tree expression in a callable function
    if flag_offdesign is True:
        fT = toolx.compile(expr=individual)
    else:
        fT = toolbox.compile(expr=individual)

    x_ini = [obj.Re, 0.0, obj.M0]  # initial conditions

    def sys(t, x):
        # State Variables
        R = x[0]
        V = x[1]
        m = x[2]

        if R < 0 or np.isnan(R):
            R = obj.Re
            flag = True
        if np.isinf(R) or R > obj.Re+80e3:
            R = obj.Re + 80e3
            flag = True
        if m < obj.M0*obj.Mc or np.isnan(m):
            m = obj.M0*obj.Mc
            flag = True
        elif m > obj.M0 or np.isinf(m):
            m = obj.M0
            flag = True
        if abs(V) > 1e3 or np.isinf(V):
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
        dxdt = np.zeros(Nstates)
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

    tin = 0.0
    if flag_offdesign is True:
        x_ini = xnew_ini
        tin = change_time
    sol = solve_ivp(sys, [tin, tfin], x_ini)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y3 = sol.y[2, :]
    tt = sol.t

    if sol.t[-1] != tfin:
        flagDeath = True

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

    if flagDeath is True:
        y = [1e5, 1e5, 1e5]
        return y

    elif flag is True:
        x = [np.random.uniform(fitness_old1 * 1.5, fitness_old1 * 1.6),
             np.random.uniform(fitness_old2 * 1.5, fitness_old2 * 1.6),
             np.random.uniform(fitness_old3* 1.5, fitness_old3 * 1.6)]
        return x

    else:
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
        return fitness



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

creator.create("Fitness", base.Fitness, weights=(-0.5, -0.5, -1.0))  # MINIMIZATION OF THE FITNESS FUNCTION

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

toolx = base.Toolbox()
toolx.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)   #### OLD ####

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

########################################################################################################################


if __name__ == "__main__":
    obj = Rocket()
    pop, log, hof = main()

Cd_old = obj.Cd


print("\n ADD SOME ALTERATIONS TO PHYSICAL COMPONENTS OF THE PLANT AT %.2f [s]" % change_time)
print("WRITE THE NUMBER FOR THE PARAMETER THAT YOU WANT CHANGE: Cd ( 1 )")
flag = int(input())
if flag == 1:
    obj.Cd = float(input("CHANGE VALUE OF THE DRAG COEFFICIENT: "))


x_ini = [obj.Re, 0.0, obj.M0]  # initial conditions

def sys2GP(t, x):
    global Cd_old
    fT = toolbox.compile(hof[0])
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
    drag = 0.5 * rho * V ** 2 * Cd_old * obj.area

    g = obj.GMe / R ** 2
    g0 = obj.g0
    Isp = obj.Isp

    T = fT(er, ev, em)

    dxdt[0] = V
    dxdt[1] = (T-drag)/m-g
    dxdt[2] = - T / g0 / Isp

    return dxdt
passint = tfin*5
tevals = np.linspace(0.0, tfin, int(passint))

solgp = solve_ivp(sys2GP, [0.0, tfin], x_ini, t_eval=tevals)
rout = solgp.y[0, :]
vout = solgp.y[1, :]
mout = solgp.y[2, :]
ttgp = solgp.t
rR = np.zeros(len(ttgp), dtype='float')
vR = np.zeros(len(ttgp), dtype='float')
mR = np.zeros(len(ttgp), dtype='float')

ii = 0
for i in ttgp:
    rR[ii] = Rfun(i)
    vR[ii] = Vfun(i)
    mR[ii] = mfun(i)
    ii = ii + 1
errgp = rR - rout

tes = np.zeros(len(ttgp), dtype='float')                        #TOP END STOP

ii = 0
for i in ttgp:
    tes[ii] = top_endstop(i, top_end_stop)
    ii = ii + 1

bes = np.zeros(len(ttgp), dtype='float')                        #BOTTOM END STOP

ii = 0
for i in ttgp:
    bes[ii] = bot_endstop(i, bottom_end_stop)
    ii = ii + 1

plt.ion()
plt.figure(1)
plt.plot(ttgp, (rR - obj.Re) / 1e3, 'r--', label="SET POINT")
animated_plot = plt.plot(ttgp, (rout - obj.Re) / 1e3, 'ro', label="ON DESIGN")[0]
plt.figure(2)
plt.plot(ttgp, vR, 'r--', label="SET POINT")
animated_plot2 = plt.plot(ttgp, vout, 'ro', label="ON DESIGN")[0]

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
    animated_plot2.set_ydata(vout[0:i])
    plt.draw()
    # plt.pause(0.1)
    plt.pause(0.00000001)
    i = i + 1

u_design = hof[0]
print(u_design)
#####################################################################################################################

start = time()
flag_gpnew = True  # POSSO METTERE I PARAMETRI CHE MI PIACCIONO SELEZIONANDOLI CON UN FLAG
if __name__ == "__main__":
    xnew_ini = [float(rout[index]), float(vout[index]), float(mout[index])]
    flag_seed_populations = True
    flag_offdesign = True
    size_pop, size_gen, cxpb, mutpb = 70, 10, 0.6, 0.2
    Mu = int(size_pop)
    Lambda = int(size_pop * 1.4)
    pop, log, hof = main()
end = time()
t_offdesign = end - start  # CALCOLO TEMPO IMPIEGATO DAL GENETIC PROGRAMMING



#########################################################################################################################
def sys2GP_c(t, x):
    global u_design
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


passint_c = (change_time + t_offdesign - (change_time)) * 5
tevals_c = np.linspace(change_time, change_time + t_offdesign, int(passint_c))
xnew_ini = [float(rout[index]), float(vout[index]), float(mout[index])]                              ################Mi servono questi PENSARE

solgp_c = solve_ivp(sys2GP_c, [change_time, change_time + t_offdesign], xnew_ini, t_eval=tevals_c)

rout_c = solgp_c.y[0, :]
vout_c = solgp_c.y[1, :]
mout_c = solgp_c.y[2, :]
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
animated_plot_c2 = plt.plot(ttgp_c, vout_c, 'bo', label="OFF DESIGN")[0]

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
    animated_plot_c2.set_ydata(vout_c[0:i])
    plt.draw()
    # plt.pause(0.1)
    plt.pause(0.00000001)
    i = i + 1

##################################################################################################################


# Simulazione per TEMPO CON NUOVA LEGGE creata dal GENETIC PROGRAMMING

passint_gp = (total_time_simulation - (change_time + t_offdesign)) * 10
tevals_gp = np.linspace(change_time + t_offdesign, total_time_simulation, int(passint_gp))
xnew_ini_gp = [float(rout_c[index_c]), float(vout_c[index_c]), float(mout_c[index_c])]

def sys2GP_gp(t, x):

    fT = toolx.compile(hof[0])
    R = x[0]
    V = x[1]
    m = x[2]

    r = Rfun(t)
    v = Vfun(t)
    mf = mfun(t)

    er = r - R
    ev = v - V
    em = mf - m
    dxdt = np.zeros(3)

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


solgp_gp = solve_ivp(sys2GP_gp, [change_time + t_offdesign, total_time_simulation], xnew_ini_gp, t_eval=tevals_gp)

rout_gp = solgp_gp.y[0, :]
vout_gp = solgp_gp.y[1, :]
mout_gp = solgp_gp.y[2, :]
ttgp_gp = solgp_gp.t

plt.ion()
plt.figure(1)
animated_plot_gp = plt.plot(ttgp_gp, (rout_gp - obj.Re) / 1e3, 'go', label="ONLINE CONTROL")[0]
plt.figure(2)
animated_plot_gp2 = plt.plot(ttgp_gp, vout_gp, 'go', label="ONLINE CONTROL")[0]

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
    animated_plot_gp2.set_ydata(vout_gp[0:i])
    #if items==change_time+t_offdesign:
     #   plt.legend(loc='best')
    plt.draw()
    plt.pause(0.00000001)

    i = i + 1

print("\n")
print(u_design)
print(hof[0])
plt.show(block=True)