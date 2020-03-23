'''GP_Goddard_2Controls: intelligent control of goddard rocket with 2 controls

This script generates the best control law that can solve the maximum amount of gust and Cd cases

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk

References:
relevant references for the algorithm
[1] Entropy-Driven Adaptive Representation. J. P. Rosca. Proceedings of the Workshop on Genetic Programming: From Theory to Real-World Applications, 23-32. 1995
[2] Exploration and Exploitation in Evolutionary Algorithms: a Survey. M. Crepinsek, S.Liu, M. Mernik. ACM Computer Surveys. 2013
[3] Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms: A Survey of the State of the Art. C. A. Coello Coello. Computer Methods in Applied Mechanics and Engineering 191. 2002'''

from scipy.integrate import solve_ivp, simps
import numpy as np
import operator
import random
from deap import gp, base, creator, tools
import multiprocessing
from scipy.interpolate import PchipInterpolator
from time import strftime
from functools import partial
import sys
import os
import pickle
import GP_funs as funs
import matplotlib.pyplot as plt

sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")

import GP_PrimitiveSet as gpprim

######################### SAVE DATA #############################################
timestr = strftime("%Y%m%d-%H%M%S")
flag_save = True
if flag_save:
    os.makedirs("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Results/2CDelta_NoLearning_1000it_Res_{}_{}".format(
        os.path.basename(__file__), timestr))
    savefig_file = "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Results/2CDelta_NoLearning_1000it_Res_{}_{}/Plot_".format(
        os.path.basename(__file__), timestr)
    savedata_file = "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Results/2CDelta_NoLearning_1000it_Res_{}_{}/".format(
        os.path.basename(__file__), timestr)



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
        self.a = [-0.0065, 0, 0.0010, 0.0028, 0, -0.0020, -0.0040, 0]
        self.a90 = [0.0030, 0.0050, 0.0100, 0.0200, 0.0150, 0.0100, 0.0070]
        self.hv = [11000, 20000, 32000, 47000, 52000, 61000, 79000, 90000]
        self.h90 = [90000, 100000, 110000, 120000, 150000, 160000, 170000, 190000]
        self.tmcoeff = [180.65, 210.65, 260.65, 360.65, 960.65, 1110.65, 1210.65]
        self.pcoeff = [0.16439, 0.030072, 0.0073526, 0.0025207, 0.505861E-3, 0.36918E-3, 0.27906E-3]
        self.tcoeff2 = [2.937, 4.698, 9.249, 18.11, 12.941, 8.12, 5.1]
        self.tcoeff1 = [180.65, 210.02, 257.0, 349.49, 892.79, 1022.2, 1103.4]
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
mutpb = 0.7
cxpb = 0.2

limit_height = 8  # Max height (complexity) of the controller law
limit_size = 20  # Max size (complexity) of the controller law

nbCPU = multiprocessing.cpu_count()

tref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/time.npy")
total_time_simulation = tref[-1]
tfin = tref[-1]

Rref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/R.npy")
Thetaref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Theta.npy")
Vrref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Vr.npy")
Vtref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Vt.npy")
mref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/m.npy")
Ttref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Tt.npy")
Trref = np.load("/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/2Controls/Tr.npy")

Rfun = PchipInterpolator(tref, Rref)
Thetafun = PchipInterpolator(tref, Thetaref)
Vrfun = PchipInterpolator(tref, Vrref)
Vtfun = PchipInterpolator(tref, Vtref)
mfun = PchipInterpolator(tref, mref)
Ttfun = PchipInterpolator(tref, Ttref)
Trfun = PchipInterpolator(tref, Trref)


################################# M A I N ###############################################


def initPOP1():
    global old_hof
    # this function outputs the first n individuals of the hall of fame of the first GP run
    res = old_hof.shuffle()
    # for i in range(10):
    #   res.append(hof[0])
    return res


def main():
    global tfin, flag, n, old_hof
    global size_gen, size_pop, Mu, Lambda, mutpb, cxpb, Trfun, Ttfun
    global Rfun, Thetafun, Vrfun, Vtfun, mfun

    flag = False

    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)
    old_entropy = 0
    for i in range(200):
        pop = funs.POP(toolbox.population(n=size_pop_tot))
        if pop.entropy > old_entropy and len(pop.indexes) == len(pop.categories) - 1:
            best_pop = pop.items
            old_entropy = pop.entropy

    #if nt > 0:
     #   pop2 = toolbox.popx()
      #  for ind in pop2:
       #     del ind.fitness.values
        #best_pop = pop2 + best_pop
    # else:
    # pop = toolbox.population(n=size_pop)
    #   if pop.entropy > old_entropy and len(pop.indexes) == len(pop.categories) - 1:
    #      best_pop = pop.items
    #     old_entropy = pop.entropy
    hof = funs.HallOfFame(10)

    print("INITIAL POP SIZE: %d" % size_pop_tot)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", funs.Min)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   ###################################

    pop, log = funs.eaMuPlusLambdaTol(best_pop, toolbox, Mu, Lambda, size_gen, 1.0, mutpb, cxpb, limit_size, stats=mstats, halloffame=hof,
                                 verbose=True)
    ####################################################################################################################

    pool.close()
    pool.join()
    return pop, log, hof


##################################  F I T N E S S    F U N C T I O N    ##############################################


def evaluate(individual):
    global flag
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun, Ttfun
    global tfin, t_eval2, penalty, fit_old, mutpb, cxpb, Cd_new

    penalty = []

    flag = False

    fTr = toolbox.compileR(expr=individual[0])
    fTt = toolbox.compileT(expr=individual[1])

    def sys(t, x, Cd_new, change_time):
        global penalty, flag, flag_offdesign
        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if t >= change_time:
            Cd = Cd_new
        else:
            Cd = obj.Cd

        if R < obj.Re - 0.5 or np.isnan(R):
            penalty.append((R - obj.Re) / obj.Htarget)
            R = obj.Re
            flag = True

        if m < obj.M0 - obj.Mp or np.isnan(m):
            penalty.append((m - (obj.M0 - obj.Mp)) / obj.M0)
            m = obj.M0 - obj.Mp
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
        # em = mf - m

        rho = obj.air_density(R - obj.Re)
        Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
        Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
        g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
        g0 = obj.g0
        Isp = obj.Isp

        Tr = Trfun(t) + fTr(er, evr)
        Tt = Ttfun(t) + fTt(et, evt)

        if np.iscomplex(Tr):
            flag = True
            Tr = 0
        elif Tr < 0.0 or np.isnan(Tr):
            penalty.append((Tr) / obj.Tmax)
            Tr = 0.0
            flag = True
        elif Tr > obj.Tmax or np.isinf(Tr):
            penalty.append((Tr - obj.Tmax) / obj.Tmax)
            Tr = obj.Tmax
            flag = True
        if np.iscomplex(Tt):
            flag = True
            Tt = 0
        elif Tt < 0.0 or np.isnan(Tt):
            penalty.append((Tt) / obj.Tmax)
            Tt = 0.0
            flag = True
        elif Tt > obj.Tmax or np.isinf(Tt):
            penalty.append((Tt - obj.Tmax) / obj.Tmax)
            Tt = obj.Tmax
            flag = True

        dxdt = np.array((Vr,
                         Vt / R,
                         Tr / m - Dr / m - g + Vt ** 2 / R,
                         Tt / m - Dt / m - (Vr * Vt) / R,
                         -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))
        return dxdt
    max_fit1 = 0
    max_fit2 = 0
    max_pen = 0

    sol = solve_ivp(partial(sys, Cd_new=Cd_new, change_time=t_change), [t_change, tfin], x_ini)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    tt = sol.t

    if tt[-1] < tfin:
        flag = True
        penalty.append((tt[-1] - tfin) / tfin)

    r = Rfun(tt)
    theta = Thetafun(tt)
    err1 = (r - y1) / obj.Htarget
    err2 = (theta - y2) / 0.9

    fitness1 = simps(abs(err1), tt)  # np.sqrt(sum(err1**2)) #+ sum(err3**2))#simps(abs(err1), tt)
    fitness2 = simps(abs(err2), tt)  # np.sqrt(sum(err2**2))# + sum(err4**2))#simps(abs(err2), tt)
    if fitness1 > max_fit1:
        max_fit1 = fitness1
        max_fit2 = fitness2
        if penalty != []:
            max_pen = np.sqrt(sum(np.array(penalty) ** 2))

    if max_fit1 > max_fit2:
        use = max_fit1
    else:
        use = max_fit2

    if flag is True:
        x = [use, max_pen]
        return x
    else:
        return [use, 0.0]


####################################    P R I M I T I V E  -  S E T     ################################################

psetR = gp.PrimitiveSet("Radial", 2)
psetR.addPrimitive(operator.add, 2, name="Add")
psetR.addPrimitive(operator.sub, 2, name="Sub")
psetR.addPrimitive(operator.mul, 2, name='Mul')
psetR.addPrimitive(gpprim.TriAdd, 3)
psetR.addPrimitive(np.tanh, 1, name="Tanh")
psetR.addPrimitive(gpprim.Sqrt, 1)
psetR.addPrimitive(gpprim.Log, 1)
psetR.addPrimitive(gpprim.modExp, 1)
psetR.addPrimitive(gpprim.Sin, 1)
psetR.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    psetR.addEphemeralConstant("randR{}".format(i), lambda: round(random.uniform(-10, 10), 6))

psetR.renameArguments(ARG0='errR')
psetR.renameArguments(ARG1='errVr')
# psetR.renameArguments(ARG2='errm')


psetT = gp.PrimitiveSet("Tangential", 2)
psetT.addPrimitive(operator.add, 2, name="Add")
psetT.addPrimitive(operator.sub, 2, name="Sub")
psetT.addPrimitive(operator.mul, 2, name='Mul')
psetT.addPrimitive(gpprim.TriAdd, 3)
psetT.addPrimitive(np.tanh, 1, name="Tanh")
psetT.addPrimitive(gpprim.Sqrt, 1)
psetT.addPrimitive(gpprim.Log, 1)
psetT.addPrimitive(gpprim.modExp, 1)
psetT.addPrimitive(gpprim.Sin, 1)
psetT.addPrimitive(gpprim.Cos, 1)

for i in range(nEph):
    psetT.addEphemeralConstant("randT{}".format(i), lambda: round(random.uniform(-10, 10), 6))

psetT.renameArguments(ARG0='errTheta')
psetT.renameArguments(ARG1='errVt')
# psetT.renameArguments(ARG2='errm')

################################################## TOOLBOX #############################################################

creator.create("Fitness", funs.FitnessMulti,
               weights=(-1.0, -1.0))  # , -0.1, -0.08, -1.0))    # MINIMIZATION OF THE FITNESS FUNCTION
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", gp.PrimitiveTree)

toolbox = base.Toolbox()

toolbox.register("exprR", gp.genFull, pset=psetR, type_=psetR.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("exprT", gp.genFull, pset=psetT, type_=psetT.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("legR", tools.initIterate, creator.SubIndividual, toolbox.exprR)  ### NEW ###
toolbox.register("legT", tools.initIterate, creator.SubIndividual, toolbox.exprT)  ### NEW ###
toolbox.register("legs", tools.initCycle, list, [toolbox.legR, toolbox.legT], n=1)  ### NEW ###
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)  ### NEW ###
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###
toolbox.register("popx", tools.initIterate, list, initPOP1)
toolbox.register("compileR", gp.compile, pset=psetR)
toolbox.register("compileT", gp.compile, pset=psetT)
toolbox.register("evaluate", evaluate)
toolbox.register("select", funs.InclusiveTournament)
# toolbox.register("select", xselDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True) ### NEW ###
# toolbox.register("select", tools.selNSGA2) ### NEW ###
toolbox.register("mate", funs.xmate)  ### NEW ##
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)  ### NEW ###
toolbox.register("mutate", funs.xmut, expr=toolbox.expr_mut, unipb=0.6, shrpb=0.2, psetR=psetR, psetT=psetT)  ### NEW ###

toolbox.decorate("mate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))

toolbox.decorate("mate", funs.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", funs.staticLimit(key=len, max_value=limit_size))

########################################################################################################################


def sys2GP_wind(t, x, expr1, expr2, v_wind, height_start, delta):
    fTr = toolbox.compileR(expr1)
    fTt = toolbox.compileT(expr2)

    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    if height_start < R < height_start + delta:
        Vt = Vt - v_wind*np.cos(theta)
        Vr = Vr - v_wind * np.sin(theta)

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt

    Tr = Trfun(t) + fTr(er, evr)
    Tt = Ttfun(t) + fTt(et, evt)
    if m <= obj.M0 - obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp
    if Tr > obj.Tmax:
        Tr = obj.Tmax
    elif Tr < 0:
        Tr = 0.0
    if Tt > obj.Tmax:
        Tt = obj.Tmax
    elif Tt < 0:
        Tt = 0.0
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

def sys2GP_Cd(t, x, expr1, expr2, Cd_new, t_change):
    fTr = toolbox.compileR(expr1)
    fTt = toolbox.compileT(expr2)

    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    if t >= t_change:
        Cd = Cd_new
    else:
        Cd = obj.Cd

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt

    Tr = Trfun(t) + fTr(er, evr)
    Tt = Ttfun(t) + fTt(et, evt)
    if m <= obj.M0 - obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp
    if Tr > obj.Tmax:
        Tr = obj.Tmax
    elif Tr < 0:
        Tr = 0.0
    if Tt > obj.Tmax:
        Tt = obj.Tmax
    elif Tt < 0:
        Tt = 0.0
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt

def sys2GP_Cd_noContr(t, x, Cd_new, t_change):

    R = x[0]
    theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

    if t >= t_change:
        Cd = Cd_new
    else:
        Cd = obj.Cd

    Tr = Trfun(t)
    Tt = Ttfun(t)
    if m <= obj.M0 - obj.Mp:
        Tr = 0.0
        Tt = 0.0
        m = obj.M0 - obj.Mp
    if Tr > obj.Tmax:
        Tr = obj.Tmax
    elif Tr < 0:
        Tr = 0.0
    if Tt > obj.Tmax:
        Tt = obj.Tmax
    elif Tt < 0:
        Tt = 0.0
    rho = obj.air_density(R - obj.Re)
    Dr = 0.5 * rho * Vr * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    Dt = 0.5 * rho * Vt * np.sqrt(Vr ** 2 + Vt ** 2) * Cd * obj.A  # [N]
    g = obj.g0 * (obj.Re / R) ** 2  # [m/s2]
    g0 = obj.g0
    Isp = obj.Isp

    dxdt = np.array((Vr,
                     Vt / R,
                     Tr / m - Dr / m - g + Vt ** 2 / R,
                     Tt / m - Dt / m - (Vr * Vt) / R,
                     -np.sqrt(Tt ** 2 + Tr ** 2) / g0 / Isp))

    return dxdt

obj = Rocket()
nt = 0
stats = []
r_diff = []
theta_diff = []
ntot = 100
stats.append(["Cd new", "time change", "Delta eval", "Real time eval"])
cases = []
laws = []
successes = []
s = 0
size_gen = 300  # Gen size
size_pop_tot = 300
Mu = int(size_pop_tot)
Lambda = int(size_pop_tot * 1.4)
DATA = []
### First case and law evaluation ###
while nt < ntot:
    t_change = random.uniform(20, 200)
    if t_change < 80:
        Cd_new = random.uniform(0.61, 1)
    else:
        Cd_new = random.uniform(0.61, 2)
    cases.append(["Cd new", "Time change [s]"])
    cases.append([round(Cd_new, 2), round(t_change,2)])
    x_ini = [Rfun(t_change), Thetafun(t_change), Vrfun(t_change), Vtfun(t_change), mfun(t_change)]
    pop, log, hof = main()
    laws.append(hof[-1])
    tev = np.linspace(t_change, tfin, 10000)
    res = solve_ivp(partial(sys2GP_Cd, expr1=hof[-1][0], expr2=hof[-1][1], Cd_new=Cd_new, t_change=t_change), [t_change, tfin], x_ini, t_eval=tev)
    res_noC = solve_ivp(partial(sys2GP_Cd_noContr, Cd_new=Cd_new, t_change=t_change), [t_change, tfin], x_ini, t_eval=tev)
    save = False
    if (Rref[-1] - obj.Re) * 0.99 < (res.y[0, -1] - obj.Re) < (Rref[-1] - obj.Re) * 1.01 and Thetaref[-1] * 0.99 < \
            res.y[1, -1] < Thetaref[-1] * 1.01:  # tolerance of 1%
        print('Success')
        if (Rref[-1] - obj.Re) * 0.99 < (res_noC.y[0, -1] - obj.Re) < (Rref[-1] - obj.Re) * 1.01 and Thetaref[
            -1] * 0.99 < res_noC.y[1, -1] < Thetaref[-1] * 1.01:  # tolerance of 1%
            print('Controller Useless')
        else:
            print("Controller Useful")
            save = True
    else:
        print('Failure')

    if save:
        database = []
        t_tot = np.linspace(0.0, tfin, 10000)
        eR = np.zeros(len(t_tot))
        eT = np.zeros(len(t_tot))
        eVR = np.zeros(len(t_tot))
        eVT = np.zeros(len(t_tot))
        R = Rfun(t_tot)
        T = Thetafun(t_tot)
        VR = Vrfun(t_tot)
        VT = Vtfun(t_tot)
        TR = Trfun(t_tot)
        TT = Ttfun(t_tot)

        er = Rfun(res.t) - res.y[0, :]
        et = Thetafun(res.t) - res.y[1, :]
        evr = Vrfun(res.t) - res.y[2, :]
        evt = Vtfun(res.t) - res.y[3, :]
        fTr = toolbox.compileR(hof[-1][0])
        fTt = toolbox.compileT(hof[-1][1])
        Tr_ref = Trfun(res.t)
        Tt_ref = Ttfun(res.t)
        Tr = Tr_ref + fTr(er, evr)
        Tt = Tt_ref + fTt(et, evt)

        for i in range(len(res.t) - 1):
            eR[-1 - i] = er[-1 - i]
            eT[-1 - i] = et[-1 - i]
            eVR[-1 - i] = evr[-1 - i]
            eVT[-1 - i] = evt[-1 - i]
            R[-1 - i] = res.y[0, -1 - i]
            T[-1 - i] = res.y[1, -1 - i]
            VR[-1 - i] = res.y[2, -1 - i]
            VT[-1 - i] = res.y[3, -1 - i]
            TR[-1 - i] = Tr[-1 - i]
            TT[-1 - i] = Tt[-1 - i]

        dataset = np.column_stack((R.T, T.T, VR.T, VT.T, eR.T, eT.T, eVR.T, eVT.T, TR.T, TT.T))
        DATA.append(dataset)
        nt += 1
np.save('Dataset_GoddardForLSTM_GPlaw_manySamples.npy', DATA)
np.save('Dataset_GoddardForLSTM_GPlaw_manySamples_cases.npy', cases)
'''plt.figure(1)
plt.plot(res.t, er, label='er')
plt.plot(res.t, et, label='et')
plt.plot(res.t, evr, label='evr')
plt.plot(res.t, evt, label='evt')
plt.legend(loc='best')

plt.figure(2)
plt.plot(res.t, Tr, label='Tr')
plt.plot(res.t, Tt, label='Tt')
plt.plot(res.t, Tr_ref, label='Tr ref')
plt.plot(res.t, Tt_ref, label='Tt ref')
plt.legend(loc='best')

plt.show(block=True)'''




