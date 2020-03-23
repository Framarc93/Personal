'''G2C_MGGPLawMultiGenerator

This script generates the best MGGP control law that can solve the maximum amount of gust and Cd cases

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
import matplotlib.pyplot as plt
import multiprocessing
from scipy.interpolate import PchipInterpolator
from time import time, strftime
from copy import deepcopy
from functools import partial, wraps
from operator import eq, mul, truediv
from collections import Sequence
import sys
import os
import pickle

laptop = True
if laptop:
    initial_path = "/home/francesco/Desktop/PhD/Git_workspace/IC4A2S"
else:
    initial_path = "/home/francesco/Desktop/Git_workspace/IC4A2S"

sys.path.insert(1, initial_path + "/Goddard_Problem/Testing/Goddard_1Control")
import GP_PrimitiveSet as gpprim
sys.path.insert(2, initial_path + "/Goddard_Problem/Testing/wMGGP")
import my_modFuns as Modfuns
import MGGP_functions as funs

######################### SAVE DATA #############################################
timestr = strftime("%Y%m%d-%H%M%S")
flag_save = True
if flag_save:
    os.makedirs(initial_path + "/Goddard_Problem/Results/2CDelta_NoLearning_1000it_Res_{}_{}".format(
        os.path.basename(__file__), timestr))
    savefig_file = initial_path + "/Goddard_Problem/Results/2CDelta_NoLearning_1000it_Res_{}_{}/Plot_".format(
        os.path.basename(__file__), timestr)
    savedata_file = initial_path + "/Goddard_Problem/Results/2CDelta_NoLearning_1000it_Res_{}_{}/".format(
        os.path.basename(__file__), timestr)


###############################  S Y S T E M - P A R A M E T E R S  ####################################################
def initPOP1():
    global old_hof
    # this function outputs the first n individuals of the hall of fame of the first GP run
    res = old_hof
    # for i in range(10):
    #   res.append(hof[0])
    return res


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

def evaluate_lst(d, individual, Cd_new, delta_eval, change_time, x_init):
    global flag
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun, Ttfun
    global tfin, t_eval2, penalty, fit_old, mutpb, cxpb

    penalty = []
    flag = False

    ws = 1
    eqR = str(d[0]) + "+"
    while ws < len(individual[0].w):
        eqR = eqR + str(d[ws]) + "*" + str(individual[0][ws - 1]) + "+"
        ws += 1
    eqR = list(eqR)
    del eqR[-1]
    eqR = "".join(eqR)

    eqT = str(d[ws]) + "+"
    ws += 1
    while ws < len(individual[1].w)+len(individual[0].w):
        eqT = eqT + str(d[ws]) + "*" + str(individual[1][ws - len(individual[0].w) - 1]) + "+"
        ws += 1
    eqT = list(eqT)
    del eqT[-1]
    eqT = "".join(eqT)


    fTr = toolbox.compileR(eqR)
    fTt = toolbox.compileT(eqT)

    def sys(t, x, change_time, delta_eval, Cd_new):
        global penalty, flag, flag_offdesign, flag_thrust
        # State Variables
        R = x[0]
        theta = x[1]
        Vr = x[2]
        Vt = x[3]
        m = x[4]

        if t >= change_time - delta_eval:
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
        #mf = mfun(t)

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
    tv = np.linspace(change_time, tfin, 200)
    sol = solve_ivp(partial(sys, change_time=change_time, delta_eval=delta_eval, Cd_new=Cd_new), [change_time, tfin], x_init, t_eval=tv, method='RK23')
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    tt = sol.t

    if tt[-1] < tfin:
        flag = True
        penalty.append((tt[-1] - tfin) / tfin)

    r = Rfun(tt)
    th = Thetafun(tt)

    err1 = (r - y1) / obj.Htarget
    err2 = (th - y2) / 0.9

    if flag is True:
        x = err1 + err2 + sum(abs(np.array(penalty))) #np.sqrt(sum(np.array(penalty) ** 2))
        return x
    else:
        return err1 + err2


Ngenes = 1
limit_height = 10
limit_size = 15
nCost = 2
nVars = 2

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

for i in range(nCost):
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

for i in range(nCost):
    psetT.addEphemeralConstant("randT{}".format(i), lambda: round(random.uniform(-10, 10), 6))

psetT.renameArguments(ARG0='errTheta')
psetT.renameArguments(ARG1='errVt')
# psetT.renameArguments(ARG2='errm')


################################################## TOOLBOX #############################################################

d = np.ones((Ngenes+1))*1.0  # weights for linear combination of genes
d[0] = 0

creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)
creator.create("SubIndividual", list, w=list(d), height=1)
creator.create("Trees", gp.PrimitiveTree)

toolbox = base.Toolbox()

toolbox.register("exprR", gp.genFull, pset=psetR, type_=psetR.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("exprT", gp.genFull, pset=psetT, type_=psetT.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("legR", tools.initIterate, creator.Trees, toolbox.exprR)  ### NEW ###
toolbox.register("legT", tools.initIterate, creator.Trees, toolbox.exprT)  ### NEW ###
toolbox.register("legsR", tools.initRepeat, list, toolbox.legR, n=Ngenes)
toolbox.register("legsT", tools.initRepeat, list, toolbox.legT, n=Ngenes)
toolbox.register("SindR", tools.initIterate, creator.SubIndividual, toolbox.legsR)  ### NEW ###
toolbox.register("SindT", tools.initIterate, creator.SubIndividual, toolbox.legsT)  ### NEW ###
toolbox.register("Sind", tools.initCycle, list, [toolbox.SindR, toolbox.SindT], n=1)  ### NEW ###
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.Sind)  ### NEW ###
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###
toolbox.register("popx", tools.initIterate, list, initPOP1)
toolbox.register("compileR", gp.compile, pset=psetR)
toolbox.register("compileT", gp.compile, pset=psetT)
toolbox.register("select", Modfuns.InclusiveTournament)
toolbox.register("mate", funs.xmate, Ngenes=Ngenes) ### NEW ##
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4) ### NEW ###
toolbox.register("mutate", funs.xmut, expr=toolbox.expr_mut, unipb=0.6, shrpb=0.3, Ngenes=Ngenes, psetR=psetR, psetT=psetT)
toolbox.decorate("mate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", funs.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", funs.staticLimit(key=len, max_value=limit_size))


################################# M A I N ###############################################


def main(cxpb, mutpb, x_init):
    nbCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nbCPU)

    toolbox.register('map', pool.map)
    old_entropy = 0
    for i in range(50):
        pop = Modfuns.POP(toolbox.population(n=size_pop))
        if pop.entropy > old_entropy and len(pop.indexes) == len(pop.categories) - 1:
            best_pop = pop.items
            old_entropy = pop.entropy
    if nt > 0:
        pop2 = toolbox.popx()
        for ind in pop2:
            del ind.fitness.values
        best_pop = pop2 + best_pop
    pop = toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, Cd_new=Cd_new, delta_eval=0.0, change_time=change_time, x_ini=x_init), best_pop)  # optimization on pop
    hof = tools.HallOfFame(10)
    hof.update(pop)
    best_ind = funs.selBest(pop, 1)[0]
    best_fit = best_ind.fitness.values[0]
    tol = 0.045
    n = 0
    while n <= Ngen and best_fit > tol:
        #to_mate = funs.selBest(pop, int(len(pop)/2))
        sub_div, good_index = Modfuns.subset_diversity(pop)
        offspring, mutpb, cxpb = Modfuns.varOr(pop, toolbox, int(len(pop)), sub_div, good_index, cxpb, mutpb, limit_size)
        offspring = toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, Cd_new=Cd_new, delta_eval=0.0, change_time=change_time, x_ini=x_init), offspring)  # optimization on all pop
        hof.update(offspring)
        pop[:] = toolbox.select(offspring + pop, len(pop)-1, to_mate=False)
        best_ind = funs.selBest(pop, 1)[0]
        pop.append(best_ind)
        best_fit = best_ind.fitness.values[0]
        print("GEN: {}, BEST FIT: {}".format(n, best_fit))
        n += 1

    print(best_fit)
    best_string1 = str(best_ind[0].w[0])
    st = 1
    while st <= Ngenes:
        best_string1 = best_string1 + "+" + str(best_ind[0].w[st]) + "*" + str(best_ind[0][st - 1])
        st += 1
    best_string2 = str(best_ind[1].w[0])
    st = 1
    while st <= Ngenes:
        best_string2 = best_string2 + "+" + str(best_ind[1].w[st]) + "*" + str(best_ind[1][st - 1])
        st += 1
    print("Tr: ", best_string1)
    print("Tt: ", best_string2)
    pool.close()

    return best_fit, best_string1, best_string2, hof

obj = Rocket()

tref = np.load(initial_path + "/Goddard_Problem/2Controls/time.npy")
total_time_simulation = tref[-1]
tfin = tref[-1]

Rref = np.load(initial_path + "/Goddard_Problem/2Controls/R.npy")
Thetaref = np.load(initial_path + "/Goddard_Problem/2Controls/Theta.npy")
Vrref = np.load(initial_path + "/Goddard_Problem/2Controls/Vr.npy")
Vtref = np.load(initial_path + "/Goddard_Problem/2Controls/Vt.npy")
mref = np.load(initial_path + "/Goddard_Problem/2Controls/m.npy")
Ttref = np.load(initial_path + "/Goddard_Problem/2Controls/Tt.npy")
Trref = np.load(initial_path + "/Goddard_Problem/2Controls/Tr.npy")

Rfun = PchipInterpolator(tref, Rref)
Thetafun = PchipInterpolator(tref, Thetaref)
Vrfun = PchipInterpolator(tref, Vrref)
Vtfun = PchipInterpolator(tref, Vtref)
mfun = PchipInterpolator(tref, mref)
Ttfun = PchipInterpolator(tref, Ttref)
Trfun = PchipInterpolator(tref, Trref)


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
    else:
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

    r = Rfun(t)
    th = Thetafun(t)
    vr = Vrfun(t)
    vt = Vtfun(t)

    er = r - R
    et = th - theta
    evr = vr - Vr
    evt = vt - Vt

    if t >= t_change:
        Cd = Cd_new
        Tr = Trfun(t) + fTr(er, evr)
        Tt = Ttfun(t) + fTt(et, evt)
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


#obj = Rocket()
nt = 0
stats = []
r_diff = []
theta_diff = []
ntot = 5
old_hof = funs.HallOfFame(3)
stats.append(["Cd new", "time change", "Delta eval", "Real time eval"])
cases = []
laws = []
successes = []
s = 0
mutpb = 0.8
cxpb = 0.2
size_pop = 40 - len(old_hof) # Pop size
Ngen = 150  # Gen size
size_pop_tot = 40
Mu = int(size_pop_tot)
Lambda = int(size_pop_tot)

### First case and law evaluation ###

change_time = random.uniform(20, 80)
Cd_new = random.uniform(1.0, 2.0)
cases.append(["Cd new", "Time change [s]"])
cases.append([round(Cd_new, 2), round(change_time,2)])
x_ini = [Rfun(change_time), Thetafun(change_time), Vrfun(change_time), Vtfun(change_time), mfun(change_time)]  # initial conditions

best_fit, best_string1, best_string2, hof = main(cxpb, mutpb, x_ini)

laws.append(hof[-1])

### Test of newly found law on multiple cases ###
success = True
it = 0
while it<100:
    ####  RANDOM VARIATIONS DEFINITION  ####
    height_start = obj.Re + random.uniform(0, 40000)
    delta = random.uniform(10000, 15000)
    v_wind = random.uniform(0, 24)
    cases.append([round((height_start-obj.Re)/1e3, 2), round(delta/1e3,2), round(v_wind,2)])
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]
    res = solve_ivp(partial(sys2GP_wind, expr1=best_string1, expr2=best_string2, v_wind=v_wind, height_start=height_start, delta=delta), [0, tfin], x_ini, method='BDF')
    if (Rref[-1]-obj.Re)*0.99 < (res.y[0, -1]-obj.Re) < (Rref[-1]-obj.Re)*1.01 and Thetaref[-1]*0.99 < res.y[1, -1] < Thetaref[-1]*1.01:  # tolerance of 1%
        s += 1
    it += 1
print("The control law could solve {}/100 wind cases".format(s))

success = True
s = 0
it = 0
while it<100:
    t_change = random.uniform(20, 200)
    Cd_new = random.uniform(0.61, 2)
    x_ini = [obj.Re, 0.0, 0.0, 0.0, obj.M0]
    res = solve_ivp(partial(sys2GP_Cd, expr1=best_string1, expr2=best_string2, Cd_new=Cd_new, t_change=t_change), [0, tfin], x_ini, method='BDF')
    if (Rref[-1]-obj.Re)*0.99 < (res.y[0, -1]-obj.Re) < (Rref[-1]-obj.Re)*1.01 and Thetaref[-1]*0.99 < res.y[1, -1] < Thetaref[-1]*1.01:  # tolerance of 1%
        s += 1
    it += 1
print("The control law could solve {}/100 Cd cases".format(s))

output = open("hof_G2C_forNN_MGGP.pkl".format(os.path.basename(__file__), timestr), "wb")  # save of hall of fame after first GP run
pickle.dump(hof, output, -1)
output.close()

print(best_string1)
print(best_string2)







