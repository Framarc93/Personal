"""MultiGeneGP_test.py = MGGP implementation in Python

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk

Rely on DEAP library and uses the same approach used for multi tree output. Each individual is compose by a ceratin number fo trees.
These trees are then combined linealry with some weights.
"""

import numpy as np
import operator
import random
from deap import gp
import matplotlib.pyplot as plt
from deap import base, creator, tools
from copy import deepcopy
import sys
import multiprocessing
import MGGP_functions as funs
from functools import partial
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp, simps
import time

laptop = True
if laptop:
    initial_path = "/home/francesco/Desktop/PhD/Git_workspace/IC4A2S"
else:
    initial_path = "/home/francesco/Desktop/Git_workspace/IC4A2S"

sys.path.insert(1, initial_path + "/Goddard_Problem/Testing/Goddard_1Control")
import GP_PrimitiveSet as gpprim

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

def evaluate_lst(d, individual, Cd_new, delta_eval, change_time, x_ini_real):
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
    tv = np.linspace(change_time, tfin, 100)
    sol = solve_ivp(partial(sys, change_time=change_time, delta_eval=delta_eval, Cd_new=Cd_new), [change_time, tfin], x_ini_real, t_eval=tv)
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

def evaluate_lst_withInput(d, individual, Cd_new, delta_eval, change_time, x_ini_real):
    global flag
    global Rfun, Thetafun, Vrfun, Vtfun, mfun, Trfun, Ttfun
    global tfin, t_eval2, penalty, fit_old, mutpb, cxpb, ER, ET, EVR, EVT

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
    ER = []
    ET = []
    EVR = []
    EVT = []
    def sys(t, x, change_time, delta_eval, Cd_new):
        global penalty, flag, flag_offdesign, flag_thrust, ER, ET, EVR, EVT
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
        #ER.append(er)
        #ET.append(et)
        #EVR.append(evr)
        #EVT.append(evt)
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
    tv = np.linspace(change_time, tfin, 100)
    sol = solve_ivp(partial(sys, change_time=change_time, delta_eval=delta_eval, Cd_new=Cd_new), [change_time, tfin], x_ini_real, t_eval=tv)
    y1 = sol.y[0, :]
    y2 = sol.y[1, :]
    y3 = sol.y[0, :]
    y4 = sol.y[1, :]
    tt = sol.t

    if tt[-1] < tfin:
        flag = True
        penalty.append((tt[-1] - tfin) / tfin)

    r = Rfun(tt)
    th = Thetafun(tt)
    vr = Vrfun(tt)
    vt = Vtfun(tt)

    err1 = (r - y1) / obj.Htarget
    err2 = (th - y2) / 0.9
    ER = r-y1
    ET = th-y2
    EVR = vr - y3
    EVT = vt - y4
    if flag is True:
        x = err1 + err2 + sum(abs(np.array(penalty))) #np.sqrt(sum(np.array(penalty) ** 2))
        return x, ER, ET, EVR, EVT
    else:
        return err1 + err2, ER, ET, EVR, EVT


def evaluate(d, individual):
    global input1, input2, output, flag

    c = len(individual[0].w) + len(individual[1].w)
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            for k in range(len(individual[i][j])):
                if type(individual[i][j][k]) == gp.Terminal and individual[i][j][k].name[0] != "A":
                    individual[i][j][k] = deepcopy(individual[i][j][k])
                    individual[i][j][k].value = float(d[c])
                    individual[i][j][k].name = str(d[c])
                    c += 1

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
    while ws < len(individual[1].w) + len(individual[0].w):
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
        # mf = mfun(t)

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

    tv = np.linspace(change_time, tfin, 100)
    sol = solve_ivp(partial(sys, change_time=change_time, delta_eval=delta_eval, Cd_new=Cd_new), [change_time, tfin],
                    x_ini_real, t_eval=tv)
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
    res = np.sqrt(sum(err1**2)) + np.sqrt(sum(err2**2))
    if flag is True:
        x = res + sum(abs(np.array(penalty)))  # np.sqrt(sum(np.array(penalty) ** 2))
        return x
    else:
        return res


####     PARAMETERS DEFINITION  #######

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
    psetR.addTerminal("randR{}".format(i), round(random.uniform(-10, 10), 6))

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
    psetT.addTerminal("randT{}".format(i), round(random.uniform(-10, 10), 6))

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
toolbox.register("evaluate", evaluate)
toolbox.register("select", funs.selDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True)
toolbox.register("mate", funs.xmate, Ngenes=Ngenes) ### NEW ##
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4) ### NEW ###
toolbox.register("mutate", funs.xmut, expr=toolbox.expr_mut, unipb=0.6, shrpb=0.3, Ngenes=Ngenes, psetR=psetR, psetT=psetT)
toolbox.decorate("mate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", funs.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", funs.staticLimit(key=len, max_value=limit_size))

def initPOP1():
    global old_hof
    # this function outputs the first n individuals of the hall of fame of the first GP run
    res = old_hof.shuffle()
    # for i in range(10):
    #   res.append(hof[0])
    return res

def main():
    nbCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nbCPU)
    toolbox.register('map', pool.map)
    pop = toolbox.population(size_pop)  # creation of initial population
    if nt > 0:
        pop2 = toolbox.popx()
        for ind in pop2:
            del ind.fitness.values
        pop = pop2 + pop
    pop_start1 = deepcopy(pop)
    pop_start2 = deepcopy(pop)
    pop1 = toolbox.map(partial(funs.lstInversion, evaluate_lst=evaluate_lst_withInput, Cd_new=Cd_new, delta_eval=delta_eval, change_time=change_time, x_ini=x_ini_real), pop_start1)  # optimization on pop
    pop2 = toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, Cd_new=Cd_new, delta_eval=delta_eval, change_time=change_time, x_ini=x_ini_real), pop_start2)  # optimization on pop
    hof = tools.HallOfFame(10)
    hof.update(pop)
    best_ind = funs.selBest(pop, 1)[0]
    best_fit = best_ind.fitness.values[0]

    tol = 0.045
    n = 0
    while n <= Ngen and best_fit > tol:
        print("------------------------------------------------------------------------------------------------------------- GEN {}".format(n))

        #to_mate = funs.selBest(pop, int(len(pop)/2))
        offspring = funs.varOr(pop, toolbox, int(len(pop)), cxpb=cxpb, mutpb=mutpb)
        #offspring = list(map(evaluate_nonOpt, offspring))  # evaluate fitness of offspring

        #if n % 1 == 0 and n >= 1:
        #    offspring = toolbox.map(partial(funs.optimize_ind, evaluate=evaluate, Ngenes=Ngenes), offspring)  # optimization on all pop
        #else:
        offspring = toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, Cd_new=Cd_new, delta_eval=delta_eval, change_time=change_time, x_ini=x_ini_real), offspring)  # optimization on all pop
        hof.update(offspring)
        pop[:] = toolbox.select(offspring + pop, len(pop)-1)
        best_ind = funs.selBest(pop, 1)[0]
        pop.append(best_ind)
        best_fit = best_ind.fitness.values[0]
        print("------------------------------------------------------------------------------------------------------------- {}".format(best_fit))
        '''string1 = str(best_ind[0].w[0])
        st = 1
        while st <= Ngenes:
            string1 = string1 + "+" + str(best_ind[0].w[st]) + "*" + str(best_ind[0][st-1])
            st += 1
        string2 = str(best_ind[1].w[0])
        st = 1
        while st <= Ngenes:
            string2 = string2 + "+" + str(best_ind[1].w[st]) + "*" + str(best_ind[1][st - 1])
            st += 1
        print("Tr: ", string1)
        print("Tt: ", string2)'''
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
    #oo = evaluate_forplot(best_ind)
    #plt.figure()
    #plt.plot(output, 'o', label="Experimental data")
    #plt.plot(oo, marker='.', label="Fitted data", color='r')
    #plt.legend(loc="best")
    #plt.show()
    pool.close()

    return best_fit, best_string1, best_string2, hof

def sys2GP(t, x, expr1, expr2, new_Cd, change_t):
    fTr = toolbox.compileR(expr1)
    fTt = toolbox.compileT(expr2)

    if t >= change_t:
        Cd = new_Cd
    else:
        Cd = obj.Cd

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
    #em = mf - m

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

def sys_evalTime(t, x, new_Cd):
    Cd = new_Cd

    R = x[0]
    #theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]

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

def sys_init(t,x):
    R = x[0]
    #theta = x[1]
    Vr = x[2]
    Vt = x[3]
    m = x[4]
    Cd = Cd_new
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


nt = 0
success_range_time = 0
success_time = 0
success_range = 0
ntot = 100
stats = []
r_diff = []
theta_diff = []
old_hof = tools.HallOfFame(3)
stats.append(["Cd new", "Change time", "Delta evaluation", "Real time Evaluation"])
mutpb = 0.8
cxpb = 0.2
delta_eval = 30
size_pop = 40 - len(old_hof) # Pop size
Ngen = 150  # Gen size
size_pop_tot = 40
Mu = int(size_pop_tot)
Lambda = int(size_pop_tot)

while nt < ntot:
    ####  RANDOM VARIATIONS DEFINITION ####

    change_time = random.uniform(50, 250)
    change_time = change_time + delta_eval
    Cd_new = random.uniform(0.61, 2)
    if change_time-delta_eval < 100 and Cd_new > 1:
        Cd_new = random.uniform(0.61, 1)

    print("\n")
    print(" ---------------Iter={}, Cd={}, Change Time={}, Delta eval={} -----------------".format(nt, round(Cd_new,2), round(change_time-delta_eval,2), delta_eval))
    x_ini = [Rfun(change_time - delta_eval), Thetafun(change_time - delta_eval), Vrfun(change_time - delta_eval),
             Vtfun(change_time - delta_eval), mfun(change_time - delta_eval)]  # initial conditions
    tev = np.linspace(change_time - delta_eval, change_time, 100)
    in_cond = solve_ivp(sys_init, [change_time - delta_eval, change_time], x_ini, t_eval=tev)
    x_ini_real = [in_cond.y[0, :][-1], in_cond.y[1, :][-1], in_cond.y[2, :][-1], in_cond.y[3, :][-1], in_cond.y[4, :][-1]]

    start = time.time()
    best_fit, best_ind1, best_ind2, hof = main()
    end = time.time()
    old_hof.update(hof.items)
    t_offdesign = end-start
    print(t_offdesign)

    stats.append([Cd_new, change_time - delta_eval, delta_eval, t_offdesign])

    x_ini = [Rfun(change_time - delta_eval), Thetafun(change_time - delta_eval), Vrfun(change_time - delta_eval),
             Vtfun(change_time - delta_eval), mfun(change_time - delta_eval)]  # initial conditions

    #########for plot #########
    tev = np.linspace(change_time - delta_eval, tfin, 100)
    teval_val_p = solve_ivp(partial(sys_evalTime, new_Cd=Cd_new), [change_time - delta_eval, tfin], x_ini, t_eval=tev) # only used for plot

    r_eval_p = teval_val_p.y[0, :]
    th_eval_p = teval_val_p.y[1, :]
    vr_eval_p = teval_val_p.y[2, :]
    vt_eval_p = teval_val_p.y[3, :]
    m_eval_p = teval_val_p.y[4, :]
    t_eval_p = teval_val_p.t

    ##### for next integration #####
    tev = np.linspace(change_time - delta_eval, change_time, 100)
    teval_val = solve_ivp(partial(sys_evalTime, new_Cd=Cd_new), [change_time - delta_eval, change_time], x_ini,
                          t_eval=tev)  # used for next integration

    r_eval = teval_val.y[0, :]
    th_eval = teval_val.y[1, :]
    vr_eval = teval_val.y[2, :]
    vt_eval = teval_val.y[3, :]
    m_eval = teval_val.y[4, :]
    t_eval = teval_val.t

    x_ini_new = [r_eval[-1], th_eval[-1], vr_eval[-1], vt_eval[-1], m_eval[-1]]
    teval = np.linspace(change_time, tfin, 100)
    solgp = solve_ivp(partial(sys2GP, new_Cd=Cd_new, expr1=best_ind1, expr2=best_ind2, change_t=change_time),
                      [change_time, tfin], x_ini_new, t_eval=teval)

    rout = solgp.y[0, :]
    thetaout = solgp.y[1, :]
    vrout = solgp.y[2, :]
    vtout = solgp.y[3, :]
    mout = solgp.y[4, :]
    ttgp = solgp.t

    rR = Rfun(ttgp)
    tR = Thetafun(ttgp)
    vrR = Vrfun(ttgp)
    vtR = Vtfun(ttgp)
    mR = mfun(ttgp)
    if t_offdesign < delta_eval:
        success_time += 1
    if Rref[-1]*0.99 < rout[-1] < Rref[-1]*1.01 and Thetaref[-1]*0.99 < thetaout[-1] < Thetaref[-1]*1.01:  # tolerance of 1%
        success_range += 1
    if t_offdesign < delta_eval and Rref[-1]*0.99 < rout[-1] < Rref[-1]*1.01 and Thetaref[-1]*0.99 < thetaout[-1] < Thetaref[-1]*1.01:
        success_range_time += 1
    time_ch = np.linspace(0.0, change_time - delta_eval, 100)
    time_eval = np.linspace(change_time - delta_eval, change_time, 100)

    '''plt.ion()
    plt.figure(2)
    plt.xlabel("time [s]")
    plt.ylabel("Altitude [km]")
    plt.plot(time_ch, (Rfun(time_ch) - obj.Re)/1e3, color='k', label='Nominal condition') # no need of integration to plot
    plt.plot(t_eval_p, (r_eval_p - obj.Re) / 1e3, color='b', linewidth=2, label='New Cd, Old Control law')
    plt.plot(ttgp, (rout - obj.Re) / 1e3, color='C2', linewidth=2, label="Cd={}, t change={}s+{}s".format(round(Cd_new, 2), round(change_time - delta_eval, 2), delta_eval))
    plt.legend(loc='best')

    plt.figure(3)
    plt.plot(time_ch, Vtfun(time_ch), color='k', label='Nominal condition')  # no need of integration to plot
    plt.plot(t_eval_p, vt_eval_p, color='b', label='New Cd, Old Control law')
    plt.plot(ttgp, vtout,
             label="Cd={}, t change={}s+{}s".format(round(Cd_new, 2), round(change_time - delta_eval, 2), delta_eval))
    plt.xlabel("time [s]")
    plt.ylabel("Tangential Velocity [m/s]")
    plt.legend(loc='best')

    plt.figure(4)
    plt.axhline(obj.M0 - obj.Mp, 0, ttgp[-1], color='r')
    plt.plot(time_ch, mfun(time_ch), color='k', label='Nominal condition')  # no need of integration to plot
    plt.plot(t_eval_p, m_eval_p, color='b', label='New Cd, Old Control law')
    plt.plot(ttgp, mout, label="Cd={}, t change={}s+{}s".format(round(Cd_new, 2), round(change_time - delta_eval, 2), delta_eval))
    plt.xlabel("time [s]")
    plt.ylabel("Mass [kg]")
    plt.legend(loc='best')

    plt.figure(5)
    plt.plot(time_ch, Vrfun(time_ch), color='k', label='Nominal condition')  # no need of integration to plot
    plt.plot(t_eval_p, vr_eval_p, color='b', label='New Cd, Old Control law')
    plt.plot(ttgp, vrout,
             label="Cd={}, t change={}s+{}s".format(round(Cd_new, 2), round(change_time - delta_eval, 2), delta_eval))
    plt.xlabel("time [s]")
    plt.ylabel("Radial Velocity [m/s]")
    plt.legend(loc='best')

    plt.figure(6)
    plt.plot(time_ch, np.rad2deg(Thetafun(time_ch)), color='k', label='Nominal condition')  # no need of integration to plot
    plt.plot(t_eval_p, np.rad2deg(th_eval_p), linewidth=2, color='b', label='New Cd, Old Control law')
    plt.plot(ttgp, np.rad2deg(thetaout), color='C2', linewidth=2, label="Cd={}, t change={}s+{}s".format(round(Cd_new, 2), round(change_time - delta_eval, 2), delta_eval))
    plt.xlabel("time [s]")
    plt.ylabel("Angle [deg]")
    plt.legend(loc='best')'''

    nt += 1
'''plt.ion()
plt.figure(2)
plt.plot(tref, (Rref - obj.Re) / 1e3, 'r--', linewidth=3, label="SET POINT")
plt.xlabel("time [s]")
plt.ylabel("Altitude [km]")
plt.legend(loc='best')
plt.grid()

plt.figure(3)
plt.plot(tref, Vtref, 'r--', label="SET POINT")
plt.xlabel("time [s]")
plt.ylabel("Tangential Velocity [m/s]")
plt.legend(loc='best')
plt.grid()

plt.figure(4)
plt.plot(tref, mref, 'r--', label="SET POINT")
plt.axhline(obj.M0 - obj.Mp, 0, tfin, color='r')
plt.xlabel("time [s]")
plt.ylabel("Mass [kg]")
plt.legend(loc='best')
plt.grid()

plt.figure(5)
plt.plot(tref, Vrref, 'r--', label="SET POINT")
plt.xlabel("time [s]")
plt.ylabel("Radial Velocity [m/s]")
plt.legend(loc='best')
plt.grid()

plt.figure(6)
plt.plot(tref, np.rad2deg(Thetaref), 'r--', linewidth=3, label="SET POINT")
plt.xlabel("time [s]")
plt.ylabel("Angle [deg]")
plt.legend(loc='best')
plt.grid()'''
print("Success time {}".format(success_time))
print("Success range {}".format(success_range))
print("Success total {}".format(success_range_time))
plt.show(block=True)











