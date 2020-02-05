'''MGGP: MGGP code for Lyapunov function

This script applies a python implementation of MGGP algorithm to produce a Lyapunov function for a van der pol system

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk
'''

import matplotlib.pyplot as plt
import numpy as np
import operator
import random
from deap import gp
from deap import base, creator, tools, algorithms
import multiprocessing
import sys
from functools import partial
from copy import deepcopy
import MGGP_funs as funs
import sympy
from sympy.parsing.sympy_parser import parse_expr

sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")
sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/GP_Experiments/MGGP")
import GP_PrimitiveSet as gpprim


lb = -4
ub = 4
points = np.random.rand(100, 2)
points = points * (ub - lb) + lb

def evaluate_lst(d, individual, f1, f2):
    global points
    out_trees = []

    for i in range(len(individual)):
        f = toolbox.compile(individual[i])
        res = f(points[:, 0], points[:, 1])
        if type(res) != np.ndarray:
            res = np.ones(len(points[:, 0])) * res
        out_trees.append(res * d[i + 1])
    output_eval = 0
    for j in range(len(out_trees)):
        output_eval = output_eval + out_trees[j]
    output_eval = output_eval + d[0]
    if not hasattr(output_eval, '__len__'):
        output_eval = output_eval * np.ones(len(points[:, 0]))

    positive = 0
    eq = funs.build_eq(individual)
    output_fun = toolbox.compile(eq)
    zero_cond = output_fun(0, 0)

    if zero_cond == 0:
        cond1 = True
    else:
        cond1 = False

    x1, x2 = sympy.symbols('x1 x2')
    parsed_ind = parse_expr(eq, evaluate=False, local_dict={'x1': x1, 'x2':x2, 'add':operator.add, 'sub':operator.sub, 'mul':operator.mul, 'sin':sympy.sin, 'cos':sympy.cos, 'exp':sympy.exp,
                                                                         'pow':pow2, 'E':np.e})
    try:
        vdot1 = sympy.diff(parsed_ind, x1)
        vdot2 = sympy.diff(parsed_ind, x2)
        vdot1_fun = toolbox.compile(vdot1, pset=pset)
        vdot2_fun = toolbox.compile(vdot2, pset=pset)
        vdot1_val = vdot1_fun(points[:, 0], points[:, 1]) * f1(points[:, 0], points[:, 1])
        vdot2_val = vdot2_fun(points[:, 0], points[:, 1]) * f2(points[:, 0], points[:, 1])
        vdot = vdot1_val + vdot2_val
    except TypeError:
        print("e")
    grad = 0
    if not hasattr(vdot, '__len__'):
        vdot = np.ones(len(points[:, 0]))*vdot
    all_good = 0
    for i in range(len(vdot)):
        if output_eval[i] > 0:
            positive += 1
            cond2 = True
        else:
            cond2 = False
        if vdot[i] < 0:
            grad += 1
            cond3 = True
        else:
            cond3 = False
        if cond2 is True and cond3 is True:
            all_good += 1

    if cond1 is False:
        fit = -(positive+grad)
    else:
        fit = -(all_good)+(len(points[:, 0]))
    Fit = len(points[:, 0])*2 - fit
    return Fit

def evaluate_best(individual, f1, f2):
    global points
    condition2 = []
    condition3 = []

    out_trees = []

    c = 0
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            if type(individual[i][j]) == gp.Terminal and individual[i][j].name[0] != "A":
                individual[i][j] = deepcopy(individual[i][j])
                individual[i][j].value = float(d[c + Ngenes + 1])
                individual[i][j].name = str(d[c + Ngenes + 1])
                c += 1

    for i in range(len(individual)):
        f = toolbox.compile(individual[i])

        res = f(points[:, 0], points[:, 1])

        if type(res) != np.ndarray:
            res = np.ones(len(points[:, 0])) * res
        out_trees.append(res * d[i + 1])

    output_eval = 0
    for j in range(len(out_trees)):
        output_eval = output_eval + out_trees[j]
    output_eval = output_eval + d[0]

    if not hasattr(output_eval, '__len__'):
        output_eval = output_eval * np.ones(len(points[:, 0]))

    positive = 0
    eq = funs.build_eq(individual)
    output_fun = toolbox.compile(eq)
    zero_cond = output_fun(0, 0)

    if zero_cond == 0:
        condition1 = True
    else:
        condition1 = False

    x1, x2 = sympy.symbols('x1 x2')
    parsed_ind = parse_expr(eq, evaluate=False, local_dict={'x1': x1, 'x2':x2, 'add':operator.add, 'sub':operator.sub, 'mul':operator.mul, 'sin':sympy.sin, 'cos':sympy.cos, 'exp':sympy.exp,
                                                                         'pow':pow2})

    vdot1 = sympy.diff(parsed_ind, x1)
    vdot2 = sympy.diff(parsed_ind, x2)
    vdot1_fun = toolbox.compile(vdot1, pset=pset)
    vdot2_fun = toolbox.compile(vdot2, pset=pset)
    vdot1_val = vdot1_fun(points[:, 0], points[:, 1]) * f1(points[:, 0], points[:, 1])
    vdot2_val = vdot2_fun(points[:, 0], points[:, 1]) * f2(points[:, 0], points[:, 1])
    vdot = vdot1_val + vdot2_val
    grad = 0
    if not hasattr(vdot, '__len__'):
        vdot = np.ones(len(points[:, 0]))*vdot
    for i in range(len(vdot)):
        if output_eval[i] > 0:
            positive += 1
            condition2.append(True)
        else:
            condition2.append(False)
        if vdot[i] < 0:
            grad += 1
            condition3.append(True)
        else:
            condition3.append(False)

    return condition1, condition2, condition3

def evaluate(d, individual, f1, f2):
    global points
    out_trees = []

    c = 0
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            if type(individual[i][j]) == gp.Terminal and individual[i][j].name[0] != "A":
                individual[i][j] = deepcopy(individual[i][j])
                individual[i][j].value = float(d[c + Ngenes + 1])
                individual[i][j].name = str(d[c + Ngenes + 1])
                c += 1

    for i in range(len(individual)):
        f = toolbox.compile(individual[i])

        res = f(points[:, 0], points[:, 1])

        if type(res) != np.ndarray:
            res = np.ones(len(points[:, 0])) * res
        out_trees.append(res * d[i + 1])

    output_eval = 0
    for j in range(len(out_trees)):
        output_eval = output_eval + out_trees[j]
    output_eval = output_eval + d[0]

    if not hasattr(output_eval, '__len__'):
        output_eval = output_eval * np.ones(len(points[:, 0]))

    positive = 0
    eq = funs.build_eq(individual)
    output_fun = toolbox.compile(eq)
    zero_cond = output_fun(0, 0)

    if zero_cond == 0:
        cond1 = True
    else:
        cond1 = False

    x1, x2 = sympy.symbols('x1 x2')
    parsed_ind = parse_expr(eq, evaluate=False, local_dict={'x1': x1, 'x2':x2, 'add':operator.add, 'sub':operator.sub, 'mul':operator.mul, 'sin':sympy.sin, 'cos':sympy.cos, 'exp':sympy.exp,
                                                                         'pow':pow2, 'E':np.e})
    try:
        vdot1 = sympy.diff(parsed_ind, x1)
        vdot2 = sympy.diff(parsed_ind, x2)
        vdot1_fun = toolbox.compile(vdot1, pset=pset)
        vdot2_fun = toolbox.compile(vdot2, pset=pset)
        vdot1_val = vdot1_fun(points[:, 0], points[:, 1]) * f1(points[:, 0], points[:, 1])
        vdot2_val = vdot2_fun(points[:, 0], points[:, 1]) * f2(points[:, 0], points[:, 1])
        vdot = vdot1_val + vdot2_val
    except TypeError:
        print("e")
    grad = 0
    if not hasattr(vdot, '__len__'):
        vdot = np.ones(len(points[:, 0]))*vdot
    all_good = 0
    for i in range(len(vdot)):
        if output_eval[i] > 0:
            positive += 1
            cond2 = True
        else:
            cond2 = False
        if vdot[i] < 0:
            grad += 1
            cond3 = True
        else:
            cond3 = False
        if cond2 is True and cond3 is True:
            all_good += 1

    if cond1 is False:
        fit = -(positive + grad)
    else:
        fit = -(all_good) + (len(points[:, 0]))
    Fit = len(points[:, 0]) * 2 - fit
    return (Fit,)

####     PARAMETERS DEFINITION  #######

f1 = lambda x, y: y
f2 = lambda x, y: -0.5*y*(1-x**2+0.1*x**4)-x

'''Main function called to perform GP evaluation'''
def pow2(x):
    return x**2

Ngenes = 3
limit_height = 12
limit_size = 20
cxpb = 0.5
mutpb = 0.5
nVars = 2
nEph = 1
pset = gp.PrimitiveSet("MAIN", nVars)
pset.addPrimitive(operator.add, 2, name='add')
pset.addPrimitive(operator.sub, 2, name='sub')
pset.addPrimitive(operator.mul, 2, name='mul')
pset.addPrimitive(np.exp, 1, name='exp')
pset.addPrimitive(pow2, 1, name='pow')
pset.addPrimitive(np.cos, 1, name="cos")
pset.addPrimitive(np.sin, 1, name="sin")
pset.addTerminal(np.e, name="E")

for i in range(nEph):
    pset.addTerminal(round(random.uniform(-1, 1), 4))

pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')

################################################## TOOLBOX #############################################################

d = np.ones((Ngenes + 1)) * 1.0  # weights for linear combination of genes
d[0] = 0

creator.create("Fitness", base.Fitness, weights=(-1.0,))  # , -0.1, -0.08, -1.0))
creator.create("Individual", list, fitness=creator.Fitness, w=list(d), height=1)
creator.create("SubIndividual", gp.PrimitiveTree)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("leg", tools.initIterate, creator.SubIndividual, toolbox.expr)  ### NEW ###
toolbox.register("legs", tools.initRepeat, list, toolbox.leg, n=Ngenes)  ### NEW ###
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)  ### NEW ###
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", partial(evaluate, f1=f2, f2=f2))
toolbox.register("select", funs.selDoubleTournament, fitness_size=2, parsimony_size=1.0, fitness_first=True)
toolbox.register("mate", funs.xmate, Ngenes=Ngenes)  ### NEW ##
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4)  ### NEW ###
toolbox.register("mutate", funs.xmut, pset=pset, expr=toolbox.expr_mut, unipb=0.6, shrpb=0.3, Ngenes=Ngenes)
toolbox.decorate("mate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", funs.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", funs.staticLimit(key=len, max_value=limit_size))

def main():
    nbCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nbCPU)
    toolbox.register("map", map)
    pop = toolbox.population(50)  # creation of initial population
    #pop = list(toolbox.map(partial(funs.optimize_ind, evaluate=evaluate, Ngenes=Ngenes, ref_fun=ref_fun, interval=interval, terminals=terminals), pop))  # optimization on pop
    pop = list(toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, f1=f1, f2=f2), pop))  # optimization on pop
    # pop = list(map(lst, pop)) # least square on pop
    # pop = list(map(evaluate_nonOpt, pop))  # evaluate fitness of individuals

    best_ind = funs.selBest(pop, 1)[0]
    best_fit = best_ind.fitness.values[0]
    n = 0
    Ngen = 10
    while n <= Ngen:
        print(
            "------------------------------------------------------------------------------------------------------------- GEN {}".format(
                n))
        # pop = list(map(lst, pop))  # least square on pop
        to_mate = funs.selBest(pop, int(len(pop) / 2))
        offspring = funs.varOr(to_mate, toolbox, int(len(pop)), cxpb=cxpb, mutpb=mutpb)
        # offspring = list(map(evaluate_nonOpt, offspring))  # evaluate fitness of offspring
        #offspring = list(toolbox.map(partial(funs.optimize_ind, evaluate=evaluate, Ngenes=Ngenes, ref_fun=ref_fun, interval=interval, terminals=terminals), offspring))  # optimization on offspring
        offspring = list(toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, f1=f1, f2=f2), offspring))  # optimization on all pop
        pop[:] = toolbox.select(offspring + pop, len(pop) - 1)
        best_ind = funs.selBest(pop, 1)[0]
        pop.append(best_ind)
        best_fit = best_ind.fitness.values[0]
        print(
            "------------------------------------------------------------------------------------------------------------- {}".format(
                best_fit))
        string = str(best_ind.w[0])
        st = 1
        while st <= Ngenes:
            string = string + "+" + str(best_ind.w[st]) + "*" + str(best_ind[st - 1])
            st += 1
        print(string)
        n += 1

    print(best_fit)
    string = str(best_ind.w[0])
    st = 1
    while st <= Ngenes:
        string = string + "+" + str(best_ind.w[st]) + "*" + str(best_ind[st - 1])
        st += 1
    return best_fit, string, best_ind

best_fit, best_eq, best_ind = main()


condition1, condition2, condition3 = evaluate_best(best_ind, f1=f1, f2=f2)

col = []
for i in range(len(points[:,0])):
    if condition1 is True and condition2[i] is True and condition3[i] is True:
        col.append('g')
    elif condition1 is True and condition2[i] is True and condition3[i] is False:
        col.append('k')
    elif condition1 is True and condition2[i] is False and condition3[i] is False:
        col.append('k')
    elif condition1 is False and condition2[i] is True and condition3[i] is True:
        col.append('k')
    elif condition1 is False and condition2[i] is False and condition3[i] is True:
        col.append('k')
    elif condition1 is False and condition2[i] is True and condition3[i] is False:
        col.append('k')
    elif condition1 is True and condition2[i] is False and condition3[i] is True:
        col.append('k')

plt.figure()
plt.title(str(hof[0]) + ", " + str(hof[0].fitness.values[0]*(-1)) + " points")
for i in range(len(points[:, 0])):
    plt.plot(points[i, 0], points[i, 1], '.', color=col[i])
plt.show(block=True)












