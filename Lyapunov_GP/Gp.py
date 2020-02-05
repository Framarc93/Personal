'''GP.py: GP code for Lyapunov function

This script applies the standard version of the Genetic Programming algorithm to produce a Lyapunov function for a van der pol system

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk
'''

import numpy as np
import operator
import random
from deap import gp
from deap import base, creator, tools, algorithms
import sys
from functools import partial
import multiprocessing
import sympy
import operator
import math
import GP_PrimitiveSet as gpprim
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt


random.seed()
lb = -4
ub = 4
points = np.random.rand(1000, 2)
points = points * (ub - lb) + lb

def evaluate_best(individual, f1, f2):
    global points
    condition2 = []
    condition3 = []

    output_fun = toolbox.compile(expr=individual, pset=pset)  # individual evaluated through GP
    output_eval = output_fun(points[:, 0], points[:, 1])  # output of reference function
    if not hasattr(output_eval, '__len__'):
        output_eval = output_eval * np.ones(len(points[:, 0]))
    positive = 0
    zero_cond = output_fun(0, 0)

    if zero_cond == 0:
        condition1 = True
    else:
        condition1 = False

    x1, x2 = sympy.symbols('x1 x2')
    parsed_ind = parse_expr(str(individual), evaluate=False, local_dict={'x1': x1, 'x2':x2, 'add':operator.add, 'sub':operator.sub, 'mul':operator.mul, 'sin':sympy.sin, 'cos':sympy.cos, 'exp':sympy.exp,
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

def evaluate(individual, f1, f2):
    global points

    output_fun = toolbox.compile(expr=individual, pset=pset)  # individual evaluated through GP
    output_eval = output_fun(points[:, 0], points[:, 1])  # output of reference function
    if not hasattr(output_eval, '__len__'):
        output_eval = output_eval * np.ones(len(points[:, 0]))

    positive = 0
    zero_cond = output_fun(0, 0)

    if zero_cond == 0:
        cond1 = True
    else:
        cond1 = False

    x1, x2 = sympy.symbols('x1 x2')
    parsed_ind = parse_expr(str(individual), evaluate=False, local_dict={'x1': x1, 'x2':x2, 'add':operator.add, 'sub':operator.sub, 'mul':operator.mul, 'sin':sympy.sin, 'cos':sympy.cos, 'exp':sympy.exp,
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
        fit = 0
    else:
        fit = -(all_good)

    return (fit,)

f1 = lambda x, y: y
f2 = lambda x, y: -0.5*y*(1-x**2+0.1*x**4)-x

'''Main function called to perform GP evaluation'''
def pow2(x):
    return x**2
cxpb = 0.7  # crossover rate
mutpb = 0.2  # 1 - cxpb - elite  # mutation rate
size_pop = 400
Mu = int(size_pop)
Lambda = int(size_pop * 1.3)
terminals = 2
nEph = 2
limit_height = 10
limit_size = 20

size_gen = 100
hof  = tools.HallOfFame(10)
pset = gp.PrimitiveSet("MAIN", terminals)
pset.addPrimitive(operator.add, 2, name='add')
pset.addPrimitive(operator.sub, 2, name='sub')
pset.addPrimitive(operator.mul, 2, name='mul')
pset.addPrimitive(np.exp, 1, name='exp')
pset.addPrimitive(pow2, 1, name='pow')
pset.addPrimitive(np.cos, 1, name="cos")
pset.addPrimitive(np.sin, 1, name="sin")
pset.addTerminal(np.e, name="E")
# pset.addPrimitive(gpprim.lf, 1)
for i in range(nEph):
    pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-1, 1), 4))
pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')
######## TOOLBOX OF FIRST GP##################################################

creator.create("Fitness", base.Fitness, weights=(-1.0,))  # MINIMIZATION OF THE FITNESS FUNCTION
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)  #### OLD ####
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", partial(evaluate, f1=f1, f2=f2))  ### OLD ###
toolbox.register("select", tools.selDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True)
# toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)  # gp.cxSemantic, gen_func=gp.genFull, pset=pset, min=2, max=8)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, pset=pset, expr=toolbox.expr)  # gp.mutSemantic, gen_func=gp.genFull, pset=pset, min=2, max=8)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

def main():
    pool = multiprocessing.Pool(8)
    toolbox.register("map", map)
    pop = toolbox.population(n=size_pop)

    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    #stats_size = tools.Statistics(len)
    #stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=Mu, lambda_=Lambda, cxpb=cxpb, mutpb=mutpb, ngen=size_gen, stats=mstats, halloffame=hof, verbose=True)


    return pop, log, hof

pop, log, hof = main()


print(hof[0])
v = toolbox.compile(expr=hof[0], pset=pset)
v_val = v(points[:, 0], points[:, 1])
x1, x2 = sympy.symbols('x1 x2')
parsed_ind = parse_expr(str(hof[0]), evaluate=False, local_dict={'x1': x1, 'x2':x2, 'add':operator.add, 'sub':operator.sub, 'mul':operator.mul, 'sin':sympy.sin, 'cos':sympy.cos, 'exp':sympy.exp,
                                                                         'pow':pow2})
vdot1 = sympy.diff(parsed_ind, x1)
vdot2 = sympy.diff(parsed_ind, x2)
vdot1_fun = toolbox.compile(vdot1, pset=pset)
vdot2_fun = toolbox.compile(vdot2, pset=pset)
vdot1_val = vdot1_fun(points[:, 0], points[:, 1]) * f1(points[:, 0], points[:, 1])
vdot2_val = vdot2_fun(points[:, 0], points[:, 1]) * f2(points[:, 0], points[:, 1])
vdot = vdot1_val + vdot2_val
print(v_val)
print(vdot)
#eq = '1.007*x1**2 + 1.0219*(x1 - 1.0063*x2 - 1.9265 * sin(0.77835 * x1))**2'
condition1, condition2, condition3 = evaluate_best(hof[0], f1=f1, f2=f2)
c = 0
col = []
for i in range(len(points[:,0])):
    if condition1 is True and condition2[i] is True and condition3[i] is True:
        col.append('g')
        c += 1
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
print(c)
#plt.title(str(hof[0]) + ", " + str(hof[0].fitness.values[0]*(-1)) + " points")
for i in range(len(points[:, 0])):
    plt.plot(points[i, 0], points[i, 1], '.', color=col[i])
plt.show(block=True)










