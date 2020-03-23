'''test_GradDescent.py: test implementation of gradient descent on GP

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk

References:
relevant references for the algorithm
[1] A. Topchy, W. Punch. Faster Genetic Programming based on Local Gradient Search of Numeric Leaf Values'''

import sympy as sy
import scipy.io as sio
import numpy as np
from scipy.integrate import solve_ivp
import operator
import pygraphviz as pgv
import random
from deap import gp
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import multiprocessing
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simps
from time import time
from functools import partial
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from operator import *

def tree(x, af, bf):
  return np.sin(af*(x+bf))

def eval_fit(fun):
    global input1, input2, output
    f_fit = toolbox.compile(expr=fun)
    return np.sqrt(sum((output - f_fit(input1, input2))**2))

def grad_descent(const, it, fun, alpha):
    if len(const) == 0:
        return fun
    else:
        x1, x2 = sy.symbols('x1 x2')
        new_fit = eval_fit(fun)
        pds = []
        c_dict = {}
        for i in range(len(const)):
            c_dict["c{}".format(i)] = sy.symbols('{}'.format(const[i]+(1e-4)*i))
        c = 0
        for i in range(len(fun)):
            if type(fun[i]) == gp.Terminal and fun[i].name[0] != "A":
                fun[i] = deepcopy(fun[i])
                fun[i].name = 'c{}'.format(c)
                fun[i].value = sy.sympify('c{}'.format(c))
                c += 1
        for i in range(len(const)):
            pdf = sy.lambdify((x1, x2), sy.Derivative(sy.sympify(str(fun), locals=c_dict), c_dict['c{}'.format(i)]), modules=[{'add': operator.add, 'mul': operator.mul, 'sub':operator.sub}, "numpy"])
            par_dev = []
            for j in range(len(input1)):
                par_dev.append(float(pdf(input1[j], input2[j])))
            pds.append(np.sqrt(sum(par_dev**2)))
        for i in range(len(const)):
            const[i] = const[i]- alpha*(new_fit*pds[i])
        c = 0
        for i in range(len(fun)):
            if type(fun[i]) == gp.Terminal and fun[i].name[0] != "A":
                fun[i].name = str(const[c])
                fun[i].value = const[c]
                c += 1

        for _ in range(it):
            new_fit = eval_fit(fun)
            pds = []
            c_dict = {}
            for i in range(len(const)):
                c_dict["c{}".format(i)] = sy.symbols('{}'.format(const[i] + 1e-4 * i))
            c = 0
            for i in range(len(fun)):
                if type(fun[i]) == gp.Terminal and fun[i].name[0] != "A":
                    fun[i] = deepcopy(fun[i])
                    fun[i].name = 'c{}'.format(c)
                    fun[i].value = sy.sympify('c{}'.format(c))
                    c += 1
            for i in range(len(const)):
                pdf = sy.lambdify((x1, x2), sy.diff(sy.sympify(str(fun), locals=c_dict), c_dict['c{}'.format(i)]))
                pds.append(pdf(input1, input2))
            for i in range(len(const)):
                const[i] = const[i] - alpha * (new_fit * pds[i])
            c = 0
            for i in range(len(fun)):
                if type(fun[i]) == gp.Terminal and fun[i].name[0] != "A":
                    fun[i].name = str(const[c])
                    fun[i].value = const[c]
                    c += 1
        return fun


mat_contents = sio.loadmat('dataModB118.mat')
files = mat_contents['VV']
input1 = np.zeros(len(files))
input2 = np.zeros(len(files))
output = np.zeros(len(files))
for i in range(len(files)):
    input1[i] = files[i][0]
    input2[i] = files[i][1]
    output[i] = files[i][2]

def Square(x):
    return x**2

def Mutual(x):
    try:
        y = 1/x
        return y
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
        return x

def Identity(x):
    return x

def Neg(x):
    return -x

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

def Sqrt(x):
    try:
        if x > 0:
            return np.sqrt(x)
        else:
            return abs(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0

def lf(x): return 1 / (1 + np.exp(-x))

def Log(x):
    try:
        if x > 0:
            return np.log(x)
        else:
            return abs(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 0

def Pow(x): return x**2

def Exp(x):
    try:
        return np.exp(x)
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError):
        return 1


nEph = 1  # number of ephemeral constant
limit_height = 17  # Max height (complexity) of the controller law
limit_size = 100  # Max size

def opt_trees(pop):
    for i in range(len(pop)):
        alpha = 0.1
        start_const = []
        for j in range(len(pop[i])):
            if type(pop[i][j]) == gp.Terminal and pop[i][j].name[0] != "A":
                start_const.append(pop[i][j].value)
        pop[i] = grad_descent(fun=pop[i], it=10, alpha=alpha, const=start_const)

def main():

    '''Main function called to perform GP evaluation'''
    cxpb = 0.6  # crossover rate

    mutpb = 0.3  # 1 - cxpb - elite  # mutation rate

    size_pop = 10 # Pop size
    size_gen = 10  # Gen size

    Mu = int(size_pop)
    Lambda = int(size_pop * 1.3)

    nbCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nbCPU)

    toolbox.register("map", pool.map)
    pop = toolbox.population(n=size_pop)
    opt_trees(pop)
    hof = tools.HallOfFame(size_gen)  ### OLD ###
    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")
    random.seed()
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    ####################################   EVOLUTIONARY ALGORITHM   -  EXECUTION   #####################################

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=size_gen, stats=mstats, halloffame=hof, verbose=True)
    pool.close()
    pool.join()
    return pop, log, hof


def evaluate(individual):
    global input1, input2, output

    output_fun = toolbox.compile(expr=individual)

    output_eval = output_fun(input1, input2)
    nodes = 0
    err = np.sqrt(1/len(output)*sum((output - output_eval)**2))
    #for i in range(len(individual)):
     #   if type(individual[i]) == gp.Primitive:
      #      nodes += 1
    return (err,)


if __name__ == "__main__":
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    #pset.addPrimitive(TriAdd, 3)
    pset.addPrimitive(np.tanh, 1, "tanh")
    #pset.addPrimitive(Exp, 1)
    #pset.addPrimitive(Pow, 1)
    #pset.addPrimitive(np.log, 1, "log")
    pset.addPrimitive(np.sqrt, 1, "sqrt")
    pset.addPrimitive(np.cos, 1, "cos")
    pset.addPrimitive(np.sin, 1, "sin")
    #pset.addPrimitive(lf, 1)
    for i in range(nEph):
        pset.addTerminal(round(random.uniform(-10, 10), 4))

    pset.renameArguments(ARG0='x1')
    pset.renameArguments(ARG1='x2')

    ################################################## TOOLBOX OF FIRST GP##################################################

    creator.create("Fitness", base.Fitness, weights=(-1.0,))  # MINIMIZATION OF THE FITNESS FUNCTION
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=6)   #### OLD ####
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate)  ### OLD ###
    toolbox.register("select", tools.selDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True)
    #toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)#gp.cxSemantic, gen_func=gp.genFull, pset=pset, min=2, max=8)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=6)
    toolbox.register("mutate", gp.mutUniform, pset=pset, expr=toolbox.expr)#gp.mutSemantic, gen_func=gp.genFull, pset=pset, min=2, max=8)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

############################################  START OF FIRST GP RUN ####################################################

    pop, log, hof = main()

for ind in hof.items:
    if np.isnan(ind.fitness.values[0]) == False:
        expr1 = ind
        break
print(expr1.fitness)
print(expr1)
np.save("function_python", expr1)
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

vals = []
output_fun = toolbox.compile(expr=expr1)

vals.append(output_fun(input1, input2))
plt.figure()
plt.plot(output, 'o', label="Experimental data")
#if len(vals) == 1:
#    plt.axhline(vals, xmin=0, xmax=len(output), label="Fitted data", color='r')
#else:
plt.plot(output_fun(input1, input2), marker='.', label="Fitted data", color='r')
plt.legend(loc="best")
plt.show()





