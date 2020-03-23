"""MultiGeneGP_test.py = MGGP implementation in Python

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk

Rely on DEAP library and uses the same approach used for multi tree output. Each individual is compose by a ceratin number fo trees.
These trees are then combined linealry with some weights.
"""

import scipy.io as sio
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

sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")
import GP_PrimitiveSet as gpprim


####   IMPORT DATA  ###########

mat_contents = sio.loadmat('/home/francesco/Desktop/Git_workspace/IC4A2S/GP_Experiments/dataModB118.mat')
files = mat_contents['VV']
input1 = np.zeros(len(files))
input2 = np.zeros(len(files))
output = np.zeros(len(files))
for i in range(len(files)):
    input1[i] = files[i][0]
    input2[i] = files[i][1]
    output[i] = files[i][2]



def evaluate_lst(d, individual):
    global input1, input2, output

    ws = 1
    eq = str(d[0]) + "+"
    while ws < len(d):
        eq = eq + str(d[ws]) + "*" + str(individual[ws-1]) + "+"
        ws += 1
    eq = list(eq)
    del eq[-1]
    eq = "".join(eq)

    f = toolbox.compile(eq)
    output_eval = f(input1, input2)
    if not hasattr(output_eval, "__len__"):
        output_eval = np.ones(len(input1)) * output_eval

    err = output - output_eval
    return err


def evaluate_nonOpt(individual):
    global input1, input2, output
    ws = 1
    eq = str(individual.w[0]) + "+"
    while ws < len(individual.w):
        eq = eq + str(individual.w[ws]) + "*" + str(individual[ws - 1]) + "+"
        ws += 1
    eq = list(eq)
    del eq[-1]
    eq = "".join(eq)

    f = toolbox.compile(eq)
    output_eval = f(input1, input2)
    if not hasattr(output_eval, "__len__"):
        output_eval = np.ones(len(input1)) * output_eval
    err = 1/np.std(output) * np.sqrt(1/len(output) * sum((output - output_eval) ** 2))  # output-output_eval
    individual.fitness.values = err,
    return individual

def evaluate(d, individual):
    global input1, input2, output

    c = 0
    for i in range(len(individual)):
        for j in range(len(individual[i])):
            if type(individual[i][j]) == gp.Terminal and individual[i][j].name[0] != "A":
                individual[i][j] = deepcopy(individual[i][j])
                individual[i][j].value = float(d[c+Ngenes+1])
                individual[i][j].name = str(d[c+Ngenes+1])
                c += 1

    ws = 1
    eq = str(d[0]) + "+"
    while ws < len(individual.w):
        eq = eq + str(d[ws]) + "*" + str(individual[ws - 1]) + "+"
        ws += 1
    eq = list(eq)
    del eq[-1]
    eq = "".join(eq)

    f = toolbox.compile(eq)
    output_eval = f(input1, input2)
    if not hasattr(output_eval, "__len__"):
        output_eval = np.ones(len(input1)) * output_eval
    err = 1/np.std(output) * np.sqrt(1/len(output) * sum((output - output_eval) ** 2))  # output-output_eval
    return err

def evaluate_forplot(individual):
    global input1, input2, output
    ws = 1
    eq = str(individual.w[0]) + "+"
    while ws < len(individual.w):
        eq = eq + str(individual.w[ws]) + "*" + str(individual[ws - 1]) + "+"
        ws += 1
    eq = list(eq)
    del eq[-1]
    eq = "".join(eq)

    f = toolbox.compile(eq)
    output_eval = f(input1, input2)
    if not hasattr(output_eval, "__len__"):
        output_eval = np.ones(len(input1)) * output_eval

    return output_eval


####     PARAMETERS DEFINITION  #######

Ngenes = 3
limit_height = 10
limit_size = 15
Ngen = 500
cxpb = 0.5
mutpb = 0.5
nCost = 2
nVars = 2


pset = gp.PrimitiveSet("MAIN", nVars)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(gpprim.TriAdd, 3)
pset.addPrimitive(gpprim.TriMul, 3)
pset.addPrimitive(np.tanh, 1, name="Tanh")
pset.addPrimitive(gpprim.lf, 1)
pset.addPrimitive(gpprim.square, 1)
pset.addPrimitive(gpprim.cubic, 1)
#pset.addPrimitive(gpprim.Abs, 1)
#pset.addPrimitive(Identity, 1)
#pset.addPrimitive(gpprim.Neg, 1)
pset.addPrimitive(gpprim.ProtDiv, 2, name="Div")
#pset.addPrimitive(operator.pow, 2, name="Pow")
pset.addPrimitive(gpprim.Sqrt, 1)
pset.addPrimitive(gpprim.Log, 1)
pset.addPrimitive(funs.modExp, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)
pset.addPrimitive(gpprim.arcsin, 1)
pset.addPrimitive(gpprim.arccos, 1)
pset.addPrimitive(np.tan, 1)
#pset.addPrimitive(np.sinh, 1)
#pset.addPrimitive(np.cosh, 1)
pset.addPrimitive(np.arcsinh, 1)
pset.addPrimitive(gpprim.arccosh, 1)
pset.addPrimitive(gpprim.arctanh, 1)


for i in range(nCost):
    pset.addTerminal(round(random.uniform(-10, 10), 4))

pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')

################################################## TOOLBOX #############################################################

d = np.ones((Ngenes+1))*1.0  # weights for linear combination of genes
d[0] = 0

creator.create("Fitness", base.Fitness, weights=(-1.0,))#, -0.1, -0.08, -1.0))
creator.create("Individual", list, fitness=creator.Fitness, w=list(d), height=1)
creator.create("SubIndividual", gp.PrimitiveTree)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("leg", tools.initIterate, creator.SubIndividual, toolbox.expr)  ### NEW ###
toolbox.register("legs", tools.initRepeat, list, toolbox.leg, n=Ngenes)  ### NEW ###
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)  ### NEW ###
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###
toolbox.register("compile", gp.compile, pset=pset)
#toolbox.register("evaluate", evaluate_mggp)
toolbox.register("select", funs.selDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True)
toolbox.register("mate", funs.xmate, Ngenes=Ngenes) ### NEW ##
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4) ### NEW ###
toolbox.register("mutate", funs.xmut, pset=pset, expr=toolbox.expr_mut, unipb=0.6, shrpb=0.3, Ngenes=Ngenes)
toolbox.decorate("mate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", funs.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", funs.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", funs.staticLimit(key=len, max_value=limit_size))

'''for i in range(len(pop)):
    d = pop[i].w
    vv = []
    for j in range(len(pop[i])):  # for loop to save variables to be optimized by least square
        for h in range(len(pop[i][j])):
            if type(pop[i][j][h]) == gp.Terminal and pop[i][j][h].name[0] != "A":
                vv.append(pop[i][j][h].value)
    for t in range(len(vv)):
        d = np.hstack((d, vv[t]))  # add variables to array with parameters to be optimzed

    try:
        #lst = least_squares(partial(evaluate, individual=pop[i]), d, method='lm')
        lst = minimize(evaluate, d, args=(pop[i],), method='BFGS')
        pop[i].w = lst.x[0:Ngenes + 1] # update the linear combination weights after optmization

        fit = lst.fun #np.sqrt(sum((lst.fun)**2))
        #if len(np.unique(lst.fun)) == 1:
         #   pop[i].fitness.values = 1e5,
        #else:
        pop[i].fitness.values = fit,
        if fit < best_fit:
            best_fit = fit
            best_ind = pop[i]
    except(ValueError, TypeError):
        pop[i].fitness.values = 1e5,
        continue'''
def main():
    nbCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nbCPU)
    toolbox.register("map", pool.map)

    pop = toolbox.population(100)  # creation of initial population

    pop = toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, output=output), pop)  # optimization on pop
    #pop = list(map(lst, pop)) # least square on pop
    #pop = list(map(evaluate_nonOpt, pop))  # evaluate fitness of individuals

    best_ind = funs.selBest(pop, 1)[0]
    best_fit = best_ind.fitness.values[0]
    n = 0
    while n <= Ngen:
        print("------------------------------------------------------------------------------------------------------------- GEN {}".format(n))
        #pop = list(map(lst, pop))  # least square on pop
        to_mate = funs.selBest(pop, int(len(pop)/2))
        offspring = funs.varOr(to_mate, toolbox, int(len(pop)), cxpb=cxpb, mutpb=mutpb)
        #offspring = list(map(evaluate_nonOpt, offspring))  # evaluate fitness of offspring

        if n % 50 == 0 and n >= 50:
            offspring = toolbox.map(partial(funs.optimize_ind, evaluate=evaluate, Ngenes=Ngenes), offspring)  # optimization on all pop
        else:
            offspring = toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, output=output), offspring)  # optimization on all pop
        pop[:] = toolbox.select(offspring + pop, len(pop)-1)
        best_ind = funs.selBest(pop, 1)[0]
        pop.append(best_ind)
        best_fit = best_ind.fitness.values[0]
        print("------------------------------------------------------------------------------------------------------------- {}".format(best_fit))
        string = str(best_ind.w[0])
        st = 1
        while st <= Ngenes:
            string = string + "+" + str(best_ind.w[st]) + "*" + str(best_ind[st-1])
            st += 1
        print(string)
        n += 1

    print(best_fit)
    string = str(best_ind.w[0])
    st = 1
    while st <= Ngenes:
        string = string + "+" + str(best_ind.w[st]) + "*" + str(best_ind[st-1])
        st += 1
    print(string)
    oo = evaluate_forplot(best_ind)
    plt.figure()
    plt.plot(output, 'o', label="Experimental data")
    plt.plot(oo, marker='.', label="Fitted data", color='r')
    plt.legend(loc="best")
    plt.show()

main()











