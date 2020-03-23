'''SNGP_test.py: implementation of Single Node Genetic Programming

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk

References:
[1] J. Kubalik, E. Alibekov, J. Zegklitz, R. Babuska. Hybrid Single Node Genetic Programming for Symbolic Regression.
    Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics). 2016.'''

import scipy.io as sio
import numpy as np
import operator
import random
from deap import gp
import matplotlib.pyplot as plt
from deap import base, creator, tools
from copy import deepcopy
import SNGP_Functions as funs
import multiprocessing
import sys
sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")
import GP_PrimitiveSet as gpprim

'''mat_contents = sio.loadmat('/home/francesco/Desktop/Git_workspace/IC4A2S/GP_Experiments/dataModB118.mat')
files = mat_contents['VV']
input1 = np.zeros(len(files))
input2 = np.zeros(len(files))
output = np.zeros(len(files))
for i in range(len(files)):
    input1[i] = files[i][0]
    input2[i] = files[i][1]
    output[i] = files[i][2]'''
mat_contents = sio.loadmat('/home/francesco/Desktop/Git_workspace/IC4A2S/GP_Experiments/xydata_1D.mat')
input1 = mat_contents['x'].T[0]
output = mat_contents['y'].T[0]


def evaluate(individual):
    global input1, output
    output_fun = toolbox.compile(expr=individual)
    output_eval = output_fun(input1)
    #err = sum(abs(output - output_eval))
    err = np.sqrt(sum((output-output_eval)**2))
    try:
        return list(output_eval), err
    except(TypeError):
        output_eval = np.ones((len(output))) * output_eval
        return list(output_eval), err


terminals = 1
nEph = 1

pset = gp.PrimitiveSet("MAIN", terminals)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(gpprim.modExp, 1)
#pset.addPrimitive(gpprim.lf, 1)
#pset.addPrimitive(gpprim.ProtDiv, 2)
pset.addPrimitive(np.cos, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.tanh, 1)
#pset.addPrimitive(gpprim.Sqrt, 1)
#pset.addPrimitive(gpprim.TriAdd, 3)
#pset.addPrimitive(gpprim.TriMul, 3)
rand_terminals = []
for i in range(nEph):
    #pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 4))
    rand_terminals.append(1.0)#round(random.uniform(-1,1), 5))
    pset.addTerminal("rand{}".format(i), rand_terminals[i])
pset.renameArguments(ARG0='x')
#pset.renameArguments(ARG1='y')

creator.create("Individual", funs.Individual)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)

if __name__ == "__main__":
    #nbCPU = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(nbCPU)
    #toolbox.register("map", pool.map)

    pop = toolbox.population(100)  # create population of random individuals with lenght 1

    for i in range(nEph):
        pop.insert(i, funs.Individual(pset.terminals[object][i+terminals])) # insert constants at begininng of population

    for i in range(terminals): ## insert terminals after the constant variables
        pop.insert(i+nEph, funs.Individual(pset.terminals[object][i]))

    pop = funs.InitialisePop(deepcopy(pop), output, toolbox, terminals, nEph, pset)  # initialize population of nodes P

    i = nEph + terminals
    #print("Start optimization")
    while i < len(pop):  # evaluate and optimize P
        opt_ind = funs.optimize_weights(deepcopy(pop[i]), output, deepcopy(pop), pset)
        pop[i] = opt_ind
        i += 1
    #print("End optimization")
    Ngen = 200
    t = 5 # nodes selected in modified tournament
    N = 10 # number of mutations performed at each epoch
    ###########################   EVOLUTION  #######################################

    fit_old = sum(pop[ll].r for ll in range(len(pop)))
    best_soFar = 0   # find best individual
    best_soFar_fit = pop[0].r
    best_soFar_index = 0
    best_soFar_eq = []
    for i in range(len(pop)):
        if pop[i].r < best_soFar_fit:
            best_soFar_fit = pop[i].r
            best_soFar_index = i
    best_soFar = deepcopy(pop[best_soFar_index])
    best_soFar_eq = funs.retrieveFun_fromInd(pop, pop[best_soFar_index], terminals, nEph, rand_terminals)

    for run in range(Ngen):
        work_pop = deepcopy(pop)  # work with copy of pop
        n_mut = random.randint(1, N)  # choose the number of mutations
        print("----------- Evaluation: {}, Number of mutations: {} --------------------------".format(run, n_mut))
        chosen_nodes = []
        for _ in range(n_mut):
            work_pop[:], chosen = funs.smut(work_pop, t, terminals, nEph)  ###  MUTATION ####
            chosen_nodes.append(chosen)
        chosen_nodes.sort()
        ################### REEVALUATE POPULATION  ########################
        chosen = chosen_nodes[0]
        while chosen < len(work_pop):
            for j in range(len(work_pop[chosen].w)):
                work_pop[chosen].w[j] = 1.0
            opt_ind = funs.optimize_weights(deepcopy(work_pop[chosen]), output, deepcopy(work_pop), pset)
            work_pop[chosen] = opt_ind
            chosen += 1

        ### EVALUATE CURRENT BEST INDIVIDUAL
        fit_new = sum(work_pop[ll].r for ll in range(len(work_pop)))  ##### FITNESS OF THE NEW POPULATION
        currBest_fit = work_pop[0].r
        currBest = work_pop[0]
        currBest_eq = []
        currBest_index = 0
        for i in range(len(work_pop)):
            if work_pop[i].r < currBest_fit:
                currBest_fit = deepcopy(work_pop[i].r)
                currBest_index = i
        currBest = deepcopy(work_pop[currBest_index])
        currBest_eq = funs.retrieveFun_fromInd(work_pop, work_pop[currBest_index], terminals, nEph, rand_terminals)
        if currBest_fit < best_soFar_fit and best_soFar_index >= nEph+terminals:
            best_soFar_fit = currBest_fit
            best_soFar = currBest
            best_soFar_eq = currBest_eq
            best_soFar_index = currBest_index
            fit_old = fit_new
            #ch = random.random()
            #if ch < 0.5:
            pop = deepcopy(work_pop)
            pop = funs.moveLeft(deepcopy(pop), best_soFar_index, terminals, nEph, pset, output, toolbox)
            #else:
             #   pop = deepcopy(funs.moveRight(pop, best_soFar_index, terminals, nEph, pset, output, toolbox))
            #print("Start optimization after move")
            #start = nEph + terminals
            #while start < len(work_pop):
            #    opt_ind = funs.optimize_weights(deepcopy(work_pop[start]), output, deepcopy(pop), pset)
            #    work_pop[start] = opt_ind
            #    start += 1
            #print("End optimization after move")
            print(best_soFar_fit)
            eq = funs.convert_equation(best_soFar_eq, nEph, rand_terminals)
            print(eq)
            print(fit_old)

        else:
            print("Negative outcome")
            print(best_soFar_fit)
            eq = funs.convert_equation(best_soFar_eq, nEph, rand_terminals)
            print(eq)
            print(fit_old)

        '''fit_new = sum(pop[ll].r for ll in range(len(pop)))  ##### FITNESS OF THE NEW POPULATION
    
        ###################### CHECK IF MUTATION WAS BENEFICIAL, ELSE DISCARD CHANGES ########################
    
        if fit_new < fit_old:
            fit_old = fit_new
            currBest_fit = pop[0].r
            currBest = pop[0]
            currBest_eq = []
            currBest_index = 0
            for i in range(len(pop)):
                if pop[i].r < currBest_fit:
                    currBest_fit = deepcopy(pop[i].r)
                    currBest_index = i
            currBest = deepcopy(pop[currBest_index])
            currBest_eq = funs.retrieveFun_fromInd(pop, pop[currBest_index], terminals, nEph, rand_terminals)
            eq = funs.convert_equation(currBest_eq, nEph, rand_terminals)
            print(currBest_fit)
            print(eq)
            print(fit_new)
        else:
            pop = old_pop
            print("Negative outcome")
            print(fit_old)'''


    print("\n")
    print("Fitness of best population: {}".format(fit_old))
    print("Best root of population: {}".format(best_soFar_index))
    print("Fitness of best individual: {}".format(best_soFar_fit))
    #print(eq)
    eq = list(best_soFar_eq)

    final_eq = funs.convert_equation(eq, nEph, rand_terminals)
    print(final_eq)
    oo, test = evaluate(final_eq)
    #print(test)
    plt.figure()
    plt.plot(output, 'o', label="Experimental data")
    plt.plot(oo, marker='.', label="Fitted data", color='r')
    plt.legend(loc="best")
    plt.show()











