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
    # err = sum(abs(output - output_eval))
    err = np.sqrt(sum((output - output_eval) ** 2))
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
# pset.addPrimitive(gpprim.lf, 1)
pset.addPrimitive(gpprim.ProtDiv, 2)
pset.addPrimitive(np.cos, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.tanh, 1)
# pset.addPrimitive(gpprim.Sqrt, 1)
# pset.addPrimitive(gpprim.TriAdd, 3)
# pset.addPrimitive(gpprim.TriMul, 3)
rand_terminals = []
for i in range(nEph):
    # pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 4))
    rand_terminals.append(1.0)  # round(random.uniform(-1,1), 5))
    pset.addTerminal("rand{}".format(i), rand_terminals[i])
pset.renameArguments(ARG0='x')
# pset.renameArguments(ARG1='y')

creator.create("Individual", funs.Individual)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)

if __name__ == "__main__":
    # nbCPU = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(nbCPU)
    # toolbox.register("map", pool.map)
    L = 500  # population size
    ne = 20  # number of epochs
    nt = 2  # number of threads
    d = 7  # maximum tree depth
    le = 500  # epoch length
    li = 150  # numnber of iterations for local search algorithm for identity nodes
    lv = 150  # numnber of iterations for local search algorithm for trnasformed variables
    nh = 120  # number of head partition nodes
    nv = terminals  # number of transformed variables
    pop = toolbox.population(L)  # create population of random individuals with lenght 1

    for i in range(nEph):
        pop.insert(i, funs.Individual(pset.terminals[object][i + terminals]))  # insert constants at begininng of population

    for i in range(terminals):  ## insert terminals after the constant variables
        pop.insert(i + nEph, funs.Individual(pset.terminals[object][i]))

    pop, true_nh = funs.InitialisePopTV(deepcopy(pop), output, toolbox, terminals, nEph, pset, nh, nv)  # initialize population of nodes P

    Ngen = 200

    ###########################   EVOLUTION  #######################################

    fit_old = sum(pop[ll].r for ll in range(len(pop)))  # fitness of old pupulation
    best_soFar = 0    # find best individual
    best_soFar_fit = pop[0].r
    best_soFar_index = 0
    best_soFar_eq = []
    for i in range(len(pop)):
        if pop[i].r < best_soFar_fit:
            best_soFar_fit = pop[i].r
            best_soFar_index = i
    best_soFar = deepcopy(pop[best_soFar_index])
    best_soFar_eq = funs.retrieveFun_fromIndTV(pop, pop[best_soFar_index], terminals, nEph, rand_terminals, true_nh)

    for run in range(Ngen):
        work_pop = deepcopy(pop)  # work with copy of pop
        #n_mut = random.randint(1, N)  # choose the number of mutations
        print("----------- Evaluation: {}, Number of mutations: 1 --------------------------".format(run))
        #for _ in range(n_mut):
        work_pop[:], chosen = funs.smutTV(work_pop, terminals, nEph, true_nh)  ###  MUTATION ####
        ################### REEVALUATE POPULATION  ########################
        while chosen < len(work_pop):
            work_pop[chosen].O = []
            arity = work_pop[chosen].u[0].arity
            vals = np.zeros((arity, len(output)))
            for l in range(len(work_pop[chosen].S)):
                vals[l] = np.array(work_pop[chosen].w[l]) * np.array(pop[work_pop[chosen].S[l]].O)
            fun = eval(work_pop[chosen].u[0].name, pset.context)
            if arity == 1:
                work_pop[chosen].O = fun(vals[0])
            elif arity == 2:
                work_pop[chosen].O = fun(vals[0], vals[1])
            elif arity == 3:
                work_pop[chosen].O = fun(vals[0], vals[1], vals[2])
            work_pop[chosen].r = np.sqrt(sum((np.array(work_pop[chosen].O) - output) ** 2))
            chosen += 1
        ### EVALUATE CURRENT BEST INDIVIDUAL
        fit_new = sum(work_pop[ll].r for ll in range(len(work_pop)))  ##### FITNESS OF THE NEW POPULATION

        for _ in range(lv):
            ww_pop = deepcopy(work_pop)
            ch = random.random()
            if ch <= 0.5:
                ww_pop[nEph+terminals+true_nh].optimize_weights()
                ww_pop[nEph + terminals + true_nh].eval_v(ww_pop, output)
            else:
                ww_pop[nEph + terminals + true_nh].change_connections(true_nh, nEph)
                ww_pop[nEph + terminals + true_nh].eval_v(ww_pop, output)
            chosen = nEph + terminals*2 + true_nh
            while chosen < len(ww_pop):
                ww_pop[chosen].O = []
                arity = ww_pop[chosen].u[0].arity
                vals = np.zeros((arity, len(output)))
                for l in range(len(ww_pop[chosen].S)):
                    vals[l] = np.array(ww_pop[chosen].w[l]) * np.array(pop[ww_pop[chosen].S[l]].O)
                fun = eval(ww_pop[chosen].u[0].name, pset.context)
                if arity == 1:
                    ww_pop[chosen].O = fun(vals[0])
                elif arity == 2:
                    ww_pop[chosen].O = fun(vals[0], vals[1])
                elif arity == 3:
                    ww_pop[chosen].O = fun(vals[0], vals[1], vals[2])
                ww_pop[chosen].r = np.sqrt(sum((np.array(ww_pop[chosen].O) - output) ** 2))
                chosen += 1

            currBest_fit = ww_pop[0].r
            currBest = ww_pop[0]
            currBest_eq = []
            currBest_index = 0
            for i in range(len(ww_pop)):
                if ww_pop[i].r < currBest_fit:
                    currBest_fit = deepcopy(ww_pop[i].r)
                    currBest_index = i
            currBest = deepcopy(ww_pop[currBest_index])
            currBest_eq = funs.retrieveFun_fromIndTV(ww_pop, ww_pop[currBest_index], terminals, nEph, rand_terminals, true_nh)
            if currBest_fit < best_soFar_fit and best_soFar_index >= nEph + terminals*2 + true_nh:
                best_soFar_fit = currBest_fit
                best_soFar = currBest
                best_soFar_eq = currBest_eq
                best_soFar_index = currBest_index
                fit_old = fit_new
                # ch = random.random()
                # if ch < 0.5:
                work_pop = deepcopy(ww_pop)
                #print(best_soFar_fit)
                #eq = funs.convert_equationTV(best_soFar_eq, nEph, rand_terminals, true_nh, terminals)
                #print(eq)
                #print(fit_old)
            else:
                pass
                #print("Negative outcome")
                #print(best_soFar_fit)
                #eq = funs.convert_equationTV(best_soFar_eq, nEph, rand_terminals, true_nh, terminals)
                #print(eq)
                #print(fit_old)
        print(best_soFar_fit)
        eq = funs.convert_equationTV(best_soFar_eq, nEph, rand_terminals, true_nh, terminals)
        print(eq)


    print("\n")
    print("Fitness of best population: {}".format(fit_old))
    print("Best root of population: {}".format(best_soFar_index))
    print("Fitness of best individual: {}".format(best_soFar_fit))
    # print(eq)
    eq = list(best_soFar_eq)

    final_eq = funs.convert_equationTV(eq, nEph, rand_terminals)
    print(final_eq)
    oo, test = evaluate(final_eq)
    # print(test)
    plt.figure()
    plt.plot(output, 'o', label="Experimental data")
    plt.plot(oo, marker='.', label="Fitted data", color='r')
    plt.legend(loc="best")
    plt.show()

