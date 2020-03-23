"""MGGP_SNGP_variant_test.py: mix of MGGP and SNGP

Author(s): Francesco Marchetti
email: francesco.marchetti@strath.ac.uk

IDEA: to use the graph recombination of SNGP using small trees instead of single nodes as individuals and using the feature of MGGP
to optimize connections and constants.
Algorithm:
1 - initialize population randomly. Randomly create trees, constants and connections. There is a maximum number fo connections defined by the user,
    to limit the size of the final equation. Find N best connections among the ones created to define a "Hall of fame of best connections (must save also individuals and
    connection weights and constants in order to retrieve them in future. Comparison made on fitness"
2 - Loop goes on for N generations. Inside loop:
    - execute crossover between individuals involved in the best combinations and mutations on the other individuals. Mutation can be only beneficials.
      i.e. if new fitness worse than old, old inidividual is restored.
    - execute smut n times to define new combinations and after each new combination execute optimization of connection weights and variables.

Problem: how enforce exploitation??
"""

import scipy.io as sio
import numpy as np
import operator
import random
from deap import gp, algorithms
import matplotlib.pyplot as plt
from deap import base, creator, tools
from functools import partial
from scipy.optimize import least_squares
from copy import deepcopy, copy
from operator import eq
import re
from scipy.optimize import minimize, least_squares
import sys
import multiprocessing
import MGGP_SNGP_Functions as funs
from operator import attrgetter

sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")
import GP_PrimitiveSet as gpprim

#### FUNCTIONS DEFINITION ###########

def evaluate_p(individual):
    global input1, input2, output
    f = toolbox.compile(individual)
    output_eval = f(input1, input2)
    err = np.sqrt(sum((output - output_eval) ** 2))
    return err

def evaluate_ind(individual):
    global input1, input2, output
    f = toolbox.compile(individual)
    output_eval = f(input1, input2)
    err = np.sqrt(sum((output - output_eval) ** 2))
    return (err,)

def evaluate_indOpt(d, individual):
    global input1, input2, output
    cc = 0
    for i in range(len(individual)):
        if type(individual[i]) == gp.Terminal and individual[i].name[0] != "A":
            individual[i] = deepcopy(individual[i])
            individual[i].name = str(d[cc])
            individual[i].value = d[cc]
            cc += 1
    f = toolbox.compile(individual)
    output_eval = f(input1, input2)
    #err = np.sqrt(sum((output - output_eval) ** 2))
    err = 1 / np.std(output) * np.sqrt(1 / len(output) * sum((output - output_eval) ** 2))  # output-output_eval
    return err

def evaluate(d, individual):
    global input1, input2, output
    c = 0
    for i in range(len(individual.weights)):
        individual.weights[i] = d[c]
        c += 1
    cc = 0
    for i in range(len(individual.ind)):
        for j in range(len(individual.ind[i])):
            if type(individual.ind[i][j]) == gp.Terminal and individual.ind[i][j].name[0] != "A":
                individual.ind[i][j] = deepcopy(individual.ind[i][j])
                individual.ind[i][j].name = str(d[cc + c])
                individual.ind[i][j].value = d[cc + c]
                cc += 1
    individual.equation = []
    expr = individual.build_eq()
    individual.equation = expr
    f = toolbox.compile(expr)
    output_eval = f(input1, input2)
    #err = np.sqrt(sum((output-output_eval)**2))
    err = 1 / np.std(output) * np.sqrt(1 / len(output) * sum((output - output_eval) ** 2))  # output-output_eval
    return err

def evaluate_forplot(comb):
    global input1, input2, output
    f = toolbox.compile(comb.equation)
    output_eval = f(input1, input2)
    if np.shape(output_eval) == ():
        output_eval = output_eval * np.ones((len(input1)))
    return output_eval

def evaluateInd_forplot(ind):
    global input1, input2, output
    f = toolbox.compile(ind)
    output_eval = f(input1, input2)
    if np.shape(output_eval) == ():
        output_eval = output_eval * np.ones((len(input1)))
    return output_eval

def evaluate_lst(d, individual):
    global input1, input2, output
    c = 0
    for i in range(len(individual.weights)):
        individual.weights[i] = d[c]
        c += 1
    individual.equation = []
    expr = individual.build_eq()
    individual.equation = expr
    f = toolbox.compile(expr)
    output_eval = f(input1, input2)
    err = output - output_eval
    return err

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


####     PARAMETERS DEFINITION  #######

limit_height = 10  # height limit on individual tree
limit_size = 15  # size limit on individual tree
Ngen = 300  # maximum number of generations
cxpb = 0.4
mutpb = 0.6
nCost = 3  # number of constants created as terminals for tree
nVars = 2  # number fo variables
nElem_max = 2  # maximum number of trees involved in the final equation
Nmut = 300 # maximum number of connections mutations performed at each generation
nCross = 100
conn_max = 2
Ncomb_pop = 100  # size of combination population
Npop = 300  # size of individuals population
New_comb = 200  # new combinations to create at the end of evolutionary process to enrich combination population
Nopt = 50  # number of combination to optimize

pset = gp.PrimitiveSet("MAIN", nVars)
pset.addPrimitive(operator.add, 2, name="Add")
pset.addPrimitive(operator.sub, 2, name="Sub")
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(gpprim.TriAdd, 3)
#pset.addPrimitive(gpprim.TriMul, 3)
pset.addPrimitive(np.tanh, 1, name="Tanh")
#pset.addPrimitive(gpprim.lf, 1)
pset.addPrimitive(gpprim.square, 1)
pset.addPrimitive(gpprim.cubic, 1)
#pset.addPrimitive(gpprim.Abs, 1)
#pset.addPrimitive(Identity, 1)
#pset.addPrimitive(gpprim.Neg, 1)
#pset.addPrimitive(gpprim.ProtDiv, 2, name="Div")
#pset.addPrimitive(operator.pow, 2, name="Pow")
pset.addPrimitive(gpprim.Sqrt, 1)
pset.addPrimitive(gpprim.Log, 1)
pset.addPrimitive(funs.modExp, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)
#pset.addPrimitive(gpprim.arcsin, 1)
#pset.addPrimitive(gpprim.arccos, 1)
#pset.addPrimitive(np.tan, 1)
#pset.addPrimitive(np.sinh, 1)
#pset.addPrimitive(np.cosh, 1)
#pset.addPrimitive(np.arcsinh, 1)
#pset.addPrimitive(gpprim.arccosh, 1)
#pset.addPrimitive(gpprim.arctanh, 1)


for i in range(nCost):
    val = random.uniform(-1, 1)
    pset.addTerminal(val)
#pset.addTerminal(1.0)
pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')
################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0,))#, -0.1, -0.08, -1.0))    # MINIMIZATION OF THE FITNESS FUNCTION
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=4)  ### NEW ###
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  ### NEW ###
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate_ind)
toolbox.register("select", tools.selDoubleTournament, fitness_size=2, parsimony_size=1.0, fitness_first=True)#InclusiveTournament)
toolbox.register("mate", gp.cxOnePoint) ### NEW ##
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=4) ### NEW ###
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) ### NEW ###
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))


#############################   CREATION OF INITIAL COMBINATION     ##############################

def main():
    nbCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(nbCPU)
    toolbox.register("map", pool.map)

    old_entropy = 0
    for i in range(200):
        try_pop = funs.POP(toolbox.population(Npop))
        if try_pop.entropy > old_entropy and len(try_pop.indexes) == len(try_pop.categories) - 1:
            pop = try_pop.items
            old_entropy = try_pop.entropy

    #pop = toolbox.population(Npop)  # population creation
    comb_pop = []
    invalid_ind = [ind for ind in pop if not ind.fitness.valid] # individuals fitness evaluation
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof_ind = tools.HallOfFame(10) # initialization of hall of fame for individuals
    hof_ind.update(pop)

    hof_comb = funs.HallofFame(10)  # initialization of hall of fame to store best combinations
    for _ in range(Ncomb_pop):
        p_combination = funs.create_combinations(pop, nElem_max, evaluate_ind)  # creates first Ncomb_pop combinations
        hof_comb.update(p_combination)
        comb_pop.append(p_combination)
    combination = hof_comb[0] # start from best combination
    w = []  # creation of array of parameters to optimize
    for i in range(len(combination.weights)):
        w.append(combination.weights[i])
    d = np.array(w)
    for i in combination.constants:
        d = np.hstack((d, float(i)))

    ### parameters optimization through optimization
    if len(d) != 0:
        #res = minimize(evaluate, d, args=(combination,), method='BFGS')
        res = least_squares(evaluate_lst, d, args=(combination,), method='lm', loss='linear')
        combination, extra_inds = funs.update_comb_afterOpt(pop, combination, res.x, toolbox, evaluate_ind)

    hof_comb.update(combination)

    n = 0
    while n <= Ngen:
        extra = []
        ################ perform mutation and crossover between combinations  ########################
        # the population is updated after each mutation and crossover with "optimized" individuals
        ##########################  MUTATION  ##################################
        Coffspring = []
        if n % 50 == 0 and n >= 50:
            opt_res = toolbox.map(partial(funs.multi_opt, comb_pop=comb_pop, evaluate=evaluate, Nind=Nopt), range(Nopt))
            for ress in opt_res:
                comb = ress[0]
                opt_a = ress[1]
                comb_up, extra_inds = funs.update_comb_afterOpt(pop, comb, opt_a, toolbox, evaluate_ind)
                comb_up = funs.check_length(comb_up, nElem_max, pop, evaluate_ind)
                hof_comb.update(comb_up)
                Coffspring.append(comb_up)
                extra.extend(extra_inds)
        #opt_res = toolbox.map(partial(funs.multi_mutation, comb_pop=hof_comb, pop=pop, nElem_max=nElem_max, evaluate_ind=evaluate_ind, evaluate=evaluate), range(Nmut))
        #sel_pop = deepcopy(funs.selBest(comb_pop, len(comb_pop)))
        opt_res = list(toolbox.map(partial(funs.multi_mutation, comb_pop=comb_pop, pop=pop, nElem_max=nElem_max, evaluate_lst=evaluate_lst, evaluate_ind=evaluate_ind, Nind=Nmut), range(Nmut)))
        for ress in opt_res:
            comb = ress[0]
            opt_a = ress[1]
            comb_up, extra_inds = funs.update_comb_afterOpt(pop, comb, opt_a, toolbox, evaluate_ind)
            comb_up = funs.check_length(comb_up, nElem_max, pop, evaluate_ind)
            hof_comb.update(comb_up)
            Coffspring.append(comb_up)
            extra.extend(extra_inds)
        ###################  CROSSOVER  #######################################
        opt_res_cross = toolbox.map(partial(funs.multi_cross, comb_pop=comb_pop, pop=pop, evaluate_ind=evaluate_ind, evaluate_lst=evaluate_lst), range(nCross))
        for res_cross in opt_res_cross:
            comb1 = res_cross[0]
            comb2 = res_cross[1]
            opt_arr = res_cross[2]
            which = res_cross[3]
            if which == 1:
                comb1, extra_inds = funs.update_comb_afterOpt(pop, comb1, opt_arr, toolbox, evaluate_ind)
            else:
                comb2, extra_inds = funs.update_comb_afterOpt(pop, comb2, opt_arr, toolbox, evaluate_ind)
            comb1 = funs.check_length(comb1, nElem_max, pop, evaluate_ind)
            comb2 = funs.check_length(comb2, nElem_max, pop, evaluate_ind)
            hof_comb.update(comb1)
            hof_comb.update(comb2)
            Coffspring.append(comb1)
            Coffspring.append(comb2)
            extra.extend(extra_inds)

        ############### perform mutation or crossover between individuals ######################

        offspring = funs.varOr(pop+extra, toolbox, int(len(pop)*1.5), cxpb=cxpb, mutpb=mutpb) # creation of offspring through crossover and mutation among individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hof_ind.update(offspring)
        pop[:] = toolbox.select(offspring+pop+extra, len(pop))
        #if n % 10 == 0 and n >= 10:
         #   pop_toOpt = deepcopy(pop)
          #  opt_res = toolbox.map(partial(funs.optimize_ind, evaluate=evaluate_indOpt), pop_toOpt)
           # pop = opt_res
        ############ create new combinations with best new individuals ##############
        best_inds = funs.find_best(pop, int(len(pop) / 2))
        best_inds = funs.shuffle(best_inds)
        for _ in range(New_comb):
            new_comb = funs.create_combinations(best_inds, nElem_max, evaluate_ind)
            comb_pop.append(new_comb)
            hof_comb.update(new_comb)

        ####### update combination hall of fame with new individuals #####
        tot_cpop = Coffspring + comb_pop
        for cc in tot_cpop:
            cc.update(pop, cc.connections, cc.weights, evaluate_ind)
        #comb_pop = funs.comb_selDoubleTournament(tot_cpop, Ncomb_pop, fitness_size=2, parsimony_size=1.3, fitness_first=True)
        comb_pop = funs.selBest(tot_cpop, Ncomb_pop)


        '''w = []
        for i in range(len(new_comb.weights)):
            w.append(new_comb.weights[i])
        d = np.array(w)
        for i in new_comb.constants:
            d = np.hstack((d, float(i)))
        if len(d) != 0:
            #print("optimize new comb")
            res = minimize(evaluate, d, args=(new_comb,), method='BFGS', options={'disp':False})
            new_comb, pop = funs.update_comb_afterOpt(pop, new_comb, res.x, toolbox, evaluate_ind)
            hof_comb.update(new_comb)'''

        best_fit = hof_comb[0].fitness
        #print also best individual, not only best combination
        #best_node = funs.find_best(pop,1)[0]
        print("\n")
        print("GEN:{} BEST FIT COMB:{} BEST FIT NODE:{}".format(n, best_fit, hof_ind[0].fitness.values[0]))
        print(hof_comb[0].equation)
        print(hof_ind[0])
        n += 1

    pool.close()
    pool.join()
    oo = evaluate_forplot(hof_comb[0])
    uu = evaluateInd_forplot(hof_ind[0])
    plt.figure()
    plt.plot(output, 'o', label="Experimental data")
    plt.plot(oo, marker='.', label="Best combination", color='r')
    plt.plot(uu, marker='.', label="Best individual", color='k')
    plt.legend(loc="best")
    plt.show()

main()









