

#### FUNCTIONS DEFINITION ###########


def mggp_sngp_fun(ref_fun, interval, Ngen, nEph, terminals):
    import numpy as np
    import operator
    import random
    from deap import gp
    from deap import base, creator, tools
    from functools import partial
    from copy import deepcopy
    from scipy.optimize import minimize, least_squares
    import sys
    import multiprocessing
    import MGGP_SNGP_funs as funs

    sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")

    import GP_PrimitiveSet as gpprim

    def evaluate_p(individual,ref_fun, interval):
        f = toolbox.compile(individual)
        output_eval = f(interval)
        output = ref_fun(interval)
        err = np.sqrt(sum((output - output_eval) ** 2))
        return err

    def evaluate_ind(individual, ref_fun, interval, terminals):
        f = toolbox.compile(individual)
        if terminals == 1:
            output_eval = f(interval)  # output of reference function
            output = ref_fun(interval)
        elif terminals == 2:
            output_eval = f(interval[0], interval[1])  # output of reference function
            output = ref_fun(interval[0], interval[1])
        elif terminals == 3:
            output_eval = f(interval[0], interval[1], interval[2])  # output of reference function
            output = ref_fun(interval[0], interval[1], interval[2])
        elif terminals == 4:
            output_eval = f(interval[0], interval[1], interval[2], interval[3])  # output of reference function
            output = ref_fun(interval[0], interval[1], interval[2], interval[3])
        err = np.sqrt(sum((output - output_eval) ** 2))
        return (err,)

    def evaluate_lst(d, individual, ref_fun, interval, terminals):
        c = 0
        for i in range(len(individual.weights)):
            individual.weights[i] = d[c]
            c += 1
        individual.equation = []
        expr = individual.build_eq()
        individual.equation = expr
        f = toolbox.compile(expr)
        if terminals == 1:
            output_eval = f(interval)  # output of reference function
            output = ref_fun(interval)
        elif terminals == 2:
            output_eval = f(interval[0], interval[1])  # output of reference function
            output = ref_fun(interval[0], interval[1])
        elif terminals == 3:
            output_eval = f(interval[0], interval[1], interval[2])  # output of reference function
            output = ref_fun(interval[0], interval[1], interval[2])
        elif terminals == 4:
            output_eval = f(interval[0], interval[1], interval[2], interval[3])  # output of reference function
            output = ref_fun(interval[0], interval[1], interval[2], interval[3])
        err = output - output_eval
        return err

    def evaluate(d, individual, ref_fun, interval):
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
        output_eval = f(interval)
        output = ref_fun(interval)
        err = np.sqrt(sum((output-output_eval)**2))
        return err

    def evaluate_forplot(comb):
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



    ####     PARAMETERS DEFINITION  #######

    limit_height = 15  # height limit on individual tree
    limit_size = 20  # size limit on individual tree
    cxpb = 0.9
    mutpb = 0.1
    nCost = nEph  # number of constants created as terminals for tree
    nVars = terminals  # number fo variables
    nElem_max = 2  # maximum number of trees involved in the final equation
    Nmut = 10 # maximum number of connections mutations performed at each generation
    nCross = 24
    conn_max = 2
    Ncomb_pop = 100
    Npop = 300
    New_comb = 100

    pset = gp.PrimitiveSet("MAIN", nVars)
    pset.addPrimitive(operator.add, 2, name="Add")
    pset.addPrimitive(operator.sub, 2, name="Sub")
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(gpprim.TriAdd, 3)
    pset.addPrimitive(np.tanh, 1, name="Tanh")
    #pset.addPrimitive(gpprim.Abs, 1)
    #pset.addPrimitive(Identity, 1)
    #pset.addPrimitive(gpprim.Neg, 1)
    #pset.addPrimitive(protDiv, 2, name="Div")
    #pset.addPrimitive(operator.pow, 2, name="Pow")
    pset.addPrimitive(gpprim.Sqrt, 1)
    pset.addPrimitive(gpprim.Log, 1)
    pset.addPrimitive(funs.modExp, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)


    for i in range(nCost):
        val = random.uniform(-1, 1)
        pset.addTerminal(val)
    #pset.addTerminal(1.0)
    if terminals == 1:
        pset.renameArguments(ARG0='x1')
    elif terminals == 2:
        pset.renameArguments(ARG0='x1')
        pset.renameArguments(ARG1='x2')
    elif terminals == 3:
        pset.renameArguments(ARG0='x1')
        pset.renameArguments(ARG1='x2')
        pset.renameArguments(ARG2='x3')
    elif terminals == 4:
        pset.renameArguments(ARG0='x1')
        pset.renameArguments(ARG1='x2')
        pset.renameArguments(ARG2='x3')
        pset.renameArguments(ARG3='x4')

    ################################################## TOOLBOX #############################################################

    creator.create("Fitness", base.Fitness, weights=(-1.0,))#, -0.1, -0.08, -1.0))    # MINIMIZATION OF THE FITNESS FUNCTION
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=6)  ### NEW ###
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  ### NEW ###
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", partial(evaluate_ind, ref_fun=ref_fun, interval=interval, terminals=terminals))
    toolbox.register("select", tools.selDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True)#InclusiveTournament)
    toolbox.register("mate", gp.cxOnePoint) ### NEW ##
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=6) ### NEW ###
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) ### NEW ###
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))


    #############################   CREATION OF INITIAL COMBINATION     ##############################

    def main():
        #nbCPU = multiprocessing.cpu_count()
        #pool = multiprocessing.Pool(nbCPU)
        toolbox.register("map", map)

        old_entropy = 0
        for i in range(200):
            try_pop = funs.POP(toolbox.population(Npop))
            if try_pop.entropy > old_entropy and len(try_pop.indexes) == len(try_pop.categories) - 1:
                pop = try_pop.items
                old_entropy = try_pop.entropy

        #pop = toolbox.population(Npop)  # population creation
        comb_pop = []
        invalid_ind = [ind for ind in pop if not ind.fitness.valid] # individuals fitness evaluation
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof_ind = tools.HallOfFame(10) # initialization of hall of fame for individuals
        hof_ind.update(pop)

        hof_comb = funs.HallofFame(5)  # initialization of hall of fame to store best combinations
        for _ in range(Ncomb_pop):
            p_combination = funs.create_combinations(pop, nElem_max, evaluate_ind, ref_fun, interval, terminals)  # creates first Ncomb_pop combinations
            hof_comb.update(p_combination)
            comb_pop.append(p_combination)
        combination = hof_comb[0] # start from best combination
        w = []  # creation of array of parameters to optimize
        for i in range(len(combination.weights)):
            w.append(combination.weights[i])
        d = np.array(w)
        #for i in combination.constants:
         #   d = np.hstack((d, float(i)))
        ### parameters optimization through optimization
        if len(d) != 0:
            #res = minimize(evaluate, d, args=(combination, ref_fun, interval), method='BFGS')
            res = least_squares(evaluate_lst, d, args=(combination, ref_fun, interval, terminals), method='lm', loss='linear')
            combination, extra_inds = funs.update_comb_afterOpt(pop, combination, res.x, toolbox, evaluate_ind, ref_fun, interval, terminals)
        hof_comb.update(combination)

        n = 0
        while n <= Ngen:
            extra = []
            ################ perform mutation and crossover between combinations  ########################
            # the population is updated after each mutation and crossover with "optimized" individuals
            ##########################  MUTATION  ##################################
            Coffspring = []
            #opt_res = list(toolbox.map(partial(funs.multi_mutation, comb_pop=hof_comb, pop=pop, nElem_max=nElem_max, evaluate_ind=evaluate_ind, evaluate=evaluate, ref_fun=ref_fun, interval=interval), range(Nmut)))
            opt_res = list(toolbox.map(partial(funs.multi_mutation, comb_pop=hof_comb, pop=pop, nElem_max=nElem_max, evaluate_lst=evaluate_lst, evaluate_ind=evaluate_ind, ref_fun=ref_fun, interval=interval, terminals=terminals), range(Nmut)))

            for ress in opt_res:
                comb = ress[0]
                opt_a = ress[1]
                comb_up, extra_inds = funs.update_comb_afterOpt(pop, comb, opt_a, toolbox, evaluate_ind, ref_fun, interval, terminals)
                comb_up = funs.check_length(comb_up, nElem_max, pop, evaluate_ind)
                hof_comb.update(comb_up)
                Coffspring.append(comb_up)
                extra.extend(extra_inds)
            ###################  CROSSOVER  #######################################
            opt_res_cross = toolbox.map(partial(funs.multi_cross, hof_comb=comb_pop, pop=pop, evaluate_ind=evaluate_ind, evaluate=evaluate), range(nCross))
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
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof_ind.update(offspring)
            pop[:] = toolbox.select(offspring+pop+extra, len(pop))

            ############ create new combinations with best new individuals ##############
            best_inds = funs.find_best(pop, int(len(pop) / 2))
            best_inds = funs.shuffle(best_inds)
            for _ in range(New_comb):
                new_comb = funs.create_combinations(best_inds, nElem_max, evaluate_ind, ref_fun, interval, terminals)
                comb_pop.append(new_comb)
                hof_comb.update(new_comb)

            ####### update combination hall of fame with new individuals #####
            tot_cpop = Coffspring + comb_pop
            for cc in tot_cpop:
                cc.update(pop, cc.connections, cc.weights, evaluate_ind, ref_fun, interval, terminals)
            #comb_pop = funs.comb_selDoubleTournament(tot_cpop, Ncomb_pop, fitness_size=2, parsimony_size=1.4, fitness_first=True)
            comb_pop = funs.shuffle(funs.selBest(tot_cpop, Ncomb_pop))

            best_fit = hof_comb[0].fitness
            #print also best individual, not only best combination
            #best_node = funs.find_best(pop,1)[0]
            print("GEN:{} BEST FIT COMB:{} BEST FIT NODE:{}".format(n, best_fit, hof_ind[0].fitness.values[0]))
            print(hof_comb[0].equation)
            print(hof_ind[0])
            n += 1

        #pool.close()
        #pool.join()
        return best_fit, hof_comb[0].equation
    best_fit, best_ind = main()
    del pset, toolbox, creator.Fitness, creator.Individual, creator
    return best_fit, best_ind









