


def mggp_fun(nEph, ref_fun, interval, Ngen, terminals):
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

    sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")
    sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/GP_Experiments/MGGP")
    import GP_PrimitiveSet as gpprim

    def evaluate_lst(d, individual, ref_fun, interval, terminals):
        out_trees = []
        if terminals == 1:
            output = ref_fun(interval)
        elif terminals == 2:
            output = ref_fun(interval[0], interval[1])
        elif terminals == 3:
            output = ref_fun(interval[0], interval[1], interval[2])
        elif terminals == 4:
            output = ref_fun(interval[0], interval[1], interval[2], interval[3])
        for i in range(len(individual)):
            f = toolbox.compile(individual[i])
            if terminals == 1:
                res = f(interval)
            elif terminals == 2:
                res = f(interval[0], interval[1])
            elif terminals == 3:
                res = f(interval[0], interval[1], interval[2])
            elif terminals == 4:
                res = f(interval[0], interval[1], interval[2], interval[3])
            if type(res) != np.ndarray:
                res = np.ones(len(output)) * res
            out_trees.append(res * d[i + 1])
        output_eval = 0
        for j in range(len(out_trees)):
            output_eval = output_eval + out_trees[j]
        output_eval = output_eval + d[0]
        err = output - output_eval
        return err

    def evaluate_nonOpt(individual, ref_fun, interval, terminals):

        out_trees = []
        if terminals == 1:
            output = ref_fun(interval)
        elif terminals == 2:
            output = ref_fun(interval[0], interval[1])
        elif terminals == 3:
            output = ref_fun(interval[0], interval[1], interval[2])
        elif terminals == 4:
            output = ref_fun(interval[0], interval[1], interval[2], interval[3])
        for i in range(len(individual)):
            f = toolbox.compile(individual[i])
            if terminals == 1:
                res = f(interval)
            elif terminals == 2:
                res = f(interval[0], interval[1])
            elif terminals == 3:
                res = f(interval[0], interval[1], interval[2])
            elif terminals == 4:
                res = f(interval[0], interval[1], interval[2], interval[3])
            if type(res) != np.ndarray:
                res = np.ones(len(output)) * res
            out_trees.append(res * individual.w[i + 1])

        output_eval = 0
        for j in range(len(out_trees)):
            output_eval = output_eval + out_trees[j]
        output_eval = output_eval + individual.w[0]
        err = np.sqrt(sum((output - output_eval) ** 2))  # output-output_eval
        individual.fitness.values = err,
        return individual

    def evaluate(d, individual, ref_fun, interval, terminals):

        out_trees = []
        if terminals == 1:
            output = ref_fun(interval)
        elif terminals == 2:
            output = ref_fun(interval[0], interval[1])
        elif terminals == 3:
            output = ref_fun(interval[0], interval[1], interval[2])
        elif terminals == 4:
            output = ref_fun(interval[0], interval[1], interval[2], interval[3])
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
            if terminals == 1:
                res = f(interval)
            elif terminals == 2:
                res = f(interval[0], interval[1])
            elif terminals == 3:
                res = f(interval[0], interval[1], interval[2])
            elif terminals == 4:
                res = f(interval[0], interval[1], interval[2], interval[3])
            if type(res) != np.ndarray:
                res = np.ones(len(interval)) * res
            out_trees.append(res * d[i + 1])

        output_eval = 0
        for j in range(len(out_trees)):
            output_eval = output_eval + out_trees[j]
        output_eval = output_eval + d[0]
        err = np.sqrt(sum((output - output_eval) ** 2))  # output-output_eval
        return err

    ####     PARAMETERS DEFINITION  #######

    Ngenes = 2
    limit_height = 10
    limit_size = 15
    cxpb = 0.5
    mutpb = 0.5
    nVars = terminals

    pset = gp.PrimitiveSet("MAIN", nVars)
    pset.addPrimitive(operator.add, 2, name="Add")
    pset.addPrimitive(operator.sub, 2, name="Sub")
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(gpprim.TriAdd, 3)
    pset.addPrimitive(gpprim.TriMul, 3)
    pset.addPrimitive(np.tanh, 1, name="Tanh")
    # pset.addPrimitive(gpprim.Abs, 1)
    # pset.addPrimitive(Identity, 1)
    # pset.addPrimitive(gpprim.Neg, 1)
    # pset.addPrimitive(protDiv, 2, name="Div")
    # pset.addPrimitive(operator.pow, 2, name="Pow")
    pset.addPrimitive(gpprim.Sqrt, 1)
    pset.addPrimitive(gpprim.Log, 1)
    pset.addPrimitive(funs.modExp, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    # pset.addPrimitive(sigmoid, 1)
    # pset.addPrimitive(softplus, 1)

    for i in range(nEph):
        pset.addTerminal(round(random.uniform(-10, 10), 4))

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
    toolbox.register("evaluate", partial(evaluate, ref_fun=ref_fun, interval=interval, terminals=terminals))
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
        pop = list(toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, ref_fun=ref_fun, interval=interval, terminals=terminals), pop))  # optimization on pop
        # pop = list(map(lst, pop)) # least square on pop
        # pop = list(map(evaluate_nonOpt, pop))  # evaluate fitness of individuals

        best_ind = funs.selBest(pop, 1)[0]
        best_fit = best_ind.fitness.values[0]
        n = 0
        while n <= Ngen:
            print(
                "------------------------------------------------------------------------------------------------------------- GEN {}".format(
                    n))
            # pop = list(map(lst, pop))  # least square on pop
            to_mate = funs.selBest(pop, int(len(pop) / 2))
            offspring = funs.varOr(to_mate, toolbox, int(len(pop)), cxpb=cxpb, mutpb=mutpb)
            # offspring = list(map(evaluate_nonOpt, offspring))  # evaluate fitness of offspring
            #offspring = list(toolbox.map(partial(funs.optimize_ind, evaluate=evaluate, Ngenes=Ngenes, ref_fun=ref_fun, interval=interval, terminals=terminals), offspring))  # optimization on offspring
            offspring = list(toolbox.map(partial(funs.lst, evaluate_lst=evaluate_lst, ref_fun=ref_fun, interval=interval, terminals=terminals), offspring))  # optimization on all pop
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
        return best_fit, string

    best_fit, best_ind = main()
    del pset, toolbox, creator.Fitness, creator.Individual, creator
    return best_fit, best_ind













