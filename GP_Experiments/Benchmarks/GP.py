
def gp_fun(nEph, limit_size, limit_height, size_pop, ref_fun, interval, size_gen, terminals, ev):
    import numpy as np
    import operator
    import random
    from deap import gp
    from deap import base, creator, tools, algorithms
    import sys
    from functools import partial
    import multiprocessing

    sys.path.insert(1, "/home/francesco/Desktop/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")
    import GP_PrimitiveSet as gpprim

    def evaluate(individual, ref_fun, interval, terminals, pset):
        '''evaluate R2 error'''
        output_fun = toolbox.compile(expr=individual, pset=pset)  # individual evaluated through GP
        if terminals == 1:
            output_eval = output_fun(interval)  # output of reference function
            output = ref_fun(interval)
        elif terminals == 2:
            output_eval = output_fun(interval[0], interval[1])  # output of reference function
            output = ref_fun(interval[0], interval[1])
        elif terminals == 3:
            output_eval = output_fun(interval[0], interval[1], interval[2])  # output of reference function
            output = ref_fun(interval[0], interval[1], interval[2])
        elif terminals == 4:
            output_eval = output_fun(interval[0], interval[1], interval[2], interval[3])  # output of reference function
            output = ref_fun(interval[0], interval[1], interval[2], interval[3])
        err = np.sqrt(sum((output - output_eval) ** 2))
        return (err,)

    '''Main function called to perform GP evaluation'''

    cxpb = 0.8  # crossover rate
    mutpb = 0.1  # 1 - cxpb - elite  # mutation rate

    Mu = int(size_pop)
    Lambda = int(size_pop * 1.3)


    pset = gp.PrimitiveSet("MAIN", terminals)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(gpprim.TriAdd, 3)
    pset.addPrimitive(np.tanh, 1)
    # pset.addPrimitive(gpprim.Exp, 1)
    #pset.addPrimitive(gpprim.Pow, 1)
    pset.addPrimitive(gpprim.Log, 1)
    pset.addPrimitive(gpprim.Sqrt, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.sin, 1)
    # pset.addPrimitive(gpprim.lf, 1)
    for i in range(nEph):
        pset.addEphemeralConstant("rand{}{}".format(i, ev), lambda: round(random.uniform(-10, 10), 4))
    if terminals == 1:
        pset.renameArguments(ARG0='input1')
    elif terminals == 2:
        pset.renameArguments(ARG0='input1')
        pset.renameArguments(ARG1='input2')
    elif terminals == 3:
        pset.renameArguments(ARG0='input1')
        pset.renameArguments(ARG1='input2')
        pset.renameArguments(ARG2='input3')
    elif terminals == 4:
        pset.renameArguments(ARG0='input1')
        pset.renameArguments(ARG1='input2')
        pset.renameArguments(ARG2='input3')
        pset.renameArguments(ARG3='input4')
    ################################################## TOOLBOX OF FIRST GP##################################################

    creator.create("Fitness", base.Fitness, weights=(-1.0,))  # MINIMIZATION OF THE FITNESS FUNCTION
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=6)  #### OLD ####
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  #### OLD ####
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  #### OLD ####
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", partial(evaluate, ref_fun=ref_fun, interval=interval, terminals=terminals, pset=pset))  ### OLD ###
    toolbox.register("select", tools.selDoubleTournament, fitness_size=2, parsimony_size=1.2, fitness_first=True)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)  # gp.cxSemantic, gen_func=gp.genFull, pset=pset, min=2, max=8)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=6)
    toolbox.register("mutate", gp.mutUniform, pset=pset, expr=toolbox.expr)  # gp.mutSemantic, gen_func=gp.genFull, pset=pset, min=2, max=8)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))
    #pool = multiprocessing.Pool(8)
    toolbox.register("map", map)
    pop = toolbox.population(n=size_pop)
    hof = tools.HallOfFame(size_gen)  ### OLD ###
    print("INITIAL POP SIZE: %d" % size_pop)
    print("GEN SIZE: %d" % size_gen)
    print("\n")
    random.seed()
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    #stats_size = tools.Statistics(len)
    #stats_height = tools.Statistics(operator.attrgetter("height"))
    mstats = tools.MultiStatistics(fitness=stats_fit)

    mstats.register("avg", np.mean, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=Mu, lambda_=Lambda, cxpb=cxpb, mutpb=mutpb, ngen=size_gen, stats=mstats, halloffame=hof, verbose=True)
    del pset, toolbox, creator.Fitness, creator.Individual, creator
    return hof[0].fitness.values[0], hof[0]













