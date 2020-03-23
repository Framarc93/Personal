

def sngp_fun(ref_fun, interval, Ngen, nEph, terminals):
    import numpy as np
    import operator
    import random
    from deap import gp
    from deap import base, creator, tools
    from copy import deepcopy
    import SNGP_funs as funs
    from functools import partial
    def evaluate(individual, terminals):
        output_fun = toolbox.compile(expr=individual)
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
        # err = sum(abs((output - output_eval)))
        err = np.sqrt(sum((output - output_eval) ** 2))
        try:
            return list(output_eval), err
        except(TypeError):
            output_eval = np.ones((len(output))) * output_eval
            return list(output_eval), err


    pset = gp.PrimitiveSet("MAIN", terminals)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(funs.modExp, 1)
    pset.addPrimitive(funs.protDiv, 2)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.tanh, 1)
    # pset.addPrimitive(np.sqrt, 1)
    pset.addPrimitive(funs.TriAdd, 3)
    rand_terminals = []
    for i in range(nEph):
        # pset.addEphemeralConstant("rand{}".format(i), lambda: round(random.uniform(-10, 10), 4))
        rand_terminals.append(round(random.uniform(-5, 5), 3))
        pset.addTerminal("rand{}".format(i), rand_terminals[i])
    if terminals == 1:
        pset.renameArguments(ARG0='x')
    elif terminals == 2:
        pset.renameArguments(ARG0='x')
        pset.renameArguments(ARG1='y')
    elif terminals == 3:
        pset.renameArguments(ARG0='x')
        pset.renameArguments(ARG1='y')
        pset.renameArguments(ARG2='z')
    elif terminals == 4:
        pset.renameArguments(ARG0='x')
        pset.renameArguments(ARG1='y')
        pset.renameArguments(ARG2='z')
        pset.renameArguments(ARG3='w')

    creator.create("Individual", funs.Individual)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", partial(evaluate, terminals=terminals))

    pop = toolbox.population(400)

    for i in range(nEph):
        pop.insert(i, funs.Individual(pset.terminals[object][i + terminals]))

    for i in range(terminals):  ## insert terminals after the constant variables
        pop.insert(i + nEph, funs.Individual(pset.terminals[object][i]))

    def InitialisePop(pop, output):
        i = terminals + nEph
        N = len(pop)
        tot_fit = 0
        # for j in range(nEph):
        #   pop[j].O, pop[j].r = toolbox.evaluate(pop[j].u.value)
        for j in range(terminals + nEph):
            pop[j].O, pop[j].r = toolbox.evaluate(str(pop[j].u.value))
            tot_fit += pop[j].r
        while i < N:
            arity = pop[i].u[0].arity
            s = []
            for ar in range(arity):
                new = random.randint(0, i - 1)
                while new in s:
                    new = random.randint(0, i - 1)
                s.append(new)

            for l in range(len(s)):
                if s[l] > nEph + terminals - 1:
                    pop[s[l]].P.append(i)
                pop[i].S.append(s[l])
            for k in range(len(output)):
                string = "("
                for l in range(len(s)):
                    string = string + str(pop[s[l]].O[k]) + ","
                string = string + ")"
                pop[i].O.append(eval(pop[i].u[0].name + string, pset.context))
            pop[i].r = np.sqrt(sum((pop[i].O - output) ** 2))
            data = np.vstack((pop[i].O, output))
            pop[i].rho = np.corrcoef(data)[0][1]
            tot_fit += pop[i].r
            i += 1

    if terminals == 1:
        output = ref_fun(interval)
    elif terminals == 2:
        output = ref_fun(interval[0], interval[1])
    elif terminals == 3:
        output = ref_fun(interval[0], interval[1], interval[2])
    elif terminals == 4:
        output = ref_fun(interval[0], interval[1], interval[2], interval[3])

    InitialisePop(pop, output)

    def moveLeft(pop_orig, best_node, terminals):
        '''moveLeft operator from [1]:
            1. Extract nodes of the graph G best rooted in the best-performing node and put the nodes into a compact ordered list L.
            2. Set all successor and predecessor links of nodes within L so that L represents the same graph as the original graph G best.
            3. Place L to the beginning of the population, i.e. the first node of L being at the first function node position in the population.
            4. Update the successor links of nodes of the original graph G best so that it retains the same functionality as it had before the action.
               It must be made sure that all nodes of the original G best have properly set their successors. If for example some successor of a node of
               the original G best gets modified (i.e. the successor falls into the portion of the population newly occupied by the compact form of the G best,
               then the successor reference is updated accordingly.
            5. Update the predecessor lists of nodes in the compact form of G best in order to reestablish links to other nodes in the population that use the nodes as successors.'''
        print("move left")
        if terminals == 1:
            output = ref_fun(interval)
        elif terminals == 2:
            output = ref_fun(interval[0], interval[1])
        elif terminals == 3:
            output = ref_fun(interval[0], interval[1], interval[2])
        elif terminals == 4:
            output = ref_fun(interval[0], interval[1], interval[2], interval[3])
        pop = deepcopy(pop_orig)
        #####  1  ######
        L_set_indexes = funs.getTree(pop, best_node, w_terminals=False, terminals=terminals, nEph=nEph)  # set of successors of the best performing node  NOT WORKING PROPERLY
        L_set_indexes.append(best_node)
        L_set_indexes.sort()
        L_set = []
        L_set_indexes_new = np.linspace(nEph + terminals, nEph + terminals + len(L_set_indexes) - 1, len(L_set_indexes),
                                        dtype=int)

        for ind in L_set_indexes:
            L_set.append(deepcopy(pop[ind]))

        #####  2  #####
        p = len(L_set) - 1
        while p >= 0:
            for s in range(len(L_set[p].S)):  # update successors list
                if L_set[p].S[s] < nEph + terminals:
                    pass
                else:
                    ind = np.where(np.array(L_set_indexes) == L_set[p].S[s])[0][0]
                    L_set[p].S[s] = L_set_indexes_new[ind]  # replace old successor with new from compact set
            to_remove = []
            for pp in range(len(L_set[p].P)):  # update predecessors list
                try:
                    indp = np.where(np.array(L_set_indexes) == L_set[p].P[pp])[0][0]
                    L_set[p].P[pp] = L_set_indexes_new[indp]
                except(IndexError):
                    to_remove.append(L_set[p].P[pp])

            if to_remove:
                # L_set[p].P.remove(to_remove)
                L_set[p].P = [e for e in L_set[p].P if e not in to_remove]

            p = p - 1

        ####  3  ####
        pop[nEph + terminals:nEph + terminals + len(L_set_indexes)] = deepcopy(L_set)

        ####  4  #####
        for i in range(len(pop)):
            if i in L_set_indexes:
                if i > nEph + terminals:
                    for s in range(len(pop[i].S)):
                        try:
                            ind = np.where(np.array(L_set_indexes) == pop[i].S[s])[0][0]
                            pop[i].S[s] = L_set_indexes_new[ind]
                        except(IndexError):
                            pass
            else:
                for s in range(len(pop[i].S)):
                    if pop[i].S[s] in L_set_indexes_new:
                        pop[pop[i].S[s]].P.append(i)
            pop[i].P = list(dict.fromkeys(pop[i].P))

        i = terminals + nEph
        N = len(pop)
        tot_fit = 0
        for j in range(terminals + nEph):
            pop[j].O, pop[j].r = toolbox.evaluate(str(pop[j].u.value))
            tot_fit += pop[j].r
        while i < N:
            pop[i].O = []
            for k in range(len(output)):
                string = "("
                for l in range(len(pop[i].S)):
                    string = string + str(pop[pop[i].S[l]].O[k]) + ","
                string = string + ")"
                pop[i].O.append(eval(pop[i].u[0].name + string, pset.context))
            pop[i].r = np.sqrt(sum((pop[i].O - output) ** 2))
            data = np.vstack((pop[i].O, output))
            pop[i].rho = np.corrcoef(data)[0][1]
            tot_fit += pop[i].r
            i += 1
        return pop

    def moveRight(pop_orig, best_node, terminals):
        '''moveRight operator from [1]:
                1. Extract nodes of the graph G best rooted in the best-performing node and put the nodes into a compact ordered list L.
                2. Set all successor and predecessor links of nodes within L so that L represents the same graph as the original graph G best.
                3. Place L to the beginning of the population, i.e. the first node of L being at the first function node position in the population.
                4. Update the successor links of nodes of the original graph G best so that it retains the same functionality as it had before the action.
                   It must be made sure that all nodes of the original G best have properly set their successors. If for example some successor of a node of
                   the original G best gets modified (i.e. the successor falls into the portion of the population newly occupied by the compact form of the G best,
                   then the successor reference is updated accordingly.
                5. Update the predecessor lists of nodes in the compact form of G best in order to reestablish links to other nodes in the population that use the nodes as successors.'''
        print("move right")
        if terminals == 1:
            output = ref_fun(interval)
        elif terminals == 2:
            output = ref_fun(interval[0], interval[1])
        elif terminals == 3:
            output = ref_fun(interval[0], interval[1], interval[2])
        elif terminals == 4:
            output = ref_fun(interval[0], interval[1], interval[2], interval[3])
        pop = deepcopy(pop_orig)
        #####  1  ######
        L_set_indexes = funs.getTree(pop, best_node, w_terminals=False, terminals=terminals,
                                     nEph=nEph)  # set of successors of the best performing node  NOT WORKING PROPERLY
        L_set_indexes.append(best_node)
        L_set_indexes.sort()
        L_set = []
        L_set_indexes_new = np.linspace(len(pop) - len(L_set_indexes), len(pop) - 1, len(L_set_indexes), dtype=int)

        for ind in L_set_indexes:
            L_set.append(deepcopy(pop[ind]))

        #####  2  #####
        p = len(L_set) - 1
        while p >= 0:
            for s in range(len(L_set[p].S)):  # update successors list
                if L_set[p].S[s] < nEph + terminals:
                    pass
                else:
                    ind = np.where(np.array(L_set_indexes) == L_set[p].S[s])[0][0]
                    L_set[p].S[s] = L_set_indexes_new[ind]  # replace old successor with new from compact set
            to_remove = []
            for pp in range(len(L_set[p].P)):  # update predecessors list
                try:
                    indp = np.where(np.array(L_set_indexes) == L_set[p].P[pp])[0][0]
                    L_set[p].P[pp] = L_set_indexes_new[indp]
                except(IndexError):
                    to_remove.append(L_set[p].P[pp])
            if to_remove:
                # L_set[p].P.remove(to_remove)
                L_set[p].P = [e for e in L_set[p].P if e not in to_remove]

            p = p - 1

        ####  3  ####
        pop[-len(L_set):] = deepcopy(L_set)

        ####  4  #####

        for i in range(len(pop) - len(L_set)):
            to_remove = []
            for p in range(len(pop[i].P)):
                if pop[i].P[p] in L_set_indexes_new:
                    to_remove.append(pop[i].P[p])
            pop[i].P = [e for e in pop[i].P if e not in to_remove]
            pop[i].P = list(dict.fromkeys(pop[i].P))
        i = terminals + nEph
        N = len(pop)
        tot_fit = 0
        for j in range(terminals + nEph):
            pop[j].O, pop[j].r = toolbox.evaluate(str(pop[j].u.value))
            tot_fit += pop[j].r
        while i < N:
            pop[i].O = []
            for k in range(len(output)):
                string = "("
                for l in range(len(pop[i].S)):
                    string = string + str(pop[pop[i].S[l]].O[k]) + ","
                string = string + ")"
                pop[i].O.append(eval(pop[i].u[0].name + string, pset.context))
            pop[i].r = np.sqrt(sum((pop[i].O - output) ** 2))
            data = np.vstack((pop[i].O, output))
            pop[i].rho = np.corrcoef(data)[0][1]
            tot_fit += pop[i].r
            i += 1

        return pop

    t = 5  # nodes selected in modified tournament
    N = 5  # number of mutations performed at each epoch
    ###########################   EVOLUTION  #######################################
    if terminals == 1:
        output = ref_fun(interval)
    elif terminals == 2:
        output = ref_fun(interval[0], interval[1])
    elif terminals == 3:
        output = ref_fun(interval[0], interval[1], interval[2])
    elif terminals == 4:
        output = ref_fun(interval[0], interval[1], interval[2], interval[3])
    fit_old = sum(pop[ll].r for ll in range(len(pop)))
    best_soFar = pop[0]
    best_soFar_fit = pop[0].r
    best_soFar_index = 0
    best_soFar_eq = []
    for i in range(len(pop)):
        if pop[i].r < best_soFar_fit:
            best_soFar_fit = deepcopy(pop[i].r)
            best_soFar = deepcopy(pop[i])
            best_soFar_index = i
            best_soFar_eq = funs.retrieveFun_fromInd(pop, pop[i], terminals, nEph)
    fit_old = 0
    for ind in pop:
        fit_old = fit_old + ind.r
    old_pop = deepcopy(pop)

    for run in range(Ngen):
        print("----------------------- Evaluation: {} --------------------------".format(run))

        n_mut = random.randint(1, N)
        for _ in range(n_mut):
            pop[:], chosen = funs.smut(pop, t, terminals, nEph)  ###  MUTATION ####

            ################### REEVALUATE THE MUTATED INDIVIDUAL  ########################

            predecessors_list = []
            pop[chosen].O = []
            successors = []
            for i in range(len(pop[chosen].S)):
                successors.append(pop[chosen].S[i])
            for k in range(len(output)):
                string = "("
                for l in range(len(successors)):
                    string = string + str(pop[successors[l]].O[k]) + ","
                string = string + ")"
                pop[chosen].O.append(eval(pop[chosen].u[0].name + string, pset.context))

            pop[chosen].r = np.sqrt(sum((pop[chosen].O - output) ** 2))

            ############## REEVALUATE ALL THE INDIVIDUALS REALTED TO THE MUTATED ONE ###################

            k = chosen
            if pop[chosen].P:

                ##### FILL PREDECESSOR LIST #####

                while k < len(pop):
                    if pop[k].P:
                        for i in range(len(pop[k].P)):
                            predecessors_list.append(pop[k].P[i])
                    k += 1
                predecessors_list = list(dict.fromkeys(predecessors_list))
                predecessors_list.sort()

                ##### LOOP THROUGH PREDECESSOR LIST TO UPDATED INVOLVED NODES ##########

                for i in predecessors_list:
                    pop[i].O = []
                    successors = []
                    for j in range(len(pop[i].S)):
                        successors.append(pop[i].S[j])
                    for kk in range(len(output)):
                        string = "("
                        for l in range(len(successors)):
                            string = string + str(pop[successors[l]].O[kk]) + ","
                        string = string + ")"
                        pop[i].O.append(eval(pop[i].u[0].name + string, pset.context))
                    pop[i].r = np.sqrt(sum((pop[i].O - output) ** 2))
                    data = np.vstack((pop[i].O, output))
                    pop[i].rho = np.corrcoef(data)[0][1]

        ### evaluate current best individual
        fit_new = 0
        for ind in pop:
            fit_new = fit_new + ind.r
        currBest_fit = pop[0].r
        currBest = pop[0]
        currBest_eq = []
        currBest_index = 0
        for i in range(len(pop)):
            if pop[i].r < currBest_fit:
                currBest_fit = deepcopy(pop[i].r)
                currBest = deepcopy(pop[i])
                currBest_eq = funs.retrieveFun_fromInd(pop, pop[i], terminals, nEph)
                currBest_index = i
        if currBest_fit <= best_soFar_fit:
            best_soFar_fit = currBest_fit
            best_soFar = currBest
            best_soFar_eq = currBest_eq
            best_soFar_index = currBest_index
            fit_old = fit_new
            if best_soFar_index >= nEph + terminals:
                ch = random.random()
                if ch < 0.5:
                    pop = deepcopy(moveLeft(pop, best_soFar_index, terminals))
                else:
                    pop = deepcopy(moveRight(pop, best_soFar_index, terminals))
            print(best_soFar_fit)
            print(best_soFar_eq)
        else:
            pop = deepcopy(old_pop)
            print("Negative outcome")
            print(best_soFar_fit)
            print(best_soFar_eq)

        '''fit_new = sum(pop[ll].r for ll in range(len(pop)))  ##### FITNESS OF THE NEW POPULATION

        ###################### CHECK IF MUTATION WAS BENEFICIAL, ELSE DISCARD CHANGES ########################

        if fit_new < fit_old:
            fit_old = fit_new
            print(fit_new)
        else:
            pop = old_pop
            print("Negative outcome")
            print(fit_old)'''

    print("\n")
    print("Fitness of best population: {}".format(fit_old))
    print("Best root of population: {}".format(best_soFar_index))
    print("Fitness of best individual: {}".format(best_soFar_fit))
    # print(eq)
    eq = list(best_soFar_eq)
    v = 0
    while v < len(eq):
        if eq[v].isdigit():
            if int(eq[v]) == nEph:
                eq[v] = 'x'
            elif int(eq[v]) == nEph + 1:
                eq[v] = 'y'
            elif int(eq[v]) == nEph + 2:
                eq[v] = 'z'
            elif int(eq[v]) == nEph + 3:
                eq[v] = 'w'
            else:
                for i in range(nEph):
                    if int(eq[v]) == i:
                        eq[v] = str(rand_terminals[i])
                        break
                pass
        elif eq[v] == "," and (eq[v + 1] == ")" or eq[v + 1] == ",") and v < len(eq) - 1:
            del eq[v]
            v = v - 2
        v += 1

    final_eq = "".join(eq)
    del pset, toolbox, creator.Individual, creator
    return best_soFar_fit, final_eq