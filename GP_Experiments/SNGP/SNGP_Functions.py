import numpy as np
import re
import random
from deap import gp
from copy import copy, deepcopy
from scipy.optimize import minimize, basinhopping
from functools import partial


class Transf_var(object):
    def __init__(self, c_arr, x_arr):
        self.c = c_arr
        self.x = x_arr
        self.ww = np.ones(len(self.x))
        self.O = []
        self.u = []
        self.S = []
        self.r = []
        self.v = []

    def eval_v(self, pop, output):
        self.O = self.ww[0]*np.array(pop[self.c[0]].O)
        for i in range(len(self.ww)):
            self.O += self.ww[i]*np.array(pop[self.c[i]].O)*self.x[i]
        self.r = np.sqrt(sum((self.O - output)**2))
        self.v = sum(self.O)

    def optimize_weights(self):
        ch = random.randint(0, len(self.ww)-1)
        self.ww[ch] = self.ww[ch] + np.random.normal(0, 1)

    def change_connections(self, true_nh, nEph):
        ch = random.randint(0, len(self.c)-1)
        self.c[ch] = random.choice(list(np.linspace(0, nEph+true_nh-1, nEph+true_nh, dtype=int)))


class Individual(object):

    def __init__(self, item):
        if not type(item) == gp.Terminal:
            try:
                if len(item)>3:
                    del item[3], item[2], item[1]
                elif len(item)>2:
                    del item[2], item[1]
                else:
                    del item[1]
            except(TypeError):
                pass
        self.u = item  # individual
        self.r = []  # combination fitness
        self.S = []  # successors list
        self.P = []  # predecessors list
        self.O = []  # combination value
        #self.rho = [] # node's utility
        self.w = []  # combination weights

def getTree(pop, start_index, w_terminals, terminals, nEph):
    '''return UNSORTED list of successors from selected node'''
    done = False
    S_set = []
    S_set.extend(pop[start_index].S)
    expend = S_set
    start = 0
    while done is False:
        for i in range(len(expend)):
            if pop[expend[i]].S:
                S_set.extend(pop[expend[i]].S)
            start += 1
        S_set = list(dict.fromkeys(S_set))
        expend = S_set[start:]
        done = True
        for n in S_set[start:]:
            if int(n) > terminals + nEph - 1:
                done = False

    if w_terminals is False:
        for i in range(0, nEph + terminals):
            try:
                S_set.remove(i)
            except(ValueError):
                continue
    return S_set

def getTreeTV(pop, start_index, w_terminals, terminals, nEph, true_nh):
    '''return UNSORTED list of successors from selected node'''
    done = False
    S_set = []
    S_set.extend(pop[start_index].S)
    expend = S_set
    start = 0
    while done is False:
        for i in range(len(expend)):
            if pop[expend[i]].S:
                S_set.extend(pop[expend[i]].S)
            start += 1
        S_set = list(dict.fromkeys(S_set))
        expend = S_set[start:]
        done = True
        for n in S_set[start:]:
            if int(n) > terminals*2 + nEph +true_nh - 1:
                done = False

    if w_terminals is False:
        for i in range(0, nEph + terminals*2 + true_nh):
            try:
                S_set.remove(i)
            except(ValueError):
                continue
    return S_set

def retrieve_Predecessors(pop, start_index):
    done = False
    P_set = []
    P_set.extend(pop[start_index].P)
    expend = P_set
    start = 0
    old_set = []
    while done is False:
        for i in range(len(expend)):
            if pop[expend[i]].P:
                P_set.extend(pop[expend[i]].P)
            start += 1
        P_set= list(dict.fromkeys(P_set))  # remove duplicates
        expend = P_set[start:]
        if old_set == P_set:
            done = True
        else:
            old_set = deepcopy(P_set)
    return sorted(P_set)

def InitialisePop(pop, output, toolbox, terminals, nEph, pset):
    i = terminals+nEph
    for j in range(terminals+nEph):
        pop[j].O, pop[j].r = toolbox.evaluate(str(pop[j].u.value))
    while i < len(pop):
        arity = pop[i].u[0].arity
        s = []  # successors list
        w = []  # connections weights
        ss = []
        for ar in range(arity):
            av_nodes = np.linspace(0, i-1, i, dtype=int)
            ss.extend([v for v in s if v > nEph+terminals-1])
            av_nodes = np.delete(av_nodes, ss)
            new = random.choice(av_nodes)  # select random node to insert in successors list
            s.append(new)
            w.append(1.0)
        for l in range(len(s)):
            if s[l] > nEph+terminals-1:
                pop[s[l]].P.append(i) # update predecessors list of successors node if they are not terminals
        pop[i].S.extend(s) # update list of successors nodes
        pop[i].w.extend(w) # update list of connections weights
        vals = np.zeros((arity, len(output)))
        for l in range(len(s)):
            vals[l] = np.array(pop[i].w[l]) * np.array(pop[s[l]].O)
        fun = eval(pop[i].u[0].name, pset.context)
        if arity == 1:
            pop[i].O = fun(vals[0])
        elif arity == 2:
            pop[i].O = fun(vals[0], vals[1])
        elif arity == 3:
            pop[i].O = fun(vals[0], vals[1], vals[2])
        pop[i].r = np.sqrt(sum((pop[i].O - output)**2))
        #data = np.vstack((pop[i].O, output))
        #pop[i].rho = np.corrcoef(data)[0][1]
        i += 1
    return pop

def InitialisePopTV(pop, output, toolbox, terminals, nEph, pset, nh, nv):
    ##### THIS PART CREATE THE DEPENDENCIES IN THE POPULATION. CREATE THE FIRST COMBINATIONS  ##########
    i = terminals+nEph
    ind_head = []
    count = 0
    for j in range(terminals+nEph):
        pop[j].O, pop[j].r = toolbox.evaluate(str(pop[j].u.value))
    while i < len(pop):
        arity = pop[i].u[0].arity
        s = []  # successors list
        w = []  # connections weights
        ss = []
        for ar in range(arity):
            av_nodes = np.linspace(0, i-1, i, dtype=int)
            ss.extend([v for v in s if v > nEph+terminals-1])
            av_nodes = np.delete(av_nodes, ss)
            new = random.choice(av_nodes)  # select random node to insert in successors list
            s.append(new)
            w.append(1.0)
        for l in range(len(s)):
            if s[l] > nEph+terminals-1:
                pop[s[l]].P.append(i) # update predecessors list of successors node if they are not terminals
        pop[i].S.extend(s) # update list of successors nodes
        pop[i].w.extend(w) # update list of connections weights
        vals = np.zeros((arity, len(output)))
        for l in range(len(s)):
            vals[l] = np.array(pop[i].w[l]) * np.array(pop[s[l]].O)
        fun = eval(pop[i].u[0].name, pset.context)
        if arity == 1:
            pop[i].O = fun(vals[0])
        elif arity == 2:
            pop[i].O = fun(vals[0], vals[1])
        elif arity == 3:
            pop[i].O = fun(vals[0], vals[1], vals[2])
        pop[i].r = np.sqrt(sum((pop[i].O - output)**2))
        if np.all([pop[i].O[j] == pop[i].O[0] for j in range(len(pop[i].O))]) and count < nh:
           ind_head.append(i)
           count += 1
        i += 1

    ########### THIS PART PLACE THE CONSTANT NODES ADTER THE CONSTANT VALUES  ##############

    pop_subset = []
    for i in range(len(ind_head)):
        pop_subset.append(pop[ind_head[i]])
    pop = list(np.delete(pop, ind_head)) # remove the constant individuals
    terminals_index = []
    ind_terms = []
    for i in range(terminals):
        terminals_index.append(i+nEph)
        ind_terms.append(pop[i+nEph])
    pop = list(np.delete(pop, terminals_index)) # remove the variables from the beginning of the population
    for i in range(len(ind_head)):
        pop.insert(i+nEph, pop_subset[i])  # insert constant individuals after constant numbers

    #######  THIS PART INSERT THE TERMINALS AFTER THE CONSTANT NODES AND VALUES AND UPDATES THE DEPENDENCIES IN THE SUCCESSORS LIST OF ALL POPULATION

    for i in range(terminals):
        pop.insert(i + nEph + len(ind_head), ind_terms[i])
    for i in range(len(pop)):
        for j in range(len(pop[i].S)):
            for z in range(terminals):
                if pop[i].S[j] == z+nEph:
                    pop[i].S[j] = z+nEph+len(ind_head)

    ###### THIS PART CREATES THE TRANSFORMED VARIABLES AND PLACE THEM IN THE RIGHT PLACE  ########

    for i in range(nv):
        x_matrix = np.ones((terminals+1, len(output)))
        j = 1
        while j < terminals+1:
            x_matrix[j][:] = pop[j+nEph+len(ind_head)-1].O
            j += 1
        c_arr = random.sample(list(np.linspace(0, nEph+len(ind_head)-1, nEph+len(ind_head), dtype=int)), terminals+1)
        v = Transf_var(c_arr, x_matrix)
        v.u = "v{}".format(i)
        v.eval_v(pop, output)
        pop.insert(i+nEph+len(ind_head)+terminals, v)

    return pop, len(ind_head)

def moveLeft(pop, best_node, terminals, nEph, pset, output, toolbox):
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
    #pop = deepcopy(pop_orig)
    #####  1  ######
    L_set_indexes = getTree(pop, best_node, w_terminals=False, terminals=terminals, nEph=nEph)  # set of successors of the best performing node  NOT WORKING PROPERLY
    L_set_indexes.append(best_node)
    L_set_indexes.sort()
    L_set = []
    L_set_indexes_new = np.linspace(nEph+terminals, nEph+terminals+len(L_set_indexes)-1, len(L_set_indexes), dtype=int)

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
            L_set[p].P = [e for e in L_set[p].P if e not in to_remove]

        p = p - 1

    ####  3  ####
    pop[nEph+terminals:nEph+terminals+len(L_set_indexes)] = deepcopy(L_set)

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

    #i = terminals + nEph
    #tot_fit = 0
    #for j in range(terminals + nEph):
     #   pop[j].O, pop[j].r = toolbox.evaluate(str(pop[j].u.value))
        #tot_fit += pop[j].r
    '''while i < len(pop):
        pop[i].O = []
        arity = pop[i].u[0].arity
        vals = np.zeros((arity, len(output)))
        for l in range(len(pop[i].S)):
            vals[l] = np.array(pop[i].w[l]) * np.array(pop[pop[i].S[l]].O)
        fun = eval(pop[i].u[0].name, pset.context)
        if arity == 1:
            pop[i].O = fun(vals[0])
        elif arity == 2:
            pop[i].O = fun(vals[0], vals[1])
        elif arity == 3:
            pop[i].O = fun(vals[0], vals[1], vals[2])
        pop[i].r = np.sqrt(sum((pop[i].O - output) ** 2))'''
        #data = np.vstack((pop[i].O, output))
        #pop[i].rho = np.corrcoef(data)[0][1]
        #tot_fit += pop[i].r
        #i += 1
    return pop


def moveRight(pop_orig, best_node, terminals, nEph, pset, output, toolbox):
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
    pop = deepcopy(pop_orig)
    #####  1  ######
    L_set_indexes = getTree(pop, best_node, w_terminals=False, terminals=terminals, nEph=nEph)  # set of successors of the best performing node  NOT WORKING PROPERLY
    L_set_indexes.append(best_node)
    L_set_indexes.sort()
    L_set = []
    L_set_indexes_new = np.linspace(len(pop)-len(L_set_indexes), len(pop)-1, len(L_set_indexes), dtype=int)

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

    for i in range(len(pop)-len(L_set)):
        to_remove = []
        for p in range(len(pop[i].P)):
            if pop[i].P[p] in L_set_indexes_new:
                to_remove.append(pop[i].P[p])
        pop[i].P = [e for e in pop[i].P if e not in to_remove]
        pop[i].P = list(dict.fromkeys(pop[i].P))
    i = terminals + nEph
    tot_fit = 0
    for j in range(terminals + nEph):
        pop[j].O, pop[j].r = toolbox.evaluate(str(pop[j].u.value))
        tot_fit += pop[j].r
    while i < len(pop):
        pop[i].O = []
        arity = pop[i].u[0].arity
        vals = np.zeros((arity, len(output)))
        for l in range(len(pop[i].S)):
            vals[l] = np.array(pop[i].w[l]) * np.array(pop[pop[i].S[l]].O)
        fun = eval(pop[i].u[0].name, pset.context)
        if arity == 1:
            pop[i].O = fun(vals[0])
        elif arity == 2:
            pop[i].O = fun(vals[0], vals[1])
        elif arity == 3:
            pop[i].O = fun(vals[0], vals[1], vals[2])
        pop[i].r = np.sqrt(sum((pop[i].O - output) ** 2))
        #data = np.vstack((pop[i].O, output))
        #pop[i].rho = np.corrcoef(data)[0][1]
        #tot_fit += pop[i].r
        i += 1

    return pop


def depthwise_selection(pop, terminals, nEph):
    ############ depthwise selection strategy ##########################
    '''Depthwide selection from [1]:
        1. A function node n is chosen at random.
        2. A tree t with the best fitness out of all trees that use the node n is chosen.
        3. All nodes of the tree t are collected in a set S. Each node is assigned a score equal to its depth in the tree t.
        4. One node is chosen from the set S using a binary tournament selection considering the score values in the higher the better manner'''

    #######  1  ######
    selected = random.choice(range(terminals + nEph, len(pop)))
    #######  2  ######
    best_node_fit = pop[selected].r
    index_best = selected
    predec = [selected]
    start = 0
    done = False
    while done is False:
        if pop[predec[start]].P:
            predec.extend(pop[start].P)
            start += 1
        else:
            start += 1
        if start >= len(predec):
            done = True

    for i in predec:
        if pop[i].r < best_node_fit:
            best_node_fit = pop[i].r
            index_best = i

    #######  3  ######
    S_set = getTree(pop, index_best, w_terminals=False, terminals=terminals, nEph=nEph)
    #######  4  ######
    if S_set:
        if len(S_set) >= 2:
            chosen = random.sample(S_set, 2)
            index = min(chosen)
        else:
            chosen = random.choice(S_set)
            index = chosen
    else:
        chosen = selected
        index = chosen

    return index

def depthwise_selectionTV(pop, terminals, nEph, true_nh):
    ############ depthwise selection strategy ##########################
    '''Depthwide selection from [1]:
        1. A function node n is chosen at random.
        2. A tree t with the best fitness out of all trees that use the node n is chosen.
        3. All nodes of the tree t are collected in a set S. Each node is assigned a score equal to its depth in the tree t.
        4. One node is chosen from the set S using a binary tournament selection considering the score values in the higher the better manner'''

    #######  1  ######
    selected = random.choice(range(terminals*2 + nEph + true_nh, len(pop)))
    #######  2  ######
    best_node_fit = pop[selected].r
    index_best = selected
    predec = [selected]
    start = 0
    done = False
    while done is False:
        if pop[predec[start]].P:
            predec.extend(pop[start].P)
            start += 1
        else:
            start += 1
        if start >= len(predec):
            done = True

    for i in predec:
        if pop[i].r < best_node_fit:
            best_node_fit = pop[i].r
            index_best = i

    #######  3  ######
    S_set = getTreeTV(pop, index_best, w_terminals=False, terminals=terminals, nEph=nEph, true_nh=true_nh)
    #######  4  ######
    if S_set:
        if len(S_set) >= 2:
            chosen = random.sample(S_set, 2)
            index = min(chosen)
        else:
            chosen = random.choice(S_set)
            index = chosen
    else:
        chosen = selected
        index = chosen

    return index


def utility_value_selection(pop, t, terminals, nEph):
    #### selection based on utility value using pearson correlation ############
    selected = random.sample(range(terminals + nEph, len(pop)), t)  # choose which individual to mutate
    old_utility = pop[selected[0]].rho
    index = selected[0]
    for i in selected:
        if pop[i].rho > old_utility:
            index = i
    return index


def smut(pop, t, terminals, nEph):
    #print("Start mutation")
    #index = utility_value_selection(pop, t)  # selection using utility value

    index = depthwise_selection(pop, terminals, nEph)  # depthwise selection strategy

    arity = pop[index].u[0].arity
    which = random.randint(0, arity-1)  # choose wich element in the successors list to mutate
    old_successor = pop[index].S[which]  # old successor of "index" individual
    pop[index].w[which] = 1.0
    if old_successor > terminals+nEph-1:
        try:
            pop[old_successor].P.remove(index)  # "index" individual removed from predecessor list of old successor
        except(ValueError):
            pass
    ss = []
    ss.extend([v for v in pop[index].S if v > nEph + terminals - 1])
    av_nodes = np.linspace(0, index - 1, index, dtype=int)
    av_nodes = np.delete(av_nodes, ss)
    replacement = random.choice(av_nodes)  # select random node to insert in successors list

    #replacement = random.randint(0, index-1)  # choose new successor
    #while replacement in pop[index].S or replacement == index:
     #   replacement = random.randint(0, index - 1)  # choose new successor
    #while replacement == old_successor:
     #   replacement = random.randint(0, index - 1)
    pop[index].S[which] = replacement  # replace the successor
    if replacement > terminals+nEph-1:
        pop[replacement].P.append(index)  # add "index" element to new successor predecessors list
    return pop, index

def smutTV(pop, terminals, nEph, true_nh):
    #print("Start mutation")
    #index = utility_value_selection(pop, t)  # selection using utility value

    #index = depthwise_selectionTV(pop, terminals, nEph, true_nh)  # depthwise selection strategy
    index = random.choice(range(terminals*2 + nEph + true_nh, len(pop)))
    arity = pop[index].u[0].arity
    which = random.randint(0, arity-1)  # choose wich element in the successors list to mutate
    old_successor = pop[index].S[which]  # old successor of "index" individual
    pop[index].w[which] = 1.0
    if old_successor > terminals*2+nEph+true_nh-1:
        try:
            pop[old_successor].P.remove(index)  # "index" individual removed from predecessor list of old successor
        except(ValueError):
            pass
    ss = []
    ss.extend([v for v in pop[index].S if v > nEph + terminals*2 + true_nh - 1])
    av_nodes = np.linspace(0, index - 1, index, dtype=int)
    av_nodes = np.delete(av_nodes, ss)
    replacement = random.choice(av_nodes[:int(len(av_nodes)/2)])  # select random node to insert in successors list

    #replacement = random.randint(0, index-1)  # choose new successor
    #while replacement in pop[index].S or replacement == index:
     #   replacement = random.randint(0, index - 1)  # choose new successor
    #while replacement == old_successor:
     #   replacement = random.randint(0, index - 1)
    pop[index].S[which] = replacement  # replace the successor
    if replacement > terminals*2+nEph+true_nh-1:
        pop[replacement].P.append(index)  # add "index" element to new successor predecessors list
    return pop, index


def retrieveFun_fromInd(pop, ind, terminals, nEph, rand_terminals):

    if len(ind.S) == 0:
        for i in range(nEph):
            if ind.u.value == rand_terminals[i]:
                string = str(i)
                #print(string)
                return string
        if ind.u.value == 'x':
            string = str(nEph)
            #print(string)
            return string
        elif ind.u.value == 'y':
            string = str(nEph+1)
            #print(string)
            return string
    else:
        count_w = 0
        string = ind.u[0].name + "("
        for l in range(len(ind.S)):
            string = string + str(round(ind.w[l],4)) + " * ".format(count_w) + str(ind.S[l])
            count_w += 1
            if l < len(ind.S)-1:
                string = string + ","
        string = string + ")"
        done = False
        extra = 0
        while done is False:
            done = True
            all_vals = list(map(lambda v: float(v) if '.' in v else int(v), re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', string)))
            vals = []
            for vv in range(len(all_vals)):
                if type(all_vals[vv]) == int:
                    vals.append(all_vals[vv])
            # ALTERNATIVE VERSION
            if vals[extra] > terminals + nEph - 1:
                string = list(string)
                to_check = len(str(vals[extra]))  # how many digit is the number it interests me
                for s in range(len(string)):
                    if (string[s].isdigit() and string[s-2] == "*") or (string[s].isdigit() and string[s-1] == "*"):
                        ss = s
                        check_num = []
                        while ss < to_check+s:
                            if string[ss].isdigit():
                                check_num.append(string[ss])
                            ss += 1
                        if vals[extra] == int("".join(map(str, check_num))):
                            for _ in range(len(str(vals[extra]))):
                                del string[s]
                            suc = pop[vals[extra]].S
                            old_index = vals[extra]
                            string.insert(s, pop[vals[extra]].u[0].name + "(")
                            l = 0
                            while l < len(suc):
                                string.insert(s+4*l+1, str(round(pop[old_index].w[l],4)))
                                string.insert(s+4*l+2, "*")
                                string.insert(s+4*l+3, str(pop[old_index].S[l]))
                                string.insert(s+4*l+4, ",")
                                l += 1
                            string.insert(s+4*(l-1)+5, ")")
                            break
                string = "".join(string)
                extra = 0
            else:
                extra += 1
            string = "".join(string)
            nums = list(map(lambda v: float(v) if '.' in v else int(v), re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', string)))
            for n in nums:
                if type(n) == int and n >= terminals+nEph:
                    done = False
                    break
        #print(string)
    return string

def retrieveFun_fromIndTV(pop, ind, terminals, nEph, rand_terminals, true_nh):

    if len(ind.S) == 0:
        if type(ind.u) == str:
            if ind.u == "v0":
                string = str(nEph + true_nh + terminals)
                # print(string)
                return string
            elif ind.u == "v1":
                string = str(nEph + true_nh + terminals + 1)
                # print(string)
                return string
        else:
            for i in range(nEph):
                if ind.u.value == rand_terminals[i]:
                    string = str(i)
                    #print(string)
                    return string
            if ind.u.value == 'x':
                string = str(nEph+true_nh)
                #print(string)
                return string
            elif ind.u.value == 'y':
                string = str(nEph+true_nh+1)
                #print(string)
                return string
    else:
        count_w = 0
        string = ind.u[0].name + "("
        for l in range(len(ind.S)):
            string = string + str(round(ind.w[l],4)) + " * ".format(count_w) + str(ind.S[l])
            count_w += 1
            if l < len(ind.S)-1:
                string = string + ","
        string = string + ")"
        done = False
        extra = 0
        while done is False:
            done = True
            all_vals = list(map(lambda v: float(v) if '.' in v else int(v), re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', string)))
            vals = []
            for vv in range(len(all_vals)):
                if type(all_vals[vv]) == int:
                    vals.append(all_vals[vv])
            # ALTERNATIVE VERSION
            if vals[extra] > terminals*2 + true_nh + nEph - 1:
                string = list(string)
                to_check = len(str(vals[extra]))  # how many digit is the number it interests me
                for s in range(len(string)):
                    if (string[s].isdigit() and string[s-2] == "*") or (string[s].isdigit() and string[s-1] == "*"):
                        ss = s
                        check_num = []
                        while ss < to_check+s:
                            if string[ss].isdigit():
                                check_num.append(string[ss])
                            ss += 1
                        if vals[extra] == int("".join(map(str, check_num))):
                            for _ in range(len(str(vals[extra]))):
                                del string[s]
                            suc = pop[vals[extra]].S
                            old_index = vals[extra]
                            string.insert(s, pop[vals[extra]].u[0].name + "(")
                            l = 0
                            while l < len(suc):
                                string.insert(s+4*l+1, str(round(pop[old_index].w[l],4)))
                                string.insert(s+4*l+2, "*")
                                string.insert(s+4*l+3, str(pop[old_index].S[l]))
                                string.insert(s+4*l+4, ",")
                                l += 1
                            string.insert(s+4*(l-1)+5, ")")
                            break
                string = "".join(string)
                extra = 0
            else:
                extra += 1
            string = "".join(string)
            nums = list(map(lambda v: float(v) if '.' in v else int(v), re.findall('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?', string)))
            for n in nums:
                if type(n) == int and n >= terminals*2+nEph+true_nh:
                    done = False
                    break
        #print(string)
    return string


def convert_equation(eq, nEph, rand_terminals):
    eq = list(eq)
    v = 0
    while v < len(eq):
        if (eq[v].isdigit() and eq[v-1] == "*") or (eq[v].isdigit() and eq[v-2] == "*"):
            if int(eq[v]) == nEph:
                eq[v] = 'x'
            elif int(eq[v]) == nEph + 1:
                eq[v] = 'y'
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
    return final_eq

def convert_equationTV(eq, nEph, rand_terminals, true_nh, terminals):
    eq = list(eq)
    v = 0
    while v < len(eq):
        if (eq[v].isdigit() and eq[v-1] == "*") or (eq[v].isdigit() and eq[v-2] == "*"):
            if int(eq[v]) == nEph + true_nh:
                eq[v] = 'x'
            elif int(eq[v]) == nEph + true_nh + 1:
                eq[v] = 'y'
            elif int(eq[v]) == nEph + true_nh + terminals:
                eq[v] = str(eq[v].r)
            elif int(eq[v]) == nEph + true_nh + terminals + 1:
                eq[v] = str(eq[v].r)
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
    return final_eq

def optimize_weights(ind, output, pop, pset):

    def evaluate_ind(w, ind, output, pop, pset):
        ind.O = []
        arity = ind.u[0].arity
        vals = np.zeros((arity, len(output)))
        for l in range(len(ind.S)):
            vals[l] = np.array(w[l]) * np.array(pop[ind.S[l]].O)
        fun = eval(ind.u[0].name, pset.context)
        if arity == 1:
            ind.O = fun(vals[0])
        elif arity == 2:
            ind.O = fun(vals[0], vals[1])
        elif arity == 3:
            ind.O = fun(vals[0], vals[1], vals[2])
        return np.sqrt(sum((ind.O - output)**2))

    w = ind.w
    res = minimize(evaluate_ind, w, args=(ind, output, pop, pset), method='BFGS')#, options={'maxiter': 10})
    #res = basinhopping(partial(evaluate_ind, ind=ind, output=output, pop=pop, pset=pset), w, disp=False)
    ind.w = res.x
    ind.O = []
    arity = ind.u[0].arity
    vals = np.zeros((arity, len(output)))
    for l in range(len(ind.S)):
        vals[l] = np.array(ind.w[l]) * np.array(pop[ind.S[l]].O)
    fun = eval(ind.u[0].name, pset.context)
    if arity == 1:
        ind.O = fun(vals[0])
    elif arity == 2:
        ind.O = fun(vals[0], vals[1])
    elif arity == 3:
        ind.O = fun(vals[0], vals[1], vals[2])
    ind.r = np.sqrt(sum((np.array(ind.O) - output) ** 2))
    #data = np.vstack((ind.O, output))
    #ind.rho = np.corrcoef(data)[0][1]

    return ind