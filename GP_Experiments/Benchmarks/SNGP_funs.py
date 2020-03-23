import numpy as np
import re
import random
from deap import gp
from copy import copy

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
        self.u = item
        self.r = []
        self.S = []
        self.P = []
        self.O = []
        self.rho = [] # node's utility


def modExp(x):
    if hasattr(x, "__len__"):
        x = np.array(x, dtype=float)
        out = []
        for i in x:
            if -100<=i<=100:
                out.append(np.exp(i))
            else:
                if i>0:
                    out.append(np.exp(100))
                else:
                    out.append(np.exp(-100))
        return np.array(out)
    else:
        if -100<=x<=100:
            return np.exp(x)
        else:
            if x>0:
                return np.exp(100)
            else:
                return np.exp(-100)


def protDiv(left, right):
    try:
        x = left / right
        return x
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
        return 0.0

def TriAdd(x, y, z): return x + y + z



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
    index_best = copy(selected)
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
            index = max(chosen)
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

    #index = utility_value_selection(pop, t)  # selection using utility value

    index = depthwise_selection(pop, terminals, nEph)  # depthwise selection strategy

    arity = pop[index].u[0].arity
    which = random.randint(0, arity-1)  # choose wich element in the successors list to mutate
    old_successor = pop[index].S[which]  # old successor of "index" individual
    if old_successor > terminals+nEph-1:
        try:
            pop[old_successor].P.remove(index)  # "index" individual removed from predecessor list of old successor
        except(ValueError):
            pass
    replacement = random.randint(0, index-1)  # choose new successor
    while replacement in pop[index].S:
        replacement = random.randint(0, index - 1)  # choose new successor
    #while replacement == old_successor:
     #   replacement = random.randint(0, index - 1)
    if replacement == index:
        print("error")
    pop[index].S[which] = replacement  # replace the successor
    if replacement > terminals+nEph-1:
        pop[replacement].P.append(index)  # add "index" element to new successor predecessors list
    return pop, index


def retrieveFunction(pop, terminals, nEph):
    best_fit = 1e50
    best_root = 0
    for i in range(len(pop)):
        if pop[i].r < best_fit:
            best_root = i
            best_fit = pop[i].r

    string = pop[best_root].u[0].name + "("
    for l in range(len(pop[best_root].S)):
        string = string + str(pop[best_root].S[l])
        if l < len(pop[best_root].S)-1:
            string = string + ","
    string = string + ")"
    done = False
    while done is False:
        done = True
        extra = 0
        vals = re.findall('\d+', string)
        for s in range(len(string)):
            if string[s].isdigit():
                if int(vals[0+extra]) > terminals + nEph - 1:
                    string = list(string)
                    for _ in range(len(vals[0+extra])):
                        del string[s]
                    if int(vals[0+extra]) > terminals + nEph-1:
                        suc = pop[int(vals[0+extra])].S
                        old_index = int(vals[0+extra])
                        string.insert(s, pop[int(vals[0+extra])].u[0].name + "(")
                        l = 0
                        while l < len(suc):
                            string.insert(s+2*l+1, str(pop[old_index].S[l]))
                            string.insert(s+2*l+2, ",")
                            l += 1
                        string.insert(s+2*(l-1)+3, ")")
                    vals = re.findall('\d+', "".join(string))
                else:
                    extra += 1
            string = "".join(string)

        string = "".join(string)
        nums = re.findall('\d+', string)
        for n in nums:
            if int(n) >= terminals+nEph:
                done = False
        string = "".join(string)
    return string, best_root

def retrieveFun_fromInd(pop, ind, terminals, nEph):

    try:
        string = ind.u[0].name + "("
    except TypeError:
        #string = ind.u.name + '('
        string = ind.u.name
        return string
    for l in range(len(ind.S)):
        string = string + str(ind.S[l])
        if l < len(ind.S)-1:
            string = string + ","
    string = string + ")"
    done = False
    while done is False:
        done = True
        extra = 0
        vals = re.findall('\d+', string)
        for s in range(len(string)):
            if string[s].isdigit():
                if int(vals[0+extra]) > terminals + nEph - 1:
                    string = list(string)
                    for _ in range(len(vals[0+extra])):
                        del string[s]
                    if int(vals[0+extra]) > terminals + nEph-1:
                        try:
                            suc = pop[int(vals[0+extra])].S
                        except IndexError:
                            print("d")
                        old_index = int(vals[0+extra])
                        string.insert(s, pop[int(vals[0+extra])].u[0].name + "(")
                        l = 0
                        while l < len(suc):
                            string.insert(s+2*l+1, str(pop[old_index].S[l]))
                            string.insert(s+2*l+2, ",")
                            l += 1
                        string.insert(s+2*(l-1)+3, ")")
                    vals = re.findall('\d+', "".join(string))
                else:
                    extra += 1
            string = "".join(string)

        string = "".join(string)
        nums = re.findall('\d+', string)
        for n in nums:
            if int(n) >= terminals+nEph:
                done = False
        string = "".join(string)
    return string