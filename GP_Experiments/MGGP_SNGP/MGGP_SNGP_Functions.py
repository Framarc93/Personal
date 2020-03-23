import numpy as np
from operator import attrgetter, eq
from copy import deepcopy
import random
from deap import gp, tools
from scipy.optimize import minimize, least_squares, basinhopping
from functools import partial
########### CLASSES DEFINITION  #############

class Combination(object):
    def __init__(self):
        self.ind = []
        self.weights = []
        self.fitness = []
        self.connections = []
        self.equation = []
        self.constants = []

    def clear(self):
        self.ind = []
        self.weights = []
        self.fitness = []
        self.connections = []
        self.equation = []
        self.constants = []

    def update(self, pop, connections, weights, evaluate_ind):
        self.clear()
        for i in range(len(connections)):
            self.ind.append(pop[connections[i]])

        self.connections = connections
        self.weights = weights
        for i in range(len(self.ind)):
            for j in range(len(self.ind[i])):
                if type(self.ind[i][j]) == gp.Terminal and self.ind[i][j].name[0] != "A":
                    self.constants.append(self.ind[i][j].name)
        expr = self.build_eq()
        fit = evaluate_ind(expr)
        self.fitness = fit[0]
        self.equation = expr

    def build_eq(self):
        ''' setup to build equations in the form:
        w0 + w1*39 + w2*34 + w3*40 + w4*38 + w5*4
        if roots: 39, 34, 40, 4
        connections: 34, (40 38), 4
        It builds equations starting from what's inside the combination individual. Doesn't consider the population'''
        cw = 1
        con = 0
        #expr = str(self.connections[0]) + '+'
        expr = "w0 + "
        while con < len(self.connections):  # this part creates the equation with weights and indices
            expr = expr + "mul(" + "w{}".format(cw) + "," + str(self.connections[con]) + ")" + "+"
            cw += 1
            con += 1
        expr = list(expr)
        del expr[-1]

        # this part put the trees equations in place of the indices
        llen = len(expr)
        ww = []
        for i in range(len(self.weights)):
            ww.append(self.weights[i])
        i = 0
        while i < llen:
            if expr[i].isdigit() and expr[i-1] != "w":
                if llen > 1:
                    j = 1
                    val = expr[i]
                    digit = True
                    while digit is True:
                        if expr[i+j].isdigit():
                            val = val + expr[i+j]
                            j += 1
                            if "".join(expr) == val:
                                digit = False
                        else:
                            digit = False

                    for d in range(len(val)):
                        del expr[i]
                    else:
                        for ii in range(len(self.connections)):
                            if val == str(self.connections[ii]):
                                expr.insert(i, str(self.ind[ii]))
                                break
                    llen = len(expr)  # llen - (j - 1)
                else:
                    val = expr[i]
                    del expr[i]
                    for ii in range(len(self.connections)):
                        if val == str(self.connections[ii]):
                            expr.insert(i, str(self.ind[ii]))
                            break
            if expr[i] == "w":
                if not expr[i+2].isdigit():
                    subs = str(ww[int(expr[i + 1])])
                    del expr[i + 1], expr[i]
                    expr.insert(i, subs)
                    llen = len(expr) #llen - 1
                else:
                    subs = str(ww[int(expr[i + 1]+expr[i+2])])
                    del expr[i + 2], expr[i+1], expr[i]
                    expr.insert(i, subs)
                    llen = len(expr)  # llen - 1
            i += 1
        expr = "".join(expr)
        return expr

class HallofFame(object):
    def __init__(self, maxsize, similar=eq):
        self.items = []
        self.maxsize = maxsize
        self.similar = similar

    def update(self, comb):
        comb = deepcopy(comb)
        if len(self) == 0 and self.maxsize !=0:
            # Working on an empty hall of fame is problematic for the
            # "for else"
            self.items.append(comb)

        elif comb.fitness < self[-1].fitness or len(self) < self.maxsize and not np.isnan(comb.fitness):
            if len(self) >= self.maxsize:
                self.remove(-1)
            self.items.append(comb)
            self.items = sorted(self.items, key=attrgetter("fitness"))

    def sort(self):
        self.items = sorted(self.items, key=attrgetter("fitness"))

    def insert_comb(self, item):

        def insert_fun(a, x):
            index = 0
            try:
                while x < a[index].fitness:
                    index += 1
            except IndexError:
                print("err")
            if index == len(a):
                index += 1
            return index

        item = deepcopy(item)
        i = insert_fun(self.items, item.fitness)
        self.items.insert(i, item)


    def remove(self, index):
        """Remove the specified *index* from the hall of fame.

        :param index: An integer giving which item to remove.
        """
        del self.items[index]

    def clear(self):
        """Clear the hall of fame."""
        del self.items[:]


    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __str__(self):
        return str(self.items)

class pointer(object):
    def __init__(self, index, fitness):
        self.index = index
        self.fitness = fitness

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

def find_best(pop, k):
    return sorted(pop, key=attrgetter("fitness"), reverse=True)[:k]

def clean_clones(elems):
    to_remove = []
    for i in range(len(elems)):
        for j in range(len(elems)):
            if elems[i] == elems[j] and i != j and j not in to_remove:
                if elems[j] not in to_remove:
                    to_remove.append(j)
    if to_remove:
        to_remove.sort()
        r = 0
        while r >= 0:
            elems.pop(to_remove[r])
            r = r - 1
    return elems

def multi_mutation(count, comb_pop, pop, nElem_max, evaluate_lst, evaluate_ind, Nind):
    #best_cpop = shuffle(tools.selBest(comb_pop, int(len(comb_pop)/2)))
    #start_comb = deepcopy(tools.selBest(comb_pop, 1))[0] #deepcopy(random.choice(comb_pop))
    '''delta = len(comb_pop) / Nind
    p = []
    for i in range(len(comb_pop)):
        p.append(pointer(i, comb_pop[i].fitness))
    p = sorted(p, key=attrgetter("fitness"), reverse=True)
    if count * delta >= len(comb_pop):
        start_comb = deepcopy(comb_pop[-1])
    else:
        start_comb = deepcopy(comb_pop[p[int(count * delta)].index])  # deepcopy(random.choice(comb_pop))'''

    start_comb = deepcopy(random.choice(comb_pop))
    new_comb = mutate_combination(start_comb, pop, unif=0.4, shr=0.33, nElem_max=nElem_max, evaluate_ind=evaluate_ind)
    w = []
    for i in range(len(new_comb.weights)):
        w.append(new_comb.weights[i])
    d = np.array(w)
    for i in new_comb.constants:
        d = np.hstack((d, float(i)))
    if len(d) != 0:
        #res = minimize(evaluate, d, args=(new_comb,), method='BFGS', options={'disp': False})
        res = least_squares(evaluate_lst, d, args=(new_comb,), method='lm', loss='linear')

    return [new_comb, res.x]

def multi_opt(count, comb_pop, evaluate, Nind):
    #best_cpop = shuffle(tools.selBest(comb_pop, int(len(comb_pop)/2)))
    '''delta = len(comb_pop)/Nind
    p = []
    for i in range(len(comb_pop)):
        p.append(pointer(i, comb_pop[i].fitness))
    p = sorted(p, key=attrgetter("fitness"), reverse=True)
    if count*delta >= len(comb_pop):
        comb = deepcopy(comb_pop[-1])
    else:
        comb = deepcopy(comb_pop[p[int(count*delta)].index]) #deepcopy(random.choice(comb_pop))'''
    comb = deepcopy(random.choice(comb_pop))
    w = []
    for i in range(len(comb.weights)):
        w.append(comb.weights[i])
    d = np.array(w)
    for i in comb.constants:
        d = np.hstack((d, float(i)))
    if len(d) != 0:
        #res = basinhopping(partial(evaluate, individual=comb), d, niter=20)
        res = minimize(evaluate, d, args=(comb,), method='BFGS', options={'disp': False})
    return [comb, res.x]

def multi_cross(count, comb_pop, pop, evaluate_ind, evaluate_lst):
    ch1 = deepcopy(random.choice(comb_pop))
    ch2 = deepcopy(random.choice(comb_pop))
    while ch1 == ch2:
        ch2 = deepcopy(random.choice(comb_pop))
    comb1, comb2 = cross_combination(ch1, ch2, pop, evaluate_ind)
    w = []
    if comb1.fitness < comb2.fitness:
        for i in range(len(comb1.weights)):
            w.append(comb1.weights[i])
        d = np.array(w)
        for i in comb1.constants:
            d = np.hstack((d, float(i)))
        if len(d) != 0:
            #res = minimize(evaluate, d, args=(comb1,), method='BFGS', options={'disp': False})
            res = least_squares(evaluate_lst, d, args=(comb1,), method='lm', loss='linear')
        which = 1
    else:
        for i in range(len(comb2.weights)):
            w.append(comb2.weights[i])
        d = np.array(w)
        for i in comb2.constants:
            d = np.hstack((d, float(i)))
        if len(d) != 0:
            #res = minimize(evaluate, d, args=(comb2,), method='BFGS', options={'disp': False})
            res = least_squares(evaluate_lst, d, args=(comb2,), method='lm', loss='linear')
        which = 2
    return [comb1, comb2, res.x, which]

def cross_combination(orig_comb1, orig_comb2, pop, evaluate_ind):
    comb1 = deepcopy(orig_comb1)
    comb2 = deepcopy(orig_comb2)
    node1 = random.randint(0, len(comb1.connections) - 1) # chose index of node for crossover in combination 1
    node2 = random.randint(0, len(comb2.connections) - 1) # chose index of node for crossover in combination 2
    nodes1 = comb1.connections[node1:]
    #weights1 = comb1.weights[node1:]
    nodes2 = comb2.connections[node2:]
    #weights2 = comb2.weights[node2:]
    comb1.connections = comb1.connections[:node1]
    comb1.connections.extend(nodes2)
    #comb1.weights = comb1.weights[:node1]
    #comb1.weights = np.hstack((comb1.weights, weights2))
    comb2.connections = comb2.connections[:node2]
    comb2.connections.extend(nodes1)
    #comb2.weights = comb1.weights[:node2]
    #comb2.weights = np.hstack((comb1.weights, weights1))
    comb1.connections = clean_clones(comb1.connections)
    comb1.weights = np.ones(len(comb1.connections)+1)
    comb1.weights[0] = 0.0
    comb2.connections = clean_clones(comb2.connections)
    comb2.weights = np.ones(len(comb2.connections)+1)
    comb2.weights[0] = 0.0

    comb1.update(pop, comb1.connections, comb1.weights, evaluate_ind)
    comb2.update(pop, comb2.connections, comb2.weights, evaluate_ind)

    return comb1, comb2

def mutate_combination(combination, pop, unif, shr, nElem_max, evaluate_ind):
    ch = random.random()
    if ch <= unif:
        combination = mutate_uniform(combination, pop, nElem_max, evaluate_ind)
    elif ch <= unif + shr and len(combination.connections) > 1:
        combination = mut_shrink(combination, pop, evaluate_ind)
    else:
        combination = mut_replacement(combination, pop, evaluate_ind)
    return combination

def mutate_uniform(combination, pop, nElem_max, evaluate_ind):
    '''Function to perform mutation on combination'''
    #indexes = list(np.linspace(0, len(pop) - 1, len(pop), dtype=int))

    insertion_node = random.randint(1, len(combination.connections))  # where to insert the new combination
    if nElem_max - insertion_node <= 0: # control to avoid errors and respect the max number of nodes
        n_conn = 0
    elif nElem_max-insertion_node == 1:
        n_conn = 1
    else:
        n_conn = random.randint(1, nElem_max-insertion_node)  # select how many new nodes to insert
    indexes = selBest(pop, len(pop))  # select the best individuals in the population
    nodes_list = []
    for i in range(len(pop)):
        if pop[i] in indexes:
            nodes_list.append(i)  # recover the indexes of the best individuals
    for nod in nodes_list:
        if nod in combination.connections:
            nodes_list.remove(nod)  # remove from the best individuals the nodes present in the original combination

    new_conn = []
    for i in range(insertion_node):
        if combination.connections[i] not in new_conn:
            new_conn.append(combination.connections[i])  # add to the new connections list the old nodes up to the insertion node
            #indexes.remove(combination.connections[i])

    new_nodes = random.sample(nodes_list, n_conn)  # select the new nodes from the list of the best individuals
    new_conn.extend(new_nodes)  #
    new_conn = clean_clones(new_conn)
    weights = np.ones(len(new_conn)+1)
    weights[0] = 0.0
    combination.update(pop, new_conn, weights, evaluate_ind)

    return combination

def mut_replacement(combination, pop, evaluate_ind):
    n_replace = random.randint(1, len(combination.connections))  # how many nodes to replace
    chosen_nodes = random.sample(combination.connections, n_replace) # which nodes are replaced
    #nodes_list = list(np.linspace(0, len(pop)-1, len(pop), dtype=int))  # all indexes in the population
    inds_list = selBest(pop, len(pop))
    nodes_list = []
    for i in range(len(pop)):
        if pop[i] in inds_list:
            nodes_list.append(i)
    #for ch_n in chosen_nodes:
     #   if ch_n in nodes_list:
      #      nodes_list.remove(ch_n)
    #if len(chosen_nodes) == 1:
     #   nodes_list.remove(chosen_nodes)  # remove the chosen node from the index list so to avoid replacing the old node with the same node
    #else:
    for nod in nodes_list:
        if nod in combination.connections:
            nodes_list.remove(nod) # remove the chosen nodes from the index list so to avoid replacing the old node with the same node

    new_nodes_indexes = random.sample(nodes_list, n_replace)  # choose the new nodes
    nn = 0
    combination.weights[0] = 0.0  # set value of first weight (offset)
    for i in range(len(combination.connections)):
        if combination.connections[i] in chosen_nodes:
            combination.connections[i] = int(new_nodes_indexes[nn])  # replace old node with new one
            if i != 0:
                combination.weights[i] = 1.0 # reset values of weights
            nn += 1
    combination.update(pop, combination.connections, combination.weights, evaluate_ind)
    return combination

def mut_shrink(combination, pop, evaluate_ind):
    to_remove = random.randint(1, len(combination.connections) - 1)
    if to_remove > 0:
        while to_remove > 0:
            rem = random.choice(combination.connections)
            combination.connections.remove(rem)
            to_remove = to_remove - 1
        new_conn = combination.connections
        weights = list(np.ones(len(new_conn)+1))
        weights[0] = 0.0
        combination.update(pop, new_conn, weights, evaluate_ind)
    return combination

def create_combinations(pop, nElem_max, evaluate_ind):
    '''Function to crate the initial combination'''
    combinations = Combination()
    indexes = list(np.linspace(0, len(pop) - 1, len(pop), dtype=int))
    n_elem = random.randint(1, nElem_max)
    connections1 = list(random.sample(indexes, n_elem))
    connections2 = clean_clones(connections1)
    while len(connections1) != len(connections2):
        connections1 = list(random.sample(indexes, n_elem))
        connections2 = clean_clones(connections1)
    connections = connections2
    weights = list(np.ones((n_elem+1)))
    weights[0] = 0.0
    combinations.update(pop, connections, weights, evaluate_ind=evaluate_ind)
    return combinations

def update_comb_afterOpt(pop, comb, d, toolbox, evaluate_ind):
    pop_new = deepcopy(pop)
    up_comb = deepcopy(comb)
    n_w = len(up_comb.weights)
    # update weights of comb with optimized ones
    up_comb.weights = d[0:n_w]
    extra_inds = []
    if len(d) > n_w:
        # update constants with optimized ones
        up_comb.constants = []
        for i in d[n_w:]:
            up_comb.constants.append(str(i))
        # update individuals with new constants
        c = 0
        for i in range(len(up_comb.ind)):
            for j in range(len(up_comb.ind[i])):
                if type(up_comb.ind[i][j]) == gp.Terminal and up_comb.ind[i][j].name[0] != "A":
                    up_comb.ind[i][j] = deepcopy(up_comb.ind[i][j])
                    up_comb.ind[i][j].value = float(up_comb.constants[c])
                    up_comb.ind[i][j].name = up_comb.constants[c]
                    c += 1
            # update population with new individuals
            new_ind = deepcopy(up_comb.ind[i])
            del new_ind.fitness.values
            pop_new[up_comb.connections[i]] = deepcopy(new_ind)
            extra_inds.append(new_ind)
    # update fitness of optimized memebers of population
    invalid_ind = [ind for ind in pop_new if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    invalid_ind = [ind for ind in extra_inds if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    up_comb.update(pop_new, up_comb.connections, up_comb.weights, evaluate_ind)
    return up_comb, extra_inds

def shuffle(arr):
    arr_start = deepcopy(arr)
    arr_end = []
    while len(arr_start) > 0:
        ind = random.randint(0, len(arr_start) - 1)
        arr_end.append(arr_start[ind])
        arr_start.pop(ind)
    return arr_end

def varOr(pop, toolbox, lambda_, cxpb, mutpb):

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(pop, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            if ind1.fitness.values[0] < ind2.fitness.values[0]:
                del ind1.fitness.values
                offspring.append(ind1)
            else:
                del ind2.fitness.values
                offspring.append(ind2)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(pop))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(tools.selBest(pop, 1)[0])

    return offspring

def protDiv(left, right):
    if type(right) == int:
        right = float(right)
    if type(right) != float:
        if any(right) == 0:
            return left
        else:
            return left/right
    else:
        if right == 0:
            return left
        else:
            return left/right

def comb_selDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first, fit_attr="fitness"):

    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            lind1 = sum([len(gpt) for gpt in ind1.ind])
            lind2 = sum([len(gpt) for gpt in ind2.ind])
            if lind1 > lind2:
                ind1, ind2 = ind2, ind1
            elif lind1 == lind2:
                # random selection in case of a tie
                prob = 0.5

            # Since size1 <= size2 then ind1 is selected
            # with a probability prob
            chosen.append(ind1 if random.random() < prob else ind2)

        return chosen

    def _fitTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            aspirants = select(individuals, k=fitness_size)
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)

def check_length(comb_in, nElem_max, pop, evaluate_ind):
    comb = deepcopy(comb_in)
    cut = False
    while len(comb.connections) > nElem_max:
        del comb.connections[-1]
        np.delete(comb.weights, -1)
        cut = True
    if cut is True:
        comb.update(pop, comb.connections, comb.weights, evaluate_ind)
    return comb

def subset_diversity(population):
    cat_number = 10  # here the number of categories is selected
    lens = []
    categories = {}
    distribution = []
    distr_stats = {}
    invalid_ind = []

    for ind in population:
        lens.append(len(ind))
    cat = np.linspace(min(lens), max(lens), cat_number + 1)
    useful_ind = np.linspace(0, len(cat) - 2, len(cat) - 1)

    for i in range(len(cat) - 1):
        categories["cat{}".format(i)] = []
    for ind in population:
        for i in range(len(cat) - 1):
            if len(ind) >= cat[i] and len(ind) <= cat[i + 1]:
                categories["cat{}".format(i)].append(ind)

    for i in range(len(cat) - 1):
        if categories["cat{}".format(i)] == []:
            invalid_ind.append(i)
        distribution.append(len(categories["cat{}".format(i)]))
    distr_stats["individuals"] = distribution
    distr_stats["percentage"] = np.array(distribution) / len(population)
    categories["distribution"] = distr_stats
    if invalid_ind != []:
        useful_ind = np.delete(useful_ind, invalid_ind, 0)
    return categories, np.asarray(useful_ind, dtype=int)

class POP(object):
    '''This class is used to collect data about a population. Used at the beginning for the selection of the initial
        population. Entropy measure comes from [1]'''

    def __init__(self, population):
        self.items = list()
        self.max = len(max(population, key=len))
        self.min = len(min(population, key=len))
        self.maxDiff = self.max - self.min
        self.categories, self.indexes = subset_diversity(population)
        self.entropy = -sum(
            self.categories["distribution"]["percentage"] * np.log(self.categories["distribution"]["percentage"]))  # entropy evaluation from [1]
        for ind in population:
            item = deepcopy(ind)
            self.items.append(item)

    def output_stats(self, Title):
        print("\n")
        print("---------- {} STATISTICS ------------".format(Title))
        print("-- Min len: {}, Max len: {}, Max Diff: {} ---".format(self.min, self.max, self.maxDiff))
        print("-- Entropy: {0:.3f} -------------------------".format(self.entropy))
        print("-- Distribution: {} --------------------".format(self.categories["distribution"]["percentage"] * 100))
        print("---------------------------------------------")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        print("it is used")
        return reversed(self.items)

    def __str__(self):
        return str(self.items)

def optimize_ind(ind, evaluate):
    d = []
    for j in range(len(ind)):
        if type(ind[j]) == gp.Terminal and ind[j].name[0] != "A":
            d.append(ind[j].value)
    if len(d) != 0:
        res = minimize(evaluate, d, args=(ind,), method='BFGS', options={'disp':False})
    else:
        return ind
    c = 0
    for j in range(len(ind)):
        if type(ind[j]) == gp.Terminal and ind[j].name[0] != "A":
            ind[j] = deepcopy(ind[j])
            ind[j].value = float(res.x[c])
            ind[j].name = str(res.x[c])
            c += 1

    ind.fitness.values = (res.fun,)
    return ind


def selBest(individuals, k, fit_attr="fitness"):
    return sorted(individuals, key=attrgetter(fit_attr), reverse=False)[:k]
