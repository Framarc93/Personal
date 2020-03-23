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
from copy import deepcopy
from operator import eq
import re
from scipy.optimize import minimize
import sys

sys.path.insert(1, "/home/francesco/Desktop/PhD/Git_workspace/IC4A2S/Goddard_Problem/Testing/Goddard_1Control")
import GP_PrimitiveSet as gpprim
def varOr(pop_cx, pop_mut, toolbox, lambda_, cxpb, mutpb):

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            if len(pop_cx) > 1:
                ind1, ind2 = map(toolbox.clone, random.sample(pop_cx, 2))
            else:
                ind1, ind2 = map(toolbox.clone, random.sample(pop_mut, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(pop_mut))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(pop_cx))

    return offspring

class Combination(object):
    def __init__(self):
        self.ind = []
        self.root = []
        self.connections = []
        self.weights = []
        self.fitness = []
        self.equation = []
        self.constants = []

    def update(self, item, root, connections, weights):
        self.ind.append(item)
        self.root.append(root)
        self.connections.append(connections)
        self.weights.append(weights)
        for ind in item:
            if type(ind) == gp.Terminal and ind.name[0] != "A":
                self.constants.append(ind.name)

    def build_eq(self, pop, new):
        ''' setup to build equations in the form:
        39 + w0*34 + w1*40 + w2*38 + w3*4
        if roots: 39, 34, 40, 4
        connections: 34, (40 38), 4'''
        cw = 0
        expr_list = []
        expr = str(self.root[0]) + '+'

        for r in range(len(self.connections)):  # this part creates the equation with weights and indices

            #if len(self.connections[r]) == 2:
             #   expr = expr + "Add("
            #elif len(self.connections[r]) == 3:
             #   expr = expr + "TriAdd("
            for c in range(len(self.connections[r])):
                expr = expr + "mul(" + "w{}".format(cw) + "," + str(self.connections[r][c]) + ")" + "+"
                cw += 1
        expr = list(expr)
        #if len(self.connections[r]) > 1:
         #   expr[-1] = ")"
        #else:
        del expr[-1]

        expr = "".join(expr)
        #expr_list.append(expr)
        #expr_list[0] = list(expr_list[0])
        #if "+" in expr_list[0]:
         #   while expr_list[0][0] != "+":
          #      del expr_list[0][0]
           # del expr_list[0][0]
        for l in range(len(expr_list)-1):
            expr_list[0] = list(expr_list[0])
            val2 = re.findall('\d+', "".join(expr_list[l + 1]))
            if len(val2) != 1:
                wh = len(val2[0])-1
                for e in range(len(expr_list[0])):
                    if expr_list[0][e] == val2[0][0]:
                        comp = expr_list[0][e]
                        for ccc in range(wh):
                            comp = comp + expr_list[0][e+ccc+1]
                        if wh!=0 and comp == val2[0]:
                            #expr_list[0][e] = "("
                            if len(expr_list[l+1]) == len(val2[0]):
                                break
                            else:
                                for ccc in range(len(val2[0])):
                                    del expr_list[0][e]
                                for j in range(len(expr_list[l+1])-2-wh):
                                    expr_list[0].insert(e+j, expr_list[l + 1][wh+2+j])
                                break
                        elif wh == 0 and expr_list[0][e] == val2[0] and not expr_list[0][e+1].isdigit() and expr_list[0][e-1] != "w":
                            #expr_list[0][e] = "("
                            if len(expr_list[l+1]) == 1:
                                break
                            else:
                                del expr_list[0][e]
                                for j in range(len(expr_list[l + 1])-2-wh):
                                    expr_list[0].insert(e + j, expr_list[l + 1][wh+2+j])
                                break
                #expr_list[0] = str(expr_list[0])
                expr_list[0] = "".join(expr_list[0])

            else:
                continue
        for i in range(len(expr_list[0])):
            if expr_list[0][i] == "," and expr_list[0][i+1] == "," and i < len(expr_list[0])-1:
                del expr_list[0][i]

        expr = list(expr_list[0]) # this part put the trees equations in place of the indices
        llen = len(expr)
        i = 0
        while i < llen:
            try:
                if expr[i].isdigit() and expr[i-1] != "w":
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
                        llen = llen - (j - 1)

                    for d in range(len(val)):
                        del expr[i]
                    if new is True:
                        expr.insert(i, str(pop[int(val)]))
                    else:
                        for ii in range(len(self.root)):
                            if val == str(self.root[ii]):
                                expr.insert(i, str(self.ind[ii]))
            except(IndexError):
                if expr[i].isdigit():
                    j = 1
                    val = expr[i]
                    digit = True
                    try:
                        while digit is True:
                            if expr[i + j].isdigit():
                                val = val + expr[i + j]
                                j += 1
                                if "".join(expr) == val:
                                    digit = False
                            else:
                                digit = False
                            llen = llen - (j - 1)
                    except(IndexError):
                        pass

                    for d in range(len(val)):
                        del expr[i]

                    if new is True:
                        expr.insert(i, str(pop[int(val)]))
                    else:
                        for ii in range(len(self.root)):
                            if val == str(self.root[ii]):
                                expr.insert(i, str(self.ind[ii]))
            i += 1

        expr = "".join(expr)
        # this part substitute the weights indeces with the values
        ww = []
        for i in range(len(self.weights)):
            ww.extend(self.weights[i])
        i = 0
        expr = list(expr)
        llen = len(expr)
        while i < llen:
            if expr[i] == "w":
                subs = str(ww[int(expr[i+1])])
                del expr[i+1], expr[i]
                expr.insert(i, subs)
                llen = llen - 1
            i += 1
        expr = "".join(expr)
        return expr

    def fit_eval(self, pop, new, update):
        if update is True:
            self.update_ind_const(pop)
        expr = self.build_eq(pop, new)
        fit = evaluate_p(expr)
        self.fitness = fit
        self.equation = expr

    def update_ind_const(self, pop):
        self.ind = []
        self.constants = []
        for i in range(len(self.root)):
            self.ind.append(pop[self.root[i]])
            for ind in pop[self.root[i]]:
                if type(ind) == gp.Terminal and ind.name[0] != "A":
                    self.constants.append(ind.name)



class HallofFame(object):
    def __init__(self, maxsize, similar=eq):
        self.items = []
        self.maxsize = maxsize
        self.similar = similar

    def update(self, comb):
        inserted = False
        if len(self) == 0 and self.maxsize !=0:
            # Working on an empty hall of fame is problematic for the
            # "for else"
            self.items.append(comb)

        elif comb.fitness < self[0].fitness or len(self) < self.maxsize:
            #for hofer in self:
             #   # Loop through the hall of fame to check for any
              #  # similar individual
               # if self.similar(comb, hofer):
                #    break
            #else:
                # The individual is unique and strictly better than
                # the worst
            if len(self) >= self.maxsize:
                self.remove(0)
            self.insert(comb)
            inserted = True
        return inserted


    def insert(self, item):

        def bisect_right(a, x, lo=0, hi=None):
            if lo < 0:
                raise ValueError('lo must be non-negative')
            if hi is None:
                hi = len(a)
            i = 0
            while lo < hi and i < len(a):
                mid = (lo + hi) // 2
                if x < a[i].fitness:
                     hi = mid
                else:
                    lo = mid + 1
                i += 1
            return lo

        item = deepcopy(item)
        i = bisect_right(self.items, item.fitness)
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

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


####   IMPORT DATA  ###########

mat_contents = sio.loadmat('dataModB118.mat')
files = mat_contents['VV']
input1 = np.zeros(len(files))
input2 = np.zeros(len(files))
output = np.zeros(len(files))
for i in range(len(files)):
    input1[i] = files[i][0]
    input2[i] = files[i][1]
    output[i] = files[i][2]

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


def modExp(x):
    if type(x) == int:
        x = float(x)
    if type(x) != float:
        if x.any()<5 and x.any()>-5:
            return np.exp(x)
        else:
            if x.any()>0:
                return np.exp(5)
            else:
                return np.exp(-5)
    else:
        if x<5 and x>-5:
            return np.exp(x)
        else:
            if x>0:
                return np.exp(5)
            else:
                return np.exp(-5)


def evaluate_p(individual):
    global input1, input2, output
    f = toolbox.compile(individual)
    try:
        output_eval = f(input1, input2)
    except(TypeError):
        print("e")
    err = np.sqrt(sum((output - output_eval) ** 2))
    return err


def evaluate(d, individual):
    global input1, input2, output

    c = 0
    for i in range(len(individual.weights)):
        for j in range(len(individual.weights[i])):
            individual.weights[i][j] = d[c]
            c += 1
    for i in range(len(individual.constants)):
        try:
            pset.terminals[object][i + nVars].value = d[i + c]
            pset.terminals[object][i + nVars].name = str(d[i + c])
        except(IndexError):
            pset.addTerminal(d[i+c])
    cc = 0
    for i in range(len(individual.ind)):
        for j in range(len(individual.ind[i])):
            if type(individual.ind[i][j]) == gp.Terminal and individual.ind[i][j].name[0] != "A":
                if individual.ind[i][j].name == individual.constants[0]:
                    individual.ind[i][j].name = str(d[cc + c])
                    individual.ind[i][j].value = d[cc + c]
                    cc += 1
    individual.fit_eval(pop, new=False, update=False)
    f = toolbox.compile(individual.equation)
    output_eval = f(input1, input2)
    err = np.sqrt(sum((output-output_eval)**2))
    return err

def evaluate_forplot(comb):
    global input1, input2, output
    f = toolbox.compile(comb.equation)
    output_eval = f(input1, input2)
    if np.shape(output_eval) == ():
        output_eval = output_eval * np.ones((len(input1)))
    return output_eval

####     PARAMETERS DEFINITION  #######

limit_height = 3  # height limit on individual tree
limit_size = 4  # size limit on individual tree
Ngen = 500  # maximum number of generations
cxpb = 0.8
mutpb = 0.2
nCost = 2  # number of constants created as terminals for tree
nVars = 2  # number fo variables
maxArity = 2  # how many connections can be established between nodes
nElem_max = 5  # maximum number of trees involved in the final equation
Nmut = 10  # maximum number of connections mutations performed at each generation


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
pset.addPrimitive(modExp, 1)
pset.addPrimitive(gpprim.Sin, 1)
pset.addPrimitive(gpprim.Cos, 1)


for i in range(nCost):
    val = random.uniform(-5, 5)
    pset.addTerminal(val)
#pset.addTerminal(1.0)
pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')

################################################## TOOLBOX #############################################################

creator.create("Fitness", base.Fitness, weights=(-1.0,))#, -0.1, -0.08, -1.0))    # MINIMIZATION OF THE FITNESS FUNCTION
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genFull, pset=pset, type_=pset.ret, min_=1, max_=3)  ### NEW ###
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  ### NEW ###
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ### NEW ###
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)
#toolbox.register("select", tools.selDoubleTournament, fitness_size=2, parsimony_size=1.4, fitness_first=True)#InclusiveTournament)
toolbox.register("mate", gp.cxOnePoint) ### NEW ##
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3) ### NEW ###
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) ### NEW ###
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limit_height))
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limit_size))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limit_size))

pop = toolbox.population(50)  # population creation

def mutate_combination(combination, pop):
    '''Function to perform mutation on combination'''
    choice = random.random()
    used = combination.root
    avlbl = [i for i in range(len(pop)) if i not in used]
    new = random.choice(avlbl)
    if len(used) > 1:
        old = random.choice(used[1:])
    else:
        old = used[0]
        choice = 0.1
    #if choice <= 0.5:
    for i in range(len(combination.root)):
        if combination.root[i] == old:
            combination.root[i] = new
    for i in range(len(combination.connections)):
        for j in range(len(combination.connections[i])):
            if combination.connections[i][j] == old:
                combination.connections[i][j] = new
    else: # choice <= 0.8:
        for i in range(len(combination.root)):
            if combination.root[i] == old:
                combination.root.insert(i, new)
                combination.connections.insert(i, [])

    '''else:
        for i in range(len(combination.root)):
            if combination.root[i] == old:
                del combination.root[i]
                break
        for j in range(len(combination.connections)):
            for h in range(len(combination.connections[j])):
                if combination.connections[j][h] == old:
                    del combination.connections[j][h]
                    break
        if len(combination.connections) != len(combination.root):
            del combination.connections[i]'''
    return combination

def create_combinations(pop):
    '''Function to crate the initial combination'''
    combinations = Combination()
    elem = 1
    indexes = list(np.linspace(0, len(pop) - 1, len(pop), dtype=int))
    root = random.choice(indexes)
    indexes.remove(root)
    arity = random.randint(1, maxArity)
    connections = list(random.sample(indexes, arity))
    indexes = [ele for ele in indexes if ele not in connections]
    elem += arity
    weights = list(np.ones((arity)))
    combinations.update(pop[root], root, connections, weights)
    leaf = []
    if connections:
        while elem < nElem_max:# and len(combinations.constants) < nCost-3:
            leaf.extend(connections)
            arity = random.randint(1, maxArity)
            connections = list(random.sample(indexes, arity))
            indexes = [ele for ele in indexes if ele not in connections]
            elem += arity
            weights = list(np.ones((arity)))
            combinations.update(pop[leaf[0]], leaf[0], connections, weights)
            del leaf[0]
        for i in connections:
            combinations.update(pop[i], i, [], [])
    return combinations

#############################   CREATION OF INITIAL COMBINATION     ##############################
init_combination = create_combinations(pop)
while len(init_combination.connections) == 1:  # to avoid having single node indivduals
    init_combination = create_combinations(pop)
init_combination.fit_eval(pop, new=False, update=False)  # evaluation of fitness function of combination

w = []  # creation of array of parameters to optimize
for i in range(len(init_combination.weights)):
    for j in range(len(init_combination.weights[i])):
        w.append(init_combination.weights[i][j])
d = np.array(w)

for i in init_combination.constants:
    d = np.hstack((d, float(i)))

### parameters optimization through least square
if len(d) != 0:
    res = minimize(evaluate, d, args=(init_combination,), method='BFGS')
    best_fit = res.fun
    c = 0
    #for i in range(len(init_combination.weights)):
     #   for j in range(len(init_combination.weights[i])):
      #      init_combination.weights[i][j] = lst.x[c]
       #     c += 1
    #for i in range(len(init_combination.constants)):
     #   init_combination.constants[i] = lst.x[i + c]
    init_combination.fit_eval(pop, new=False, update=True)
else:
    best_fit = init_combination.fitness


combination = deepcopy(init_combination)
best_combination = combination
hof_comb = HallofFame(15)  # initialization of hall of fame to store best combinations
hof_comb.update(combination)
good = []
cx_inds = []  # individuals to use for crossover (best)
mt_inds = []  # individuals to use for mutation (worst)
for i in range(len(hof_comb[0].root)):
    good.append(hof_comb[0].root[i])
for i in range(len(pop)):
    if i in good:
        cx_inds.append(pop[i])
    else:
        mt_inds.append(pop[i])
n = 0
while n <= Ngen:
    offspring = varOr(cx_inds, mt_inds, toolbox, len(pop), cxpb=cxpb, mutpb=mutpb) # creation of offspring through crossover and mutation among individuals
    pop[:] = offspring[:] #toolbox.select(offspring+pop, len(pop))
    combination.fit_eval(pop, new=True, update=True)  # update combination with new individuals
    ins = []
    old_comb = deepcopy(combination)
    nmut = random.randint(1, Nmut)
    for m in range(nmut):
        combination = mutate_combination(combination, pop)
        w = []
        for i in range(len(combination.weights)):
            for j in range(len(combination.weights[i])):
                w.append(combination.weights[i][j])
        d = np.array(w)
        for i in combination.constants:
            d = np.hstack((d, float(i)))
        if len(d) != 0:
            res = minimize(evaluate, d, args=(combination,), method='BFGS')

            best_fit = res.fun
            c = 0
            for i in range(len(combination.weights)):
                for j in range(len(combination.weights[i])):
                    combination.weights[i][j] = res.x[c]
                    c += 1
            for i in range(len(combination.constants)):
                combination.constants[i] = res.x[i+c]
            combination.fit_eval(pop, new=False, update=False)
        else:
            best_fit = combination.fitness
        ins.append(hof_comb.update(combination))

    new_comb = create_combinations(pop)
    w = []
    for i in range(len(new_comb.weights)):
        for j in range(len(new_comb.weights[i])):
            w.append(new_comb.weights[i][j])
    d = np.array(w)
    for i in new_comb.constants:
        d = np.hstack((d, float(i)))
    if len(d) != 0:
        res = minimize(evaluate, d, args=(new_comb,), method='BFGS')
        best_fit = res.fun
        c = 0
        for i in range(len(new_comb.weights)):
            for j in range(len(new_comb.weights[i])):
                new_comb.weights[i][j] = res.x[c]
                c += 1
        for i in range(len(new_comb.constants)):
            new_comb.constants[i] = res.x[i + c]
        new_comb.fit_eval(pop, new=False, update=False)
    else:
        best_fit = new_comb.fitness
    ins.append(hof_comb.update(new_comb))
    #if True not in ins:
     #   combination = deepcopy(old_comb)
    cx_inds = []
    mt_inds = []
    good = []
    for j in range(len(hof_comb)):
        for i in range(len(hof_comb[j].root)):
            good.append(hof_comb[j].root[i])
        for i in range(len(pop)):
            if i in good:
                cx_inds.append(pop[i])
            else:
                mt_inds.append(pop[i])
    best_fit = hof_comb[0].fitness
    print("GEN:{} BEST FIT:{}".format(n, best_fit))
    n += 1


oo = evaluate_forplot(hof_comb[0])
plt.figure()
plt.plot(output, 'o', label="Experimental data")
plt.plot(oo, marker='.', label="Fitted data", color='r')
plt.legend(loc="best")
plt.show()











