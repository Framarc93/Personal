import tkinter as tk
import time
from scipy.integrate import solve_ivp
import numpy as np
#import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
import operator
import pygraphviz as pgv
import random
from deap import gp
import matplotlib.pyplot as plt
import sys
import timeit
import pandas as pd
from functools import reduce
from operator import add, itemgetter
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
from scipy.interpolate import PchipInterpolator
from functools import partial


def xmate(ind1, ind2):
    i1 = random.randrange(len(ind1))
    i2 = random.randrange(len(ind2))
    ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
    return ind1, ind2


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr)
    ind[i1] = indx[0]
    return ind,


# Direct copy from tools - modified for individuals with GP trees in an array
def xselDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first):
    assert (1 <= parsimony_size <= 2), "Parsimony tournament size has to be in the range [1, 2]."

    def _sizeTournament(individuals, k, select):
        chosen = []
        for i in range(k):
            # Select two individuals from the population
            # The first individual has to be the shortest
            prob = parsimony_size / 2.
            ind1, ind2 = select(individuals, k=2)

            lind1 = sum([len(gpt) for gpt in ind1])
            lind2 = sum([len(gpt) for gpt in ind2])
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
            chosen.append(max(aspirants, key=getattr("fitness")))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)

# >>>> program part example
toolbox = base.Toolbox()
pset = gp.PrimitiveSet("MAIN", 2)
toolbox.register("map", map)
toolbox.register("expr", gp.genRamped, pset=pset, type_=pset.ret, min_=1, max_=2)
toolbox.register("leg", tools.initIterate, creator.SubIndividual, toolbox.expr)
toolbox.register("legs", tools.initRepeat, list, toolbox.leg, n=numLegs)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.legs)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("lambdify", gp.lambdify, pset=pset)
toolbox.register("stringify", gp.lambdify, pset=pset)

toolbox.register('evaluate', evalFitness, toolbox=toolbox, sourceData=data, minTrades=minTrades, log=False)
toolbox.register("select", xselDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=True)

toolbox.register("mate", xmate)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", xmut, expr=toolbox.expr_mut)

pop = toolbox.population(n=individuals)
hof = tools.HallOfFame(1)

pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=individuals, lambda_=individuals,
                          cxpb=0.5, mutpb=0.1, ngen=generations, stats=stats, halloffame=hof, verbose=True)



gen = log.select("gen")
fit_max = log.chapters["fitness"].select('max')

perform = []
p = 0
for items in fit_max:
    perform.append(fit_max[p][0])
    p = p + 1

size_avgs = log.chapters["size"].select("avg")
fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, perform, "b-", label="Maximum Fitness Performance")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
ax2.set_ylabel("Size", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")
textstr = ('Total Running Time:\n  %dh %dm %.3fs' % (hours, mins, secs))
ax1.text(0.65, 0.9, textstr, transform=ax1.transAxes, fontsize=10,
         horizontalalignment='right')

plt.savefig('Stats')
plt.show()

print("\n")
print("THE BEST VALUE IS:")
print(hof[0])
print("\n")
print("THE HEIGHT OF THE BEST INDIVIDUAL IS:")
print(hof[0].height)
print("\n")
print("THE SIZE OF THE BEST INDIVIDUAL IS:")
print(len(hof[0]))