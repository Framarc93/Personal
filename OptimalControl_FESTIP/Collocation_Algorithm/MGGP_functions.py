import numpy as np
import random
from operator import attrgetter
from copy import deepcopy
from functools import wraps, partial
from deap import gp, tools
from scipy.optimize import minimize, least_squares
import sys

############## MODIFIED BLOAT CONTROL #########################################
def staticLimit(key, max_value):
    """Implement a static limit on some measurement on a GP tree, as defined
    by Koza in [Koza1989]. It may be used to decorate both crossover and
    mutation operators. When an invalid (over the limit) child is generated,
    it is simply replaced by one of its parents, randomly selected.

    This operator can be used to avoid memory errors occuring when the tree
    gets higher than 90 levels (as Python puts a limit on the call stack
    depth), because it can ensure that no tree higher than this limit will ever
    be accepted in the population, except if it was generated at initialization
    time.

    :param key: The function to use in order the get the wanted value. For
                instance, on a GP tree, ``operator.attrgetter('height')`` may
                be used to set a depth limit, and ``len`` to set a size limit.
    :param max_value: The maximum value allowed for the given measurement.
    :returns: A decorator that can be applied to a GP operator using \
    :func:`~deap.base.Toolbox.decorate`

    .. note::
       If you want to reproduce the exact behavior intended by Koza, set
       *key* to ``operator.attrgetter('height')`` and *max_value* to 17.

    .. [Koza1989] J.R. Koza, Genetic Programming - On the Programming of
        Computers by Means of Natural Selection (MIT Press,
        Cambridge, MA, 1992)

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                to_check = []
                for indd in new_inds[i]:
                    to_check.append(indd)
                check = max(to_check, key=key)
                j = 0
                while j<len(to_check)-1:
                    if key(ind[j]) == key(check):
                        break
                    j += 1
                if key(check) > max_value:
                    new_inds[i][j] = random.choice(keep_inds)[j]
            return new_inds

        return wrapper

    return decorator

def selBest(pop, k):
    return sorted(pop, key=attrgetter("fitness"), reverse=True)[:k]

def selDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first, fit_attr="fitness"):
    """Tournament selection which use the size of the individuals in order
    to discriminate good solutions. This kind of tournament is obviously
    useless with fixed-length representation, but has been shown to
    significantly reduce excessive growth of individuals, especially in GP,
    where it can be used as a bloat control technique (see
    [Luke2002fighting]_). This selection operator implements the double
    tournament technique presented in this paper.
    The core principle is to use a normal tournament selection, but using a
    special sample function to select aspirants, which is another tournament
    based on the size of the individuals. To ensure that the selection
    pressure is not too high, the size of the size tournament (the number
    of candidates evaluated) can be a real number between 1 and 2. In this
    case, the smaller individual among two will be selected with a probability
    *size_tourn_size*/2. For instance, if *size_tourn_size* is set to 1.4,
    then the smaller individual will have a 0.7 probability to be selected.
    .. note::
        In GP, it has been shown that this operator produces better results
        when it is combined with some kind of a depth limit.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fitness_size: The number of individuals participating in each \
    fitness tournament
    :param parsimony_size: The number of individuals participating in each \
    size tournament. This value has to be a real number\
    in the range [1,2], see above for details.
    :param fitness_first: Set this to True if the first tournament done should \
    be the fitness one (i.e. the fitness tournament producing aspirants for \
    the size tournament). Setting it to False will behaves as the opposite \
    (size tournament feeding fitness tournaments with candidates). It has been \
    shown that this parameter does not have a significant effect in most cases\
    (see [Luke2002fighting]_).
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.
    .. [Luke2002fighting] Luke and Panait, 2002, Fighting bloat with
        nonparametric parsimony pressure
    """
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
            chosen.append(max(aspirants, key=attrgetter(fit_attr)))
        return chosen

    if fitness_first:
        tfit = partial(_fitTournament, select=tools.selRandom)
        return _sizeTournament(individuals, k, tfit)
    else:
        tsize = partial(_sizeTournament, select=tools.selRandom)
        return _fitTournament(individuals, k, tsize)


def varOr(population, toolbox, lambda_, cxpb, mutpb):

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
            offspring.append(ind1)
            offspring.append(ind2)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def xmate(ind1, ind2, Ngenes):
    choice = random.random()
    ch = random.randint(0, Ngenes-1)
    #if choice <= 0.8:
    #for i in range(len(ind1)):
     #   ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    #else:
     #   gene1 = ind1[ch]
      #  w1 = ind1.w[ch]
       # gene2 = ind2[ch]
        #w2 = ind2.w[ch]
        #ind1[ch] = deepcopy(gene2)
        #ind1.w[ch] = copy(w2)
        #ind2[ch] = deepcopy(gene1)
        #ind2.w[ch] = copy(w1)

    if choice <= 0.7:
        ind1[ch], ind2[ch] = gp.cxOnePoint(ind1[ch], ind2[ch])
    else: # 0.7 < choice <= 0.9:
        ch2 = Ngenes-ch-1
        ind1[ch], ind2[ch2] = gp.cxOnePoint(ind1[ch], ind2[ch2])
        ind1[ch2], ind2[ch] = gp.cxOnePoint(ind1[ch2], ind2[ch])

    return ind1, ind2


def xmut(ind, expr, pset, unipb, shrpb, Ngenes):
    ch = random.randint(0, Ngenes - 1)
    choice = random.random()
    #if choice < opti:
     #   ind = deepcopy(ind)
      #  ind = optimize_ind(ind)
    if choice < unipb:
        indx1 = gp.mutUniform(ind[ch], expr, pset=pset)
        ind[ch] = indx1[0]
    elif choice < unipb + shrpb :
        indx1 = gp.mutShrink(ind[ch])
        ind[ch] = indx1[0]
    else:
        indx1 = gp.mutInsert(ind[ch], pset=pset)
        ind[ch] = indx1[0]
    return ind,

def protDiv(left, right):
    try:
        x = left / right
        return x
    except (RuntimeError, RuntimeWarning, TypeError, ArithmeticError, BufferError, BaseException, NameError, ValueError, FloatingPointError, OverflowError):
        return left

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

def sigmoid(x):
    return 1/(1+np.exp(x))

def softplus(x):
    return np.log(1+np.exp(x))

def optimize_ind(ind, evaluate, Ngenes):

    d = ind.w
    vv = []
    for j in range(len(ind)):  # for loop to save variables to be optimized by least square
        for h in range(len(ind[j])):
            if type(ind[j][h]) == gp.Terminal and ind[j][h].name[0] != "A":
                vv.append(ind[j][h].value)
    for t in range(len(vv)):
        d = np.hstack((d, vv[t]))  # add variables to array with parameters to be optimzed
    #try:
    lst = minimize(evaluate, d, args=(ind,), method='BFGS')
    for i in range(len(lst.x)):
        if np.isnan(lst.x[i]):
            lst.x[i] = 0.0
    ind.w = lst.x[0:Ngenes + 1]  # update the linear combination weights after optmization
    fit = lst.fun  # np.sqrt(sum((lst.fun)**2))
    ind.fitness.values = fit,
    #except(ValueError, TypeError):
     #   ind.fitness.values = 1e5,

    return ind

def lst(ind, evaluate):
    d = ind.w
    for i in range(len(d)):
        if np.isnan(d[i]):
            d[i] = 0
        elif np.isinf(d[i]):
            d[i] = 1e5
    #try:
    #print("d", d)
    lst = least_squares(evaluate, d, args=(ind,), method='lm', loss='linear')
    #print("res", lst.x)
    ind.w = lst.x  # update the linear combination weights after optmization
    fit = np.sqrt(sum((lst.fun)**2))
    ind.fitness.values = fit,
    #except(ValueError, TypeError):
     #   ind.fitness.values = 1e5,
    return ind