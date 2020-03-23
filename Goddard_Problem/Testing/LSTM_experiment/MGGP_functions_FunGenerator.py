import numpy as np
import random
from operator import attrgetter, eq
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

class HallOfFame(object):
    """The hall of fame contains the best individual that ever lived in the
    population during the evolution. It is lexicographically sorted at all
    time so that the first element of the hall of fame is the individual that
    has the best first fitness value ever seen, according to the weights
    provided to the fitness at creation time.

    The insertion is made so that old individuals have priority on new
    individuals. A single copy of each individual is kept at all time, the
    equivalence between two individuals is made by the operator passed to the
    *similar* argument.

    :param maxsize: The maximum number of individual to keep in the hall of
                    fame.
    :param similar: An equivalence operator between two individuals, optional.
                    It defaults to operator :func:`operator.eq`.

    The class :class:`HallOfFame` provides an interface similar to a list
    (without being one completely). It is possible to retrieve its length, to
    iterate on it forward and backward and to get an item or a slice from it.
    """

    def __init__(self, maxsize, similar=eq):
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()
        self.similar = similar

    def shuffle(self):
        arr_start = deepcopy(self.items)
        arr_end = []
        while len(arr_start) > 0:
            ind = random.randint(0, len(arr_start) - 1)
            arr_end.append(arr_start[ind])
            arr_start.pop(ind)
        return arr_end

    def update(self, population, for_feasible):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        if len(self) == 0 and self.maxsize != 0 and len(population) > 0:
            # Working on an empty hall of fame is problematic for the
            # "for else"
            self.insert(population[0], for_feasible)

        if for_feasible is True:
            for ind in population:
                if ind.fitness.values[-1] == 0.0:  ### NEW PART - REMOVE IF DOESNT WORK ####
                    if self[0].fitness.values[-1] == 0.0:
                        if sum(ind.fitness.values) < sum(self[0].fitness.values) or len(self) < self.maxsize:
                            for hofer in self:
                                # Loop through the hall of fame to check for any
                                # similar individual
                                if self.similar(ind, hofer):
                                    break
                            else:
                                # The individual is unique and strictly better than
                                # the worst
                                if len(self) >= self.maxsize:
                                    self.remove(0)
                                self.insert(ind, for_feasible)
                    else:
                        for hofer in self:
                            # Loop through the hall of fame to check for any
                            # similar individual
                            if self.similar(ind, hofer):
                                break
                        else:
                            # The individual is unique and strictly better than
                            # the worst
                            if len(self) >= self.maxsize:
                                self.remove(0)
                            self.insert(ind, for_feasible)  #### END NEW PART ######
                elif (sum(ind.fitness.values) < sum(self[0].fitness.values)) or len(self) < self.maxsize:
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(0)
                        self.insert(ind, for_feasible)
        else:
            for ind in population:
                if ind.fitness.values[0] < 10 and ind.fitness.values[1] < 40 and sum(ind.fitness.values) < sum(
                        self[0].fitness.values) or len(self) < self.maxsize:
                    for hofer in self:
                        # Loop through the hall of fame to check for any
                        # similar individual
                        if self.similar(ind, hofer):
                            break
                    else:
                        # The individual is unique and strictly better than
                        # the worst
                        if len(self) >= self.maxsize:
                            self.remove(0)
                        self.insert(ind, for_feasible)

    def insert(self, item, for_feasible):
        """Insert a new individual in the hall of fame using the
        :func:`~bisect.bisect_right` function. The inserted individual is
        inserted on the right side of an equal individual. Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """

        def bisect_right(a, x, lo=0, hi=None):
            """Return the index where to insert item x in list a, assuming a is sorted.
            The return value i is such that all e in a[:i] have e <= x, and all e in
            a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
            insert just after the rightmost x already there.
            Optional args lo (default 0) and hi (default len(a)) bound the
            slice of a to be searched.
            """

            if lo < 0:
                raise ValueError('lo must be non-negative')
            if hi is None:
                hi = len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                '''must indentify 4 cases: if both are feasible, if the new is feasible and the one in the list is not, viceversa and if both are infeasible'''
                if for_feasible is True:
                    # 1st case: both are feasible
                    if x.values[-1] == 0 and a[mid].values[-1] == 0:
                        if sum(x.values) < sum(a[mid].values):
                            hi = mid
                        else:
                            lo = mid + 1
                    # 2nd case: value to insert is feasible, the one in the list is not
                    elif x.values[-1] == 0 and a[mid].values[-1] != 0:
                        hi = mid
                    # 3rd case: value to insert is not feasible, the one in the list is feasible
                    elif x.values[-1] != 0 and a[mid].values[-1] == 0:
                        lo = mid + 1
                    # 4th case: both are infeasible
                    elif x.values[-1] != 0 and a[mid].values[-1] != 0:
                        if x.values[-1] < a[mid].values[-1]:
                            hi = mid
                        else:
                            lo = mid + 1
                else:
                    if sum(x.values) < sum(a[mid].values):
                        hi = mid
                    else:
                        lo = mid + 1
            return lo

        item = deepcopy(item)
        i = bisect_right(self.keys, item.fitness)
        self.items.insert(len(self) - i, item)
        self.keys.insert(i, item.fitness)

    def remove(self, index):
        """Remove the specified *index* from the hall of fame.

        :param index: An integer giving which item to remove.
        """
        del self.keys[len(self) - (index % len(self) + 1)]
        del self.items[index]

    def clear(self):
        """Clear the hall of fame."""
        del self.items[:]
        del self.keys[:]

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

            lind1 = 0
            lind2 = 0
            for l in range(len(ind1)):
                for ll in range(len(ind1[l])):
                    lind1 += len(ind1[l][ll])
            for l in range(len(ind2)):
                for ll in range(len(ind2[l])):
                    lind2 += len(ind2[l][ll])
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
    for i in range(len(ind1)):
        if choice <= 0.7:
            ind1[i][ch], ind2[i][ch] = gp.cxOnePoint(ind1[i][ch], ind2[i][ch])
        else: # 0.7 < choice <= 0.9:
            ch2 = Ngenes-ch-1
            ind1[i][ch], ind2[i][ch2] = gp.cxOnePoint(ind1[i][ch], ind2[i][ch2])
            ind1[i][ch2], ind2[i][ch] = gp.cxOnePoint(ind1[i][ch2], ind2[i][ch])

    return ind1, ind2


def xmut(ind, expr, psetR, psetT, unipb, shrpb, Ngenes):
    ch = random.randint(0, Ngenes - 1)
    choice = random.random()
    #if choice < opti:
     #   ind = deepcopy(ind)
      #  ind = optimize_ind(ind)
    if choice < unipb:
        indx1 = gp.mutUniform(ind[0][ch], expr, pset=psetR)
        ind[0][ch] = indx1[0]
        indx2 = gp.mutUniform(ind[1][ch], expr, pset=psetT)
        ind[1][ch] = indx2[0]
    elif choice < unipb + shrpb :
        indx1 = gp.mutShrink(ind[0][ch])
        ind[0][ch] = indx1[0]
        indx2 = gp.mutShrink(ind[1][ch])
        ind[1][ch] = indx2[0]
    else:
        indx1 = gp.mutInsert(ind[0][ch], pset=psetR)
        ind[0][ch] = indx1[0]
        indx2 = gp.mutInsert(ind[1][ch], pset=psetT)
        ind[1][ch] = indx2[0]
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
    d = []
    for i in range(len(ind)):
        d.extend(ind[i].w)
    vv = []
    for j in range(len(ind)):  # for loop to save variables to be optimized by least square
        for h in range(len(ind[j])):
            for l in range(len(ind[j][h])):
                if type(ind[j][h][l]) == gp.Terminal and ind[j][h][l].name[0] != "A":
                    vv.append(ind[j][h][l].value)
    for t in range(len(vv)):
        d = np.hstack((d, vv[t]))  # add variables to array with parameters to be optimzed
    #try:
    lst = minimize(evaluate, d, args=(ind,), method='BFGS')
    ind[0].w = lst.x[0:Ngenes + 1]  # update the linear combination weights after optmization
    ind[1].w = lst.x[Ngenes+1:(Ngenes+1)*2]
    fit = lst.fun  # np.sqrt(sum((lst.fun)**2))
    c = (Ngenes+1)*2
    for i in range(len(ind)):
        for j in range(len(ind[i])):
            for k in range(len(ind[i][j])):
                if type(ind[i][j][k]) == gp.Terminal and ind[i][j][k].name[0] != "A":
                    ind[i][j][k] = deepcopy(ind[i][j][k])
                    ind[i][j][k].value = float(lst.x[c])
                    ind[i][j][k].name = str(lst.x[c])
                    c += 1
    ind.fitness.values = fit,
    #except(ValueError, TypeError):
        #ind.fitness.values = 1e5,

    return ind

def lst(ind, evaluate_lst, Cd_new, change_time, x_ini):
    d = []
    for i in range(len(ind)):
        d.extend(ind[i].w)
    #try:
    lst = least_squares(evaluate_lst, d, args=(ind, Cd_new, change_time, x_ini), method='lm', loss='linear')
    ind[0].w = lst.x[:len(ind[0].w)]
    ind[1].w = lst.x[len(ind[1].w):]# update the linear combination weights after optmization
    fit = np.sqrt(sum((lst.fun)**2))
    ind.fitness.values = fit,
    #except(ValueError, TypeError):
     #   ind.fitness.values = 1e5,
    return ind